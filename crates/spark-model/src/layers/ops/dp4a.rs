// SPDX-License-Identifier: AGPL-3.0-only

//! W4A8 integer-DP4A decode GEMV dispatch (strix-hip / gfx1151 only).
//!
//! These wrap the additive DP4A kernels in
//! `kernels/strix-hip/common/w4a16_gemv_dp4a.cu`. They engage wherever the
//! target ships them (gfx1151 today; see `layers::dense_ffn`); the float
//! E2M1-LUT path (`w4a16_gemv*`) is untouched and remains the fallback on
//! every target. The win is on the bandwidth-bound LPDDR5X part: int8 v_dot4
//! (`__builtin_amdgcn_sudot4`) + branchless v_perm codebook replace per-weight
//! FP32 FMA, validated cosine 0.999991 vs the float oracle on real gfx1151.
//!
//! The activation int8 quant is HOISTED: it runs once per distinct activation
//! (gate/up share the post-norm input; down uses silu(gate)*up), not once per
//! GEMV — per-call quant is break-even, hoisted quant is the +12% lever.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// Group size: one weight scale + one activation scale per 16 elements.
/// SSOT with `DP4A_GROUP_SIZE` in `w4a16_gemv_dp4a.cu`.
pub const DP4A_GROUP_SIZE: u32 = 16;

/// Runtime gate for the W4A8 integer-DP4A decode path. ON by default:
/// accuracy-validated on gfx1151 (bfcl-subset 83.02/80.41 on the 995-row
/// golden draw; 511/511 M=1 argmax match vs the float path — see the strix
/// BENCH.toml notes) and every call site still requires resolved kernel
/// handles, so targets without the DP4A kernel set miss on `KernelHandle(0)`
/// and take the float E2M1-LUT path unchanged. `ATLAS_W4A16_DP4A=0` forces
/// the float path (rollback / A/B); `=1` is accepted for harness back-compat.
pub fn dp4a_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("ATLAS_W4A16_DP4A").as_deref() != Ok("0"))
}

/// `ATLAS_GEMV_VL2=0` kills the virtual-lane (VLANES=2) verify GEMV tier —
/// `_vl2` kernels map two of the original 64 logical lanes onto one physical
/// thread for 2x memory-level parallelism. Default ON on HIP; the vl2 kernels
/// only exist in the strix-hip target, so non-HIP callers never resolve them.
/// Bit-identical outputs; A/B from one binary.
pub fn gemv_vl2_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("ATLAS_GEMV_VL2").as_deref() != Ok("0"))
}

/// Eligibility for the guard-free M=4 DP4A batch4 GEMV arm. `m` must be
/// EXACTLY 4: `w4a16_gemv_dp4a_batch4_d4` writes all four rows
/// unconditionally, so dispatching at any other M would write rows the
/// caller never asked for. Both kernel handles must be resolved and the
/// int8 activation scratch must be non-null.
pub fn dp4a_batch4_eligible(
    m: u32,
    enabled: bool,
    quant_kernel: KernelHandle,
    gemv_kernel: KernelHandle,
    a_q: DevicePtr,
    a_scale: DevicePtr,
) -> bool {
    m == 4
        && enabled
        && quant_kernel.0 != 0
        && gemv_kernel.0 != 0
        && !a_q.is_null()
        && !a_scale.is_null()
}

/// One `tracing::info!` per process the first time any GDN/attention M=4
/// DP4A arm dispatches — the serve-log proof the arm engaged.
pub fn dp4a_arm_active_once() {
    static ON: std::sync::Once = std::sync::Once::new();
    ON.call_once(|| {
        tracing::info!("GDN/attention W4A8 DP4A M=4 arm ACTIVE");
    });
}

/// Quantize one BF16 activation row `[1, K]` to int8 `[1, K]` + per-16-group
/// f32 scales `[K/16]` (symmetric block-q8_1, d = amax/127). Hoisted: call once
/// per distinct activation, then feed `aq`/`a_scale` to one or more
/// [`w4a16_gemv_dp4a`] GEMVs.
///
/// Kernel: `quantize_act_int8_g16(A, a_q, a_scale, K)`
/// Grid: (K/16, 1, 1)  Block: (16, 1, 1)
pub fn quantize_act_int8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k, DP4A_GROUP_SIZE), 1, 1])
        .block([DP4A_GROUP_SIZE, 1, 1])
        .arg_ptr(input)
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn quantize_act_int8_batch4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k, DP4A_GROUP_SIZE), m, 1])
        .block([DP4A_GROUP_SIZE, 1, 1])
        .arg_ptr(input)
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)
}

/// Fused `silu(gate)*up` activation prep for the down-proj: materializes the
/// hidden then int8-quantizes it (identical math to the float
/// `w4a16_gemv_silu_input` inline activation + [`quantize_act_int8`]). Hoists the
/// down-proj input quant out of the GEMV.
///
/// Kernel: `silu_mul_quant_int8_g16(gate, up, a_q, a_scale, K)`
/// Grid: (K/16, 1, 1)  Block: (16, 1, 1)
pub fn silu_mul_quant_int8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(k, DP4A_GROUP_SIZE), 1, 1])
        .block([DP4A_GROUP_SIZE, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_u32(k)
        .launch(stream)
}

/// W4A8 integer-DP4A GEMV (M=1) from a PRE-QUANTIZED int8 activation.
/// `C[1,N] = A_int8 @ dequant(B)`. Same weight bandwidth/layout as the float
/// `w4a16_gemv`; the activation is int8 with per-16-group scales.
///
/// Kernel: `w4a16_gemv_dp4a(a_q, a_scale, B_packed, B_scale, scale2, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
pub fn w4a16_gemv_dp4a(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dp4a_batch4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Strided-output variant (`w4a16_gemv_dp4a_batch4_d4_os` /
/// `_dyn_os`): identical GEMV, but output row `t` lands at
/// `output[t*out_stride + col]` — the K=4..8 attention QKV path writes each
/// projection straight into its interleaved `qkv_buf` slice with this,
/// removing the per-row D2D scatter. `out_stride` is in BF16 ELEMENTS.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dp4a_batch4_os(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// vl2 twins of the batch8 DP4A GEMVs: identical args and signature, but the
/// kernel covers 8 outputs per 256-thread block — grid `ceil(n/8)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dp4a_batch8_vl2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dp4a_dual_batch8_vl2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    gate_weight: &QuantizedWeight,
    gate_out: DevicePtr,
    up_weight: &QuantizedWeight,
    up_out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_ptr(gate_weight.weight)
        .arg_ptr(gate_weight.weight_scale)
        .arg_f32(gate_weight.weight_scale_2)
        .arg_ptr(gate_out)
        .arg_ptr(up_weight.weight)
        .arg_ptr(up_weight.weight_scale)
        .arg_f32(up_weight.weight_scale_2)
        .arg_ptr(up_out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dp4a_dual_batch4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_q: DevicePtr,
    a_scale: DevicePtr,
    gate_weight: &QuantizedWeight,
    gate_out: DevicePtr,
    up_weight: &QuantizedWeight,
    up_out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_q)
        .arg_ptr(a_scale)
        .arg_ptr(gate_weight.weight)
        .arg_ptr(gate_weight.weight_scale)
        .arg_f32(gate_weight.weight_scale_2)
        .arg_ptr(gate_out)
        .arg_ptr(up_weight.weight)
        .arg_ptr(up_weight.weight_scale)
        .arg_f32(up_weight.weight_scale_2)
        .arg_ptr(up_out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dp4a_batch4_eligible_predicate() {
        let k = KernelHandle(1);
        let p = DevicePtr(1);
        // All conditions met -> true.
        assert!(dp4a_batch4_eligible(4, true, k, k, p, p));
        // Each single negation -> false.
        assert!(!dp4a_batch4_eligible(3, true, k, k, p, p));
        assert!(!dp4a_batch4_eligible(5, true, k, k, p, p));
        assert!(!dp4a_batch4_eligible(4, false, k, k, p, p));
        assert!(!dp4a_batch4_eligible(4, true, KernelHandle(0), k, p, p));
        assert!(!dp4a_batch4_eligible(4, true, k, KernelHandle(0), p, p));
        assert!(!dp4a_batch4_eligible(4, true, k, k, DevicePtr::NULL, p));
        assert!(!dp4a_batch4_eligible(4, true, k, k, p, DevicePtr::NULL));
    }
}
