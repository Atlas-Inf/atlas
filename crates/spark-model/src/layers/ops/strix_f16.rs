// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel picks for the gfx1151 F16-operand prefill GEMM twins
//! (`*_f16a` in `kernels/strix-hip/qwen3.6-{27b,35b-a3b}/nvfp4/`).
//!
//! Each twin has the SAME signature, grid and launcher as the BF16 kernel it
//! replaces, so the pick is a handle swap and every caller keeps its launch
//! code. Why the twins exist: gfx1151 has no f32->bf16 instruction, and the
//! BF16 kernels spend most of their hot loop rounding dequantized weights to
//! bf16 in software (829 instructions per 32-K step for 32 WMMA in
//! `w4a16_gemm_t_m128`; 688 for 16 in the routed MoE kernels). The twins
//! dequantize NVFP4 to f16 exactly (no rounding) and convert the bf16 A tile in
//! LDS (378 and 293 instructions). See the kernel headers.
//!
//! The twin lookup is issued only under `cfg!(atlas_hip)`: the kernels exist
//! only in `kernels/strix-hip/`, and a lookup that is never made leaves no
//! failed row in the boot audit (the #101 / #127 class).

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

use crate::layers::try_kernel;

fn f16_twin(gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    if cfg!(atlas_hip) {
        try_kernel(gpu, module, &format!("{func}_f16a"))
    } else {
        KernelHandle(0)
    }
}

/// `w4a16_gemm_t_m128`, or its F16 twin on gfx1151. Optional (0 if absent).
#[track_caller]
pub fn w4a16_m128_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    let h = f16_twin(gpu, "w4a16", "w4a16_gemm_t_m128");
    let base = try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128");
    if h.0 != 0 && base.0 != 0 {
        record_m128_pair(h, base);
        return h;
    }
    base
}

/// The dense twin now wins at every measured shape, so there is no cut by
/// default. The 40M N*K cut this replaces was fitted on the double-buffered
/// twin (strix job 146), where the BF16 base won the small weights. Since the
/// single-LDS + B-reuse twin (#235) it is the other way round. strix s104
/// (gfx1151, M=2048, three alternating rounds), base -> twin TF/s:
///
///   27B gate/up 17408x5120  20.7-23.1 -> 27.1-29.6   27B down 5120x17408   18.0-22.1 -> 29.2-29.4
///   FNext qkvz  16384x2560  20.5-22.3 -> 28.8-29.0   FNext attn q 12288x2560 20.7-22.4 -> 27.7-28.9
///   27B qkv      5120x5120  21.9-22.4 -> 28.2-28.7   FNext out    2560x6144  19.6-21.8 -> 25.4-28.0
///   FNext kv     1024x2560  18.2-18.5 -> 22.6-22.9
///
/// End to end, Flash-Next NVFP4 prefill is -3% at 1.9k / 7.4k / 12.5k
/// (framework f82, interleaved x3). The twin's dequant is exact in F16 where the
/// base rounds every weight to BF16, so outputs change: the f80 gate (ppl dense
/// 9.6654 -> 9.6383, above-bound 1.0000, needles 12/12) and the Qwen3.6-27B gate
/// (s105) cleared that. `ATLAS_F16_M128_MIN_NK` still restores a cut (e.g.
/// 40000000 for the old behaviour).
const F16_M128_MIN_NK_DEFAULT: u64 = 0;

static M128_PAIR: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();

fn record_m128_pair(twin: KernelHandle, base: KernelHandle) {
    let _ = M128_PAIR.set((twin.0, base.0));
}

fn f16_m128_min_nk() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_F16_M128_MIN_NK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(F16_M128_MIN_NK_DEFAULT)
    })
}

/// Pick the kernel for one `w4a16_gemm_t_m128` launch: the F16 twin on large
/// weights, its BF16 base below the cut. Any other handle passes through.
pub(crate) fn m128_for_shape(kernel: KernelHandle, n: u32, k: u32) -> KernelHandle {
    match M128_PAIR.get() {
        Some(&(twin, base)) if kernel.0 == twin && (n as u64) * (k as u64) < f16_m128_min_nk() => {
            KernelHandle(base)
        }
        _ => kernel,
    }
}

/// [`w4a16_m128_kernel`] for callers that require the kernel.
#[track_caller]
pub fn w4a16_m128_kernel_required(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    let h = f16_twin(gpu, "w4a16", "w4a16_gemm_t_m128");
    let base = gpu.kernel("w4a16", "w4a16_gemm_t_m128")?;
    if h.0 != 0 {
        record_m128_pair(h, base);
        return Ok(h);
    }
    Ok(base)
}

/// A routed-expert MoE prefill kernel (`moe_w4a16_fused_gate_up_t`,
/// `moe_w4a16_grouped_gemm_ptrtable_t`), or its F16 twin on gfx1151.
#[track_caller]
pub fn moe_w4a16_t_kernel(gpu: &dyn GpuBackend, func: &str) -> Result<KernelHandle> {
    let h = f16_twin(gpu, "moe_w4a16", func);
    if h.0 != 0 {
        return Ok(h);
    }
    gpu.kernel("moe_w4a16", func)
}

/// The `_f16a` twin of a routed-MoE prefill kernel (`moe_w4a16_fused_gate_up_t`,
/// `moe_w4a16_grouped_gemm_ptrtable_t`), or `KernelHandle(0)` when the target
/// does not ship it (or is not native HIP). Unlike [`moe_w4a16_t_kernel`] this
/// never falls back to the BF16 base — it reports twin-ness, which is what
/// engages the twins in production prefill (layers/moe/
/// forward_prefill_routed.rs): that path dispatches the `_k64` handles, so
/// the init-time swap alone engaged nothing (winbox B2 vs B1: TTFT 1.01x,
/// byte-identical greedy output).
///
/// The twins carry the SAME signatures as their BF16 bases and the same
/// 64-row `blockIdx.y` M tile with NO grid striding, so the n128 launchers
/// pass `max_m_tiles`, not `grid_m_strided`.
#[track_caller]
pub fn moe_w4a16_t_f16_twin(gpu: &dyn GpuBackend, func: &str) -> KernelHandle {
    f16_twin(gpu, "moe_w4a16", func)
}

/// Should routed-MoE prefill run the F16 twins? Only when BOTH resolved. The
/// pair is added and validated together — a half-resolved set is not a
/// supported configuration, so it keeps the BF16 `_k64` arms rather than
/// mixing operand widths across the gate_up/down boundary. Pure, for unit
/// testing.
pub fn moe_f16_prefill_engaged(gate_up_twin: KernelHandle, down_twin: KernelHandle) -> bool {
    gate_up_twin.0 != 0 && down_twin.0 != 0
}

/// Resolve the routed-MoE prefill pair for `MoeLayer::new_*`: returns
/// `(grouped_gemm_t, fused_gate_up_t, both_are_f16)` where each handle is the
/// `_f16a` twin when the target ships it else the BF16 base (required — every
/// MoE target ships the base kernels).
#[track_caller]
pub fn moe_f16_prefill_handles(gpu: &dyn GpuBackend) -> Result<(KernelHandle, KernelHandle, bool)> {
    let down = moe_w4a16_t_f16_twin(gpu, "moe_w4a16_grouped_gemm_ptrtable_t");
    let gate_up = moe_w4a16_t_f16_twin(gpu, "moe_w4a16_fused_gate_up_t");
    let both = moe_f16_prefill_engaged(gate_up, down);
    let down = if down.0 != 0 {
        down
    } else {
        gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_t")?
    };
    let gate_up = if gate_up.0 != 0 {
        gate_up
    } else {
        gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t")?
    };
    Ok((down, gate_up, both))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_f16_prefill_requires_both_twins() {
        assert!(moe_f16_prefill_engaged(KernelHandle(7), KernelHandle(9)));
        assert!(!moe_f16_prefill_engaged(KernelHandle(0), KernelHandle(9)));
        assert!(!moe_f16_prefill_engaged(KernelHandle(7), KernelHandle(0)));
        assert!(!moe_f16_prefill_engaged(KernelHandle(0), KernelHandle(0)));
    }

    // The only test that touches M128_PAIR, so the process-global set is safe.
    #[test]
    fn m128_shape_gate_matches_the_measured_wins() {
        record_m128_pair(KernelHandle(11), KernelHandle(22));
        if std::env::var_os("ATLAS_F16_M128_MIN_NK").is_some() {
            return; // an override changes the cut; the table below is for the default
        }
        let twin = KernelHandle(11);
        // Twin won (strix job 146): 27B gate/up, 27B down, Flash-Next GDN qkvz.
        for (n, k) in [(17408, 5120), (5120, 17408), (16384, 2560)] {
            assert_eq!(
                m128_for_shape(twin, n, k).0,
                11,
                "{n}x{k} should keep the twin"
            );
        }
        // Twin lost: Flash-Next attn q, 27B qkv, Flash-Next out.
        for (n, k) in [(12288, 2560), (5120, 5120), (2560, 6144)] {
            assert_eq!(
                m128_for_shape(twin, n, k).0,
                22,
                "{n}x{k} should fall back to bf16"
            );
        }
        // Any other handle passes through untouched.
        assert_eq!(m128_for_shape(KernelHandle(33), 2560, 6144).0, 33);
    }
}
