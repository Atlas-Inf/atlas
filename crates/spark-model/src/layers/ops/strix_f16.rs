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

/// The dense twin wins only on large weights. strix job 146 (gfx1151), BF16 vs
/// F16-twin TF/s at M=2048, N x K:
///
///   27B gate/up 17408x5120   17.68 -> 21.31   27B down 5120x17408   18.32 -> 20.71
///   FNext qkvz  16384x2560   17.49 -> 20.14   FNext attn q 12288x2560 21.65 -> 20.45
///   27B qkv      5120x5120   22.17 -> 21.77   FNext out    2560x6144  22.67 -> 20.58
///
/// The BF16 kernel already reaches ~22 TF/s on the smaller weights, where its
/// software bf16 rounding is not the limiter, and the twin's extra LDS pass for
/// the A tile then costs more than it saves. `N*K >= 40M` separates the six
/// measured shapes exactly; it is a fitted cut from six points, so
/// `ATLAS_F16_M128_MIN_NK` overrides it (0 = always the twin). The routed-MoE
/// twins are not gated: they won 2.1x / 1.9x on the Flash-Next shapes.
const F16_M128_MIN_NK_DEFAULT: u64 = 40_000_000;

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

#[cfg(test)]
mod tests {
    use super::*;

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
