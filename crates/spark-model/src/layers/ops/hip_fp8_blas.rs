// SPDX-License-Identifier: AGPL-3.0-only

//! gfx1151 FP8 block-scaled prefill projections as dequant-to-scratch + hipBLASLt.
//!
//! `w8a16_gemm_n_m128` dequantizes every B tile inside its K loop, once per
//! 128-row M tile: 16 times over at a 2048-row prefill chunk. Dequantizing the
//! whole weight once (`dequant_fp8_blockscaled_bf16`, E4M3 x block scale in
//! FP32 -> BF16 RNE, the same rounding the fused kernel applies before its
//! BF16 WMMA) and running a vendor BF16 GEMM is faster at every 27B FP8
//! projection shape. strix in-process bench, M=2048, medians of 7x5 (s61,
//! 2026-10-28):
//!
//! ```text
//!   shape (N x K)        fused w8a16_n_m128   dequant + rocBLAS bf16 TN
//!   GDN qkvz 16384x5120  18.3-19.2 ms         13.5 ms
//!   attn q   12288x5120  13.6-15.1 ms         10.0-10.9 ms
//!   attn k/v  1024x5120   1.2-1.3  ms          0.93 ms
//!   GDN out   5120x6144   7.2      ms          6.7 ms
//! ```
//!
//! Same BF16 operands, so only the FP32 accumulation order differs: 99.1% of
//! outputs bit-equal, relL2 2e-4. The scratch is one per stream, grown to the
//! largest projection seen (~168 MB for the 27B GDN qkvz) and never shrunk.
//! It is NOT a cached copy of the weights: those would cost ~14 GB on the 27B.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Rows at or above which the dequant + BLAS route is taken. The dequant is a
/// fixed per-call cost (~1 ms for the 27B qkvz), so short chunks stay on the
/// fused kernel. `ATLAS_HIP_FP8_BLAS_MIN_M=0` disables the route.
fn min_rows() -> u32 {
    static MIN: OnceLock<u32> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("ATLAS_HIP_FP8_BLAS_MIN_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
    })
}

/// The dequant kernel, preferring the 8-per-thread twin. `true` = `_v8`.
///
/// `dequant_fp8_blockscaled_bf16` maps one element to each thread and measured
/// 3.8 ms per launch on this route, 7.3% of a 16k 27B prefill (strix s69).
/// `dequant_fp8_bs128_bf16_v8` computes the same expression on 8 consecutive
/// elements per thread, so the scratch is bit-identical either way.
fn dequant_kernel(gpu: &dyn GpuBackend) -> Option<(KernelHandle, bool)> {
    static K: OnceLock<Option<(KernelHandle, bool)>> = OnceLock::new();
    *K.get_or_init(|| {
        if let Ok(k) = gpu.kernel("dequant_fp8_bs128_bf16_v8", "dequant_fp8_bs128_bf16_v8") {
            return Some((k, true));
        }
        gpu.kernel("dequant_fp8_blockscaled_bf16", "dequant_fp8_blockscaled_bf16")
            .map(|k| (k, false))
            .map_err(|e| tracing::warn!("hip fp8 prefill: no dequant kernel ({e}); fused FP8 GEMM only"))
            .ok()
    })
}

/// Per-stream BF16 scratch of at least `bytes`.
fn scratch(gpu: &dyn GpuBackend, stream: u64, bytes: usize) -> Result<DevicePtr> {
    static S: OnceLock<Mutex<HashMap<u64, (u64, usize)>>> = OnceLock::new();
    let mut map = S.get_or_init(Default::default).lock().unwrap_or_else(|p| p.into_inner());
    let slot = map.entry(stream).or_insert((0, 0));
    if slot.1 < bytes {
        if slot.0 != 0 {
            // Earlier work on this stream may still read the old buffer.
            gpu.synchronize(stream)?;
            gpu.free(DevicePtr(slot.0))?;
            *slot = (0, 0);
        }
        *slot = (gpu.alloc(bytes)?.0, bytes);
    }
    Ok(DevicePtr(slot.0))
}

/// `output[M,N] = input[M,K] @ dequant(weight[N,K], block_scale[N/128,K/128])^T`
/// via scratch + hipBLASLt. `Ok(false)`: not taken (short M, or a piece is
/// unavailable) and nothing was launched that the caller must account for.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_w8a16_n(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<bool> {
    let min = min_rows();
    if min == 0 || m < min {
        return Ok(false);
    }
    let Some((kernel, v8)) = dequant_kernel(gpu) else {
        return Ok(false);
    };
    let w_bf16 = match scratch(gpu, stream, n as usize * k as usize * 2) {
        Ok(p) => p,
        Err(e) => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| tracing::warn!("hip fp8 prefill: scratch alloc failed ({e}); fused FP8 GEMM"));
            return Ok(false);
        }
    };
    if v8 && k.is_multiple_of(8) {
        let threads = u64::from(n) * u64::from(k) / 8;
        KernelLaunch::new(gpu, kernel)
            .grid([threads.div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(weight)
            .arg_ptr(block_scale)
            .arg_ptr(w_bf16)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(div_ceil(k, 128)) // sk: scale row stride
            .launch(stream)?;
    } else if v8 {
        return Ok(false);
    } else {
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(k, 64), div_ceil(n, 4), 1])
            .block([64, 4, 1])
            .arg_ptr(weight)
            .arg_ptr(block_scale)
            .arg_ptr(w_bf16)
            .arg_u32(n)
            .arg_u32(k)
            .arg_u32(128) // block_n
            .arg_u32(128) // block_k
            .arg_u32(div_ceil(k, 128)) // sk: scale row stride
            .arg_u32(1) // scale_is_fp32
            .launch(stream)?;
    }
    match super::bf16_gemm(input, w_bf16.0, output, m, n, k, stream) {
        Ok(()) => Ok(true),
        Err(e) => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| tracing::warn!("hip fp8 prefill: BF16 GEMM unavailable ({e}); fused FP8 GEMM"));
            Ok(false)
        }
    }
}
