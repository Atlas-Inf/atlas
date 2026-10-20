// SPDX-License-Identifier: AGPL-3.0-only

//! MXFP8 weight twins (OCP microscaling FP8): E4M3 values plus one E8M0
//! scale per 32 input columns.
//!
//! Two sources produce them. GLM quantizes BF16 projections into MXFP8 twins
//! at load (`ops::mxfp8_quantize`). Checkpoints can also SHIP them: ModelOpt's
//! `quant_algo: "MXFP8"` (nvidia Qwen3.8-Flash-Next-FP4-Local's whole
//! backbone) stores `X.weight` F8_E4M3 `[n, k]` next to `X.weight_scale` U8
//! UE8M0 `[n, k/32]` — byte-for-byte the untiled [`Mxfp8Weight`] layout.
//!
//! For those, `mxfp8_load::raw_mxfp8` is the per-tensor probe ("is this
//! projection MXFP8, and where are its bytes"), and [`dequant_mxfp8_to_bf16`]
//! is the exact BF16 expansion every existing BF16 arm can load unchanged.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{DenseWeight, RawMxfp8};
use crate::layers::ops::MXFP8_BLOCK;

/// MXFP8 twin of a projection: E4M3 `[n, k]` + E8M0 `[n, k/32]`.
#[derive(Debug, Clone, Copy)]
pub struct Mxfp8Weight {
    pub data: DevicePtr,
    pub scales: DevicePtr,
    /// The bytes are in `mxfp8_tile16`'s chunk-tiled layout
    /// (`ops::mxfp8_tile_if_requested`), served only by the `_tiled` tiers.
    pub tiled: bool,
}

/// E8M0 exponents whose scale keeps EVERY E4M3 value exact in BF16.
///
/// E4M3 values are multiples of 2^-9 with at most 4 significant bits, up to
/// 448 = 1.75 * 2^8. Times 2^s they stay exact in BF16 (8 significant bits,
/// f32's exponent range, subnormals down to 2^-133) iff `s - 9 >= -133` and
/// `8 + s <= 127`, i.e. `-124 <= s <= 119`, i.e. biased `3 <= e <= 246`.
/// Real checkpoints sit far inside it (FP4-Local: 111..=119), so anything
/// outside is refused rather than rounded: this path promises exactness.
const EXACT_E8M0: std::ops::RangeInclusive<u8> = 3..=246;

impl Mxfp8Weight {
    /// Untiled twin buffers for an `[n, k]` weight: E4M3 data `n*k` bytes and
    /// E8M0 scales `n*k/32`, the data freed again if the scales do not fit.
    pub fn alloc(gpu: &dyn GpuBackend, n: usize, k: usize) -> Result<Self> {
        let data = gpu.alloc(n * k)?;
        let scales = match gpu.alloc(n * k / MXFP8_BLOCK) {
            Ok(p) => p,
            Err(e) => {
                gpu.free(data)?;
                return Err(e);
            }
        };
        Ok(Self {
            data,
            scales,
            tiled: false,
        })
    }
}

/// UE8M0 scale bytes as f32 multipliers, refusing any that would make the
/// BF16 expansion inexact (see [`EXACT_E8M0`]; 0xFF is the NaN code).
pub(crate) fn mxfp8_scales_f32(scales: &[u8]) -> Result<Vec<f32>> {
    if let Some(at) = scales.iter().position(|e| !EXACT_E8M0.contains(e)) {
        bail!(
            "MXFP8 scale block {at} has E8M0 exponent {} (2^{}), outside the \
             {EXACT_E8M0:?} range where the BF16 expansion is exact",
            scales[at],
            scales[at] as i32 - 127
        );
    }
    Ok(scales
        .iter()
        .map(|&e| f32::from_bits((e as u32) << 23))
        .collect())
}

/// Host reference: MXFP8 `[n, k]` to BF16 bits. The kernel path below must
/// match it bit for bit; it is also what the golden test pins against
/// PyTorch's own `float8_e4m3fn -> float32 -> bfloat16`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn mxfp8_to_bf16_host(
    data: &[u8],
    scales: &[u8],
    n: usize,
    k: usize,
) -> Result<Vec<u16>> {
    ensure!(
        data.len() == n * k && scales.len() * MXFP8_BLOCK == n * k,
        "MXFP8 host dequant: {} data / {} scale bytes for [{n}, {k}]",
        data.len(),
        scales.len()
    );
    let mult = mxfp8_scales_f32(scales)?;
    Ok(data
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let v = atlas_core::numeric::fp8_e4m3_to_f32(b) * mult[i / MXFP8_BLOCK];
            // Exact by construction (`EXACT_E8M0`), so RNE never rounds here:
            // the result's low 16 f32 bits are zero.
            atlas_core::numeric::f32_to_bf16(v)
        })
        .collect())
}

/// Expand a checkpoint MXFP8 projection to a fresh BF16 `[n, k]` buffer.
///
/// EXACT: every E4M3 value times a power of two in [`EXACT_E8M0`] is
/// representable in BF16, so no rounding happens anywhere — the BF16 copy is
/// the checkpoint's own math, not an approximation of it.
///
/// Runs the shared `dequant_fp8_blockscaled_bf16` kernel (in `common/`, so
/// every kernel target has it) with a `1 x 32` block and the E8M0 bytes widened
/// to f32 on the host: the scale matrix is k/32 bytes a row, so the D2H, the
/// widening and the H2D are a few MB for the largest projection.
///
/// The source stays in the store: the MXFP8 decode twins (`mxfp8_load`) read
/// the same bytes, and the model loader releases them once both have
/// (`release_raw_mxfp8`).
pub fn dequant_mxfp8_to_bf16(gpu: &dyn GpuBackend, src: &RawMxfp8) -> Result<DenseWeight> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
    let (n, k) = (src.n, src.k);
    let mut e8m0 = vec![0u8; n * k / MXFP8_BLOCK];
    gpu.copy_d2h(src.scales, &mut e8m0)
        .context("MXFP8 scale D2H")?;
    let mult = mxfp8_scales_f32(&e8m0)?;
    let mult_bytes: Vec<u8> = mult.iter().flat_map(|f| f.to_le_bytes()).collect();
    let scale_f32 = gpu.alloc(mult_bytes.len())?;
    let out = match gpu
        .copy_h2d(&mult_bytes, scale_f32)
        .and_then(|()| gpu.alloc(n * k * 2))
    {
        Ok(out) => out,
        Err(error) => {
            let _ = gpu.free(scale_f32);
            return Err(error.context("preparing MXFP8 expansion buffers"));
        }
    };
    let stream = gpu.default_stream();
    let launched = gpu
        .kernel(
            "dequant_fp8_blockscaled_bf16",
            "dequant_fp8_blockscaled_bf16",
        )
        .and_then(|kernel| {
            KernelLaunch::new(gpu, kernel)
                .grid([div_ceil(k as u32, 64), div_ceil(n as u32, 4), 1])
                .block([64, 4, 1])
                .arg_ptr(src.data)
                .arg_ptr(scale_f32)
                .arg_ptr(out)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(1) // block_n: one scale row per weight row
                .arg_u32(MXFP8_BLOCK as u32) // block_k
                .arg_u32((k / MXFP8_BLOCK) as u32) // scale row stride
                .arg_u32(1) // scale_is_fp32
                .launch(stream)
        })
        .and_then(|()| gpu.synchronize(stream));
    // The widened scales are dead once the kernel has run (synchronized
    // above), and are freed on the error path too.
    let released = gpu.free(scale_f32);
    if let Err(e) = launched {
        let _ = gpu.free(out);
        return Err(e.context(format!("MXFP8 -> BF16 dequant [{n}, {k}]")));
    }
    if let Err(error) = released {
        let _ = gpu.free(out);
        return Err(error.context("releasing MXFP8 expansion scales"));
    }
    Ok(DenseWeight { weight: out })
}

#[cfg(test)]
#[path = "mxfp8_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "mxfp8_gpu_tests.rs"]
mod gpu_tests;
