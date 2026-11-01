// SPDX-License-Identifier: AGPL-3.0-only
//! Native FP8 block (E4M3 + f32 per 128 x 128 block, ModelOpt `FP8_PB_WO`)
//! and MXFP8 projections on the checkpoint's own bytes: the row-invariant
//! small-row GEMV tiers, their multi-projection launches, and the prefill
//! GEMMs.
//!
//! Small rows (1..=32 a launch, any count chunked): `fp8b_gemv_tc{8,16,32}`
//! (`mxfp8_gemv.cu`, B128) read the E4M3 values unscaled into the BF16 MMA
//! (exact) and scale each warp's 64-value FP32 partial by its block's f32.
//! The weights are never rounded. Like the MXFP8 tiers, every 8-row
//! activation tile keeps its own accumulators and the eight K-split warps sum
//! in one fixed order, so a row's output bits do not depend on how many rows
//! share the launch: a 1-row decode, a k-row DFlash verify and a
//! multi-sequence batch agree bit for bit.
//!
//! `*_multi` launches run up to three projections of the same rows (KDA
//! q/k/v, MLA q_a + kv_a, an FFN's gate + up) as one grid, each CTA the body
//! of its own projection's tier: the same output words, one launch.
//!
//! Prefill (more rows than a verify block), three ways ([`Fp8bPrefill`]):
//! - `W8a8` (NVIDIA's / vLLM's FP8_BLOCK_SCALES recipe: activations quantized
//!   per token and 128 values to E4M3 with an f32 scale, then cuBLASLt's block
//!   FP8 GEMM on one pinned in-order kernel per shape,
//!   `fp8_gemm_act_weight_t_blkscaled_pinned`: ~1.8x the BF16 GEMM's FLOPs);
//! - `W8a16` (exact weights on BF16 activations: `w8a16_gemm_pipelined`, the
//!   same two-level FP32 accumulation as the GEMV tiers);
//! - `Bf16` (production's: the BF16 copy, one rounding of e4m3 x scale).
//!
//! All three are row-invariant (each output word depends only on its own row).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::{MXFP8_MAX_ROWS, mxfp8_tier};
use crate::weight_map::{FP8_BLOCK, Fp8BlockWeight, Mxfp8Weight};

/// Kernel module of every tier here.
pub const MX_GEMV_MODULE: &str = "mxfp8_gemv";

/// Row-major and tiled tiers (8/16/32 rows) of one weight format.
#[derive(Debug, Clone, Copy)]
pub struct MxTiers {
    pub row_major: [KernelHandle; 3],
    pub tiled: [KernelHandle; 3],
    /// The family reads FP8 block scales (an f32 grid) rather than UE8M0.
    pub fp8b_scales: bool,
}

impl Default for MxTiers {
    fn default() -> Self {
        Self {
            row_major: [KernelHandle(0); 3],
            tiled: [KernelHandle(0); 3],
            fp8b_scales: false,
        }
    }
}

/// The function names of a tier family in [`MX_GEMV_MODULE`]: row-major
/// then tiled, 8 / 16 / 32 rows (`{stem}_tc{rows}{suffix}[_tiled]`).
fn tier_names(stem: &str, suffix: &str) -> [[String; 3]; 2] {
    ["", "_tiled"].map(|tiled| ["8", "16", "32"].map(|t| format!("{stem}_tc{t}{suffix}{tiled}")))
}

/// The strided FP8 block tiers' names (`fp8b_gemv_tc{8,16,32}_ld`).
fn ld_names() -> [String; 3] {
    ["8", "16", "32"].map(|r| format!("fp8b_gemv_tc{r}_ld"))
}

/// The tier families [`native_kernels`] resolves: (stem, suffix).
const TIER_FAMILIES: [(&str, &str); 4] = [
    ("mxfp8_gemv", ""),
    ("fp8b_gemv", ""),
    ("fp8b_gemv", "_multi"),
    ("mxfp8_gemv", "_multi"),
];

/// The single kernels [`native_kernels`] resolves: (module, function).
const SINGLE_KERNELS: [(&str, &str); 5] = [
    (MX_GEMV_MODULE, "mxfp8_to_bf16"),
    (
        "per_token_group_quant_fp8",
        "per_token_group_quant_fp8_warp",
    ),
    ("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled"),
    ("w8a16_gemm_pipelined", "w8a16_gemm_pipelined"),
    (super::MXFP8_ACT_MODULE, super::MX_QUANT_KERNEL),
];

/// Every (module, function) [`native_kernels`] looks up: what a target serving
/// the native GLM paths must build (`fp8b_tests` checks the glm-5.3-flash
/// sources declare each one).
pub fn native_kernel_names() -> Vec<(&'static str, String)> {
    let mut names: Vec<(&'static str, String)> = TIER_FAMILIES
        .iter()
        .flat_map(|(stem, suffix)| tier_names(stem, suffix).into_iter().flatten())
        .chain(ld_names())
        .map(|f| (MX_GEMV_MODULE, f))
        .collect();
    names.extend(SINGLE_KERNELS.iter().map(|(m, f)| (*m, f.to_string())));
    names
}

impl MxTiers {
    fn resolve(gpu: &dyn GpuBackend, stem: &str, suffix: &str) -> Self {
        let [row_major, tiled] = tier_names(stem, suffix)
            .map(|tier| tier.map(|f| crate::layers::try_kernel(gpu, MX_GEMV_MODULE, &f)));
        Self {
            row_major,
            tiled,
            fp8b_scales: stem == "fp8b_gemv",
        }
    }

    /// The tier for `m` rows (1..=32) of a twin `tiled` or not; zero when the
    /// target lacks it.
    pub fn kernel(&self, tiled: bool, m: u32) -> KernelHandle {
        let tiers = if tiled { &self.tiled } else { &self.row_major };
        tiers[mxfp8_tier(m)]
    }

    /// Whether every tier of the layout is present.
    pub fn complete(&self, tiled: bool) -> bool {
        let tiers = if tiled { &self.tiled } else { &self.row_major };
        tiers.iter().all(|k| k.0 != 0)
    }
}

/// The native kernels, resolved once per process (zero handles when the
/// target lacks them: every caller checks before routing to them).
#[derive(Debug, Clone, Copy)]
pub struct NativeKernels {
    /// `mxfp8_gemv_tc{8,16,32}[_tiled]`.
    pub mx: MxTiers,
    /// `fp8b_gemv_tc{8,16,32}[_tiled]`.
    pub fp8b: MxTiers,
    /// `fp8b_gemv_tc{8,16,32}_multi[_tiled]`.
    pub fp8b_multi: MxTiers,
    /// `fp8b_gemv_tc{8,16,32}_ld` (a K-slice of a wider row-major weight).
    pub fp8b_ld: [KernelHandle; 3],
    /// `mxfp8_to_bf16` (exact transient expansion for prefill).
    pub mx_expand: KernelHandle,
    /// `mxfp8_gemv_tc{8,16,32}_multi[_tiled]`.
    pub mx_multi: MxTiers,
    /// `per_token_group_quant_fp8` (W8A8 prefill activations).
    pub act_quant: KernelHandle,
    /// `fp8_gemm_t_blockscaled` (W8A8 prefill where cuBLASLt cannot run
    /// block FP8).
    pub w8a8_gemm: KernelHandle,
    /// `w8a16_gemm_pipelined` (exact W8A16 prefill GEMM).
    pub w8a16_gemm: KernelHandle,
    /// `mxfp8_quant_act_bf16` (MXFP8 W8A8 prefill activations).
    pub mx_quant: KernelHandle,
}

/// [`NativeKernels`] for this process's target.
pub fn native_kernels(gpu: &dyn GpuBackend) -> NativeKernels {
    static K: std::sync::OnceLock<NativeKernels> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        let [mx, fp8b, fp8b_multi, mx_multi] =
            TIER_FAMILIES.map(|(stem, suffix)| MxTiers::resolve(gpu, stem, suffix));
        let [mx_expand, act_quant, w8a8_gemm, w8a16_gemm, mx_quant] =
            SINGLE_KERNELS.map(|(m, f)| crate::layers::try_kernel(gpu, m, f));
        let k = NativeKernels {
            mx,
            fp8b,
            fp8b_multi,
            fp8b_ld: ld_names().map(|f| crate::layers::try_kernel(gpu, MX_GEMV_MODULE, &f)),
            mx_expand,
            mx_multi,
            act_quant,
            w8a8_gemm,
            w8a16_gemm,
            mx_quant,
        };
        // A target that serves GLM builds `mxfp8_gemv`; one whose existing
        // MXFP8 tiers resolve but the native ones do not runs a kernel build
        // older than this source (e.g. a cached atlas-kernels build-script
        // output after an mtime-preserving sync of the tree).
        if k.mx.complete(false) && !(k.fp8b.complete(false) && k.mx_multi.complete(false)) {
            tracing::error!(
                "the mxfp8_gemv module lacks the native FP8 block / multi-projection tiers: \
                 the kernel build is stale (touch kernels/gb10/glm-5.3-flash/nvfp4/mxfp8_gemv.cu \
                 or `cargo clean -p atlas-kernels` and rebuild)"
            );
        }
        k
    })
}

/// The native kernels of [`native_kernels`] this target lacks, by name
/// (empty when all resolve).
pub fn missing_native_kernels(gpu: &dyn GpuBackend) -> Vec<String> {
    native_kernel_names()
        .into_iter()
        .filter(|(m, f)| gpu.kernel(m, f).is_err())
        .map(|(m, f)| format!("{m}::{f}"))
        .collect()
}

/// `output[r, n] = input[r, k] · W^T` for `rows` rows of FP8 block twin `w`
/// (`[n, k]`, both multiples of 128), output row `r` at `output + r *
/// out_stride` elements, in launches of up to 32 rows on `tiers`.
/// Row-invariant (see the module doc).
#[allow(clippy::too_many_arguments)]
pub fn fp8b_gemv_rows(
    gpu: &dyn GpuBackend,
    tiers: &MxTiers,
    w: &Fp8BlockWeight,
    input: DevicePtr,
    output: DevicePtr,
    rows: usize,
    n: usize,
    k: usize,
    out_stride: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        out_stride >= n && n.is_multiple_of(16) && k.is_multiple_of(FP8_BLOCK),
        "fp8b_gemv_rows: n={n} k={k} out_stride={out_stride}"
    );
    for first in (0..rows).step_by(MXFP8_MAX_ROWS) {
        let m = (rows - first).min(MXFP8_MAX_ROWS) as u32;
        let kernel = tiers.kernel(w.tiled, m);
        ensure!(
            kernel.0 != 0,
            "fp8b_gemv_rows: no kernel for {m} rows (tiled={})",
            w.tiled
        );
        let grid = div_ceil(n as u32, 16);
        let (rows, ctas) = super::native_touch(m, n as u32, k as u32, grid);
        KernelLaunch::new(gpu, kernel)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input.offset(first * k * 2))
            .arg_ptr(w.data)
            .arg_ptr(w.scales)
            .arg_ptr(output.offset(first * out_stride * 2))
            .arg_u32(m)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(out_stride as u32)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)?;
    }
    Ok(())
}

/// [`super::mxfp8_gemv_rows`] through the `mxfp8_gemv_tc*_touch` twin (PDL
/// entry plus the weight touch, `ATLAS_GLM_DECODE_GEMV_BATCH=1` under PDL) on
/// a row-major twin for up to 16 rows, as production's verify tiers run; the
/// plain tiers otherwise. Same body: the same output bits.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_gemv_rows_touched(
    gpu: &dyn GpuBackend,
    row_major: &[KernelHandle; 3],
    w: &Mxfp8Weight,
    input: DevicePtr,
    output: DevicePtr,
    rows: usize,
    n: usize,
    k: usize,
    out_stride: usize,
    stream: u64,
) -> Result<()> {
    if !w.tiled
        && (1..=16).contains(&rows)
        && let Some(touch) = super::gemv_touch(super::mxfp8_tc_twin(rows as u32))
    {
        return touch.mxfp8_tc(
            gpu,
            input,
            w.data,
            w.scales,
            output,
            rows as u32,
            n as u32,
            k as u32,
            out_stride as u32,
            stream,
        );
    }
    super::mxfp8_gemv_rows(
        gpu, row_major, w, input, output, rows, n, k, out_stride, stream,
    )
}

/// [`fp8b_gemv_rows`] on a K-slice of a wider row-major FP8 block weight:
/// `data` at the slice's first value, `scales` at its first grid column, rows
/// `ldw` values apart (`k` and the slice's start on the 128 grid).
#[allow(clippy::too_many_arguments)]
pub fn fp8b_gemv_ld_rows(
    gpu: &dyn GpuBackend,
    tiers: &[KernelHandle; 3],
    data: DevicePtr,
    scales: DevicePtr,
    input: DevicePtr,
    output: DevicePtr,
    rows: usize,
    n: usize,
    k: usize,
    ldw: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        n.is_multiple_of(16)
            && k.is_multiple_of(FP8_BLOCK)
            && ldw >= k
            && ldw.is_multiple_of(FP8_BLOCK),
        "fp8b_gemv_ld_rows: n={n} k={k} ldw={ldw}"
    );
    for first in (0..rows).step_by(MXFP8_MAX_ROWS) {
        let m = (rows - first).min(MXFP8_MAX_ROWS) as u32;
        let kernel = tiers[mxfp8_tier(m)];
        ensure!(kernel.0 != 0, "fp8b_gemv_ld_rows: no kernel for {m} rows");
        let grid = div_ceil(n as u32, 16);
        let (rows, ctas) = super::native_touch(m, n as u32, k as u32, grid);
        KernelLaunch::new(gpu, kernel)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input.offset(first * k * 2))
            .arg_ptr(data)
            .arg_ptr(scales)
            .arg_ptr(output.offset(first * n * 2))
            .arg_u32(m)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(n as u32)
            .arg_u32(ldw as u32)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)?;
    }
    Ok(())
}

/// The exact BF16 expansion of an untiled MXFP8 `[n, k]` twin into `out`
/// (`n * k * 2` bytes): `mxfp8_to_bf16`.
pub fn mxfp8_expand_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w: &Mxfp8Weight,
    out: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0 && !w.tiled && (n * k).is_multiple_of(8) && k.is_multiple_of(32),
        "mxfp8_to_bf16: [{n}, {k}] (tiled={}, kernel {})",
        w.tiled,
        kernel.0
    );
    let count8 = (n * k / 8) as u64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(count8 as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(w.data)
        .arg_ptr(w.scales)
        .arg_ptr(out)
        .arg_u64(count8)
        .launch(stream)
}

/// One projection of a multi-projection launch: its weight's values and
/// scales (MXFP8 E8M0 bytes or an FP8 block f32 grid), its output (rows `n`
/// apart) and `n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MxPart {
    pub data: DevicePtr,
    pub scales: DevicePtr,
    pub out: DevicePtr,
    pub n: usize,
}

impl MxPart {
    pub fn mx(w: &Mxfp8Weight, out: DevicePtr, n: usize) -> Self {
        Self {
            data: w.data,
            scales: w.scales,
            out,
            n,
        }
    }

    pub fn fp8b(w: &Fp8BlockWeight, out: DevicePtr, n: usize) -> Self {
        Self {
            data: w.data,
            scales: w.scales,
            out,
            n,
        }
    }
}

/// `parts[i].out[r] = input[r] · W_i^T` for 1..=3 projections of the same
/// `rows` rows (`[rows, k]`, contiguous) in one launch per 32 rows, on the
/// `_multi` `tiers` of their format (all parts tiled or all not, as `tiled`
/// says). Each output word is the one its own single-projection tier writes.
#[allow(clippy::too_many_arguments)]
pub fn mx_multi_rows(
    gpu: &dyn GpuBackend,
    tiers: &MxTiers,
    tiled: bool,
    input: DevicePtr,
    parts: &[MxPart],
    rows: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=3).contains(&parts.len())
            && k.is_multiple_of(64)
            && parts.iter().all(|p| p.n > 0 && p.n.is_multiple_of(16)),
        "mx_multi_rows: {} parts, k={k}",
        parts.len()
    );
    let tiles: u32 = parts.iter().map(|p| div_ceil(p.n as u32, 16)).sum();
    let none = MxPart {
        data: DevicePtr::NULL,
        scales: DevicePtr::NULL,
        out: DevicePtr::NULL,
        n: 0,
    };
    let part = |i: usize| parts.get(i).copied().unwrap_or(none);
    for first in (0..rows).step_by(MXFP8_MAX_ROWS) {
        let m = (rows - first).min(MXFP8_MAX_ROWS) as u32;
        let kernel = tiers.kernel(tiled, m);
        ensure!(
            kernel.0 != 0,
            "mx_multi_rows: no kernel for {m} rows (tiled={tiled})"
        );
        let mut launch = KernelLaunch::new(gpu, kernel)
            .grid([tiles, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input.offset(first * k * 2));
        for i in 0..3 {
            let p = part(i);
            let out = if p.n == 0 {
                p.out
            } else {
                p.out.offset(first * p.n * 2)
            };
            launch = launch
                .arg_ptr(p.data)
                .arg_ptr(p.scales)
                .arg_ptr(out)
                .arg_u32(p.n as u32);
        }
        // Touch geometry of the narrowest projection (MX rows carry K/32 scale
        // bytes; the FP8 block grid is not touched).
        let n_min = parts.iter().map(|p| p.n as u32).min().unwrap_or(0);
        let row_bytes = if tiers.fp8b_scales { k } else { k + k / 32 } as u32;
        let (rows, ctas) = super::native_touch(m, n_min, row_bytes, div_ceil(n_min, 16));
        launch
            .arg_u32(m)
            .arg_u32(k as u32)
            .arg_u32(rows)
            .arg_u32(ctas)
            .launch(stream)?;
    }
    Ok(())
}

/// Rewrite FP8 block twin `w` (`[n, k]`, `n % 16 == 0`, `k % 64 == 0`) in
/// place into `mxfp8_tile16`'s value tiling (the grid stays row-major) under
/// `ATLAS_GLM_MXFP8_TILED=1`; a no-op otherwise. Load time: synchronizes.
pub fn fp8b_tile_if_requested(
    gpu: &dyn GpuBackend,
    w: &mut Fp8BlockWeight,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    if !super::mxfp8_tiled_requested()? || w.tiled {
        return Ok(());
    }
    ensure!(
        native_kernels(gpu).fp8b.complete(true),
        "ATLAS_GLM_MXFP8_TILED=1 but the fp8b_gemv_tc*_tiled tiers are missing"
    );
    super::mx_tile_values_in_place(gpu, w.data, n, k, stream)?;
    w.tiled = true;
    Ok(())
}

#[path = "fp8b_prefill.rs"]
mod prefill;
pub use prefill::*;

#[cfg(test)]
#[path = "fp8b_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "fp8b_cutlass_test_gpu.rs"]
mod cutlass_gpu_tests;
