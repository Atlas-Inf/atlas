// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_SKINNY_MMA=1` (default off): the NVFP4, Q+gate and block-FP8
//! projections of 1..=16 rows run on one row-invariant tensor-core kernel
//! family (`skinny_mma.cu`, `kernels/gb10/qwen3.6-27b/nvfp4/`, so the
//! qwen3.6-27b and qwen3.8-27b targets), serial decode (M = 1) and every
//! verify width alike.
//!
//! | family | kernels | replaces |
//! |---|---|---|
//! | [`Family::W4a16`] | `skinny_w4a16_t{1,2}` | `w4a16_gemv[_sw]`, `_dual[_sw]`, `_batch2/3`, `_batch4..8/16/32`, `_dual_batch2/3` |
//! | [`Family::Qg`] | `skinny_qg_t{1,2}` | `w4a16_gemv_qg`, `_qg_batch2/3/4` |
//! | [`Family::W8a16`] | `skinny_w8a16_t{1,2}` | `w8a16_gemv`, `_dual`, `_batch4/16` |
//!
//! A row's bits depend only on its own activation row (the kernel's K order
//! and cross-warp combine follow (N, K) alone), so with the lever on every
//! launch of these projections — decode or verify, any width — computes the
//! same bytes for a row, and an exact verify (`ATLAS_EXACT_VERIFY`) stays
//! exact against the new serial decode. Gate: `examples/skinny_mma_microtest.rs`;
//! bench: the skinny rows of `examples/wide_gemv_bench.rs`.
//!
//! The hook sits in the shared launch ops, behind an allowlist of the kernel
//! handles above: a launch of any other kernel through those ops (the FP32
//! `w4a16_gemv_logits`, tensor-core or DP4A tiers) keeps its kernel. Shapes
//! the kernels do not serve (K not a multiple of [`k_chunk`], more than 16
//! rows) keep the old kernels at every width, decode and verify alike, so
//! they stay each other's exact twins. Off, nothing is resolved and every
//! launch is today's.

use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::try_kernel;
use crate::weight_map::QuantizedWeight;

/// The kernels' module (file stem).
pub const MODULE: &str = "skinny_mma";
/// Widest launch: two 8-row token tiles.
pub const MAX_ROWS: u32 = 16;
/// Token rows a tile (the MMA's N).
const TILE_ROWS: u32 = 8;
/// Outputs a CTA (the MMA's M; `SK_ROWS`).
const CTA_OUTPUTS: u32 = 16;
/// Threads a CTA (`SK_WARPS` x 32).
const BLOCK: u32 = 256;

/// The three projection families, as the wide twins name them: the NVFP4
/// GEMV, the NVFP4 Q+gate and the block-FP8 GEMV.
pub use super::wide_gemv::Family;

fn prefix(family: Family) -> &'static str {
    match family {
        Family::W4a16 => "skinny_w4a16",
        Family::Qg => "skinny_qg",
        Family::W8a16 => "skinny_w8a16",
    }
}

/// `(module, function)` of the GEMV kernels `family`'s skinny kernel stands
/// in for.
pub fn replaces(family: Family) -> &'static [(&'static str, &'static str)] {
    match family {
        Family::W4a16 => &[
            ("w4a16_gemv", "w4a16_gemv"),
            ("w4a16_gemv", "w4a16_gemv_sw"),
            ("w4a16_gemv", "w4a16_gemv_batch2"),
            ("w4a16_gemv", "w4a16_gemv_batch3"),
            ("w4a16_gemv", "w4a16_gemv_batch4"),
            ("w4a16_gemv", "w4a16_gemv_batch5"),
            ("w4a16_gemv", "w4a16_gemv_batch6"),
            ("w4a16_gemv", "w4a16_gemv_batch7"),
            ("w4a16_gemv", "w4a16_gemv_batch8"),
            ("w4a16_gemv", "w4a16_gemv_batch16"),
            ("w4a16_gemv", "w4a16_gemv_batch32"),
            ("w4a16_gemv", "w4a16_gemv_dual_batch2"),
            ("w4a16_gemv", "w4a16_gemv_dual_batch3"),
            ("w4a16_gemv_fused", "w4a16_gemv_dual"),
            ("w4a16_gemv_fused", "w4a16_gemv_dual_sw"),
        ],
        Family::Qg => &[
            ("w4a16_gemv", "w4a16_gemv_qg"),
            ("w4a16_gemv", "w4a16_gemv_qg_batch2"),
            ("w4a16_gemv", "w4a16_gemv_qg_batch3"),
            ("w4a16_gemv", "w4a16_gemv_qg_batch4"),
        ],
        Family::W8a16 => &[
            ("w8a16_gemv", "w8a16_gemv"),
            ("w8a16_gemv_fused", "w8a16_gemv_dual"),
            ("w8a16_gemv_batch4", "w8a16_gemv_batch4"),
            ("w8a16_gemv_batch4", "w8a16_gemv_batch16"),
        ],
    }
}

/// `skinny_w4a16_t1`, `skinny_qg_t2`, ...: `tiles` 8-row token tiles.
pub fn kernel_name(family: Family, tiles: u32) -> String {
    format!("{}_t{tiles}", prefix(family))
}

/// K values a warp step of `family`'s kernel (`SkFam::KC`: four lanes'
/// runs of 64 NVFP4 or 32 FP8 values); K must be a multiple.
pub fn k_chunk(family: Family) -> u32 {
    match family {
        Family::W4a16 | Family::Qg => 256,
        Family::W8a16 => 128,
    }
}

/// Whether the kernels serve `m` rows of a `k`-wide weight. A function of
/// the shape only (and `m <= 16`), so a projection takes the same kernel
/// family at every width.
pub fn serves(family: Family, m: u32, k: u32) -> bool {
    (1..=MAX_ROWS).contains(&m) && k > 0 && k.is_multiple_of(k_chunk(family))
}

/// The resolved kernels: `[family][tiles - 1]`.
#[derive(Clone, Copy, Debug)]
pub struct SkinnyKernels {
    handles: [[KernelHandle; 2]; 3],
}

impl SkinnyKernels {
    /// Look every kernel up; zero handles where the target does not ship them.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        Self {
            handles: Family::ALL
                .map(|f| [1, 2].map(|t| try_kernel(gpu, MODULE, &kernel_name(f, t)))),
        }
    }

    /// Every kernel resolved.
    pub fn complete(&self) -> bool {
        self.handles.iter().flatten().all(|h| h.0 != 0)
    }

    /// The kernel of `family` for `m` rows (one tile to 8, two to 16).
    pub fn kernel(&self, family: Family, m: u32) -> KernelHandle {
        let tiles = div_ceil(m.max(1), TILE_ROWS).min(2) as usize;
        self.handles[family as usize][tiles - 1]
    }

    fn check(&self, family: Family, m: u32, k: u32) -> Result<KernelHandle> {
        let h = self.kernel(family, m);
        ensure!(
            h.0 != 0 && serves(family, m, k),
            "{}: m={m} k={k} not served",
            prefix(family)
        );
        Ok(h)
    }
}

fn launch<'a>(gpu: &'a dyn GpuBackend, kernel: KernelHandle, ctas: u32) -> KernelLaunch<'a> {
    KernelLaunch::new(gpu, kernel)
        .grid([ctas.max(1), 1, 1])
        .block([BLOCK, 1, 1])
}

/// `m` rows of `input` (`[m, k]` BF16) through one or two NVFP4 weights:
/// part `(w, out, n)` writes `out[t * c_stride + j]` for its `n` outputs.
pub fn launch_w4a16(
    gpu: &dyn GpuBackend,
    kernels: &SkinnyKernels,
    input: DevicePtr,
    parts: &[(&QuantizedWeight, DevicePtr, u32)],
    (m, k): (u32, u32),
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    let kernel = kernels.check(Family::W4a16, m, k)?;
    ensure!(
        (1..=2).contains(&parts.len()),
        "skinny_w4a16: {} parts",
        parts.len()
    );
    let ctas: u32 = parts.iter().map(|p| div_ceil(p.2, CTA_OUTPUTS)).sum();
    let mut l = launch(gpu, kernel, ctas)
        .arg_ptr(input)
        .arg_u32(m)
        .arg_u32(k)
        .arg_u32(c_stride);
    for p in 0..2 {
        let (packed, scale, scale2, out, n) = parts.get(p).map_or(
            (DevicePtr(0), DevicePtr(0), 0.0, DevicePtr(0), 0),
            |&(w, out, n)| (w.weight, w.weight_scale, w.weight_scale_2, out, n),
        );
        l = l
            .arg_ptr(packed)
            .arg_ptr(scale)
            .arg_f32(scale2)
            .arg_ptr(out)
            .arg_u32(n);
    }
    l.launch(stream)
}

/// `m` rows through the interleaved NVFP4 Q+gate weight (`n` = heads x
/// head_dim x 2), output row `t` deinterleaved `[Q | G]` at `output + t*n`.
#[allow(clippy::too_many_arguments)]
pub fn launch_qg(
    gpu: &dyn GpuBackend,
    kernels: &SkinnyKernels,
    input: DevicePtr,
    w: &QuantizedWeight,
    output: DevicePtr,
    (m, n, k): (u32, u32, u32),
    (heads, head_dim): (u32, u32),
    stream: u64,
) -> Result<()> {
    let kernel = kernels.check(Family::Qg, m, k)?;
    ensure!(
        head_dim > 0 && n == heads * head_dim * 2,
        "skinny q+gate: n={n} != 2 x {heads} x {head_dim}"
    );
    launch(gpu, kernel, div_ceil(n, CTA_OUTPUTS))
        .arg_ptr(input)
        .arg_u32(m)
        .arg_u32(k)
        .arg_ptr(w.weight)
        .arg_ptr(w.weight_scale)
        .arg_f32(w.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// `m` rows through one or two block-scaled FP8 weights: part
/// `((weight, block_scale), out, n)`, FP32 block scales
/// `[ceil(n/128), k/128]`, writes `out[t * c_stride + j]`.
pub fn launch_w8a16(
    gpu: &dyn GpuBackend,
    kernels: &SkinnyKernels,
    input: DevicePtr,
    parts: &[((DevicePtr, DevicePtr), DevicePtr, u32)],
    (m, k): (u32, u32),
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    let kernel = kernels.check(Family::W8a16, m, k)?;
    ensure!(
        (1..=2).contains(&parts.len()),
        "skinny_w8a16: {} parts",
        parts.len()
    );
    let ctas: u32 = parts.iter().map(|p| div_ceil(p.2, CTA_OUTPUTS)).sum();
    let mut l = launch(gpu, kernel, ctas)
        .arg_ptr(input)
        .arg_u32(m)
        .arg_u32(k)
        .arg_u32(c_stride);
    for p in 0..2 {
        let ((weight, scale), out, n) =
            parts
                .get(p)
                .copied()
                .unwrap_or(((DevicePtr(0), DevicePtr(0)), DevicePtr(0), 0));
        l = l.arg_ptr(weight).arg_ptr(scale).arg_ptr(out).arg_u32(n);
    }
    l.launch(stream)
}

/// Whether an `ATLAS_SKINNY_MMA` value asks for the lever: exactly `1`.
pub fn requested_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// `ATLAS_SKINNY_MMA=1`, read once.
pub fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| requested_from(std::env::var("ATLAS_SKINNY_MMA").ok().as_deref()))
}

static SKINNY: OnceLock<Skinny> = OnceLock::new();

/// Resolve the lever's kernels once, when it is on (layer init calls this
/// through `W4a16BatchmTiers::resolve`); a no-op otherwise.
pub fn init(gpu: &dyn GpuBackend) {
    if requested() {
        SKINNY.get_or_init(|| Skinny::resolve(gpu));
    }
}

/// The lever's kernels, `None` when it is off or the target lacks them.
pub fn active() -> Option<&'static Skinny> {
    SKINNY.get().filter(|s| s.kernels.complete())
}

/// The lever's kernels when they take a launch of `kernel` (`family`, `m`
/// rows at `k`), else `None`.
fn taking(family: Family, kernel: KernelHandle, m: u32, k: u32) -> Option<&'static SkinnyKernels> {
    active()
        .filter(|s| s.takes(family, kernel, m, k))
        .map(|s| &s.kernels)
}

/// The launch-op hook: [`launch_w4a16`] in place of a launch of `kernel`
/// (`Ok(true)`), or nothing launched (`Ok(false)`: lever off, kernel not on
/// the list, shape not served).
pub fn try_w4a16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    parts: &[(&QuantizedWeight, DevicePtr, u32)],
    (m, k): (u32, u32),
    c_stride: u32,
    stream: u64,
) -> Result<bool> {
    let Some(ks) = taking(Family::W4a16, kernel, m, k) else {
        return Ok(false);
    };
    launch_w4a16(gpu, ks, input, parts, (m, k), c_stride, stream).map(|()| true)
}

/// [`try_w4a16`] for one weight, rows `n` apart: `w4a16_gemv`'s and the
/// batch tiers' argument shape.
#[allow(clippy::too_many_arguments)]
pub fn try_w4a16_one(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    w: &QuantizedWeight,
    output: DevicePtr,
    (m, n, k): (u32, u32, u32),
    stream: u64,
) -> Result<bool> {
    try_w4a16(gpu, kernel, input, &[(w, output, n)], (m, k), n, stream)
}

/// The Q+gate hook: [`launch_qg`] in place of a launch of `kernel`.
#[allow(clippy::too_many_arguments)]
pub fn try_qg(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    w: &QuantizedWeight,
    output: DevicePtr,
    (m, n, k): (u32, u32, u32),
    heads_dim: (u32, u32),
    stream: u64,
) -> Result<bool> {
    let Some(ks) = taking(Family::Qg, kernel, m, k) else {
        return Ok(false);
    };
    launch_qg(gpu, ks, input, w, output, (m, n, k), heads_dim, stream).map(|()| true)
}

/// The block-FP8 hook: [`launch_w8a16`] in place of a launch of `kernel`.
pub fn try_w8a16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    parts: &[((DevicePtr, DevicePtr), DevicePtr, u32)],
    (m, k): (u32, u32),
    c_stride: u32,
    stream: u64,
) -> Result<bool> {
    let Some(ks) = taking(Family::W8a16, kernel, m, k) else {
        return Ok(false);
    };
    launch_w8a16(gpu, ks, input, parts, (m, k), c_stride, stream).map(|()| true)
}

/// The resolved kernels and the handles of the kernels they replace.
#[derive(Debug)]
pub struct Skinny {
    kernels: SkinnyKernels,
    replaced: [Vec<u64>; 3],
}

impl Skinny {
    fn resolve(gpu: &dyn GpuBackend) -> Self {
        let kernels = SkinnyKernels::resolve(gpu);
        let replaced = Family::ALL.map(|f| {
            replaces(f)
                .iter()
                .map(|&(module, func)| try_kernel(gpu, module, func).0)
                .filter(|&h| h != 0)
                .collect()
        });
        if kernels.complete() {
            tracing::info!(
                "ATLAS_SKINNY_MMA=1: row-invariant tensor-core projections for 1..={MAX_ROWS} rows"
            );
        } else {
            tracing::warn!("ATLAS_SKINNY_MMA=1 ignored: this target does not ship {MODULE}");
        }
        Self { kernels, replaced }
    }

    /// Whether a launch of `kernel` for `m` rows at `k` goes to the skinny
    /// `family` kernel instead.
    pub fn takes(&self, family: Family, kernel: KernelHandle, m: u32, k: u32) -> bool {
        kernel.0 != 0 && serves(family, m, k) && self.replaced[family as usize].contains(&kernel.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_the_kernels() {
        assert_eq!(kernel_name(Family::W4a16, 1), "skinny_w4a16_t1");
        assert_eq!(kernel_name(Family::Qg, 2), "skinny_qg_t2");
        assert_eq!(kernel_name(Family::W8a16, 2), "skinny_w8a16_t2");
    }

    #[test]
    fn served_shapes_follow_k_and_the_row_cap() {
        let w4 = |m, k| serves(Family::W4a16, m, k);
        assert!(w4(1, 5120) && w4(16, 17408) && w4(9, 6144));
        assert!(!w4(0, 5120) && !w4(17, 5120));
        assert!(!w4(1, 5120 + 128) && !w4(4, 0));
        assert!(serves(Family::W8a16, 1, 5120 + 128) && !serves(Family::W8a16, 1, 5120 + 16));
    }

    #[test]
    fn one_tile_to_eight_rows_two_past() {
        let k = SkinnyKernels {
            handles: [[KernelHandle(1), KernelHandle(2)]; 3],
        };
        for m in 1..=8 {
            assert_eq!(k.kernel(Family::Qg, m).0, 1, "m={m}");
        }
        for m in 9..=16 {
            assert_eq!(k.kernel(Family::W8a16, m).0, 2, "m={m}");
        }
    }

    #[test]
    fn only_listed_kernels_are_taken() {
        let s = Skinny {
            kernels: SkinnyKernels {
                handles: [[KernelHandle(1), KernelHandle(2)]; 3],
            },
            replaced: [vec![10, 11], vec![20], vec![30]],
        };
        assert!(s.takes(Family::W4a16, KernelHandle(11), 1, 5120));
        assert!(
            !s.takes(Family::W4a16, KernelHandle(20), 1, 5120),
            "a Q+gate handle"
        );
        assert!(
            !s.takes(Family::W4a16, KernelHandle(99), 1, 5120),
            "e.g. the FP32 logits GEMV"
        );
        assert!(!s.takes(Family::W8a16, KernelHandle(30), 17, 5120));
        assert!(!s.takes(Family::W8a16, KernelHandle(0), 1, 5120));
    }

    #[test]
    fn lever_reads_exactly_one() {
        assert!(requested_from(Some("1")));
        assert!(
            !requested_from(None) && !requested_from(Some("0")) && !requested_from(Some("true"))
        );
    }
}
