// SPDX-License-Identifier: AGPL-3.0-only
//! The MXFP8 W8A8 GEMM shapes the row-invariance sweep covers
//! (`mxfp8_gemm_test_gpu.rs`, spark-model), and on which cuBLASLt versions
//! each one passed it: the one list both the sweep and `mx_algo`'s gate
//! read, so what is swept and what is trusted cannot drift apart.
//!
//! `mx_algo` lets each row bucket of an `n x k` GEMM take its own fastest
//! in-order kernel only where [`verified`] holds; every other GEMM runs ONE
//! kernel for all row counts, row-invariant by construction.
//!
//! Promoting a shape: run the sweep on a GB10 with the cuBLASLt in question
//! (the command is in `mxfp8_gemm_test_gpu.rs`). If it passes for the shape
//! (no "an in-order kernel differs" / "the tuned kernel differs" failure;
//! it prints `cuBLASLt <version>` first), add that version to the shape's
//! `verified_on`. Never add a version the sweep has not run on.

/// One projection the sweep covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MxSweepShape {
    /// The projection; the sweep finds its checkpoint cut by this label.
    pub label: &'static str,
    pub n: u32,
    pub k: u32,
    /// cuBLASLt versions ([`super::version`]) the sweep passed on.
    pub verified_on: &'static [usize],
}

/// 13.0.0 (the runtime image's) and 13.1.1: the sweep of 2026-10-15.
const SWEPT_2026_10_15: &[usize] = &[130000, 130101];
/// Added after that sweep; not run yet. See the module docs to promote.
const NOT_SWEPT: &[usize] = &[];

const fn shape(label: &'static str, n: u32, k: u32, on: &'static [usize]) -> MxSweepShape {
    MxSweepShape {
        label,
        n,
        k,
        verified_on: on,
    }
}

/// The FP4-Local prefill projections `ATLAS_QWEN4EXP_PREFILL_W8A8` runs, at
/// TP2 (a rank's cut) and TP1. K and V are separate GEMMs of one shape.
pub const MX_SWEEP: [MxSweepShape; 10] = [
    shape("tp2 gdn qkvz", 8192, 2560, SWEPT_2026_10_15),
    shape("tp2 gdn out", 2560, 3072, SWEPT_2026_10_15),
    shape("tp2 qsa q", 6144, 2560, SWEPT_2026_10_15),
    shape("tp2 qsa k, v", 256, 2560, NOT_SWEPT),
    shape("tp2 qsa o", 2560, 3072, SWEPT_2026_10_15),
    shape("tp1 gdn qkvz", 16384, 2560, NOT_SWEPT),
    shape("tp1 gdn out", 2560, 6144, NOT_SWEPT),
    shape("tp1 qsa q", 12288, 2560, NOT_SWEPT),
    // The 2026-10-15 sweep ran 512 x 2560 as a stacked TP2 k|v cut, which
    // production never runs; the kernels and their order of accumulation
    // depend on the shape and the row count, not on the weights.
    shape("tp1 qsa k, v", 512, 2560, SWEPT_2026_10_15),
    shape("tp1 qsa o", 2560, 6144, NOT_SWEPT),
];

/// Whether the sweep passed for `n x k` on cuBLASLt `version`.
pub(super) fn verified(version: usize, n: u32, k: u32) -> bool {
    MX_SWEEP
        .iter()
        .any(|s| s.n == n && s.k == k && s.verified_on.contains(&version))
}

#[cfg(test)]
mod tests {
    use super::{MX_SWEEP, verified};

    #[test]
    fn only_swept_shapes_on_their_swept_versions_are_verified() {
        for v in [130000, 130101] {
            for (n, k) in [(8192, 2560), (2560, 3072), (6144, 2560), (512, 2560)] {
                assert!(verified(v, n, k), "{v} {n}x{k}");
            }
            // Added to the sweep, not run yet.
            for (n, k) in [(256, 2560), (16384, 2560), (2560, 6144), (12288, 2560)] {
                assert!(!verified(v, n, k), "{v} {n}x{k}");
            }
            // Never swept (the indexer and shared expert don't run W8A8).
            for (n, k) in [(320, 2560), (640, 2560), (2560, 320), (384, 1536)] {
                assert!(!verified(v, n, k), "{v} {n}x{k}");
            }
        }
        for v in [0, 120900, 130001, 130100, 130200, 140000] {
            assert!(!verified(v, 8192, 2560), "{v}");
        }
        assert!(!verified(130000, 2560, 8192), "transposed");
    }

    #[test]
    fn every_swept_shape_is_a_whole_number_of_mxfp8_blocks_and_labelled_once() {
        for (i, s) in MX_SWEEP.iter().enumerate() {
            assert!(s.n > 0 && s.k > 0 && s.k.is_multiple_of(32), "{s:?}");
            assert!(MX_SWEEP[..i].iter().all(|t| t.label != s.label), "{s:?}");
        }
    }
}
