// SPDX-License-Identifier: AGPL-3.0-only

//! Which block-scaled FP8 GEMV launches take an `rows`-row projection
//! (multi-sequence decode, MTP verify): the launch plan behind the native FP8
//! attention q/k/v/o (`ATLAS_DENSE_NATIVE_FP8_ATTN`, the HIP native arm, the
//! Holo / Nemotron native-FP8 attention).
//!
//! Up to 4 rows the plan is ONE `w8a16_gemv_batch4` launch, whose row `r` is
//! `w8a16_gemv`'s bytes for that row (`examples/w8a16_gemv_batch4_microtest`,
//! `fp8_gemv_parity_gpu_tests.rs`): serial decode projects with
//! `w8a16_gemv`, so the K=2/3/4 verify rows equal serial decode's
//! (`ATLAS_EXACT_VERIFY`, `model/exact_verify.rs`). 5..=16 rows take
//! [`fp8_verify_gemv_tier`]'s kernel; past 16 rows (`wide`) chunks of 16 rows
//! each take the same pick, so the weight is read once per 16 rows instead of
//! once per row.

use spark_runtime::gpu::KernelHandle;

use super::fp8_verify_gemv_tier;

/// Rows one launch of the widest template serves (`w8a16_gemv_batch16`).
pub const FP8_ROWS_CHUNK: usize = 16;

/// The FP8 multi-row GEMV handles a layer resolved (`KernelHandle(0)` = not
/// linked; `batch8*` exist only in the Strix kernel set).
#[derive(Clone, Copy, Debug)]
pub struct Fp8RowsKernels {
    pub batch4: KernelHandle,
    pub batch8: KernelHandle,
    pub batch8_dyn_vl2: KernelHandle,
    pub batch8_dyn_vl2_n2: KernelHandle,
    pub batch16: KernelHandle,
}

/// One launch: `rows` rows from row `first`, grid `ceil(n / div)`.
#[derive(Clone, Copy, Debug)]
pub struct Fp8RowsLaunch {
    pub kernel: KernelHandle,
    pub div: u32,
    pub first: usize,
    pub rows: usize,
}

/// The launches for `rows` rows, or `None` when a launch's kernel is not
/// linked or `rows` is past [`FP8_ROWS_CHUNK`] without `wide`; the caller
/// then keeps its per-row `w8a16_gemv` loop, which is serial decode's
/// arithmetic too.
pub fn fp8_rows_plan(
    rows: usize,
    wide: bool,
    vl2_on: bool,
    k: Fp8RowsKernels,
) -> Option<Vec<Fp8RowsLaunch>> {
    if rows == 0 || (rows > FP8_ROWS_CHUNK && !wide) {
        return None;
    }
    let mut plan = Vec::with_capacity(rows.div_ceil(FP8_ROWS_CHUNK));
    let mut first = 0;
    while first < rows {
        let m = (rows - first).min(FP8_ROWS_CHUNK);
        let (kernel, div) = if m <= 4 {
            (k.batch4, 4)
        } else {
            fp8_verify_gemv_tier(
                m,
                vl2_on,
                k.batch8,
                k.batch8_dyn_vl2,
                k.batch8_dyn_vl2_n2,
                k.batch16,
            )
        };
        if kernel.0 == 0 {
            return None;
        }
        plan.push(Fp8RowsLaunch {
            kernel,
            div,
            first,
            rows: m,
        });
        first += m;
    }
    Some(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    const Z: KernelHandle = KernelHandle(0);
    /// The GB10 set: batch4 + batch16, no Strix batch8 tiers.
    const GB10: Fp8RowsKernels = Fp8RowsKernels {
        batch4: KernelHandle(4),
        batch8: Z,
        batch8_dyn_vl2: Z,
        batch8_dyn_vl2_n2: Z,
        batch16: KernelHandle(16),
    };

    fn plan(rows: usize, wide: bool, k: Fp8RowsKernels) -> Option<Vec<(u64, u32, usize, usize)>> {
        fp8_rows_plan(rows, wide, true, k).map(|p| {
            p.iter()
                .map(|l| (l.kernel.0, l.div, l.first, l.rows))
                .collect()
        })
    }

    #[test]
    fn verify_widths_are_one_batch4_launch() {
        // K=2/3/4 verify: the exact twin of `w8a16_gemv`, one weight pass.
        for rows in 1..=4 {
            assert_eq!(plan(rows, false, GB10), Some(vec![(4, 4, 0, rows)]));
            assert_eq!(plan(rows, true, GB10), Some(vec![(4, 4, 0, rows)]));
        }
    }

    #[test]
    fn up_to_16_rows_is_one_tier_launch() {
        for rows in 5..=16 {
            assert_eq!(plan(rows, false, GB10), Some(vec![(16, 4, 0, rows)]));
        }
        // Strix: the vl2 n2 tier at 5..=8 rows, as `fp8_verify_gemv_tier`.
        let strix = Fp8RowsKernels {
            batch8: KernelHandle(8),
            batch8_dyn_vl2: KernelHandle(9),
            batch8_dyn_vl2_n2: KernelHandle(10),
            ..GB10
        };
        assert_eq!(plan(6, false, strix), Some(vec![(10, 16, 0, 6)]));
    }

    #[test]
    fn past_16_rows_chunks_only_when_wide() {
        assert_eq!(plan(17, false, GB10), None);
        assert_eq!(plan(128, false, GB10), None);
        assert_eq!(
            plan(24, true, GB10),
            Some(vec![(16, 4, 0, 16), (16, 4, 16, 8)])
        );
        // A tail of at most 4 rows takes batch4.
        assert_eq!(
            plan(36, true, GB10),
            Some(vec![(16, 4, 0, 16), (16, 4, 16, 16), (4, 4, 32, 4)])
        );
        let p = fp8_rows_plan(128, true, false, GB10).unwrap();
        assert_eq!(p.len(), 8);
        assert_eq!(p.iter().map(|l| l.rows).sum::<usize>(), 128);
    }

    #[test]
    fn a_missing_kernel_keeps_the_caller_loop() {
        assert_eq!(plan(0, true, GB10), None);
        let no_b4 = Fp8RowsKernels { batch4: Z, ..GB10 };
        assert_eq!(plan(3, false, no_b4), None);
        assert_eq!(plan(20, true, no_b4), None);
        let no_b16 = Fp8RowsKernels { batch16: Z, ..GB10 };
        assert_eq!(plan(4, false, no_b16), Some(vec![(4, 4, 0, 4)]));
        assert_eq!(plan(5, false, no_b16), None);
    }
}
