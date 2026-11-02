// SPDX-License-Identifier: AGPL-3.0-only

//! Row-count pins of a packed multi-prompt prefill (`layer::glm_packed`).
//!
//! A packed pass runs each prompt's rows through every row-local kernel
//! together with other prompts' rows, so a kernel whose choice depends on
//! the pass's row count would give a prompt different bytes packed than
//! alone. While a packed pass is in scope ([`enter`]) every such choice in
//! the GLM prefill path takes ONE family for all row counts, a family whose
//! per-row result does not depend on the other rows:
//!
//! * BF16 GEMMs (`ops::bf16_gemm`: MLA projections, dense FFN, router):
//!   CUTLASS at every row count (tile config from `(N, K)` only, no split-K),
//!   else cuBLASLt on one configuration (`fixed_algo`), never the per-M
//!   heuristic. MLA head GEMMs: the CUTLASS grouped tiles.
//! * KDA q/k/v/o: the W8A8 MX GEMM when present (row-invariant per bucket,
//!   `mx_algo`), else the transposed M128 W4A16 tiles at every row count; no
//!   Lt FP8 route.
//! * mHC: the fused FP32 mixer at every row count, and the post+pre seam
//!   whenever it is enabled (not only from 512 rows); no decode seam.
//! * MoE: the generic grouped path (K128W tiles, FP8-cache shared expert,
//!   the BF16 router GEMM above) for every row count; none of the verify
//!   shapes (C2/C3/C4/K5/independent, M16 stream twins, tensor-core GEMV
//!   tiers, shared TP split).
//! * MLA: per-owner projections (no owner batching).
//!
//! The pins run on the scheduler thread only, for the duration of the
//! traversal; off it nothing changes.

use std::cell::Cell;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Whether a packed traversal is running on this thread.
pub fn active() -> bool {
    ACTIVE.with(Cell::get)
}

/// Keeps the pins on until dropped.
pub struct Scope {
    was: bool,
}

/// Turn the pins on for this thread until the returned guard drops.
pub fn enter() -> Scope {
    Scope {
        was: ACTIVE.with(|a| a.replace(true)),
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|a| a.set(self.was));
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "glm_packed_scope_test_gpu.rs"]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_nests_and_restores() {
        assert!(!active());
        {
            let _a = enter();
            assert!(active());
            {
                let _b = enter();
                assert!(active());
            }
            assert!(active());
        }
        assert!(!active());
    }
}
