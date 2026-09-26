// SPDX-License-Identifier: AGPL-3.0-only

//! Budget for the LAZY FP8→BF16 weight copies `dequant_fp8_bf16_cached`
//! materialises on first prefill (issue #71).
//!
//! `cublas_bf16_proj` / `cutlass_bf16_proj` dequantise each FP8 `[N,K]`
//! projection to BF16 on first use and cache it in `DerivedWeights`. Those
//! allocations happen after the KV pool was sized, so on a unified-memory
//! GB10 they land on top of the budget and can take the box to the OOM edge
//! (reiner job 262: 27B at util 0.88 left 2.1 GB MemAvailable after the
//! copies). `factory/build.rs` reserves the superset footprint up front.
//!
//! Only RESIDENT tensors count: `WeightStore::names()`/`get()` iterate the
//! resident map, while deferred tensors (e.g. the qwen4_exp PLE n-gram tables)
//! live in `store.deferred(..)` and are never uploaded, so nothing can
//! dequantise them to BF16.

use spark_runtime::weights::{WeightDtype, WeightStore};

use super::GemmDispatch;

/// Bytes the lazy FP8→BF16 copies can cost this run, summed over the store's
/// resident tensors. 0 when no dispatch flag reaches `dequant_fp8_bf16_cached`.
/// Logs one INFO line when the reserve is nonzero.
pub fn lazy_bf16_reserve(store: &WeightStore) -> usize {
    if !lazy_bf16_reserve_enabled(&GemmDispatch::from_env()) {
        return 0;
    }
    let bytes = lazy_bf16_copy_bytes(
        store
            .names()
            .filter_map(|n| store.get(n).ok().map(|t| (n, t.dtype, t.shape.as_slice()))),
    );
    if bytes > 0 {
        tracing::info!(
            "lazy BF16 copies (ATLAS_CUBLAS_GEMM / ATLAS_CUTLASS_GEMM / ATLAS_FP8_ROWWISE): reserving \
             {:.1} GB outside the KV pool for dequantised FP8 weights",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    bytes
}

/// Bytes of lazy BF16 copies `dequant_fp8_bf16_cached` can materialise from
/// the given `(name, dtype, shape)` tensors: `numel * 2` for every FP8 E4M3
/// tensor with ≥ 2 dims — EXCEPT routed experts (`.experts.` in the name),
/// which run the MoE kernels and never reach the dequant; counting them would
/// double-book the whole expert set. 1-D scale tensors and every non-FP8
/// dtype are excluded too; every other FP8 matrix (including `shared_expert`)
/// is counted, which is the documented superset.
pub fn lazy_bf16_copy_bytes<'a>(
    tensors: impl IntoIterator<Item = (&'a str, WeightDtype, &'a [usize])>,
) -> usize {
    tensors
        .into_iter()
        .filter(|(name, dtype, shape)| {
            !name.contains(".experts.") && matches!(dtype, WeightDtype::FP8E4M3) && shape.len() >= 2
        })
        .map(|(_, _, shape)| shape.iter().product::<usize>() * 2)
        .sum()
}

/// Whether any dispatch flag can route a projection through
/// `dequant_fp8_bf16_cached` this run: `ATLAS_CUBLAS_GEMM` and
/// `ATLAS_CUTLASS_GEMM` (both on [`GemmDispatch`]), plus `ATLAS_FP8_ROWWISE`,
/// whose prefill arms call `cublas_bf16_proj` from
/// `qwen3_ssm/trait_prefill_helper.rs` and `trait_prefill_proj.rs` — read at
/// the call sites straight from the env, so it is read from the env here too.
pub fn lazy_bf16_reserve_enabled(d: &GemmDispatch) -> bool {
    lazy_bf16_reserve_decision(d, fp8_rowwise_prefill())
}

/// Pure form (table-tested): the flags that reach `dequant_fp8_bf16_cached`.
pub(crate) fn lazy_bf16_reserve_decision(d: &GemmDispatch, fp8_rowwise: bool) -> bool {
    d.cublas_gemm || d.cutlass_gemm || fp8_rowwise
}

fn fp8_rowwise_prefill() -> bool {
    matches!(
        std::env::var("ATLAS_FP8_ROWWISE").ok().as_deref(),
        Some("1")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_bf16_copy_bytes_counts_only_fp8_matrices() {
        let fp8_2d = [4usize, 8];
        let bf16_2d = [4usize, 8];
        let u8_2d = [4usize, 8];
        let q2_2d = [4usize, 8];
        let fp8_1d = [32usize];
        // FP8 2-D counted at numel*2; BF16 / UInt8 / PackedQ2_0 2-D and FP8
        // 1-D excluded.
        let got = lazy_bf16_copy_bytes([
            ("w.fp8", WeightDtype::FP8E4M3, fp8_2d.as_slice()),
            ("w.bf16", WeightDtype::BF16, bf16_2d.as_slice()),
            ("w.u8", WeightDtype::UInt8, u8_2d.as_slice()),
            (
                "w.q2",
                WeightDtype::PackedQ2_0 { group: 32 },
                q2_2d.as_slice(),
            ),
            ("w.fp8_scale", WeightDtype::FP8E4M3, fp8_1d.as_slice()),
        ]);
        assert_eq!(got, 4 * 8 * 2);
    }

    #[test]
    fn lazy_bf16_copy_bytes_skips_routed_experts_only() {
        let m = [4usize, 8];
        let got = lazy_bf16_copy_bytes([
            (
                "model.layers.0.mlp.experts.3.gate_proj.weight",
                WeightDtype::FP8E4M3,
                m.as_slice(),
            ),
            (
                "model.layers.0.mlp.shared_expert.gate_proj.weight",
                WeightDtype::FP8E4M3,
                m.as_slice(),
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                WeightDtype::FP8E4M3,
                m.as_slice(),
            ),
        ]);
        assert_eq!(got, 2 * 4 * 8 * 2);
    }

    #[test]
    fn lazy_bf16_copy_bytes_sums_multiple_tensors() {
        let a = [2usize, 3];
        let b = [5usize, 7];
        let got = lazy_bf16_copy_bytes([
            ("a.weight", WeightDtype::FP8E4M3, a.as_slice()),
            ("b.weight", WeightDtype::FP8E4M3, b.as_slice()),
        ]);
        assert_eq!(got, (2 * 3 + 5 * 7) * 2);
    }

    #[test]
    fn lazy_bf16_reserve_decision_table() {
        let off = GemmDispatch::defaults();
        assert!(!lazy_bf16_reserve_decision(&off, false));
        assert!(lazy_bf16_reserve_decision(&off, true));
        let cublas = GemmDispatch {
            cublas_gemm: true,
            ..GemmDispatch::defaults()
        };
        assert!(lazy_bf16_reserve_decision(&cublas, false));
        let cutlass = GemmDispatch {
            cutlass_gemm: true,
            ..GemmDispatch::defaults()
        };
        assert!(lazy_bf16_reserve_decision(&cutlass, false));
        // Unrelated flags do not reach the cache.
        let nvfp4 = GemmDispatch {
            cutlass_nvfp4_gemm: true,
            ..GemmDispatch::defaults()
        };
        assert!(!lazy_bf16_reserve_decision(&nvfp4, false));
    }
}
