// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_SLAB_PACK=1` (default off): the fast loader packs each
//! shard's kept tensors into one allocation (`FastSafetensorsLoader::
//! pack_slab`) instead of one `cuMemAlloc` a tensor.
//!
//! Why: a per-tensor `cuMemAlloc` of 1 MB or more is rounded to 2 MB pages,
//! and a routed expert's packed projection is 0.8 MB with a 0.1 MB scale.
//! On the TP2 pair (nvidia/Qwen3.8-Flash-Next-FP4-Local, 2026-11-16) packing
//! took the pre-KV footprint from 60.97 GB to 55.76 GB a rank and the KV pool
//! from 152,607 to 179,403 blocks, and the routed MoE's 34 GB of weights read
//! 4-5% faster (`scripts/dev/qwen4exp_moe_c8_bench.cu`, `POOLS=48 SLAB=1`).
//!
//! Why it is safe here, on one rank or the TP=EP=2 pair: every packed tensor
//! pointer is strictly interior to its slab (`slab_layout` reserves offset
//! 0), and the CUDA backend's `free` of an interior pointer is a no-op. The
//! frees this loader issues are of its own allocations (`gpu_concat_rows`
//! buffers, the per-rank slices `tp_shard` copies out at tp > 1) or go
//! through `WeightStore::reclaim` / `free_loader_source`, which under a slab
//! return no memory and free nothing. So a reclaim no longer shrinks the
//! footprint; the measured footprint above is net of that.

use atlas_core::config::ModelConfig;

/// Whether this model's shards are slab-packed: the switch, on a qwen4_exp
/// model. Anything but `1` is off.
pub(super) fn requested(config: &ModelConfig) -> bool {
    from(
        std::env::var("ATLAS_QWEN4EXP_SLAB_PACK").ok().as_deref(),
        &config.model_type,
    )
}

fn from(value: Option<&str>, model_type: &str) -> bool {
    value == Some("1") && model_type == "qwen4_exp"
}

#[cfg(test)]
mod tests {
    use super::from;

    #[test]
    fn packing_is_opt_in_and_qwen4exp_only() {
        assert!(!from(None, "qwen4_exp"));
        assert!(!from(Some("0"), "qwen4_exp"));
        assert!(!from(Some("true"), "qwen4_exp"));
        assert!(from(Some("1"), "qwen4_exp"));
        assert!(!from(Some("1"), "qwen3_5"));
    }
}
