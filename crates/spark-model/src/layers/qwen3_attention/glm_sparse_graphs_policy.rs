// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit graph opt-in; the eager semantic-index path remains the oracle.

use anyhow::{Result, ensure};
use spark_runtime::kv_cache::PagedKvCache;

/// The single-sequence (C1) decode selector bakes host positions into its
/// launches — index length, logits stride and the dense/sparse branch (see
/// `glm_index_decode_update_and_select`) — so a captured C1 graph would replay
/// a stale selection on every later step of its slot. Any cache carrying the
/// GLM semantic index keeps C1 eager; device-length selectors exist only on
/// the explicit multi-seq graph lane.
pub fn glm_c1_decode_graph_vetoed(kv_cache: &PagedKvCache) -> bool {
    kv_cache.sparse_index_config().is_some()
}

fn graph_policy(model_type: &str, sparse: bool, graphs: bool) -> Result<bool> {
    if model_type != "glm5_next" {
        return Ok(false);
    }
    ensure!(
        !graphs || sparse,
        "ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS=1 requires ATLAS_GLM_MULTI_SEQ_SPARSE=1"
    );
    Ok(graphs)
}

pub fn glm_multi_seq_sparse_graphs_enabled(model_type: &str) -> Result<bool> {
    graph_policy(
        model_type,
        super::glm_multi_seq_sparse_enabled(model_type),
        std::env::var("ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS").as_deref() == Ok("1"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_opt_in_requires_eager_feature_and_exact_model() {
        assert!(!graph_policy("glm5_next", false, false).unwrap());
        assert!(!graph_policy("glm5_next", true, false).unwrap());
        assert!(graph_policy("glm5_next", true, true).unwrap());
        assert!(graph_policy("glm5_next", false, true).is_err());
        assert!(!graph_policy("qwen3_next", true, true).unwrap());
        assert!(!graph_policy("deepseek_v4", false, true).unwrap());
    }

    #[test]
    fn c1_decode_graphs_are_vetoed_by_the_semantic_index() {
        use spark_runtime::gpu::mock::MockGpuBackend;
        use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, SparseIndexCacheConfig};
        let gpu = MockGpuBackend::new();
        let config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 64,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mut cache = PagedKvCache::new(config, 4, &gpu).unwrap();
        assert!(!glm_c1_decode_graph_vetoed(&cache));
        cache
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), &gpu)
            .unwrap();
        assert!(glm_c1_decode_graph_vetoed(&cache));
        // The C1 decode gate ANDs the veto into the one immutable `use_graphs`
        // it computes (a captured selector would replay the capture step's
        // positions forever).
        let decode = include_str!("../../model/trait_impl/decode_a.rs");
        let lets: Vec<_> = decode.match_indices("let use_graphs =").collect();
        assert_eq!(lets.len(), 1, "decode_a.rs must bind use_graphs once");
        assert!(!decode.contains("mut use_graphs"));
        let stmt = &decode[lets[0].0..];
        let stmt = &stmt[..stmt.find(';').unwrap()];
        let veto = stmt
            .find("&& !crate::layers::qwen3_attention::glm_c1_decode_graph_vetoed(&kv_cache)")
            .expect("the veto must be a conjunct of use_graphs");
        let head = &stmt[..veto];
        assert_eq!(
            head.matches('(').count(),
            head.matches(')').count(),
            "the veto must be a top-level conjunct"
        );
    }
}
