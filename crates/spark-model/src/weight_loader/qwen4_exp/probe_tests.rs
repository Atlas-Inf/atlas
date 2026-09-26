// SPDX-License-Identifier: AGPL-3.0-only

//! `audit_namespace` must see every projection of an EXL3 checkpoint (`.trellis`
//! instead of `.weight`) and the deferred n-gram tables (never in the store map).

use super::*;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{DeferredTensor, WeightDtype, WeightTensor};

fn store(names: &[String], deferred: &[String]) -> WeightStore {
    let map = names
        .iter()
        .map(|n| {
            (
                n.clone(),
                WeightTensor {
                    ptr: DevicePtr::NULL,
                    shape: vec![1],
                    dtype: WeightDtype::BF16,
                },
            )
        })
        .collect();
    let mut s = WeightStore::from_map(map);
    for n in deferred {
        s.defer(
            n.clone(),
            DeferredTensor {
                path: "x.safetensors".into(),
                offset: 0,
                shape: vec![1],
                dtype: WeightDtype::Int16,
            },
        );
    }
    s
}

/// Two layers: GDN at 0, full attention + indexer at 1, PLE at decoder layer 1
/// (`ple_layer_ids` is ONE-indexed, exactly as `audit_namespace` consumes it).
fn config() -> ModelConfig {
    let mut c = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    c.num_hidden_layers = 2;
    c.weight_prefix = "model.language_model".into();
    c.ple_layer_ids = vec![1];
    c.split_ngram_parts = 2;
    c
}

fn n(name: String) -> String {
    name
}

#[test]
fn exl3_layout_is_counted() {
    let c = config();
    let l0 = c.layer_prefix(0);
    let l1 = c.layer_prefix(1);
    let names = [
        n(format!("{l0}.linear_attn.in_proj_qkv.trellis")),
        n(format!("{l1}.self_attn.q_proj.trellis")),
        n(format!("{l1}.self_attn.indexer.index_qk_proj.trellis")),
        n(format!("{l0}.mlp.experts.0.gate_proj.trellis")),
        n(format!("{l1}.mlp.experts.0.gate_proj.trellis")),
        n(format!("{l0}.attn_hyper_connection.hc_norm.weight")),
        n(format!("{l1}.attn_hyper_connection.hc_norm.weight")),
        n(format!("{l0}.mlp_hyper_connection.hc_norm.weight")),
        n(format!("{l1}.mlp_hyper_connection.hc_norm.weight")),
        n("model.language_model.embed_tokens.weight".into()),
        n("lm_head.trellis".into()),
    ];
    // EXL3 ships the whole n-gram table as ONE deferred tensor, no shards.
    let deferred = [n(format!("{l1}.ple.ple_embedding.ngram_embedding.trellis"))];
    let r = audit_namespace(&store(&names, &deferred), &c);
    assert_eq!(r.gdn_layers, 1);
    assert_eq!(r.attn_layers, 1);
    assert_eq!(r.indexer_tensors, 1);
    assert_eq!(r.expert_tensors, 2);
    assert!(r.has_lm_head);
    assert_eq!(r.ple_shards, 2);
    assert!(r.ensure_loadable().is_ok());
}

#[test]
fn deferred_shards_are_counted() {
    let c = config();
    let l0 = c.layer_prefix(0);
    let l1 = c.layer_prefix(1);
    let names = [
        n(format!("{l0}.linear_attn.in_proj_qkv.weight")),
        n(format!("{l1}.self_attn.q_proj.weight")),
        n(format!("{l1}.self_attn.indexer.index_qk_proj.weight")),
        n(format!("{l0}.mlp.experts.0.gate_proj.weight")),
        n(format!("{l1}.mlp.experts.0.gate_proj.weight")),
        n(format!("{l0}.attn_hyper_connection.hc_norm.weight")),
        n(format!("{l1}.attn_hyper_connection.hc_norm.weight")),
        n(format!("{l0}.mlp_hyper_connection.hc_norm.weight")),
        n(format!("{l1}.mlp_hyper_connection.hc_norm.weight")),
        n("model.language_model.embed_tokens.weight".into()),
        n("lm_head.weight".into()),
    ];
    let base = format!("{l1}.ple.ple_embedding.ngram_embedding");
    let deferred = [
        n(format!("{base}.shard_0.weight")),
        n(format!("{base}.shard_1.weight")),
    ];
    let r = audit_namespace(&store(&names, &deferred), &c);
    assert_eq!(r.ple_shards, 2);
}

#[test]
fn missing_expert_still_refuses() {
    let c = config();
    let l0 = c.layer_prefix(0);
    let l1 = c.layer_prefix(1);
    let names = [
        n(format!("{l0}.linear_attn.in_proj_qkv.trellis")),
        n(format!("{l1}.self_attn.q_proj.trellis")),
        n(format!("{l0}.mlp.experts.0.gate_proj.trellis")),
        n(format!("{l0}.attn_hyper_connection.hc_norm.weight")),
        n(format!("{l0}.mlp_hyper_connection.hc_norm.weight")),
        n("model.language_model.embed_tokens.weight".into()),
        n("lm_head.trellis".into()),
    ];
    let r = audit_namespace(&store(&names, &[]), &c);
    assert_eq!(r.expert_tensors, 1);
    assert!(r.ensure_loadable().is_err());
}
