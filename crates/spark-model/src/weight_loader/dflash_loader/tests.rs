// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter loader tests. Split out of `dflash_loader.rs` for the
//! 500-LoC cap.

use super::*;

/// Smoke-test the DFlash drafter `config.json` parser against the live
/// `z-lab/Qwen3.6-35B-A3B-DFlash` checkpoint downloaded into the user's
/// HF cache. Skipped when the cache directory isn't populated — keeps
/// CI hermetic. Asserts the locked drafter dimensions: 8 layers,
/// hidden=2048, vocab=248320, γ=16, mask=248070, layer_ids=[1,10,19,28,37].
#[test]
fn parse_qwen3_6_35b_dflash_config() {
    const SNAP: &str = "/workspace/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719/config.json";
    let json = match std::fs::read_to_string(SNAP) {
        Ok(s) => s,
        Err(_) => {
            tracing::warn!("Skipping: drafter snapshot not in cache");
            return;
        }
    };
    let config = parse_dflash_config(&json).expect("parse drafter config");
    assert_eq!(config.num_hidden_layers, 8);
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.intermediate_size, 6144);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 248320);
    assert!(!config.tie_word_embeddings);
    assert_eq!(config.block_size, 16);
    let sub = config.dflash_config.expect("dflash_config present");
    assert_eq!(sub.mask_token_id, 248070);
    assert_eq!(sub.target_layer_ids, vec![1, 10, 19, 28, 37]);
}

/// A DFlash2 drafter must be refused by name, and a v1 drafter (the DSpark
/// layout) must not be. Regression: the DFlash2 tensors used to be silently
/// ignored, and the head then ran a forward it was never built for — a
/// sticky CUDA 700 on the first propose.
#[test]
fn dflash2_architecture_is_refused_and_v1_is_not() {
    use spark_runtime::gpu::DevicePtr;
    use spark_runtime::weights::{WeightDtype, WeightTensor};
    use std::collections::HashMap;

    let dummy = || WeightTensor {
        ptr: DevicePtr::NULL,
        shape: vec![1],
        dtype: WeightDtype::BF16,
    };
    let store = |keys: &[&str]| {
        WeightStore::from_map(
            keys.iter()
                .map(|k| (k.to_string(), dummy()))
                .collect::<HashMap<_, _>>(),
        )
    };

    let dflash2 = store(&["fc.weight", "candidate_selector.hidden_projection.weight"]);
    let (probe, what) = unsupported_dflash_marker(&dflash2, "").expect("DFlash2 refused");
    assert_eq!(probe, "candidate_selector.hidden_projection.weight");
    assert!(
        what.contains("selector"),
        "reason names the feature: {what}"
    );

    let v1 = store(&["fc.weight", "markov_head.markov_w1.weight"]);
    assert!(unsupported_dflash_marker(&v1, "").is_none());
}

#[test]
fn parse_lightning_dspark_config() {
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/lightning_dspark_config.json"
    ));
    let config = parse_dflash_config(json).expect("parse Lightning DSpark config");
    assert_eq!(config.num_hidden_layers, 6);
    assert_eq!(config.hidden_size, 2688);
    assert_eq!(config.intermediate_size, 6144);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 131072);
    assert_eq!(config.block_size, 8);
    assert_eq!(config.markov_rank, Some(512));
    let sub = config.dflash_config.expect("dflash_config present");
    assert_eq!(sub.mask_token_id, 990);
    assert_eq!(sub.target_layer_ids, vec![1, 5, 19, 29, 41, 51]);
    assert_eq!(sub.causal, Some(true));
    assert_eq!(sub.use_swa, Some(true));
    assert_eq!(sub.swa_window_size, Some(1024));
    assert_eq!(sub.attention_sink_bias, Some(true));
}
