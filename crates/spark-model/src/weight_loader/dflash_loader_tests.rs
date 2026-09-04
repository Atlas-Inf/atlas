// SPDX-License-Identifier: AGPL-3.0-only

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

#[test]
fn parse_dflash2_qwen38_config() {
    let json = r#"{
      "architectures": ["DFlash2DraftModel"],
      "dflash_config": {
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 248070,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [5, 19, 33, 47, 61]
      },
      "hidden_size": 5120,
      "num_hidden_layers": 5,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "intermediate_size": 17408,
      "vocab_size": 248320
    }"#;
    let config = parse_dflash_config(json).expect("parse DFlash2 config");
    assert_eq!(
        config.architectures.as_deref(),
        Some(&["DFlash2DraftModel".to_string()][..])
    );
    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.num_hidden_layers, 5);
    assert_eq!(config.intermediate_size, 17408);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.vocab_size, 248320);
    let sub = config.dflash_config.clone().expect("dflash_config");
    assert_eq!(sub.block_size, Some(8));
    assert_eq!(sub.conv_group_size, Some(16));
    assert_eq!(sub.conv_kernel_size, Some(2));
    assert_eq!(sub.selector_rank, Some(256));
    assert_eq!(sub.selector_top_k, Some(16));
    assert_eq!(sub.target_layer_ids, vec![5, 19, 33, 47, 61]);
    assert_eq!(config.block_size(), 8);
}

#[test]
fn test_live_qwen38_dflash2_safetensors_keys() {
    let snap = "/home/azeez/code/hf/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4";
    if !std::path::Path::new(snap).exists() {
        return;
    }
    let config_json = std::fs::read_to_string(format!("{snap}/config.json")).expect("read config");
    let config = parse_dflash_config(&config_json).expect("parse config");
    assert!(config.is_dflash2());
    assert_eq!(config.hidden_size, 5120);
    assert_eq!(config.num_hidden_layers, 5);
    assert_eq!(config.block_size(), 8);

    let mut file =
        std::fs::File::open(format!("{snap}/model.safetensors")).expect("open safetensors");
    use std::io::Read;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .expect("read header len");
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf).expect("read header");
    let header_json: serde_json::Value = serde_json::from_slice(&header_buf).expect("parse header");
    let map = header_json.as_object().expect("header is json object");

    assert!(map.contains_key("fc.weight"));
    assert!(map.contains_key("hidden_norm.weight"));
    assert!(map.contains_key("norm.weight"));
    assert!(map.contains_key("candidate_selector.hidden_projection.weight"));
    assert!(map.contains_key("candidate_selector.predecessor_codebook"));
    assert!(map.contains_key("candidate_selector.successor_codebook"));

    for l in 0..5 {
        assert!(map.contains_key(&format!("layers.{l}.attention_conv.base_kernel")));
        assert!(map.contains_key(&format!(
            "layers.{l}.attention_conv.kernel_projection.weight"
        )));
        assert!(map.contains_key(&format!("layers.{l}.mlp_conv.base_kernel")));
        assert!(map.contains_key(&format!("layers.{l}.mlp_conv.kernel_projection.weight")));
    }
}
