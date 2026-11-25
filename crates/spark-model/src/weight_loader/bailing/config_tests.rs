// SPDX-License-Identifier: AGPL-3.0-only

//! Config-parse tests against the VERBATIM `config.json` of
//! `primitive-ai/Ling-3.2-air-NVFP4` (the campaign ground-truth copy).

use super::*;
use serde_json::{Value, json};

/// The checkpoint's full config.json, byte-identical to the campaign
/// ground-truth copy at ~/atlas-campaign/context/ling-nvfp4-config.json.
const CHECKPOINT_CONFIG: &str = r#####"{
  "model_type": "bailing_moe_v3_vl",
  "architectures": [
    "BailingMoeV3VLForConditionalGeneration"
  ],
  "auto_map": {
    "AutoConfig": "configuration_bailing_moe_v3_vl.BailingMoeV3VLConfig",
    "AutoModelForCausalLM": "modeling_bailing_moe_v3_vl.BailingMoeV3VLForConditionalGeneration"
  },
  "torch_dtype": "bfloat16",
  "transformers_version": "4.56.2",
  "tie_word_embeddings": false,
  "norm_query_embeds": false,
  "image_token_id": 157157,
  "video_token_id": 156909,
  "vision_start_token_id": 157158,
  "vision_end_token_id": 157159,
  "mrope_section": [
    8,
    12,
    12
  ],
  "text_config": {
    "image_patch_token": 157157,
    "video_patch_token": 156909,
    "image_start_token": 157158,
    "video_start_token": 157160,
    "num_hidden_layers": 42,
    "hidden_size": 2560,
    "intermediate_size": 6144,
    "eos_token_id": 156895,
    "pad_token_id": 156892,
    "first_k_dense_replace": 2,
    "max_position_embeddings": 262144,
    "moe_intermediate_size": 768,
    "num_experts_per_tok": 8,
    "num_attention_heads": 32,
    "q_lora_rank": null,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "num_experts": 512,
    "num_key_value_heads": 32,
    "rope_theta": 6000000,
    "rms_norm_eps": 1e-06,
    "head_dim": 128,
    "vocab_size": 157184,
    "partial_rotary_factor": 0.5,
    "router_dtype": "fp32",
    "moe_router_enable_expert_bias": true,
    "routed_scaling_factor": 2.5,
    "n_group": 8,
    "topk_group": 4,
    "use_qk_norm": true,
    "score_function": "sigmoid",
    "moe_shared_expert_intermediate_size": 768,
    "layer_group_size": 6,
    "num_kv_heads_for_linear_attn": 0,
    "group_norm_size": 1,
    "linear_silu": true,
    "rotary_dim": 64,
    "use_mla_nope": false,
    "short_conv_kernel_size": 4,
    "use_nGPT": false,
    "scale_router_input": false,
    "value_norm": false,
    "up_proj_norm": false,
    "gated_attention_proj_granularity_type": "head_wise",
    "mtp_use_kda": false,
    "no_kda_lora": true,
    "use_kda_lora": false,
    "kda_safe_gate": true,
    "kda_lower_bound": -5.0,
    "norm_topk_prob": true,
    "expert_swiglu_limit_list": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      4,
      4,
      4,
      4,
      4,
      4,
      4
    ],
    "share_expert_swiglu_limit_list": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      5,
      5,
      5,
      5,
      5,
      5,
      7,
      7
    ],
    "num_nextn_predict_layers": 1,
    "mrope_section": [
      8,
      12,
      12
    ]
  },
  "vision_config": {
    "model_type": "qwen3_moe_vit",
    "depth": 27,
    "hidden_size": 1152,
    "hidden_act": "gelu_pytorch_tanh",
    "intermediate_size": 4304,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 4608,
    "num_position_embeddings": 2304,
    "initializer_range": 0.02,
    "disable_merger_proj": true,
    "deepstack_visual_indexes": []
  },
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "quantization_status": "compressed",
    "kv_cache_scheme": null,
    "config_groups": {
      "group_0": {
        "targets": [
          "re:.*\\.layers\\.(2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42)\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"
        ],
        "weights": {
          "num_bits": 4,
          "type": "float",
          "strategy": "tensor_group",
          "group_size": 16,
          "symmetric": true,
          "dynamic": false,
          "observer": "minmax"
        },
        "input_activations": {
          "num_bits": 4,
          "type": "float",
          "strategy": "tensor_group",
          "group_size": 16,
          "symmetric": true,
          "dynamic": "local",
          "observer": null
        },
        "output_activations": null
      }
    },
    "ignore": [
      "re:^(?!.*\\.mlp\\.experts(\\.|$)).*$",
      "lm_head"
    ]
  },
  "video_start_token_id": 157160,
  "video_end_token_id": 157161
}"#####;

fn parse() -> BailingConfig {
    let raw: Value = serde_json::from_str(CHECKPOINT_CONFIG).unwrap();
    BailingConfig::from_value(&raw).unwrap()
}

fn parse_raw() -> Value {
    serde_json::from_str(CHECKPOINT_CONFIG).unwrap()
}

#[test]
fn dims_parse_exactly() {
    let config = parse();
    assert_eq!(config.hidden_size, 2560);
    assert_eq!(config.num_hidden_layers, 42);
    assert_eq!(config.vocab_size, 157184);
    assert_eq!(config.intermediate_size, 6144);
    assert_eq!(config.moe_intermediate_size, 768);
    assert_eq!(
        config.moe_shared_expert_intermediate_size, 768,
        "ONE shared expert at the routed width"
    );
    assert_eq!(config.rms_norm_eps, 1e-6);
    assert_eq!(config.max_position_embeddings, 262144);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 32);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.kv_lora_rank, 512);
    assert_eq!(config.qk_nope_head_dim, 128);
    assert_eq!(config.qk_rope_head_dim, 64);
    assert_eq!(config.v_head_dim, 128);
    assert_eq!(config.rope_theta, 6_000_000.0);
    assert_eq!(config.rotary_dim, 64);
    assert_eq!(config.partial_rotary_factor, 0.5);
    assert!(config.use_qk_norm);
    assert_eq!(config.gated_attention_proj_granularity_type, "head_wise");
}

#[test]
fn kda_constants_parse_exactly() {
    let config = parse();
    assert_eq!(config.short_conv_kernel_size, 4);
    assert!(config.linear_silu);
    assert!(config.kda_safe_gate);
    assert_eq!(config.kda_lower_bound, -5.0);
    assert!(config.no_kda_lora, "Ling ships the lora-less KDA shapes");
}

#[test]
fn layout_matches_the_checkpoint_index() {
    let config = parse();
    // Ground truth: gated MLA at exactly these 7 decoder indices.
    let mla_layers: &[usize] = &[5, 11, 17, 23, 29, 35, 41];
    assert_eq!(config.num_hidden_layers, 42);
    for idx in 0..42 {
        let expected = if mla_layers.contains(&idx) {
            AttentionKind::GatedMla
        } else {
            AttentionKind::Kda
        };
        assert_eq!(config.attention_kind(idx), expected, "layer {idx}");
    }
    // first_k_dense_replace = 2: dense MLP on 0..1, MoE on 2..=41.
    for idx in 0..42 {
        let expected = if idx >= 2 {
            MlpKind::Moe
        } else {
            MlpKind::Dense
        };
        assert_eq!(config.mlp_kind(idx), expected, "layer {idx}");
    }
    // ONE NEXTN layer, checkpoint index 42.
    assert_eq!(config.num_nextn_predict_layers, 1);
    assert!(!config.mtp_use_kda);
    assert_eq!(
        config.mtp_layer_indices(),
        std::ops::Range { start: 42, end: 43 }
    );
}

#[test]
fn router_constants_parse_exactly() {
    let config = parse();
    assert_eq!(config.gating.num_experts, 512);
    assert_eq!(config.gating.top_k, 8);
    assert_eq!(config.gating.n_group, 8);
    assert_eq!(config.gating.topk_group, 4);
    assert_eq!(config.gating.experts_per_group(), 64);
    assert_eq!(config.gating.routed_scaling_factor, 2.5);
    assert!(config.gating.norm_topk_prob);
    assert!(config.gating.expert_bias, "moe_router_enable_expert_bias");
    assert!(config.gating.router_fp32);
}

#[test]
fn swiglu_lists_parse_exactly_from_text_config() {
    let config = parse();
    // expert_swiglu_limit_list: 4 on the LAST 7 MoE layers (35..=41).
    for idx in 0..42 {
        assert_eq!(
            config.swiglu.routed_limit(idx),
            (35..=41).contains(&idx).then_some(4),
            "routed layer {idx}"
        );
    }
    // share_expert_swiglu_limit_list: 5 on 34..=39, 7 on 40..=41.
    for idx in 0..42 {
        let expected = match idx {
            34..=39 => Some(5),
            40 | 41 => Some(7),
            _ => None,
        };
        assert_eq!(
            config.swiglu.shared_limit(idx),
            expected,
            "shared layer {idx}"
        );
    }
}

#[test]
fn tokens_mrope_and_rope_scaling() {
    let config = parse();
    assert_eq!(config.eos_token_id, 156895);
    assert_ne!(config.pad_token_id, 0);
    assert_eq!(config.mrope_section, [8, 12, 12]);
    // The NVFP4 checkpoint ships NO rope_scaling: native 262144 context only.
    assert!(config.rope_scaling.is_none());
}

#[test]
fn outer_wrapper_is_required() {
    let raw: Value = serde_json::from_str(CHECKPOINT_CONFIG).unwrap();
    // Rejects the inner text_config masquerading as the outer config.
    let text_only = raw.get("text_config").cloned().unwrap();
    assert!(BailingConfig::from_value(&text_only).is_err());
    // Rejects a foreign model_type.
    let mut foreign = raw.clone();
    foreign["model_type"] = Value::String("llama".into());
    assert!(BailingConfig::from_value(&foreign).is_err());
}

#[test]
fn a_yarn_checkpoint_parses_its_rope_scaling() {
    let mut raw = parse_raw();
    raw["text_config"]["rope_scaling"] = json!({
        "type": "deepseek_yarn",
        "rope_type": "deepseek_yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 262144
    });
    let config = BailingConfig::from_value(&raw).unwrap();
    let yarn = config.rope_scaling.expect("rope_scaling present");
    assert_eq!(yarn.rope_type, "deepseek_yarn");
    assert_eq!(yarn.factor, 4.0);
    assert_eq!(yarn.original_max_position_embeddings, 262144);
}

#[test]
fn mrope_section_read_from_either_level() {
    let mut raw = parse_raw();
    raw.as_object_mut()
        .unwrap()
        .remove("mrope_section")
        .unwrap();
    let kept = raw["text_config"]["mrope_section"].clone();
    assert_eq!(kept.as_array().map(|a| a.len()), Some(3));
    let config = BailingConfig::from_value(&raw).unwrap();
    assert_eq!(config.mrope_section, [8, 12, 12]);
}
