// SPDX-License-Identifier: AGPL-3.0-only

//! `bailing_moe_v3_vl` (Ling-3.2-air) config parsing, against the verbatim
//! `config.json` of the plan-of-record checkpoint
//! `primitive-ai/Ling-3.2-air-NVFP4`.

use super::*;
use crate::config::parse_config;

/// The checkpoint's config.json, verbatim (no rope_scaling: native context
/// 262,144; routed experts NVFP4, everything else BF16).
const CONFIG: &str = r#"
{
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
}
"#;

fn text_config_with(edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> String {
    let mut raw: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    edit(raw["text_config"].as_object_mut().unwrap());
    raw.to_string()
}

fn edit_full_config(edit: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut raw: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    edit(&mut raw);
    raw.to_string()
}

#[test]
fn checkpoint_maps_hybrid_kda_mla_moe_shape() {
    let cfg = parse_config(CONFIG).unwrap();
    assert_eq!(cfg.model_type, "bailing_moe_v3_vl");
    assert!(cfg.nested_config);
    // Decoder under `model.*` directly (model.word_embeddings /
    // model.layers.N), so the weight prefix is empty.
    assert!(cfg.weight_prefix.is_empty());
    assert_eq!(cfg.layer_prefix(3), "model.layers.3");

    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.num_hidden_layers, 42);
    assert_eq!(cfg.intermediate_size, 6144);
    assert_eq!(cfg.vocab_size, 157_184);
    assert_eq!(cfg.max_position_embeddings, 262_144);
    assert_eq!(cfg.rms_norm_eps, 1e-6);
    assert_eq!(cfg.rope_theta, 6_000_000.0);
    assert_eq!(cfg.eos_token_id, 156_895);
    assert_eq!(cfg.bos_token_id, 0);
    assert!(!cfg.tie_word_embeddings);
    // No rope_scaling in this checkpoint: YaRN stays off.
    assert_eq!(cfg.yarn_factor, 0.0);

    // Gated-MLA attention (7 layers + the NEXTN draft layer 42).
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 32);
    assert_eq!(cfg.kv_lora_rank, 512);
    assert_eq!(cfg.q_lora_rank, 0);
    assert_eq!(cfg.qk_nope_head_dim, 128);
    assert_eq!(cfg.qk_rope_head_dim, 64);
    assert_eq!(cfg.v_head_dim, 128);
    // Expanded MLA Q/K head width; the config literal head_dim=128 is the
    // per-head V/output width, carried by v_head_dim.
    assert_eq!(cfg.head_dim, 192);
    assert_eq!(cfg.partial_rotary_factor, 0.5);
    assert_eq!(cfg.rotary_dim, 64);
    assert_eq!(cfg.rotary_dim(), 64);
    assert_eq!(cfg.mrope_section, [8, 12, 12]);
    assert!(!cfg.mrope_interleaved);

    // KDA linear attention (35 layers).
    assert_eq!(cfg.linear_num_key_heads, 32);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_num_value_heads, 32);
    assert_eq!(cfg.linear_value_head_dim, 128);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.kda_gate_lower_bound, -5.0);
    assert!(cfg.gdn_norm_sigmoid);

    // MoE: 512 experts top-8, sigmoid + selection-only bias, FP32 logits.
    assert_eq!(cfg.num_experts, 512);
    assert_eq!(cfg.n_routed_experts, 512);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.moe_intermediate_size, 768);
    assert_eq!(cfg.shared_expert_intermediate_size, 768);
    assert_eq!(cfg.moe_shared_expert_intermediate_size, 768);
    assert_eq!(cfg.scoring_func, "sigmoid");
    assert!(cfg.use_routing_bias);
    assert!(cfg.router_logits_f32);
    assert_eq!(cfg.router_logit_bytes(), 4);
    assert!(cfg.norm_topk_prob);
    assert_eq!(cfg.routed_scaling_factor, 2.5);
    assert_eq!(cfg.mlp_only_layers, vec![0, 1]);

    // ONE NEXTN MTP layer (model.layers.42), gated-MLA shaped.
    assert_eq!(cfg.num_mtp_modules, 1);
    assert_eq!(cfg.mtp_num_hidden_layers, 1);

    // The head-wise gate is a separate g_proj on the attention OUTPUT, not
    // the interleaved Q+gate layout `attn_gated` names. Text-only phase:
    // the ViT tower + projector are skipped by the family loader.
    assert!(!cfg.attn_gated);
    assert!(cfg.vision.is_none());

    let quant = cfg
        .quantization_config
        .expect("compressed-tensors quant config");
    assert_eq!(quant.quant_method, "compressed-tensors");
    assert_eq!(quant.format, "nvfp4-pack-quantized");
    assert_eq!(quant.group_size, 16);
}

#[test]
fn layer_layout_follows_the_modeling_group_rule() {
    let cfg = parse_config(CONFIG).unwrap();
    assert_eq!(cfg.layer_types.len(), 42);
    // (i + 1) % layer_group_size == 0 → gated MLA: {5, 11, ..., 35, 41}.
    let full: Vec<usize> = (0..42)
        .filter(|&i| cfg.layer_type(i) == LayerType::FullAttention)
        .collect();
    assert_eq!(full, vec![5, 11, 17, 23, 29, 35, 41]);
    assert_eq!(cfg.num_attention_layers(), 7);
    assert_eq!(cfg.num_ssm_layers(), 35);
    assert!(cfg.has_recurrent_state());
    assert_eq!(cfg.full_attention_interval, 6);
}

fn refuses(check: &str, edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) {
    let err = parse_config(&text_config_with(edit))
        .unwrap_err()
        .to_string();
    assert!(!err.is_empty(), "{check}: expected a refusal");
}

#[test]
fn rejects_unsupported_variants() {
    refuses("score_function", |t| {
        t.insert("score_function".into(), "softmax".into());
    });
    refuses("router_dtype", |t| {
        t.insert("router_dtype".into(), "bfloat16".into());
    });
    refuses("no_kda_lora", |t| {
        t.insert("no_kda_lora".into(), false.into());
    });
    refuses("kda_safe_gate", |t| {
        t.insert("kda_safe_gate".into(), false.into());
    });
    refuses("gating_granularity", |t| {
        t.insert(
            "gated_attention_proj_granularity_type".into(),
            "element_wise".into(),
        );
    });
    refuses("mtp_use_kda", |t| {
        t.insert("mtp_use_kda".into(), true.into());
    });
}

#[test]
fn rejects_malformed_routing_and_limit_lists() {
    // n_group must divide num_experts.
    let json = text_config_with(|t| {
        t.insert("n_group".into(), 7.into());
    });
    assert!(
        parse_config(&json)
            .unwrap_err()
            .to_string()
            .contains("n_group")
    );

    // SwiGLU limit lists must be num_hidden_layers long.
    let json = text_config_with(|t| {
        let short: Vec<serde_json::Value> =
            (0..41).map(|i| serde_json::Value::from(i.min(1))).collect();
        t.insert("expert_swiglu_limit_list".into(), short.into());
    });
    assert!(
        parse_config(&json)
            .unwrap_err()
            .to_string()
            .contains("expert_swiglu_limit_list")
    );
}

#[test]
fn inner_model_type_is_optional_but_must_be_bailing_hybrid() {
    let json = text_config_with(|t| {
        t.insert("model_type".into(), "something_else".into());
    });
    let err = parse_config(&json).unwrap_err().to_string();
    assert!(err.contains("bailing_hybrid"), "{err}");

    // The NVFP4 build omits the inner key entirely: accepted.
    let json = text_config_with(|t| {
        t.remove("model_type");
    });
    assert_eq!(parse_config(&json).unwrap().model_type, "bailing_moe_v3_vl");
}

#[test]
fn yarn_extension_is_recorded_when_present() {
    // The official FP8 build wraps deepseek_yarn factor 4; the NVFP4
    // plan-of-record build does not.
    let json = edit_full_config(|raw| {
        let scaling = serde_json::json!({
            "type": "deepseek_yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 262_144
        });
        raw["text_config"]["rope_scaling"] = scaling;
    });
    let cfg = parse_config(&json).unwrap();
    assert_eq!(cfg.yarn_factor, 4.0);
    assert_eq!(cfg.yarn_original_max_position_embeddings, 262_144);

    let json = edit_full_config(|raw| {
        raw["text_config"]["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 4.0});
    });
    let err = parse_config(&json).unwrap_err().to_string();
    assert!(err.contains("rope_scaling"), "{err}");
}

#[test]
fn layer_group_size_drives_the_full_attention_stride() {
    // Sanity of the derivation rule itself: group 7 over 42 layers puts
    // full attention at {6, 13, 20, 27, 34, 41} and KDA elsewhere.
    let json = text_config_with(|t| {
        t.insert("layer_group_size".into(), 7.into());
    });
    let cfg = parse_config(&json).unwrap();
    let full: Vec<usize> = (0..42)
        .filter(|&i| cfg.layer_type(i) == LayerType::FullAttention)
        .collect();
    assert_eq!(full, vec![6, 13, 20, 27, 34, 41]);
}
