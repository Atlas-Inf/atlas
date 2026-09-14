// SPDX-License-Identifier: AGPL-3.0-only

//! Parser tests for `qwen4_exp`, pinned to the real
//! `RadixArk/Qwen3.8-Flash-Next-NVFP4` config.json values.

use super::*;
// `LayerType` is only needed by these assertions — the parser itself no
// longer references it, and `deny(warnings)` makes an unused import fatal.
use crate::config::LayerType;

/// A trimmed copy of the shipped config — every key the parser reads, with
/// the checkpoint's actual values, so a rename upstream fails here first.
fn raw_config() -> Value {
    serde_json::from_str(RAW).expect("fixture must be valid JSON")
}

/// Verbatim JSON rather than `serde_json::json!` — the macro blows the
/// recursion limit on a literal this deep, and a raw string is closer to
/// what the checkpoint actually ships.
const RAW: &str = r#"{
  "architectures": ["Qwen4ExpForConditionalGeneration"],
  "model_type": "qwen4_exp",
  "text_config": {
    "model_type": "qwen4_exp_text",
    "hidden_size": 2560,
    "num_hidden_layers": 4,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 248320,
    "eos_token_id": 248044,
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 262144,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "full_attention_interval": 4,
    "layer_types": ["linear_attention", "linear_attention",
                    "linear_attention", "full_attention"],
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "hc_count": 4,
    "hc_lowrank": 320,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "indexer_head_dim": 128,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "ngram_size": 3,
    "ngram_vocab_size_base": 20000000,
    "heads_per_ngram": 8,
    "split_ngram_parts": 128,
    "make_ngram_vocab_size_divisible_by": 128,
    "ple_layer_ids": [2],
    "ple_embed_dim": 2560,
    "ple_conv_kernel_size": 4,
    "output_gate_type": "sigmoid",
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    }
  },
  "vision_config": {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "out_hidden_size": 2560,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  }
}"#;

#[test]
fn parses_the_shipped_checkpoint_config() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.model_type, "qwen4_exp", "top-level type, not *_text");
    assert!(c.nested_config);
    assert_eq!(c.hidden_size, 2560);
    assert_eq!(c.vocab_size, 248320);
    assert_eq!(c.head_dim, 256);
    assert_eq!(c.num_experts, 512);
    assert_eq!(c.num_experts_per_tok, 10);
    assert_eq!(c.moe_intermediate_size, 640);
    assert_eq!(c.shared_expert_intermediate_size, 640);
}

#[test]
fn hyper_connections_map_onto_the_deepseek_v4_fields() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.hc_mult, 4, "hc_count -> hc_mult");
    assert_eq!(c.hc_lowrank, 320, "selects the low-rank mixer variant");
}

/// A stream count with no mixer rank would silently take DeepSeek-V4's
/// Sinkhorn path and mix with the wrong math — refuse instead.
#[test]
fn partial_hyper_connection_config_is_refused() {
    let mut raw = raw_config();
    raw["text_config"]["hc_lowrank"] = Value::from(0);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("hyper-connection config is partial"), "{err}");
}

#[test]
fn indexer_maps_onto_index_fields_and_only_full_attention_layers() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.index_n_heads, 4);
    assert_eq!(c.index_head_dim, 128);
    assert_eq!(c.index_topk, 2048, "indexer_budget -> index_topk");
    assert_eq!(c.index_compress_ratio, 4);
    // `compress_ratios` stays EMPTY on purpose: a non-empty value turns on
    // `probes.compressed_attn` and dispatches DeepSeek-V4's compressor, which
    // is a different mechanism from Qwen's QSA indexer. Below the budget the
    // indexer is inert anyway — selection is
    // topk(min(budget/ratio, complete_blocks)), so at seq_len <= 2048 every
    // block is chosen and dense attention is EXACT.
    assert!(
        c.compress_ratios.is_empty(),
        "must not dispatch the V4 compressor for Qwen's indexer"
    );
}

#[test]
fn partial_indexer_config_is_refused() {
    let mut raw = raw_config();
    raw["text_config"]["indexer_head_dim"] = Value::from(0);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("indexer config is partial"), "{err}");
}

#[test]
fn ngram_geometry_matches_the_longcat_formula() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.emb_neighbor_num, 3, "ngram_size");
    assert_eq!(c.emb_split_num, 8, "heads_per_ngram");
    // 8 x (3-1) = 16 heads, 2560 / 16 = 160 dims each — the shard width in
    // the checkpoint.
    let heads = c.emb_split_num * (c.emb_neighbor_num - 1);
    assert_eq!(heads, 16);
    assert_eq!(c.hidden_size / heads, 160);
    assert_eq!(c.ngram_vocab_size_base, 20_000_000);
    assert_eq!(c.split_ngram_parts, 128);
    assert_eq!(c.ple_layer_ids, vec![2]);
    assert_eq!(c.ple_conv_kernel_size, 4);
}

/// The per-head slices are CONCATENATED (16 x 160 = 2560), not projected the
/// way LongCat's are, so a head count that does not divide hidden_size can
/// never reconstruct a hidden vector.
#[test]
fn ngram_head_count_must_divide_hidden_size() {
    let mut raw = raw_config();
    raw["text_config"]["heads_per_ngram"] = Value::from(7);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("must divide evenly"), "{err}");
    assert!(err.contains("concatenated, not projected"), "{err}");
}

#[test]
fn ple_layer_id_past_the_end_is_refused() {
    let mut raw = raw_config();
    raw["text_config"]["ple_layer_ids"] = Value::from(vec![99u64]);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("ONE-indexed"), "{err}");
    assert!(err.contains("[1, 4]"), "{err}");
}

/// Zero is not "layer 0" here, it is a malformed id -- the ids are
/// one-indexed, so `l < num_layers` alone would accept it and then look for
/// the tower one layer earlier than where it lives.
#[test]
fn a_zero_ple_layer_id_is_refused() {
    let mut raw = raw_config();
    raw["text_config"]["ple_layer_ids"] = Value::from(vec![0u64]);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("ONE-indexed"), "{err}");
}

#[test]
fn rope_reads_through_the_nested_rope_parameters() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert_eq!(c.rope_theta, 10_000_000.0);
    assert_eq!(c.partial_rotary_factor, 0.25);
    assert!(c.mrope_interleaved);
    assert_eq!(c.mrope_section, [11, 11, 10]);
}

#[test]
fn attention_is_gated_and_layer_types_survive() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert!(c.attn_gated, "output_gate_type: sigmoid");
    assert_eq!(c.layer_types.len(), 4);
    assert_eq!(c.layer_types[0], LayerType::LinearAttention);
    assert_eq!(c.layer_types[3], LayerType::FullAttention);
}

#[test]
fn layer_types_length_must_match_num_hidden_layers() {
    let mut raw = raw_config();
    raw["text_config"]["num_hidden_layers"] = Value::from(48);
    let err = parse_qwen4_exp(&raw).unwrap_err().to_string();
    assert!(err.contains("layer_types has 4 entries"), "{err}");
}

#[test]
fn missing_text_config_is_refused() {
    let err = parse_qwen4_exp(&serde_json::json!({"model_type": "qwen4_exp"}))
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing text_config"), "{err}");
}

#[test]
fn vision_tower_is_parsed() {
    let c = parse_qwen4_exp(&raw_config()).expect("parse");
    assert!(c.vision.is_some(), "vision_config sits at the TOP level");
}

/// The fixture above is a hand-trimmed copy, so it can drift from the real
/// file. This parses the SHIPPED config.json when the checkpoint is present
/// and skips otherwise — CI has no checkpoint, the GB10 box does.
///
///     ATLAS_QWEN4_EXP_CONFIG=/path/to/snapshot/config.json \
///       cargo test -p atlas-core --lib qwen4_exp
#[test]
fn parses_the_real_config_json_when_present() {
    let Ok(path) = std::env::var("ATLAS_QWEN4_EXP_CONFIG") else {
        eprintln!("ATLAS_QWEN4_EXP_CONFIG unset — skipping real-checkpoint parse");
        return;
    };
    let text = std::fs::read_to_string(&path).expect("read config.json");
    let raw: Value = serde_json::from_str(&text).expect("config.json is valid JSON");
    let c = parse_qwen4_exp(&raw).expect("real config.json must parse");

    // Values read off RadixArk/Qwen3.8-Flash-Next-NVFP4 on 2026-08-26.
    assert_eq!(c.num_hidden_layers, 48);
    assert_eq!(c.layer_types.len(), 48);
    assert_eq!(
        c.layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count(),
        12,
        "3 GDN : 1 full over 48 layers"
    );
    assert_eq!(c.hidden_size, 2560);
    assert_eq!(c.num_experts, 512);
    assert_eq!(c.hc_mult, 4);
    assert_eq!(c.hc_lowrank, 320);
    assert_eq!(c.index_topk, 2048);
    assert_eq!(
        c.emb_split_num * (c.emb_neighbor_num - 1),
        16,
        "n-gram heads"
    );
    assert_eq!(c.ple_layer_ids, vec![2]);
    assert!(c.vision.is_some());
}

/// Qwen3.8-Flash-Next shipped under `qwen3_8_flash_next` and was later
/// renamed `qwen4_exp`. Quantizers pinned to different transformers
/// revisions emit different names — RadixArk says `qwen4_exp`, Inferact says
/// `qwen3_8_flash_next` — but the two `text_config`s are otherwise identical
/// field-for-field. `parse_config` must route both to this parser.
#[test]
fn both_naming_revisions_route_to_this_parser() {
    for (model_type, arch) in [
        ("qwen4_exp", "Qwen4ExpForConditionalGeneration"),
        (
            "qwen3_8_flash_next",
            "Qwen3_8FlashNextForConditionalGeneration",
        ),
    ] {
        let mut raw = raw_config();
        raw["model_type"] = Value::from(model_type);
        raw["architectures"] = Value::from(vec![arch]);
        raw["text_config"]["model_type"] = Value::from(format!("{model_type}_text"));

        let c = crate::config::parse_config(&raw.to_string())
            .unwrap_or_else(|e| panic!("{model_type} must parse: {e:#}"));
        // Normalized to the canonical name so kernel-target resolution and
        // every downstream `model_type ==` check sees ONE family.
        assert_eq!(c.model_type, "qwen4_exp", "{model_type} normalizes");
        assert_eq!(c.hc_mult, 4);
        assert_eq!(c.num_experts, 512);
    }
}

/// **`quantization_config` lives at the TOP level, and this parser used to
/// return before reading it.**
///
/// ModelOpt writes it beside `text_config`, not inside — verified against both
/// vendored configs, whose `text_config` carries no `quant*` key at all. So
/// `serde_json::from_value(text_config)` cannot see it, and without the
/// `finalize_config` call at the end of `parse_qwen4_exp` an NVFP4 checkpoint
/// parses as UNQUANTIZED: 120.8 B of routed experts would be read as BF16.
///
/// This is the regression guard for that call. It runs against the real
/// published `config.json`, vendored whole, rather than a hand-written fixture
/// that could be written to agree with the bug.
#[test]
fn the_nvfp4_quantization_config_is_read_from_the_top_level() {
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen4_exp_flash_next_nvfp4_config.json"
    ));
    // The premise: it is NOT inside text_config, so serde alone cannot find it.
    let raw: Value = serde_json::from_str(json).expect("valid json");
    assert!(
        raw["text_config"]
            .as_object()
            .expect("text_config")
            .keys()
            .all(|k| !k.contains("quant")),
        "text_config gained a quant key — this test's premise needs rechecking"
    );
    assert!(
        raw.get("quantization_config").is_some(),
        "top-level, as ModelOpt writes it"
    );

    let cfg = crate::config::parse_config(json).expect("parses");
    let q = cfg
        .quantization_config
        .as_ref()
        .expect("quantization_config must survive the nested-config parse");
    assert_eq!(q.quant_method, "modelopt");
    assert_eq!(q.quant_algo, "NVFP4");
    // group 16 along the input dim is what makes a [640, 1280] expert weight
    // carry a [640, 160] E4M3 scale; a 0 here would mis-size every scale.
    assert_eq!(q.group_size, 16);
    // The ignore list is what keeps attention, GDN, mHC, PLE and the shared
    // experts in BF16 — only the routed experts are quantized.
    assert!(
        q.ignore_modules
            .iter()
            .any(|m| m.contains("hyper_connection")),
        "the mHC weights must be on the ignore list: {:?}",
        q.ignore_modules
    );
    assert!(q.ignore_modules.iter().any(|m| m.contains(".ple.")));
}

/// The FP8 release of the same model, for the same reason: its
/// `quantization_config` is also top-level, and its `weight_block_size` is what
/// tells the loader a quantized `[rows, cols]` weight carries a
/// `[ceil(rows/128), ceil(cols/128)]` scale sibling.
#[test]
fn the_fp8_release_carries_its_block_size_through_too() {
    let json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen4_exp_flash_next_config.json"
    ));
    let cfg = crate::config::parse_config(json).expect("parses");
    let q = cfg
        .quantization_config
        .as_ref()
        .expect("quantization_config");
    assert_eq!(q.quant_method, "fp8");
    assert_eq!(q.weight_block_size, vec![128, 128]);
}
