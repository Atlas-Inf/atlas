// SPDX-License-Identifier: AGPL-3.0-only

//! `ModeloptQuantPlan` parsing + the native-consumer registry.

use super::*;
use atlas_core::config::{ModelConfig, QuantLayerSpec, QuantizationConfig};
use std::collections::BTreeMap;

fn spec(algo: &str) -> QuantLayerSpec {
    QuantLayerSpec {
        quant_algo: algo.into(),
        group_size: 0,
        weight_scale_encoding: None,
        weight_block_size: vec![],
    }
}

fn cfg_with(layers: &[(&str, QuantLayerSpec)]) -> ModelConfig {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let mut map = BTreeMap::new();
    for (k, v) in layers {
        map.insert(k.to_string(), v.clone());
    }
    cfg.quantization_config = Some(QuantizationConfig {
        quant_method: "modelopt".into(),
        quant_algo: "MIXED_PRECISION".into(),
        format: String::new(),
        ignore_modules: vec![],
        weight_block_size: vec![],
        group_size: 0,
        quantized_layers: map,
        exl3: None,
    });
    cfg
}

/// Flash-Next spellings: `group_size`, `weight_scale_encoding: "csf"`,
/// `W4A16_NVFP4`, one plain NVFP4.
#[test]
fn parses_the_flash_next_map() {
    let mut mx = spec("MXFP8");
    mx.group_size = 32;
    let mut nv4 = spec("NVFP4");
    nv4.group_size = 16;
    nv4.weight_scale_encoding = Some("csf".into());
    let mut w4 = spec("W4A16_NVFP4");
    w4.group_size = 16;
    let mut plain = spec("NVFP4");
    plain.group_size = 16;
    let mut cfg = cfg_with(&[
        ("model.language_model.layers.0.self_attn.q_proj", mx),
        ("model.language_model.layers.0.mlp.experts", nv4),
        ("model.visual.blocks.0.mlp.linear_fc1", w4),
        ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding", plain),
    ]);
    cfg.model_type = "qwen4_exp".into();
    let plan = ModeloptQuantPlan::from_config(&cfg).unwrap();
    assert!(plan.is_fp4_local());
    assert_eq!(
        plan.format_of("model.language_model.layers.0.mlp.experts")
            .unwrap()
            .algo,
        QuantAlgo::Nvfp4
    );
    assert_eq!(
        plan.format_of("model.visual.blocks.0.mlp.linear_fc1")
            .unwrap()
            .algo,
        QuantAlgo::W4A16Nvfp4
    );
    assert_eq!(
        plan.format_of("model.language_model.layers.0.self_attn.q_proj")
            .unwrap()
            .group_size,
        32
    );
}

/// The other family's spellings: MXFP8 as `weight_block_size [1,32]`,
/// `FP8_PB_WO` with `[128,128]`, and `csf_32x2_v1` — all parsed, none
/// guessed at.
#[test]
fn parses_the_csf_32x2_fixture_map() {
    let mut mx = spec("MXFP8");
    mx.weight_block_size = vec![1, 32];
    let mut pb = spec("FP8_PB_WO");
    pb.weight_block_size = vec![128, 128];
    let mut csf2 = spec("NVFP4");
    csf2.weight_scale_encoding = Some("csf_32x2_v1".into());
    let mut mx_plain = spec("MXFP8");
    mx_plain.group_size = 32;
    let cfg = cfg_with(&[
        ("model.language_model.layers.0.mlp.gate_proj", mx),
        ("model.language_model.layers.0.mlp.up_proj", pb),
        ("model.language_model.layers.0.mlp.experts", csf2),
        ("lm_head", mx_plain),
    ]);
    let plan = ModeloptQuantPlan::from_config(&cfg).unwrap();
    assert!(plan.is_fp4_local());
    let gate = plan.format_of("model.language_model.layers.0.mlp.gate_proj").unwrap();
    assert_eq!(gate.algo, QuantAlgo::Mxfp8);
    assert_eq!(gate.group_size, 32);
    assert_eq!(
        plan.format_of("model.language_model.layers.0.mlp.up_proj")
            .unwrap()
            .algo,
        QuantAlgo::Fp8Block { bn: 128, bk: 128 }
    );
    assert_eq!(
        plan.format_of("model.language_model.layers.0.mlp.experts")
            .unwrap()
            .scale_encoding
            .as_deref(),
        Some("csf_32x2_v1")
    );
}

/// The qwen4_exp registry covers exactly the modules its loaders run
/// natively — no more, no less.
#[test]
fn qwen4_exp_registry_coverage() {
    let n = native_consumers("qwen4_exp");
    // MXFP8 natives: GDN + attention + shared expert + vision attention.
    for m in [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.self_attn.o_proj",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj",
        "model.visual.blocks.3.attn.qkv",
        "model.visual.blocks.3.attn.proj",
    ] {
        assert!(n.covers(m, &QuantAlgo::Mxfp8), "{m}");
    }
    // MXFP8 NOT native (expands): indexer, dense MLP, visual MLP, MTP, misc.
    for m in [
        "model.language_model.layers.0.self_attn.indexer.index_qk_proj",
        "model.language_model.layers.0.mlp.gate_proj",
        "model.visual.blocks.3.mlp.linear_fc1",
        "mtp.layers.0.mlp.experts",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding",
    ] {
        assert!(!n.covers(m, &QuantAlgo::Mxfp8), "{m}");
    }
    // NVFP4 natives: experts (LM + MTP), vision linears/mergers, PLE table.
    for m in [
        "model.language_model.layers.0.mlp.experts",
        "mtp.layers.0.mlp.experts",
        "model.visual.blocks.0.mlp.linear_fc2",
        "model.visual.merger.linear_fc1",
        "model.language_model.visual.merger.linear_fc2",
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding",
    ] {
        assert!(n.covers(m, &QuantAlgo::W4A16Nvfp4), "{m}");
    }
    // Not covered: LM projections, lm_head.
    for m in [
        "model.language_model.layers.0.self_attn.q_proj",
        "lm_head",
    ] {
        assert!(!n.covers(m, &QuantAlgo::W4A16Nvfp4), "{m}");
    }
    // An unknown model type covers nothing.
    let none = native_consumers("some_new_model");
    assert!(!none.covers(
        "model.language_model.layers.0.self_attn.q_proj",
        &QuantAlgo::Mxfp8
    ));
    assert!(!none.covers(
        "model.language_model.layers.0.mlp.experts",
        &QuantAlgo::Nvfp4
    ));
}

/// No `quantized_layers` (or no quant config) → no plan, no FP4-Local.
#[test]
fn absent_map_means_no_plan() {
    assert!(ModeloptQuantPlan::from_config(&ModelConfig::qwen3_next_80b_nvfp4()).is_none());
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.quantization_config = Some(QuantizationConfig {
        quant_method: "modelopt".into(),
        quant_algo: "NVFP4".into(),
        format: String::new(),
        ignore_modules: vec![],
        weight_block_size: vec![],
        group_size: 16,
        quantized_layers: BTreeMap::new(),
        exl3: None,
    });
    assert!(ModeloptQuantPlan::from_config(&cfg).is_none());
}
