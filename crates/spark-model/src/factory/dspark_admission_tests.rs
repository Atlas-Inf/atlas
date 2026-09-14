// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use atlas_core::config::ModelConfig;
use serde_json::Value;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::DflashBuildArgs;
use super::dspark_admission::{
    LightningRuntimeAdmission, admit_lightning_dspark, admit_lightning_dspark_build,
};
use crate::weight_loader::DflashConfig;
use crate::weight_loader::dflash_loader::parse_dflash_config;

const OFFICIAL_LIGHTNING_JSON: &str = r#"
{
  "architectures": ["Qwen3DSparkModel"],
  "attention_sink_bias": true,
  "block_size": 8,
  "dflash_config": {
    "attention_sink_bias": true,
    "causal": true,
    "mask_token_id": 990,
    "swa_window_size": 1024,
    "target_layer_ids": [1, 5, 19, 29, 41, 51],
    "use_swa": true,
    "sample_from_anchor": false
  },
  "dspark_bonus_anchor": true,
  "dspark_markov_rank": 512,
  "markov_rank": 512,
  "target_layer_ids": [1, 5, 19, 29, 41, 51],
  "hidden_size": 2688,
  "num_hidden_layers": 6,
  "intermediate_size": 6144,
  "num_attention_heads": 32,
  "num_key_value_heads": 2,
  "head_dim": 128,
  "vocab_size": 131072,
  "sample_from_anchor": false,
  "quantization_config": {
    "quant_algo": "W4A16_NVFP4",
    "kv_cache_quant_algo": null
  }
}
"#;

pub(super) fn official_value() -> Value {
    serde_json::from_str(OFFICIAL_LIGHTNING_JSON).expect("official inline JSON")
}

pub(super) fn parse_value(value: &Value) -> DflashConfig {
    parse_dflash_config(&serde_json::to_string(value).unwrap()).expect("DFlash config")
}

fn runtime() -> LightningRuntimeAdmission {
    LightningRuntimeAdmission {
        served_gamma: 4,
        num_drafts: 3,
        physical_kv_page_size: 16,
        target_kv_dtype: KvCacheDtype::Fp8,
        tp: 1,
        ep: 1,
        fc_present: true,
        markov_w1_present: true,
        markov_w2_present: true,
        all_required_sinks_present: true,
        target_model_type_is_lightning: true,
        target_hidden_size: 2688,
        target_num_hidden_layers: 52,
        target_num_experts: 128,
        target_top_k: 6,
    }
}

pub(super) fn required_store() -> WeightStore {
    let mut weights = HashMap::new();
    for name in [
        "fc.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
    ] {
        weights.insert(
            name.to_owned(),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape: vec![1],
                dtype: WeightDtype::BF16,
            },
        );
    }
    for layer in 0..6 {
        weights.insert(
            format!("layers.{layer}.self_attn.attention_sink_bias"),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape: vec![1],
                dtype: WeightDtype::BF16,
            },
        );
    }
    WeightStore::from_map(weights)
}

fn admit(
    value: &Value,
    runtime: LightningRuntimeAdmission,
) -> anyhow::Result<Option<super::super::layers::dflash_head::LightningDsparkProfile>> {
    admit_lightning_dspark(&parse_value(value), runtime)
}

fn reject_metadata(mut value: Value, mutate: impl FnOnce(&mut Value), field: &str) {
    mutate(&mut value);
    let error = admit(&value, runtime()).expect_err("metadata drift must reject");
    assert!(format!("{error:#}").contains(field), "{error:#}");
}
#[test]
fn parses_actual_official_lightning_field_names_and_admits() {
    let config = parse_dflash_config(OFFICIAL_LIGHTNING_JSON).expect("official config");
    assert_eq!(
        config.architectures.as_ref().unwrap(),
        &vec!["Qwen3DSparkModel".to_owned()]
    );
    assert_eq!(config.dspark_bonus_anchor, Some(true));
    assert_eq!(config.sample_from_anchor, Some(false));
    assert_eq!(config.attention_sink_bias, Some(true));
    assert_eq!(config.dspark_markov_rank, Some(512));
    assert_eq!(
        config.target_layer_ids.as_deref(),
        Some([1, 5, 19, 29, 41, 51].as_slice())
    );
    assert_eq!(
        config
            .quantization_config
            .as_ref()
            .unwrap()
            .kv_cache_quant_algo,
        None
    );
    let sub = config.dflash_config.as_ref().unwrap();
    assert_eq!(sub.sample_from_anchor, Some(false));
    assert_eq!(sub.causal, Some(true));
    assert_eq!(sub.use_swa, Some(true));

    let profile = admit_lightning_dspark(&config, runtime())
        .expect("admission")
        .expect("Lightning profile");
    assert_eq!(profile.checkpoint.block_size, 8);
    assert_eq!(profile.checkpoint.physical_kv_page_size, 16);
    assert_eq!(profile.taps, vec![1, 5, 19, 29, 41, 51]);
    assert_eq!(profile.markov.rank, 512);
    assert_eq!(profile.kv.target, crate::layers::dflash_head::KvDtype::Fp8);
    assert_eq!(
        profile.kv.drafter,
        crate::layers::dflash_head::KvDtype::Bf16
    );
}
#[test]
fn generic_non_lightning_architecture_returns_none() {
    let mut value = official_value();
    value["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
    assert!(
        admit(&value, runtime())
            .expect("generic admission")
            .is_none()
    );

    for field in [
        "architectures",
        "dspark_bonus_anchor",
        "dspark_markov_rank",
        "markov_rank",
    ] {
        value.as_object_mut().unwrap().remove(field);
    }
    let store = required_store();
    let args = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: parse_value(&value),
        gamma: None,
        window_size: Some(4096),
    };
    let target = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(
        admit_lightning_dspark_build(&args, &target, 7, 16, KvCacheDtype::Fp8)
            .expect("legacy generic DFlash without architectures")
            .is_none()
    );
}
#[test]
fn missing_architecture_is_not_admitted_and_wrong_is_ignored() {
    let mut missing = official_value();
    missing.as_object_mut().unwrap().remove("architectures");
    let error = admit(&missing, runtime()).expect_err("missing architecture must reject");
    assert!(format!("{error:#}").contains("architectures"), "{error:#}");

    let mut wrong = official_value();
    wrong["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
    assert!(admit(&wrong, runtime()).unwrap().is_none());

    let mut ambiguous = official_value();
    ambiguous["architectures"] = serde_json::json!(["Qwen3DSparkModel", "Qwen3ForCausalLM"]);
    let error = admit(&ambiguous, runtime()).expect_err("ambiguous Lightning identity");
    assert!(format!("{error:#}").contains("exactly"), "{error:#}");
}
#[test]
fn rejects_every_metadata_profile_family_when_it_drifts() {
    reject_metadata(
        official_value(),
        |v| v["block_size"] = serde_json::json!(16),
        "block_size",
    );
    reject_metadata(
        official_value(),
        |v| v["dflash_config"]["causal"] = serde_json::json!(false),
        "causal",
    );
    reject_metadata(
        official_value(),
        |v| v["dflash_config"]["use_swa"] = serde_json::json!(false),
        "use_swa",
    );
    reject_metadata(
        official_value(),
        |v| v["dflash_config"]["swa_window_size"] = serde_json::json!(512),
        "swa_window",
    );
    reject_metadata(
        official_value(),
        |v| v["attention_sink_bias"] = serde_json::json!(false),
        "attention_sink_bias",
    );
    reject_metadata(
        official_value(),
        |v| v["dflash_config"]["attention_sink_bias"] = serde_json::json!(false),
        "attention_sink_bias",
    );
    reject_metadata(
        official_value(),
        |v| v["markov_rank"] = serde_json::json!(256),
        "markov_rank",
    );
    reject_metadata(
        official_value(),
        |v| v["dspark_markov_rank"] = serde_json::json!(256),
        "dspark_markov_rank",
    );
    reject_metadata(
        official_value(),
        |v| v["dspark_bonus_anchor"] = serde_json::json!(false),
        "bonus.bonus_anchor",
    );
    reject_metadata(
        official_value(),
        |v| v["sample_from_anchor"] = serde_json::json!(true),
        "sample_from_anchor",
    );
    reject_metadata(
        official_value(),
        |v| v["dflash_config"]["sample_from_anchor"] = serde_json::json!(true),
        "sample_from_anchor",
    );
    reject_metadata(
        official_value(),
        |v| v["dflash_config"]["target_layer_ids"] = serde_json::json!([1, 5, 19]),
        "target_layer_ids",
    );
    reject_metadata(
        official_value(),
        |v| v["target_layer_ids"] = serde_json::json!([1, 5, 19]),
        "target_layer_ids disagree",
    );
    reject_metadata(
        official_value(),
        |v| v["dflash_config"]["causal"] = Value::Null,
        "dflash_config.causal",
    );
}
#[test]
fn required_presence_is_not_replaced_by_false_defaults() {
    let mut missing_root_bonus = official_value();
    missing_root_bonus
        .as_object_mut()
        .unwrap()
        .remove("dspark_bonus_anchor");
    let error = admit(&missing_root_bonus, runtime()).expect_err("bonus presence");
    assert!(
        format!("{error:#}").contains("dspark_bonus_anchor"),
        "{error:#}"
    );

    let mut missing_root_sample = official_value();
    missing_root_sample
        .as_object_mut()
        .unwrap()
        .remove("sample_from_anchor");
    let error = admit(&missing_root_sample, runtime()).expect_err("root sample presence");
    assert!(
        format!("{error:#}").contains("sample_from_anchor"),
        "{error:#}"
    );

    let mut missing_nested_sample = official_value();
    missing_nested_sample["dflash_config"]
        .as_object_mut()
        .unwrap()
        .remove("sample_from_anchor");
    let error = admit(&missing_nested_sample, runtime()).expect_err("nested sample presence");
    assert!(
        format!("{error:#}").contains("dflash_config.sample_from_anchor"),
        "{error:#}"
    );

    let mut missing_rank = official_value();
    missing_rank
        .as_object_mut()
        .unwrap()
        .remove("dspark_markov_rank");
    let error = admit(&missing_rank, runtime()).expect_err("DSpark rank presence");
    assert!(
        format!("{error:#}").contains("dspark_markov_rank"),
        "{error:#}"
    );

    let mut missing_taps = official_value();
    missing_taps
        .as_object_mut()
        .unwrap()
        .remove("target_layer_ids");
    let error = admit(&missing_taps, runtime()).expect_err("root taps presence");
    assert!(
        format!("{error:#}").contains("target_layer_ids"),
        "{error:#}"
    );
}
#[test]
fn explicit_drafter_kv_quantization_is_rejected() {
    let mut value = official_value();
    value["quantization_config"]["kv_cache_quant_algo"] = serde_json::json!("FP8");
    let error = admit(&value, runtime()).expect_err("explicit drafter quantization");
    assert!(
        format!("{error:#}").contains("kv_cache_quant_algo"),
        "{error:#}"
    );
}
#[test]
fn missing_markov_weights_or_sinks_are_rejected() {
    let mut no_fc = runtime();
    no_fc.fc_present = false;
    let error = admit(&official_value(), no_fc).expect_err("missing fc");
    assert!(format!("{error:#}").contains("fc.weight"), "{error:#}");

    let mut no_w1 = runtime();
    no_w1.markov_w1_present = false;
    let error = admit(&official_value(), no_w1).expect_err("missing W1");
    assert!(
        format!("{error:#}").contains("markov.w1_present"),
        "{error:#}"
    );

    let mut no_w2 = runtime();
    no_w2.markov_w2_present = false;
    let error = admit(&official_value(), no_w2).expect_err("missing W2");
    assert!(
        format!("{error:#}").contains("markov.w2_present"),
        "{error:#}"
    );

    let mut no_sinks = runtime();
    no_sinks.all_required_sinks_present = false;
    let error = admit(&official_value(), no_sinks).expect_err("missing sinks");
    assert!(
        format!("{error:#}").contains("attention_sink_bias"),
        "{error:#}"
    );
}
#[test]
fn target_dtype_and_topology_are_exact() {
    let mut bf16 = runtime();
    bf16.target_kv_dtype = KvCacheDtype::Bf16;
    let error = admit(&official_value(), bf16).expect_err("BF16 target");
    assert!(
        format!("{error:#}").contains("target KV dtype"),
        "{error:#}"
    );

    let mut tp = runtime();
    tp.tp = 2;
    let error = admit(&official_value(), tp).expect_err("TP2");
    assert!(
        format!("{error:#}").contains("tensor_parallel"),
        "{error:#}"
    );

    let mut ep = runtime();
    ep.ep = 2;
    let error = admit(&official_value(), ep).expect_err("EP2");
    assert!(
        format!("{error:#}").contains("expert_parallel"),
        "{error:#}"
    );

    let mut page = runtime();
    page.physical_kv_page_size = 8;
    let error = admit(&official_value(), page).expect_err("physical page size");
    assert!(format!("{error:#}").contains("page_size"), "{error:#}");
}
#[test]
fn optional_confidence_and_adaptive_declarations_must_be_false() {
    let mut confidence = official_value();
    confidence["confidence_head"] = serde_json::json!(true);
    let error = admit(&confidence, runtime()).expect_err("confidence head");
    assert!(
        format!("{error:#}").contains("confidence.head_present"),
        "{error:#}"
    );

    let mut adaptive = official_value();
    adaptive["dflash_config"]["adaptive"] = serde_json::json!(true);
    let error = admit(&adaptive, runtime()).expect_err("adaptive declaration");
    assert!(
        format!("{error:#}").contains("confidence.adaptive"),
        "{error:#}"
    );
}
#[test]
fn build_mapper_accepts_exact_lightning_swa_window() {
    let config = parse_dflash_config(OFFICIAL_LIGHTNING_JSON).unwrap();
    let store = required_store();
    let args = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: config,
        gamma: Some(4),
        window_size: Some(1024),
    };
    let mut target = ModelConfig::qwen3_next_80b_nvfp4();
    target.model_type = "nemotron_h".to_owned();
    target.hidden_size = 2688;
    target.num_hidden_layers = 52;
    target.num_experts = 128;
    target.num_experts_per_tok = 6;
    target.tp_world_size = 1;
    target.ep_world_size = 1;
    let profile = admit_lightning_dspark_build(&args, &target, 3, 16, KvCacheDtype::Fp8)
        .unwrap()
        .expect("exact Lightning SWA window must pass");
    assert_eq!(profile.attention.swa_window, 1024);
    let missing_gamma = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: parse_dflash_config(OFFICIAL_LIGHTNING_JSON).unwrap(),
        gamma: None,
        window_size: Some(1024),
    };
    assert!(
        admit_lightning_dspark_build(&missing_gamma, &target, 3, 16, KvCacheDtype::Fp8).is_err()
    );
}
#[test]
fn build_mapper_rejects_missing_lightning_swa_window() {
    let config = parse_dflash_config(OFFICIAL_LIGHTNING_JSON).unwrap();
    let store = required_store();
    let args = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: config,
        gamma: Some(4),
        window_size: None,
    };
    let mut target = ModelConfig::qwen3_next_80b_nvfp4();
    target.tp_world_size = 1;
    target.ep_world_size = 1;
    let error = admit_lightning_dspark_build(&args, &target, 3, 16, KvCacheDtype::Fp8)
        .expect_err("Lightning must reject an omitted served SWA window");
    assert!(
        format!("{error:#}").contains("explicit served SWA window"),
        "{error:#}"
    );
}
#[test]
fn build_mapper_rejects_wrong_lightning_swa_window() {
    let config = parse_dflash_config(OFFICIAL_LIGHTNING_JSON).unwrap();
    let store = required_store();
    let args = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: config,
        gamma: Some(4),
        window_size: Some(4096),
    };
    let mut target = ModelConfig::qwen3_next_80b_nvfp4();
    target.tp_world_size = 1;
    target.ep_world_size = 1;
    let error = admit_lightning_dspark_build(&args, &target, 3, 16, KvCacheDtype::Fp8)
        .expect_err("Lightning must reject a non-contract served SWA window");
    assert!(
        format!("{error:#}").contains("served SWA window must be 1024"),
        "{error:#}"
    );
}
