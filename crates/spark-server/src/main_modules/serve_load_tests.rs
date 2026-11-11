// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// A swap must CARRY the process-scoped stores, never rebuild them.
///
/// Asserted by POINTER identity, not equality: two freshly-built empty stores
/// compare equal, so an equality check would pass on exactly the bug this
/// guards — a swap that silently drops every stored conversation and resets
/// every rate-limit bucket while looking fine.
#[test]
fn carried_state_is_the_same_allocation_not_an_equal_one() {
    let first = Carried::from_env();
    let cloned = first.clone();

    assert!(
        std::sync::Arc::ptr_eq(&first.response_store, &cloned.response_store),
        "responses must survive a swap"
    );
    assert!(
        std::sync::Arc::ptr_eq(&first.rate_limiter, &cloned.rate_limiter),
        "rate-limit buckets must survive a swap"
    );
    assert!(
        std::sync::Arc::ptr_eq(&first.conversation_store, &cloned.conversation_store),
        "stored conversations must survive a swap"
    );
}

/// Two independent `from_env()` calls are DIFFERENT allocations — which is why
/// `load_model` takes `Carried` rather than building its own.
#[test]
fn building_from_env_twice_would_lose_the_stores() {
    let first = Carried::from_env();
    let second = Carried::from_env();
    assert!(
        !std::sync::Arc::ptr_eq(&first.conversation_store, &second.conversation_store),
        "if this ever passes, from_env has become a singleton and the carried \
         parameter is no longer what protects the stores — re-check the swap"
    );
}

#[test]
fn carried_uses_the_process_limiter_rather_than_minting_its_own() {
    // Handlers refund through `AppState.rate_limiter` and the middleware debits
    // through the host's. If those are two instances, refunds credit buckets
    // the middleware never debited and the accounting silently drifts.
    let host = crate::main_modules::model_host::ModelHost::empty();
    let carried = Carried::from_env();
    let process = carried.rate_limiter.clone();
    host.set_process(carried);
    assert!(
        std::sync::Arc::ptr_eq(&host.rate_limiter().expect("installed"), &process),
        "the host's limiter IS the one the model's AppState will hold"
    );
}

#[test]
fn glm_vision_guard_requires_the_native_glm_module() {
    let glm = [("glm_vision_encoder", &[][..])];
    let qwen = [("vision_encoder", &[][..])];

    assert_eq!(required_vision_module(true), "glm_vision_encoder");
    assert!(target_has_required_vision_module(true, &glm));
    assert!(!target_has_required_vision_module(true, &qwen));
}

#[test]
fn qwen_vision_guard_keeps_the_generic_module_name() {
    let qwen = [("vision_encoder", &[][..])];
    let glm = [("glm_vision_encoder", &[][..])];

    assert_eq!(required_vision_module(false), "vision_encoder");
    assert!(target_has_required_vision_module(false, &qwen));
    assert!(!target_has_required_vision_module(false, &glm));
}

// `nvfp4_verbatim_prompt` resolves per checkpoint: only a ModelOpt NVFP4
// pack that is NOT FP4-Local takes the override; everything else keeps the
// baked `verbatim_prompt`. Fixtures mirror
// `spark-model/src/weight_map/modelopt_plan_tests.rs`.
mod verbatim_prompt_resolution {
    use super::super::resolve_verbatim_prompt;
    use atlas_core::config::{ModelConfig, QuantLayerSpec, QuantizationConfig};
    use std::collections::BTreeMap;

    fn spec(algo: &str, encoding: Option<&str>) -> QuantLayerSpec {
        QuantLayerSpec {
            quant_algo: algo.into(),
            group_size: 16,
            weight_scale_encoding: encoding.map(str::to_string),
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

    fn behavior(nvfp4: Option<bool>) -> atlas_kernels::ModelBehavior {
        atlas_kernels::ModelBehavior {
            nvfp4_verbatim_prompt: nvfp4,
            ..Default::default()
        }
    }

    /// No ModelOpt plan (plain config): the override never applies.
    #[test]
    fn no_plan_keeps_the_baked_value() {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.quantization_config = None;
        assert!(!resolve_verbatim_prompt(&behavior(Some(true)), &cfg));
    }

    /// FP4-Local (a module carries `weight_scale_encoding`): the override is
    /// scoped away even though a plan exists.
    #[test]
    fn fp4_local_keeps_the_baked_value() {
        let cfg = cfg_with(&[(
            "model.language_model.layers.0.mlp.experts",
            spec("NVFP4", Some("csf")),
        )]);
        assert!(!resolve_verbatim_prompt(&behavior(Some(true)), &cfg));
    }

    /// Plain ModelOpt NVFP4 (no encodings anywhere): the override applies.
    #[test]
    fn nvfp4_pack_takes_the_override() {
        let cfg = cfg_with(&[
            (
                "model.language_model.layers.0.mlp.experts",
                spec("NVFP4", None),
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj",
                spec("MXFP8", None),
            ),
        ]);
        assert!(resolve_verbatim_prompt(&behavior(Some(true)), &cfg));
        assert!(!resolve_verbatim_prompt(&behavior(Some(false)), &cfg));
    }

    /// Key absent on a plain NVFP4 pack: `verbatim_prompt` applies to every
    /// checkpoint, exactly as before the key existed.
    #[test]
    fn absent_key_keeps_the_baked_value() {
        let cfg = cfg_with(&[(
            "model.language_model.layers.0.mlp.experts",
            spec("NVFP4", None),
        )]);
        assert!(!resolve_verbatim_prompt(&behavior(None), &cfg));
    }

    /// The `[behavior]` parser leaves the tri-state absent when MODEL.toml
    /// does not set the key.
    #[test]
    fn absent_key_parses_as_none() {
        let parsed = crate::api::test_app_state::build_behavior::parse_behavior(
            std::path::Path::new("/nonexistent-model-dir"),
        );
        assert!(parsed.nvfp4_verbatim_prompt.is_none());
    }
}
