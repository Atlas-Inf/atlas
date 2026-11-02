// SPDX-License-Identifier: AGPL-3.0-only

use super::{parse_qwen4_exp, raw_config};
use serde_json::json;

#[test]
fn qwen_yarn_reads_the_official_nested_config() {
    for factor in [2.0, 4.0] {
        let mut raw = raw_config();
        let rp = &mut raw["text_config"]["rope_parameters"];
        rp["rope_type"] = json!("yarn");
        rp["factor"] = json!(factor);
        rp["original_max_position_embeddings"] = json!(262144);
        let c = parse_qwen4_exp(&raw).unwrap();
        assert_eq!(c.yarn_factor, factor as f32);
        assert_eq!(c.yarn_original_max_position_embeddings, 262144);
        assert_eq!(c.yarn_beta_fast, 32.0);
        assert_eq!(c.yarn_beta_slow, 1.0);
        assert!((c.yarn_attention_factor - (1.0 + 0.1 * (factor as f32).ln())).abs() < 1e-6);
        assert_eq!(c.rotary_dim(), 64);
        assert_eq!(c.mrope_section, [11, 11, 10]);
    }
}

#[test]
fn qwen_yarn_refuses_unsupported_or_incomplete_scaling() {
    for rp in [
        json!({"rope_type": "dynamic"}),
        json!({"rope_type": "yarn"}),
        json!({"rope_type": "yarn", "factor": 0, "original_max_position_embeddings": 262144}),
        json!({"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 0}),
        json!({"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 262144,
               "beta_fast": 1, "beta_slow": 32}),
        json!({"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 262144,
               "truncate": false}),
        json!({"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 262144,
               "truncate": "true"}),
    ] {
        let mut raw = raw_config();
        let params = raw["text_config"]["rope_parameters"]
            .as_object_mut()
            .unwrap();
        params.extend(rp.as_object().unwrap().clone());
        assert!(parse_qwen4_exp(&raw).is_err(), "accepted {rp}");
    }
}

#[test]
fn qwen_native_rope_keeps_scaling_disabled() {
    let c = parse_qwen4_exp(&raw_config()).unwrap();
    assert_eq!(c.yarn_factor, 0.0);
    assert_eq!(c.yarn_attention_factor, 1.0);
}
