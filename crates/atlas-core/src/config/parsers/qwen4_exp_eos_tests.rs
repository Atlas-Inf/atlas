// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn parses_array_eos_token_id() {
    let mut raw = serde_json::json!({
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 1,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "vocab_size": 248320,
            "layer_types": ["full_attention"],
            "eos_token_id": [151643, 151645]
        }
    });
    let c = parse_qwen4_exp(&raw).expect("parse array EOS");
    assert_eq!(c.eos_token_id, 151643);
    raw["text_config"]["eos_token_id"] = serde_json::json!([151645, 151643]);
    let c = parse_qwen4_exp(&raw).expect("parse reversed array EOS");
    assert_eq!(c.eos_token_id, 151645);
}

#[test]
fn rejects_empty_or_invalid_eos_token_id() {
    for eos in [
        serde_json::json!([]),
        serde_json::json!(["invalid"]),
        serde_json::json!([4294967296_u64]),
        serde_json::json!("invalid"),
    ] {
        let mut raw = raw_config();
        raw["text_config"]["eos_token_id"] = eos;
        assert!(parse_qwen4_exp(&raw).is_err());
    }
}
