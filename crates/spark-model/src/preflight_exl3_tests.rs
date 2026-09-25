// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-flight tests for the EXL3 quant method: `exl3` must be a known
//! `quant_method` (an EXL3 checkpoint bails here otherwise), and unknown
//! methods must still bail with their name in the error.

use super::check_quant_method;
use atlas_core::config::{ModelConfig, parse_quantization_config};

#[test]
fn exl3_is_a_known_quant_method() {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.quantization_config = parse_quantization_config(&serde_json::json!({
        "quantization_config": {
            "quant_method": "exl3",
            "version": "1.4.4",
            "bits": 4.05,
            "head_bits": 6,
            "codebook": "mul1",
            "out_scales": "always",
        }
    }));
    check_quant_method(&c).unwrap();
}

#[test]
fn unknown_quant_method_still_bails() {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.quantization_config = parse_quantization_config(&serde_json::json!({
        "quantization_config": { "quant_method": "gptq" }
    }));
    let err = check_quant_method(&c).unwrap_err();
    assert!(format!("{err:#}").contains("gptq"), "{err:#}");
}
