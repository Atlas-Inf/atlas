// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::config::parse_config;

/// thinkingmachines/Inkling-2-Mini `config.json` (2026-11-20 release).
const CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../atlas-core/test_data/inkling2_mini_config.json"
));

/// Name -> [dtype, shape] of every `model.llm.` tensor in the release's
/// `model-00001-of-00001.safetensors` header (vision tensors left out).
const HEADER: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_data/inkling2_mini_text_header.json"
));

fn release() -> (Header, ModelConfig) {
    let raw: HashMap<String, (String, Vec<usize>)> = serde_json::from_str(HEADER).unwrap();
    let header = raw
        .into_iter()
        .map(|(k, (d, s))| (k, (WeightDtype::from_safetensors_str(&d).unwrap(), s)))
        .collect();
    (header, parse_config(CONFIG).unwrap())
}

#[test]
fn release_header_matches_the_plan() {
    let (header, config) = release();
    let plan = plan(&header, &config).expect("release layout");
    assert_eq!(plan.layers.len(), 24);
    let full: Vec<usize> = plan
        .layers
        .iter()
        .filter(|l| l.full)
        .map(|l| l.index)
        .collect();
    assert_eq!(full, [5, 11, 17, 23]);
    let looped: Vec<usize> = plan
        .layers
        .iter()
        .filter(|l| l.looped)
        .map(|l| l.index)
        .collect();
    assert_eq!(looped, (6..18).collect::<Vec<_>>());
    for l in &plan.layers {
        let want = if l.index < 20 {
            ExpertFormat::Nvfp4
        } else {
            ExpertFormat::Mxfp8
        };
        assert_eq!(l.experts, want, "layer {}", l.index);
    }
}

#[test]
fn missing_tensor_fails() {
    let (mut header, config) = release();
    header.remove("model.llm.layers.11.attn.k_sconv.static_w");
    let err = plan(&header, &config).unwrap_err().to_string();
    assert!(
        err.contains("missing model.llm.layers.11.attn.k_sconv.static_w"),
        "{err}"
    );
}

#[test]
fn wrong_shape_or_dtype_fails() {
    let (mut header, config) = release();
    header
        .get_mut("model.llm.layers.3.attn.wo_ud.weight")
        .unwrap()
        .1 = vec![3072, 2304];
    assert!(
        plan(&header, &config)
            .unwrap_err()
            .to_string()
            .contains("wo_ud")
    );

    let (mut header, config) = release();
    header
        .get_mut("model.llm.layers.0.mlp.gate.bias")
        .unwrap()
        .0 = WeightDtype::BF16;
    assert!(
        plan(&header, &config)
            .unwrap_err()
            .to_string()
            .contains("gate.bias")
    );
}

#[test]
fn unknown_expert_format_fails() {
    let (mut header, config) = release();
    header
        .get_mut("model.llm.layers.2.mlp.experts.w13_weight")
        .unwrap()
        .0 = WeightDtype::BF16;
    let err = plan(&header, &config).unwrap_err().to_string();
    assert!(
        err.contains("expected U8 (NVFP4) or F8_E4M3 (MXFP8)"),
        "{err}"
    );
}

#[test]
fn unread_tensor_fails() {
    let (mut header, config) = release();
    header.insert(
        "model.llm.layers.4.attn.new_thing.weight".into(),
        (WeightDtype::BF16, vec![8]),
    );
    let err = plan(&header, &config).unwrap_err().to_string();
    assert!(
        err.contains("does not read") && err.contains("new_thing"),
        "{err}"
    );
}
