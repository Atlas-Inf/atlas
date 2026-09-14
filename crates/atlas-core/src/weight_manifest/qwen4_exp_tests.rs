// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the `qwen4_exp` weight manifest. Split out for the 500-LoC cap.

use super::*;
use crate::config::parse_config;
use crate::weight_manifest::diff;

fn published() -> ModelConfig {
    parse_config(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen4_exp_flash_next_config.json"
    )))
    .expect("published config parses")
}

fn shape_of<'a>(manifest: &'a [ExpectedTensor], name: &str) -> &'a [usize] {
    manifest
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("manifest is missing {name}"))
        .shape
        .as_slice()
}

/// The count is exact, and it is checkable: the published
/// `model.safetensors.index.json` holds 152_089 tensors, of which 333 are
/// `model.visual.*` and 75_264 are FP8 `weight_scale_inv` siblings. The
/// remaining 76_492 are what this manifest describes, and a name-level diff
/// against that index reports zero missing and zero unexpected
/// (`scripts/dev/verify_qwen4_exp_manifest.py`).
#[test]
fn the_published_checkpoint_is_covered_exactly() {
    let manifest = qwen4_exp_manifest(&published()).unwrap();
    assert_eq!(manifest.len(), 76_492);

    let names: std::collections::HashSet<&str> = manifest.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names.len(),
        manifest.len(),
        "manifest must not repeat a name"
    );
    assert!(
        !names.iter().any(|n| n.starts_with("model.visual.")),
        "vision is out of scope for this manifest"
    );
}

/// Shapes read out of the real safetensors headers over HTTP range
/// requests. These are the ones that encode an architectural claim rather
/// than a restatement of the config, so a wrong claim fails here.
#[test]
fn the_load_bearing_shapes_match_the_real_headers() {
    let m = qwen4_exp_manifest(&published()).unwrap();

    // Gated Q: 2 x (24 heads x 256). o_proj consumes only the Q half.
    let attn = "model.language_model.layers.3.self_attn";
    assert_eq!(
        shape_of(&m, &format!("{attn}.q_proj.weight")),
        [12288, 2560]
    );
    assert_eq!(shape_of(&m, &format!("{attn}.o_proj.weight")), [2560, 6144]);
    assert_eq!(shape_of(&m, &format!("{attn}.k_proj.weight")), [512, 2560]);
    assert_eq!(shape_of(&m, &format!("{attn}.q_norm.weight")), [256]);
    // Fused indexer q+k: (4 heads + 1 kv head) x 128.
    assert_eq!(
        shape_of(&m, &format!("{attn}.indexer.index_qk_proj.weight")),
        [640, 2560]
    );

    // Hyper-connections run on hc_count x hidden = 10240, NOT hidden.
    let hc = "model.language_model.layers.0.attn_hyper_connection";
    assert_eq!(shape_of(&m, &format!("{hc}.hc_norm.weight")), [10240]);
    assert_eq!(
        shape_of(&m, &format!("{hc}.input_mix_weight_down.weight")),
        [320, 10240]
    );
    assert_eq!(
        shape_of(&m, &format!("{hc}.input_mix_weight_up.weight")),
        [10240, 320]
    );
    assert_eq!(
        shape_of(&m, &format!("{hc}.block_inject_weight.weight")),
        [4, 10240]
    );

    // Linear attention: 2 x (16 x 128) for q/k plus 48 x 128 for v.
    let lin = "model.language_model.layers.0.linear_attn";
    assert_eq!(
        shape_of(&m, &format!("{lin}.in_proj_qkv.weight")),
        [10240, 2560]
    );
    assert_eq!(
        shape_of(&m, &format!("{lin}.in_proj_z.weight")),
        [6144, 2560]
    );
    assert_eq!(shape_of(&m, &format!("{lin}.conv1d.weight")), [10240, 1, 4]);
    assert_eq!(shape_of(&m, &format!("{lin}.A_log")), [48]);
    assert_eq!(shape_of(&m, &format!("{lin}.norm.weight")), [128]);

    // PLE sits on decoder layer 1 (ple_layer_ids is one-indexed [2]), and
    // its shards are the published [2_500_012, 160] x 128.
    let emb = "model.language_model.layers.1.ple.ple_embedding";
    assert_eq!(
        shape_of(&m, &format!("{emb}.ngram_embedding.shard_0.weight")),
        [2_500_012, 160]
    );
    assert_eq!(
        shape_of(&m, &format!("{emb}.ngram_embedding.shard_127.weight")),
        [2_500_012, 160]
    );
    assert_eq!(
        shape_of(&m, &format!("{emb}.ngram_heads_vocab_sizes")),
        [16]
    );
    assert_eq!(shape_of(&m, &format!("{emb}.layer_multipliers")), [3]);
    assert_eq!(
        shape_of(&m, "model.language_model.layers.1.ple.key_proj.weight"),
        [10240, 2560]
    );

    // MTP carries its own indexer and its own 512 experts. Its two input
    // norms differ in width: the embedding side is hidden, the hidden side
    // is the trunk's hc_count-wide hyper-connection state.
    assert_eq!(shape_of(&m, "mtp.fc_embedding.weight"), [2560, 2560]);
    assert_eq!(shape_of(&m, "mtp.pre_fc_norm_embedding.weight"), [2560]);
    assert_eq!(shape_of(&m, "mtp.pre_fc_norm_hidden.weight"), [10240]);
    assert_eq!(
        shape_of(&m, "mtp.layers.0.mlp.experts.511.down_proj.weight"),
        [2560, 640]
    );
}

/// PLE belongs to exactly one decoder layer, and it is layer 1 — not layer
/// 2. `ple_layer_ids` is one-indexed, and putting the tower a layer off
/// would read real tensors from the wrong block.
#[test]
fn the_ple_tower_lands_on_the_one_indexed_layer() {
    let m = qwen4_exp_manifest(&published()).unwrap();
    let with_ple: std::collections::BTreeSet<&str> = m
        .iter()
        .filter(|t| t.name.contains(".ple."))
        .filter_map(|t| t.name.split(".ple.").next())
        .collect();
    assert_eq!(
        with_ple.into_iter().collect::<Vec<_>>(),
        ["model.language_model.layers.1"]
    );
}

#[test]
fn a_diff_against_itself_is_clean_and_notices_every_kind_of_drift() {
    let m = qwen4_exp_manifest(&published()).unwrap();
    let actual: Vec<(&str, &[usize])> = m
        .iter()
        .map(|t| (t.name.as_str(), t.shape.as_slice()))
        .collect();
    assert!(diff(&m, actual.iter().copied()).is_clean());

    // Drop one, add one, and corrupt one shape.
    let mut drifted = actual.clone();
    drifted.retain(|(n, _)| *n != "mtp.fc_hidden.weight");
    drifted.push(("model.language_model.layers.0.not_a_tensor", &[1]));
    let wrong: &[usize] = &[9, 9];
    let target = "model.language_model.layers.0.linear_attn.A_log";
    for entry in drifted.iter_mut() {
        if entry.0 == target {
            entry.1 = wrong;
        }
    }

    let d = diff(&m, drifted);
    assert_eq!(d.missing, ["mtp.fc_hidden.weight"]);
    assert_eq!(d.unexpected, ["model.language_model.layers.0.not_a_tensor"]);
    assert_eq!(d.mismatched.len(), 1);
    assert_eq!(d.mismatched[0].0, target);
    assert_eq!(d.mismatched[0].2, vec![9, 9]);
}

/// The FP8 release's scale siblings, exactly. The published index holds
/// 152_089 tensors; 333 are `model.visual.*`, and the manifest plus its
/// siblings account for the remaining 151_756 with nothing missing and
/// nothing extra.
#[test]
fn the_fp8_scale_siblings_are_derived_exactly() {
    let cfg = published();
    let manifest = qwen4_exp_manifest(&cfg).unwrap();
    let scales = crate::weight_manifest::quantization_siblings(&cfg, &manifest)
        .unwrap()
        .expect("fp8 block-quant is described");
    assert_eq!(scales.len(), 75_264);
    assert_eq!(manifest.len() + scales.len(), 151_756);

    // Only routed experts are block-quantized in this release.
    assert!(
        scales.iter().all(|t| t.name.contains(".mlp.experts.")),
        "a non-expert module picked up a block scale"
    );
    // [2560, 640] tiled by [128, 128] -> [20, 5].
    let down = scales
        .iter()
        .find(|t| {
            t.name == "model.language_model.layers.3.mlp.experts.0.down_proj.weight_scale_inv"
        })
        .expect("expert down_proj scale");
    assert_eq!(down.shape, vec![20, 5]);
}

/// The n-gram shards are FP8 but scale PER TENSOR -- 128 shards behind one
/// `weight_scale` -- and they are absent from `modules_to_not_convert`
/// because they are converted, just by another scheme. Treating them as
/// block-quantized over-generates by exactly 128 tensors.
#[test]
fn per_tensor_groups_take_no_block_scale() {
    let cfg = published();
    let manifest = qwen4_exp_manifest(&cfg).unwrap();
    let scales = crate::weight_manifest::quantization_siblings(&cfg, &manifest)
        .unwrap()
        .unwrap();
    assert!(
        manifest
            .iter()
            .any(|t| t.name.ends_with("ngram_embedding.shard_0.weight")),
        "the shards must be in the base manifest"
    );
    assert!(
        !scales.iter().any(|t| t.name.contains("ngram_embedding")),
        "n-gram shards must not get weight_scale_inv"
    );
}

fn radixark() -> ModelConfig {
    parse_config(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/qwen4_exp_flash_next_nvfp4_config.json"
    )))
    .expect("RadixArk NVFP4 config parses")
}

/// The other published release, and it differs in two ways that are
/// packaging rather than architecture: NVFP4 instead of block FP8, and the
/// MTP block left in HF's native stacked-expert form because it is not
/// quantized at all. Its index holds 296_475 tensors, 333 of them vision.
#[test]
fn the_nvfp4_repack_is_covered_exactly() {
    let cfg = radixark();
    let layout = Qwen4ExpLayout {
        mtp_experts: ExpertLayout::Stacked,
        ..Default::default()
    };
    let base = qwen4_exp_manifest_with(&cfg, layout).unwrap();
    let full = crate::weight_manifest::quantized_manifest(&cfg, &base)
        .unwrap()
        .expect("NVFP4 is described");
    assert_eq!(full.len(), 296_142);

    // NVFP4 repacks the weight itself: [2560, 640] is stored U8
    // [2560, 320], two FP4 values per byte.
    let expert = "model.language_model.layers.0.mlp.experts.0.down_proj";
    assert_eq!(shape_of(&full, &format!("{expert}.weight")), [2560, 320]);
    // One scale per group of 16 along the input dim.
    assert_eq!(
        shape_of(&full, &format!("{expert}.weight_scale")),
        [2560, 40]
    );
    // Both second-level scales are scalars.
    assert!(shape_of(&full, &format!("{expert}.weight_scale_2")).is_empty());
    assert!(shape_of(&full, &format!("{expert}.input_scale")).is_empty());

    // `mtp.*` is in the exclude list, so the MTP experts stay BF16 and
    // keep the stacked shapes: gate and up fused along the intermediate.
    assert_eq!(
        shape_of(&full, "mtp.layers.0.mlp.experts.gate_up_proj"),
        [512, 1280, 2560]
    );
    assert_eq!(
        shape_of(&full, "mtp.layers.0.mlp.experts.down_proj"),
        [512, 2560, 640]
    );
    assert!(
        !full
            .iter()
            .any(|t| t.name.starts_with("mtp.") && t.name.ends_with(".weight_scale")),
        "mtp.* is excluded from quantization"
    );
}

/// ModelOpt's ignore list is globbed (`*.self_attn.*`, `mtp.*`) while HF's
/// native FP8 list is 943 literal paths. Both have to work, and a `*` has
/// to span dots or `*.self_attn.*` never matches a real module path.
#[test]
fn module_globs_span_dots_and_literals_still_match() {
    use crate::weight_manifest::module_glob_matches as m;
    let q = "model.language_model.layers.3.self_attn.q_proj";
    assert!(m("*.self_attn.*", q));
    assert!(m(
        "*hyper_connection*",
        "model.language_model.layers.0.attn_hyper_connection.hc_norm"
    ));
    assert!(m("mtp.*", "mtp.layers.0.mlp.experts.0.down_proj"));
    assert!(!m(
        "mtp.*",
        "model.language_model.layers.0.mlp.experts.0.down_proj"
    ));
    // Literal, no glob.
    assert!(m("lm_head", "lm_head"));
    assert!(!m("lm_head", "lm_head.something"));
    assert!(m(
        "model.language_model.layers.1.ple.conv1d",
        "model.language_model.layers.1.ple.conv1d"
    ));
    assert!(!m(
        "*.self_attn.*",
        "model.language_model.layers.0.linear_attn.out_proj"
    ));
}

/// The hyper-connection widths are the whole reason this manifest exists,
/// so a config that does not declare them must not silently produce a
/// manifest full of zero-width tensors.
#[test]
fn a_config_without_hyper_connections_is_refused() {
    let mut cfg = published();
    cfg.hc_count = 0;
    let err = qwen4_exp_manifest(&cfg).expect_err("must refuse");
    assert!(format!("{err:#}").contains("hc_count"), "{err:#}");
}
