// SPDX-License-Identifier: AGPL-3.0-only

//! The `qwen4_exp` (Qwen3.8-Flash-Next) tensor manifest.
//!
//! Every shape here was read out of the published
//! `Qwen/Qwen3.8-Flash-Next-FP8` safetensors headers over HTTP range requests
//! — no download — and is expressed as the config arithmetic that produces it,
//! so it stays right for the tiny development checkpoint too.
//!
//! Vision (`model.visual.*`) is deliberately NOT covered: the tower is
//! independent of the language model and its dimensions live in a separate
//! config block. Callers comparing against a full checkpoint index must filter
//! that prefix out, and [`super::diff`] will otherwise report it as unexpected.

use super::ExpectedTensor;
use crate::config::{LayerType, ModelConfig};
use anyhow::{Result, ensure};

/// Language-model weight prefix in the published checkpoint.
const LM: &str = "model.language_model";

fn hyper_connection(prefix: &str, cfg: &ModelConfig, inject: bool, out: &mut Vec<ExpectedTensor>) {
    // The hyper-connection block is `hc_count * hidden` wide: the residual is
    // literally `hc_count` streams concatenated, not `hidden` with a gate.
    let wide = cfg.hc_count * cfg.hidden_size;
    out.push(ExpectedTensor::new(
        format!("{prefix}.hc_norm.weight"),
        [wide],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.input_mix_weight_down.weight"),
        [cfg.hc_lowrank, wide],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.input_mix_weight_up.weight"),
        [wide, cfg.hc_lowrank],
    ));
    if inject {
        out.push(ExpectedTensor::new(
            format!("{prefix}.block_inject_weight.weight"),
            [cfg.hc_count, wide],
        ));
    }
}

fn full_attention(prefix: &str, cfg: &ModelConfig, out: &mut Vec<ExpectedTensor>) {
    let (h, hd) = (cfg.hidden_size, cfg.head_dim);
    let q_dim = cfg.num_attention_heads * hd;
    let kv_dim = cfg.num_key_value_heads * hd;
    // 2x q_dim: Q and its gate, interleaved per head.
    out.push(ExpectedTensor::new(
        format!("{prefix}.q_proj.weight"),
        [q_dim * 2, h],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.k_proj.weight"),
        [kv_dim, h],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.v_proj.weight"),
        [kv_dim, h],
    ));
    // Consumes q_dim, not 2*q_dim -- the gate is spent inside attention.
    out.push(ExpectedTensor::new(
        format!("{prefix}.o_proj.weight"),
        [h, q_dim],
    ));
    out.push(ExpectedTensor::new(format!("{prefix}.q_norm.weight"), [hd]));
    out.push(ExpectedTensor::new(format!("{prefix}.k_norm.weight"), [hd]));

    let idx = cfg.indexer_head_dim;
    out.push(ExpectedTensor::new(
        format!("{prefix}.indexer.index_qk_proj.weight"),
        [(cfg.indexer_n_heads + cfg.indexer_kv_heads) * idx, h],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.indexer.q_layernorm.weight"),
        [idx],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.indexer.k_layernorm.weight"),
        [idx],
    ));
}

fn linear_attention(prefix: &str, cfg: &ModelConfig, out: &mut Vec<ExpectedTensor>) {
    let h = cfg.hidden_size;
    let k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    // q and k share the key head count; v is separate. Conv runs over all three.
    let qkv = k_dim * 2 + v_dim;
    out.push(ExpectedTensor::new(
        format!("{prefix}.in_proj_qkv.weight"),
        [qkv, h],
    ));
    for part in ["a", "b"] {
        out.push(ExpectedTensor::new(
            format!("{prefix}.in_proj_{part}.weight"),
            [cfg.linear_num_value_heads, h],
        ));
    }
    out.push(ExpectedTensor::new(
        format!("{prefix}.in_proj_z.weight"),
        [v_dim, h],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.conv1d.weight"),
        [qkv, 1, cfg.linear_conv_kernel_dim],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.A_log"),
        [cfg.linear_num_value_heads],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.dt_bias"),
        [cfg.linear_num_value_heads],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.norm.weight"),
        [cfg.linear_value_head_dim],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.out_proj.weight"),
        [h, v_dim],
    ));
}

fn moe(prefix: &str, cfg: &ModelConfig, out: &mut Vec<ExpectedTensor>) {
    let (h, mi) = (cfg.hidden_size, cfg.moe_intermediate_size);
    out.push(ExpectedTensor::new(
        format!("{prefix}.gate.weight"),
        [cfg.num_experts, h],
    ));
    for expert in 0..cfg.num_experts {
        for (proj, shape) in [
            ("gate_proj", [mi, h]),
            ("up_proj", [mi, h]),
            ("down_proj", [h, mi]),
        ] {
            out.push(ExpectedTensor::new(
                format!("{prefix}.experts.{expert}.{proj}.weight"),
                shape,
            ));
        }
    }
    let si = cfg.shared_expert_intermediate_size;
    for (proj, shape) in [
        ("gate_proj", [si, h]),
        ("up_proj", [si, h]),
        ("down_proj", [h, si]),
    ] {
        out.push(ExpectedTensor::new(
            format!("{prefix}.shared_expert.{proj}.weight"),
            shape,
        ));
    }
    out.push(ExpectedTensor::new(
        format!("{prefix}.shared_expert_gate.weight"),
        [1, h],
    ));
}

fn ple(
    prefix: &str,
    cfg: &ModelConfig,
    ple_layer_index: usize,
    out: &mut Vec<ExpectedTensor>,
) -> Result<()> {
    let h = cfg.hidden_size;
    let wide = cfg.hc_count * h;
    // One key per residual stream, one shared value.
    out.push(ExpectedTensor::new(
        format!("{prefix}.conv1d.weight"),
        [wide, 1, cfg.ple_conv_kernel_size],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.key_proj.weight"),
        [wide, h],
    ));
    out.push(ExpectedTensor::new(
        format!("{prefix}.value_proj.weight"),
        [h, h],
    ));
    for norm in ["norm_conv", "norm_key", "norm_query"] {
        out.push(ExpectedTensor::new(
            format!("{prefix}.{norm}.weight"),
            [wide],
        ));
    }

    let ngram = cfg
        .qwen4exp_ngram(ple_layer_index)?
        .ok_or_else(|| anyhow::anyhow!("PLE layer without n-gram geometry"))?;
    let rows = ngram.padded_rows(cfg.ngram_vocab_size_base) as usize;
    ensure!(
        cfg.split_ngram_parts > 0 && rows.is_multiple_of(cfg.split_ngram_parts),
        "the padded n-gram table ({rows} rows) must divide evenly by \
         split_ngram_parts ({})",
        cfg.split_ngram_parts
    );
    let emb = format!("{prefix}.ple_embedding");
    for shard in 0..cfg.split_ngram_parts {
        out.push(ExpectedTensor::new(
            format!("{emb}.ngram_embedding.shard_{shard}.weight"),
            [rows / cfg.split_ngram_parts, ngram.head_dim()],
        ));
    }
    out.push(ExpectedTensor::new(
        format!("{emb}.ngram_embedding.weight_scale"),
        [1],
    ));
    for (buffer, len) in [
        ("ngram_heads_vocab_sizes", ngram.num_heads()),
        ("ngram_heads_offsets", ngram.num_heads()),
        ("layer_multipliers", cfg.ngram_size),
    ] {
        out.push(ExpectedTensor::new(format!("{emb}.{buffer}"), [len]));
    }
    Ok(())
}

/// Every language-model and MTP tensor a `qwen4_exp` checkpoint must carry.
///
/// Excludes `model.visual.*`; see the module docs.
pub fn qwen4_exp_manifest(cfg: &ModelConfig) -> Result<Vec<ExpectedTensor>> {
    ensure!(
        cfg.hc_count > 0 && cfg.hc_lowrank > 0,
        "qwen4_exp requires hc_count and hc_lowrank; got {} / {}",
        cfg.hc_count,
        cfg.hc_lowrank
    );
    ensure!(
        cfg.layer_types.len() == cfg.num_hidden_layers,
        "layer_types must cover every layer"
    );

    let h = cfg.hidden_size;
    let mut out = Vec::new();
    out.push(ExpectedTensor::new(
        format!("{LM}.embed_tokens.weight"),
        [cfg.vocab_size, h],
    ));
    if !cfg.tie_word_embeddings {
        out.push(ExpectedTensor::new("lm_head.weight", [cfg.vocab_size, h]));
    }
    // The trunk mixer has no block injection -- it mixes, it does not inject.
    hyper_connection(
        &format!("{LM}.hyper_connection_mixer"),
        cfg,
        false,
        &mut out,
    );

    for layer in 0..cfg.num_hidden_layers {
        let base = format!("{LM}.layers.{layer}");
        hyper_connection(
            &format!("{base}.attn_hyper_connection"),
            cfg,
            true,
            &mut out,
        );
        hyper_connection(&format!("{base}.mlp_hyper_connection"), cfg, true, &mut out);
        match cfg.layer_types[layer] {
            LayerType::FullAttention => full_attention(&format!("{base}.self_attn"), cfg, &mut out),
            LayerType::LinearAttention => {
                linear_attention(&format!("{base}.linear_attn"), cfg, &mut out)
            }
            other => anyhow::bail!("qwen4_exp does not use layer type {other:?}"),
        }
        moe(&format!("{base}.mlp"), cfg, &mut out);
        // ple_layer_ids is ONE-indexed.
        if let Some(index) = cfg.ple_layer_ids.iter().position(|id| *id == layer + 1) {
            ple(&format!("{base}.ple"), cfg, index, &mut out)?;
        }
    }

    for layer in 0..cfg.mtp_num_hidden_layers {
        let base = format!("mtp.layers.{layer}");
        hyper_connection(
            &format!("{base}.attn_hyper_connection"),
            cfg,
            true,
            &mut out,
        );
        hyper_connection(&format!("{base}.mlp_hyper_connection"), cfg, true, &mut out);
        // The MTP block is declared `full_attention` regardless of the trunk's
        // schedule, and carries its own indexer and its own expert stack.
        full_attention(&format!("{base}.self_attn"), cfg, &mut out);
        moe(&format!("{base}.mlp"), cfg, &mut out);
    }
    if cfg.mtp_num_hidden_layers > 0 {
        hyper_connection("mtp.hyper_connection_mixer", cfg, false, &mut out);
        for fc in ["fc_embedding", "fc_hidden"] {
            out.push(ExpectedTensor::new(format!("mtp.{fc}.weight"), [h, h]));
        }
        for norm in ["pre_fc_norm_embedding", "pre_fc_norm_hidden"] {
            out.push(ExpectedTensor::new(format!("mtp.{norm}.weight"), [h]));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
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

        let names: std::collections::HashSet<&str> =
            manifest.iter().map(|t| t.name.as_str()).collect();
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

        // MTP carries its own indexer and its own 512 experts.
        assert_eq!(shape_of(&m, "mtp.fc_embedding.weight"), [2560, 2560]);
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
}
