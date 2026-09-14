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

/// How a block's routed experts are stored.
///
/// `Stacked` is HuggingFace's NATIVE layout — `Qwen4ExpTextExperts` holds
/// `gate_up_proj` as one `[experts, 2*moe_intermediate, hidden]` tensor and
/// chunks it at use. `PerExpert` is what appears once a quantizer has been
/// through: ModelOpt works per `nn.Linear`, so it splits the stack.
///
/// Both are published. Qwen3.8-Flash-Next-FP8 is `PerExpert` throughout;
/// RadixArk's NVFP4 repack is `PerExpert` for the quantized routed experts and
/// `Stacked` for the MTP block, which it leaves in BF16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExpertLayout {
    #[default]
    PerExpert,
    Stacked,
}

/// Knobs a release can differ on without differing architecturally.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Qwen4ExpLayout {
    pub trunk_experts: ExpertLayout,
    pub mtp_experts: ExpertLayout,
}

impl Qwen4ExpLayout {
    /// The layouts seen in published releases, cheapest guess first: both
    /// published checkpoints split their trunk experts, and differ only on the
    /// MTP block, which a quantizer leaves alone when it excludes `mtp.*`.
    pub fn candidates() -> [Self; 2] {
        [
            Self::default(),
            Self {
                mtp_experts: ExpertLayout::Stacked,
                ..Self::default()
            },
        ]
    }
}

fn moe(prefix: &str, cfg: &ModelConfig, layout: ExpertLayout, out: &mut Vec<ExpectedTensor>) {
    let (h, mi) = (cfg.hidden_size, cfg.moe_intermediate_size);
    out.push(ExpectedTensor::new(
        format!("{prefix}.gate.weight"),
        [cfg.num_experts, h],
    ));
    match layout {
        ExpertLayout::Stacked => {
            // gate and up fused along the intermediate dim, experts stacked.
            out.push(ExpectedTensor::new(
                format!("{prefix}.experts.gate_up_proj"),
                [cfg.num_experts, mi * 2, h],
            ));
            out.push(ExpectedTensor::new(
                format!("{prefix}.experts.down_proj"),
                [cfg.num_experts, h, mi],
            ));
        }
        ExpertLayout::PerExpert => {
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
    qwen4_exp_manifest_with(cfg, Qwen4ExpLayout::default())
}

/// [`qwen4_exp_manifest`] for a release that stores its experts differently.
pub fn qwen4_exp_manifest_with(
    cfg: &ModelConfig,
    layout: Qwen4ExpLayout,
) -> Result<Vec<ExpectedTensor>> {
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
        moe(&format!("{base}.mlp"), cfg, layout.trunk_experts, &mut out);
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
        moe(&format!("{base}.mlp"), cfg, layout.mtp_experts, &mut out);
    }
    if cfg.mtp_num_hidden_layers > 0 {
        hyper_connection("mtp.hyper_connection_mixer", cfg, false, &mut out);
        for fc in ["fc_embedding", "fc_hidden"] {
            out.push(ExpectedTensor::new(format!("mtp.{fc}.weight"), [h, h]));
        }
        // The two MTP inputs are NOT the same width, and the checkpoint is the
        // only place that says so. `pre_fc_norm_embedding` normalises a plain
        // token embedding (hidden); `pre_fc_norm_hidden` normalises the trunk's
        // hyper-connection state, which is hc_count streams wide (10240 on the
        // published model, against a hidden of 2560). Both releases agree.
        out.push(ExpectedTensor::new("mtp.pre_fc_norm_embedding.weight", [h]));
        out.push(ExpectedTensor::new(
            "mtp.pre_fc_norm_hidden.weight",
            [cfg.hc_count * h],
        ));
    }
    Ok(out)
}

#[cfg(test)]
#[path = "qwen4_exp_tests.rs"]
mod tests;
