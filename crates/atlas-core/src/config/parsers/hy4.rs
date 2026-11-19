// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

use super::super::{Hy4Attention, Hy4Config, Hy4Mlp, ModelConfig, finalize_config};

fn integer(raw: &Value, key: &str) -> Result<usize> {
    let n = raw[key]
        .as_u64()
        .with_context(|| format!("hy_v4_flash missing unsigned integer `{key}`"))?;
    usize::try_from(n).with_context(|| format!("hy_v4_flash `{key}` overflows usize"))
}

fn positive(raw: &Value, key: &str) -> Result<usize> {
    let n = integer(raw, key)?;
    ensure!(n > 0, "hy_v4_flash `{key}` must be positive");
    Ok(n)
}

fn number(raw: &Value, key: &str) -> Result<f64> {
    let n = raw[key]
        .as_f64()
        .with_context(|| format!("hy_v4_flash missing number `{key}`"))?;
    ensure!(
        n.is_finite() && n > 0.0,
        "hy_v4_flash `{key}` must be positive"
    );
    Ok(n)
}

fn boolean(raw: &Value, key: &str) -> Result<bool> {
    raw[key]
        .as_bool()
        .with_context(|| format!("hy_v4_flash missing boolean `{key}`"))
}

pub(crate) fn parse_hy4(raw: &Value) -> Result<ModelConfig> {
    let text = raw.get("text_config").unwrap_or(raw);
    ensure!(
        text["model_type"] == "hy_v4_flash",
        "hy_v4_flash text model_type mismatch"
    );
    let mut hy: Hy4Config =
        serde_json::from_value(text.clone()).context("Invalid hy_v4_flash text configuration")?;
    hy.dfly_mask_token_id = raw
        .get("dfly_mask_token_id")
        .filter(|value| !value.is_null())
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()
        .context("hy_v4_flash invalid dfly_mask_token_id")?;
    if let Some(mask) = hy.dfly_mask_token_id {
        ensure!(
            (mask as usize) < positive(text, "vocab_size")?,
            "hy_v4_flash dfly_mask_token_id exceeds vocabulary"
        );
    }
    let layers = positive(text, "num_hidden_layers")?;
    ensure!(
        hy.layer_types.len() == layers
            && hy.swa_pattern.len() == layers
            && hy.mlp_layer_types.len() == layers,
        "hy_v4_flash layer maps must match num_hidden_layers"
    );
    ensure!(
        hy.first_k_dense_replace <= layers,
        "hy_v4_flash dense prefix exceeds layers"
    );
    for i in 0..layers {
        let swa = hy.layer_types[i] == Hy4Attention::SlidingAttention;
        ensure!(
            hy.swa_pattern[i] == u8::from(swa),
            "hy_v4_flash swa_pattern disagrees at layer {i}"
        );
        ensure!(
            (hy.mlp_layer_types[i] == Hy4Mlp::Dense) == (i < hy.first_k_dense_replace),
            "hy_v4_flash mlp_layer_types disagrees with dense prefix at layer {i}"
        );
    }
    ensure!(
        hy.enable_ihc && hy.hc_magnitude.is_finite() && hy.hc_magnitude > 0.0,
        "hy_v4_flash requires positive independent HC magnitude"
    );
    ensure!(
        hy.use_dsa && hy.use_mla && hy.use_compressed_indexer,
        "hy_v4_flash requires compressed DSA MLA"
    );
    ensure!(
        hy.swa_num_attention_heads > 0
            && hy.swa_v_head_dim > 0
            && Some(hy.swa_qk_head_dim)
                == hy.swa_qk_nope_head_dim.checked_add(hy.swa_qk_rope_head_dim)
            && hy.swa_qk_rope_head_dim > 0,
        "hy_v4_flash invalid SWA dimensions"
    );
    ensure!(
        hy.routed_expert_hidden_size > 0 && hy.swiglu_limit.is_finite() && hy.swiglu_limit > 0.0,
        "hy_v4_flash invalid latent MoE dimensions/clamp"
    );
    ensure!(
        hy.compressed_indexer_compress_ratio > 0
            && hy.compressed_indexer_group_topk > 1
            && hy
                .compressed_indexer_group_topk
                .checked_mul(hy.compressed_indexer_compress_ratio)
                == Some(hy.compressed_indexer_candidate_topk)
            && hy.compressed_indexer_head_dim > 0
            && hy.compressed_indexer_num_attention_heads > 0
            && hy.compressed_indexer_key_pooling == "gated"
            && hy.compressed_indexer_skip_fine_select,
        "hy_v4_flash invalid coarse IndexPool contract"
    );
    for rope in [
        &hy.rope_parameters.deepseek_sparse_attention,
        &hy.rope_parameters.sliding_attention,
    ] {
        ensure!(
            rope.rope_type == "default" && rope.rope_theta.is_finite() && rope.rope_theta > 0.0,
            "hy_v4_flash requires positive default RoPE"
        );
        let rotary = rope
            .mrope_section
            .iter()
            .try_fold(0usize, |n, &v| n.checked_add(v))
            .and_then(|n| n.checked_mul(2));
        ensure!(
            rotary == Some(positive(text, "qk_rope_head_dim")?),
            "hy_v4_flash MRoPE sections must cover rotary dimension"
        );
    }
    ensure!(
        hy.rope_parameters.deepseek_sparse_attention.mrope_section
            == hy.rope_parameters.sliding_attention.mrope_section
            && hy
                .rope_parameters
                .deepseek_sparse_attention
                .mrope_interleaved
                == hy.rope_parameters.sliding_attention.mrope_interleaved,
        "hy_v4_flash MRoPE layouts disagree"
    );
    validate_engram(&hy, layers)?;
    let mut normalized = text.clone();
    let obj = normalized
        .as_object_mut()
        .context("hy_v4_flash text config must be object")?;
    obj.insert(
        "layer_types".into(),
        json!(
            hy.layer_types
                .iter()
                .map(|t| match t {
                    Hy4Attention::DeepseekSparseAttention => "full_attention",
                    Hy4Attention::SlidingAttention => "sliding_attention",
                })
                .collect::<Vec<_>>()
        ),
    );
    // HF uses null for absent low-rank gate/output projections.
    for key in ["o_lora_rank", "gate_lora_rank"] {
        ensure!(text.get(key).is_some(), "hy_v4_flash missing `{key}`");
        if text[key].is_null() {
            obj.remove(key);
        }
    }
    let mut c: ModelConfig =
        serde_json::from_value(normalized).context("hy_v4_flash core dimensions")?;
    for key in [
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "max_position_embeddings",
        "hc_mult",
        "sliding_window",
        "moe_intermediate_size",
    ] {
        positive(text, key)?;
    }
    ensure!(
        Some(positive(text, "qk_head_dim")?) == c.qk_nope_head_dim.checked_add(c.qk_rope_head_dim),
        "hy_v4_flash qk dimensions disagree"
    );
    c.head_dim = positive(text, "qk_head_dim")?;
    c.rms_norm_eps = number(text, "rms_norm_eps")?;
    c.hc_eps = number(text, "hc_eps")? as f32;
    ensure!(
        c.hc_eps.is_finite() && c.hc_eps > 0.0,
        "hy_v4_flash hc_eps outside f32 range"
    );
    c.hc_sinkhorn_iters = 0;
    c.num_experts = positive(text, "n_routed_experts")?;
    c.num_experts_per_tok = positive(text, "num_experts_per_tok")?;
    ensure!(
        c.num_experts_per_tok <= c.num_experts,
        "hy_v4_flash top-k exceeds experts"
    );
    c.shared_expert_intermediate_size = integer(text, "n_shared_experts")?
        .checked_mul(c.moe_intermediate_size)
        .context("hy_v4_flash shared width overflow")?;
    c.norm_topk_prob = boolean(text, "norm_topk_prob")?;
    ensure!(
        boolean(text, "moe_router_use_sigmoid")?,
        "hy_v4_flash requires sigmoid router"
    );
    c.scoring_func = "sigmoid".into();
    c.use_routing_bias = boolean(text, "moe_router_enable_expert_bias")?;
    c.router_logits_f32 = true;
    ensure!(
        positive(text, "index_topk")? == hy.compressed_indexer_candidate_topk
            && positive(text, "index_n_heads")? == hy.compressed_indexer_num_attention_heads
            && positive(text, "index_head_dim")? == hy.compressed_indexer_head_dim,
        "hy_v4_flash index dimensions disagree"
    );
    c.routed_scaling_factor = number(text, "routed_scaling_factor")?;
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"] {
        ensure!(
            integer(text, key)? < c.vocab_size,
            "hy_v4_flash invalid `{key}`"
        );
    }
    boolean(text, "tie_word_embeddings")?;
    c.mlp_only_layers = (0..hy.first_k_dense_replace).collect();
    c.num_attention_heads_per_layer = hy
        .layer_types
        .iter()
        .map(|t| match t {
            Hy4Attention::DeepseekSparseAttention => c.num_attention_heads,
            Hy4Attention::SlidingAttention => hy.swa_num_attention_heads,
        })
        .collect();
    c.rope_theta = hy.rope_parameters.deepseek_sparse_attention.rope_theta;
    c.partial_rotary_factor = c.qk_rope_head_dim as f64 / c.head_dim as f64;
    c.mrope_section = hy.rope_parameters.deepseek_sparse_attention.mrope_section;
    c.mrope_interleaved = hy
        .rope_parameters
        .deepseek_sparse_attention
        .mrope_interleaved;
    c.nested_config = raw.get("text_config").is_some();
    c.weight_prefix = "model".into();
    c.hy4 = Some(hy);
    finalize_config(&mut c, raw)?;
    Ok(c)
}

fn validate_engram(h: &Hy4Config, layers: usize) -> Result<()> {
    ensure!(
        h.engram_max_ngram_size >= 2
            && h.engram_n_head_per_ngram > 0
            && h.engram_n_embed_per_ngram > 0
            && h.engram_n_embed_per_ngram
                .is_multiple_of(h.engram_n_head_per_ngram),
        "hy_v4_flash invalid Engram dimensions"
    );
    let mut seen = std::collections::BTreeSet::new();
    for &id in &h.engram_layer_ids {
        ensure!(
            id > 0 && id <= layers && seen.insert(id),
            "hy_v4_flash invalid/duplicate 1-based Engram layer {id}"
        );
        let sizes = h
            .engram_layer_head_vocab_sizes
            .get(&id)
            .context("hy_v4_flash missing Engram head vocabularies")?;
        ensure!(
            sizes.len() == h.engram_max_ngram_size - 1
                && sizes
                    .iter()
                    .all(|s| s.len() == h.engram_n_head_per_ngram && s.iter().all(|&v| v > 0)),
            "hy_v4_flash invalid Engram head vocabularies"
        );
    }
    ensure!(
        h.engram_layer_head_vocab_sizes.len() == seen.len(),
        "hy_v4_flash unexpected Engram vocabularies"
    );
    Ok(())
}
