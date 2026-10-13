// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded, opt-in independent C4 policy shared by server and model dispatch.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferSizes;

pub struct C4Policy<'a> {
    pub model_type: &'a str,
    pub enabled: bool,
    pub world: usize,
    pub tp: usize,
    pub ep: usize,
    pub ep_v2: bool,
    pub independent: bool,
    pub kda_multi: bool,
    pub mla_multi: bool,
    pub sparse: bool,
    pub sparse_graphs: bool,
    pub c4_sparse: bool,
    pub no_multiseq_graphs: bool,
    pub kv_overcommit_disabled: bool,
    pub fp32_state: bool,
}

impl C4Policy<'_> {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.enabled && self.model_type == "glm5_next",
            "independent GLM C4 requires ATLAS_GLM_C4_DECODE=1 and exact glm5_next"
        );
        ensure!(
            self.world == 2 && self.tp == 2 && self.ep == 2 && self.ep_v2,
            "GLM C4 requires world2/TP2/EP2 and ATLAS_EP_PROTOCOL=v2"
        );
        ensure!(
            self.independent
                && self.kda_multi
                && self.mla_multi
                && self.sparse == self.c4_sparse
                && !self.sparse_graphs
                && self.fp32_state,
            "GLM C4 requires non-speculative decode, KDA/MLA_MULTI_SEQ=1, FP32 KDA state, sparse graphs off and base sparse matching C4_SPARSE"
        );
        ensure!(
            !self.c4_sparse || (self.no_multiseq_graphs && self.kv_overcommit_disabled),
            "GLM C4_SPARSE requires ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1 and ATLAS_KV_OVERCOMMIT=0"
        );
        Ok(())
    }
}

pub fn enabled(model_type: &str) -> bool {
    model_type == "glm5_next" && std::env::var("ATLAS_GLM_C4_DECODE").as_deref() == Ok("1")
}

pub const SPARSE_CONTEXT_LIMIT: usize = 16384;

pub fn validate_prefill_budget(tokens: usize, c4_sparse: bool) -> Result<()> {
    ensure!(
        !c4_sparse || (1..=1024).contains(&tokens),
        "GLM C4_SPARSE requires a configured prefill budget in 1..1024"
    );
    Ok(())
}

fn context_limit(c4_sparse: bool) -> usize {
    if c4_sparse {
        SPARSE_CONTEXT_LIMIT
    } else {
        2048
    }
}

pub fn sparse_enabled(model_type: &str) -> bool {
    model_type == "glm5_next" && std::env::var("ATLAS_GLM_C4_SPARSE").as_deref() == Ok("1")
}

/// Direct-server boundary rejects misspelled flags and other architectures.
pub fn validate_sparse_flag(model_type: &str, value: Option<&str>) -> Result<bool> {
    ensure!(
        matches!(value, None | Some("0" | "1")),
        "ATLAS_GLM_C4_SPARSE must be 0 or 1"
    );
    ensure!(
        value != Some("1") || model_type == "glm5_next",
        "ATLAS_GLM_C4_SPARSE requires glm5_next"
    );
    Ok(value == Some("1"))
}

pub fn policy_from_env(
    model_type: &str,
    world: usize,
    tp: usize,
    ep: usize,
    ep_v2: bool,
    independent: bool,
) -> C4Policy<'_> {
    C4Policy {
        model_type,
        enabled: enabled(model_type),
        world,
        tp,
        ep,
        ep_v2,
        independent,
        kda_multi: crate::layers::kda_multi_seq_enabled(),
        mla_multi: crate::layers::qwen3_attention::glm_mla_multi_seq_enabled(),
        sparse: std::env::var("ATLAS_GLM_MULTI_SEQ_SPARSE").as_deref() == Ok("1"),
        sparse_graphs: std::env::var("ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS").as_deref() == Ok("1"),
        c4_sparse: sparse_enabled(model_type),
        no_multiseq_graphs: std::env::var("ATLAS_NO_DECODE_GRAPHS_MULTISEQ").as_deref() == Ok("1"),
        kv_overcommit_disabled: matches!(
            std::env::var("ATLAS_KV_OVERCOMMIT").as_deref(),
            Ok("0" | "false")
        ),
        fp32_state: !crate::layers::qwen3_ssm::ssm_h_fp16_enabled(),
    }
}

pub fn validate_launch_limits(
    active: usize,
    admitted: usize,
    context: usize,
    bf16: bool,
) -> Result<()> {
    validate_launch_limits_for_mode(active, admitted, context, bf16, sparse_enabled("glm5_next"))
}

pub fn validate_launch_limits_for_mode(
    active: usize,
    admitted: usize,
    context: usize,
    bf16: bool,
    c4_sparse: bool,
) -> Result<()> {
    let limit = context_limit(c4_sparse);
    ensure!(
        active == 4 && admitted == 4 && (1..=limit).contains(&context) && bf16,
        "GLM C4 requires active4/admitted4, context1..{limit} and BF16 KV"
    );
    Ok(())
}

pub fn validate_runtime(
    config: &ModelConfig,
    world: usize,
    ep_v2: bool,
    independent: bool,
) -> Result<()> {
    policy_from_env(
        &config.model_type,
        world,
        config.tp_world_size,
        config.ep_world_size,
        ep_v2,
        independent,
    )
    .validate()?;
    validate_geometry(config)
}

pub(crate) fn validate_geometry(config: &ModelConfig) -> Result<()> {
    ensure!(
        config.hidden_size == 4096
            && config.num_attention_heads == 32
            && config.q_lora_rank == 1536
            && config.kv_lora_rank == 512
            && config.qk_nope_head_dim == 256
            && config.qk_rope_head_dim == 0
            && config.v_head_dim == 256
            && config.index_topk == 2048
            && config.hc_mult > 0,
        "GLM C4 requires the validated local TP2 MLA/mHC geometry"
    );
    Ok(())
}

pub fn batched_kda_rows(rows: usize, kda: bool, c4: bool) -> Result<bool> {
    if rows == 4 {
        ensure!(
            kda && c4,
            "GLM C4 must use opted-in true batched KDA, never scalar fallback"
        );
        return Ok(true);
    }
    Ok(kda && matches!(rows, 2 | 3))
}

pub fn validate_positions(positions: impl IntoIterator<Item = usize>, rows: usize) -> Result<()> {
    validate_positions_for_mode(positions, rows, sparse_enabled("glm5_next"))
}

pub fn validate_positions_for_mode(
    positions: impl IntoIterator<Item = usize>,
    rows: usize,
    c4_sparse: bool,
) -> Result<()> {
    ensure!(
        matches!(rows, 2..=4),
        "GLM C4 lane accepts only independent C2/C3/C4 batch widths"
    );
    let mut count = 0;
    let limit = context_limit(c4_sparse);
    for position in positions {
        ensure!(position < limit, "GLM C4 position must be below {limit}");
        count += 1;
    }
    ensure!(count == rows, "GLM C4 host position count mismatch");
    Ok(())
}

pub fn validate_projection_handles(nvfp4_m4: u64, bf16_batchm: u64) -> Result<()> {
    ensure!(
        nvfp4_m4 != 0 && bf16_batchm != 0,
        "GLM C4 requires nonzero exact-M4 NVFP4 and BF16 batch-M projections"
    );
    Ok(())
}

fn hc_bytes(mult: usize) -> Result<(usize, usize, usize)> {
    ensure!(mult > 0, "GLM C4 requires mHC streams");
    let post = mult
        .checked_mul(4 * 4)
        .ok_or_else(|| anyhow::anyhow!("C4 mHC size overflow"))?;
    let highway = post
        .checked_mul(4096)
        .ok_or_else(|| anyhow::anyhow!("C4 mHC size overflow"))?;
    let comb = post
        .checked_mul(mult)
        .ok_or_else(|| anyhow::anyhow!("C4 mHC size overflow"))?;
    Ok((highway, post, comb))
}

/// Fixed C4 lower bounds, checked before lookup and again before layer work.
pub fn validate_scratch(s: &BufferSizes, hc_mult: usize) -> Result<()> {
    validate_scratch_rows(s, hc_mult, 4)
}

pub(crate) fn validate_scratch_rows(s: &BufferSizes, hc_mult: usize, rows: usize) -> Result<()> {
    let (highway, post, comb) = hc_bytes(hc_mult)?;
    for (name, available, required) in [
        ("hidden", s.hidden_states, 4 * 4096 * 2),
        ("norm", s.norm_output, 4 * 4096 * 2),
        ("QKV projections", s.qkv_output, 100608),
        ("packed QKV", s.ssm_qkvz, 98304),
        ("convolved QKV", s.ssm_conv_out_f32, 98304),
        ("gates/expanded Q", s.ssm_deinterleaved, 65536),
        ("Q latent", s.ssm_ba, 12288),
        ("absorbed attention", s.attn_output, 131072),
        ("expert gate", s.expert_gate_out, 131072),
        ("expert up/absorbed Q", s.expert_up_out, 131072),
        ("expert down", s.expert_down_out, 262144),
        ("sort metadata", s.gate_logits, 1540),
        ("MoE output", s.moe_output, 32768),
        ("mHC highway", s.hc_streams, highway),
        ("mHC post", s.hc_post, post),
        ("mHC comb", s.hc_comb, comb),
    ] {
        let required = if name == "sort metadata" {
            rows.checked_mul(96).and_then(|n| n.checked_add(1156))
        } else {
            required.checked_div(4).and_then(|n| n.checked_mul(rows))
        }
        .ok_or_else(|| anyhow::anyhow!("independent scratch size overflow"))?;
        ensure!(
            available >= required,
            "GLM C4 {name} scratch needs {required} bytes, has {available}"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm_c4_tests.rs"]
mod tests;
