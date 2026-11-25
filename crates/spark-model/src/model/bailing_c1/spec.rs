// SPDX-License-Identifier: AGPL-3.0-only

//! Pure-logic Ling layer plan and the conservative memory math the driver
//! sizes its pools and scratch by. No GPU here — colocated tests pin the
//! numbers against the checkpoint facts (42 decoder layers; 35 KDA + 7
//! gated MLA at {5, 11, 17, 23, 29, 35, 41}; dense MLP prefix 2; 512
//! routed experts).

use anyhow::{Result, ensure};
use atlas_core::config::{BailingParams, LayerType, ModelConfig};

/// Ling KDA geometry the kernels support (kernels/gb10/ling-3.2-air/nvfp4/
/// kda.cu: 32 heads x head_dim 128, state [32, 128, 128] FP32).
pub(crate) const KDA_HEADS: usize = 32;
pub(crate) const KDA_HEAD_DIM: usize = 128;

/// Per-token bf16 bytes of the MLA half of the paged pools: K rows hold
/// `[kv_lora 512 | rope 64]`, V rows hold the latent `512` (the gated-MLA
/// kernels read exactly these widths — `LING_MLA_KV_DIM` / `LING_MLA_LORA`).
pub(crate) const MLA_K_ROW_BF16: usize = 576;
pub(crate) const MLA_V_ROW_BF16: usize = 512;

/// Default per-layer prefill sub-chunk the shared scratch is sized for
/// (`ATLAS_LING_PREFILL_ROWS` overrides). 512 keeps every decode/verify
/// buffer small while an ISL-4096 prompt takes 8 sub-chunks — conservative
/// on the 121.6 GiB unified box.
pub(crate) const DEFAULT_PREFILL_ROWS: usize = 512;

/// Hard prompt cap the driver accepts without a recompile-grade override
/// (`ATLAS_LING_PREFILL_ROWS`): ISL 4096-class prompts per the campaign
/// envelope, with a little slack.
pub(crate) const MAX_PREFILL_ROWS: usize = 8192;

fn prefill_rows_from_env() -> Result<usize> {
    let rows = std::env::var("ATLAS_LING_PREFILL_ROWS")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|_| anyhow::anyhow!("ATLAS_LING_PREFILL_ROWS must be an integer"))?;
    let rows = rows.unwrap_or(DEFAULT_PREFILL_ROWS);
    ensure!(
        rows > 0 && rows.is_multiple_of(8) && rows <= MAX_PREFILL_ROWS,
        "ling: ATLAS_LING_PREFILL_ROWS must be a positive multiple of 8 up to \
         {MAX_PREFILL_ROWS}, got {rows}"
    );
    Ok(rows)
}

/// The loader-facing FFN plan of one decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FfnKind {
    Dense,
    Moe,
}

/// One decoder layer's driver plan entry (attention kind from
/// `config.layer_types`, FFN kind from the dense prefix, SwiGLU clamps from
/// the checkpoint lists — all ground truth the `BailingWeightLoader`'s
/// tensor-signature plan validated against the store at load).
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerSpec {
    pub attention: LayerType,
    pub ffn: FfnKind,
    /// Routed-expert SwiGLU clamp of this layer (0 = none).
    pub routed_swiglu_limit: i64,
    /// Shared-expert SwiGLU clamp of this layer (0 = none).
    pub shared_swiglu_limit: i64,
}

/// The whole-stack plan plus the derived sizes.
#[derive(Clone, Debug)]
pub(crate) struct LingSpec {
    pub layers: Vec<LayerSpec>,
    pub hidden: usize,
    pub heads: usize,
    pub mla_layers: usize,
    pub kda_layers: usize,
    /// Routed experts per MoE layer.
    pub experts: usize,
    pub top_k: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling: f32,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub dense_inter: usize,
    pub eps: f32,
    pub lower_bound: f32,
    pub conv_kernel: usize,
    pub rope_theta: f64,
    /// Rows per prefill sub-chunk (the shared scratch is sized for it).
    pub prefill_rows: usize,
}

/// Prefill sub-chunking plan: how the layers process `total` prompt rows
/// that follow `seq_len_start` processed positions.
#[derive(Clone, Debug)]
pub(crate) struct PrefillPlan {
    /// Sub-chunk row counts (last may be the short tail).
    pub chunks: Vec<usize>,
}

impl PrefillPlan {
    /// ISL-class chunk decomposition: `cap`-wide sub-chunks, the tail last.
    pub(crate) fn for_rows(total: usize, cap: usize) -> Self {
        Self {
            chunks: (0..total)
                .step_by(cap)
                .map(|start| (total - start).min(cap))
                .collect(),
        }
    }
}

/// Bytes one sequence's KDA states occupy per pool slot (35 layers x
/// [h_state 2 MiB + conv_state 192 KiB]): 76.8 MiB per `--ssm-cache-slots`
/// slot.
pub(crate) fn kda_state_bytes_per_slot(config: &ModelConfig) -> usize {
    config.num_ssm_layers() * (config.ssm_h_state_bytes() + config.ssm_conv_state_bytes())
}

/// Bytes per token of the MLA half of the pools, all 7 layers: K paged rows
/// (house `PagedKvCache`, one 576-wide bf16 row) + V latent rows (the
/// layer-owned pool, 512-wide).
pub(crate) fn mla_kv_bytes_per_token(config: &ModelConfig) -> usize {
    config.num_attention_layers() * (MLA_K_ROW_BF16 + MLA_V_ROW_BF16) * 2
}

impl LingSpec {
    /// Build and validate the plan from the parsed config. Every hard shape
    /// the kernels assume is enforced here, loudly, at load.
    pub(crate) fn from_config(config: &ModelConfig) -> Result<Self> {
        let params = config
            .bailing
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ling: config parsed without Bailing params"))?;
        let bailing = params.clone();
        ensure!(
            config.model_type == "bailing_moe_v3_vl",
            "ling: spec from a non-bailing model_type {:?}",
            config.model_type
        );
        ensure!(
            config.linear_num_key_heads == KDA_HEADS
                && config.linear_key_head_dim == KDA_HEAD_DIM
                && config.linear_num_value_heads == KDA_HEADS
                && config.linear_value_head_dim == KDA_HEAD_DIM,
            "ling: the GB10 KDA kernels support 32 heads x head_dim 128, config has \
             {} x {}/{}/{}",
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            config.linear_num_value_heads,
            config.linear_value_head_dim
        );
        ensure!(
            config.hidden_size == 2560,
            "ling: the kernels are built for hidden 2560, config has {}",
            config.hidden_size
        );
        ensure!(
            config.num_attention_heads == 32
                && config.kv_lora_rank == 512
                && config.qk_nope_head_dim == 128
                && config.qk_rope_head_dim == 64
                && config.v_head_dim == 128
                && config.q_lora_rank == 0,
            "ling: the gated-MLA kernels support the Ling geometry \
             (32 heads, kv_lora 512, nope 128, rope 64, v 128, full-rank q_proj)"
        );
        let SpecParams {
            n_group,
            topk_group,
            routed_swiglu,
            shared_swiglu,
        } = spec_params(&bailing);
        let n_group = n_group;
        let topk_group = topk_group;
        let dense_prefix = config
            .mlp_only_layers
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m + 1);
        let layers = (0..config.num_hidden_layers)
            .map(|idx| LayerSpec {
                attention: config.layer_type(idx),
                ffn: if idx < dense_prefix {
                    FfnKind::Dense
                } else {
                    FfnKind::Moe
                },
                routed_swiglu_limit: routed_swiglu.get(idx).copied().unwrap_or(0),
                shared_swiglu_limit: shared_swiglu.get(idx).copied().unwrap_or(0),
            })
            .collect::<Vec<_>>();
        let spec = Self {
            mla_layers: layers
                .iter()
                .filter(|l| l.attention == LayerType::FullAttention)
                .count(),
            kda_layers: layers
                .iter()
                .filter(|l| l.attention == LayerType::LinearAttention)
                .count(),
            layers,
            hidden: config.hidden_size,
            heads: config.num_attention_heads,
            experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            n_group,
            topk_group,
            routed_scaling: config.routed_scaling_factor as f32,
            moe_inter: config.moe_intermediate_size,
            shared_inter: config.shared_expert_intermediate_size,
            dense_inter: config.intermediate_size,
            eps: config.rms_norm_eps as f32,
            lower_bound: config.kda_gate_lower_bound,
            conv_kernel: config.linear_conv_kernel_dim,
            rope_theta: config.rope_theta,
            prefill_rows: prefill_rows_from_env()?,
        };
        ensure!(
            spec.mla_layers == config.num_attention_layers()
                && spec.kda_layers == config.num_ssm_layers(),
            "ling: layer plan disagrees with the config layer counts"
        );
        ensure!(
            spec.experts.is_multiple_of(spec.n_group),
            "ling: experts must be divisible by n_group"
        );
        Ok(spec)
    }
}

struct SpecParams {
    n_group: usize,
    topk_group: usize,
    routed_swiglu: Vec<i64>,
    shared_swiglu: Vec<i64>,
}

fn spec_params(params: &BailingParams) -> SpecParams {
    SpecParams {
        n_group: params.n_group,
        topk_group: params.topk_group,
        routed_swiglu: params.routed_swiglu.clone(),
        shared_swiglu: params.shared_swiglu.clone(),
    }
}
