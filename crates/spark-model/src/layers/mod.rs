// SPDX-License-Identifier: AGPL-3.0-only

pub mod aux_d2h;
pub mod canonical_verify;
pub mod deepseek_v4_mtp;
pub mod dense_ffn;
pub mod dflash_head;
pub mod ep_dispatch;
pub mod fp8_calibration;
mod glm5_kda;
pub mod glm_kv_shard;
pub mod glm_sp;
pub mod glm_sp_overlap;
pub mod glm_sp_uneven;
pub mod moe;
mod moe_grouped_decode;
pub mod mtp_head;
pub(crate) mod mtp_meta;
pub mod mtp_multi;
pub(crate) mod nemotron_decode_policy;
pub mod nemotron_mamba2;
pub mod nemotron_moe;
pub mod ngram_embed;
pub mod ops;
pub mod ple;
pub mod qsa;
pub mod qwen3_attention;
pub mod qwen3_ssm;
pub mod qwen4exp_ckpt;
pub mod qwen4exp_draft_head;
pub mod qwen4exp_mtp;
pub mod qwen4exp_sp_pipe;
pub mod vision_encoder;
pub mod w4a16_gemv_tiers;

pub(crate) mod kernel_pick;
pub use kernel_pick::*;

pub(crate) use deepseek_v4_mtp::argmax_grammar_masked;
pub use deepseek_v4_mtp::{DeepseekV4MtpHead, DeepseekV4MtpProposerState};
pub use dense_ffn::{DenseFfnLayer, DenseFfnWeights, FfnActivation};
pub use dflash_head::{
    BlockDiffusionDraftHead, DflashLayer, DflashProposerState, DflashQuantization,
};
pub use glm5_kda::{Glm5KdaLayer, Glm5KdaWeights, Glm5Projection};
pub(crate) use glm5_kda::{
    kda_batched_ffn_enabled, kda_conv_planes_enabled, kda_flash_prefill_word,
    kda_multi_seq_enabled, kda_side_proj_lt_enabled, verify_fused_tp_hc_enabled,
};
pub use moe::MoeLayer;
pub use moe_grouped_decode::{
    moe_grouped_decode_decide, moe_grouped_decode_enabled, moe_grouped_decode_for,
    moe_grouped_decode_forced, moe_grouped_decode_min_rows,
};
pub use mtp_head::{MtpHead, MtpQuantization, mtp_drafter_prefill_enabled};
pub use nemotron_mamba2::NemotronMamba2Layer;
pub use nemotron_moe::NemotronMoeLayer;
pub use qwen3_attention::Qwen3AttentionLayer;
pub use qwen3_ssm::Qwen3SsmLayer;
pub use qwen4exp_mtp::{Qwen4ExpMtpHead, Qwen4ExpMtpProposerState};
pub(crate) use vision_encoder::{GlmVisionBlockWeights, GlmVisionMergerWeights, GlmVisionWeights};
pub use vision_encoder::{MergerLayer, ViTBlock, VisionEncoder, VitWeight};
pub mod vision_tower;
pub use vision_tower::VisionTower;

use crate::layer::ForwardContext;
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

/// FFN component: MoE (expert routing), dense SwiGLU, or None (standalone attention).
#[allow(clippy::large_enum_variant)]
pub enum FfnComponent {
    Moe(MoeLayer),
    Dense(DenseFfnLayer),
    /// No FFN — used by Nemotron-H standalone attention layers.
    None,
}

mod ffn_c4;
mod ffn_glm;

impl FfnComponent {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// True for a plain dense (SwiGLU) FFN. Wide-batch verify paths gate their
    /// `forward_prefill` fast path on this: batching reads dense weights once
    /// (big win at N=17), but on a 256-expert MoE the grouped-GEMM is a net
    /// loss at small batch (per-expert M~1 + sort/permute overhead), so MoE
    /// keeps its per-token loop.
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// True when this MoE FFN can serve DECODE through the grouped read-once
    /// GEMM (forward_prefill) instead of the pairwise per-slot loop. The
    /// is_dense() comment above asserts grouped is "a net loss at small batch"
    /// on a 256-expert MoE, but that was never measured for decode CONCURRENCY
    /// (n=4) where the pairwise path re-reads ~14-20 distinct experts as 40
    /// per-slot CTAs. Native-NVFP4-routed only (forward_prefill's unconditional
    /// grouped path); dense/none are false.
    pub fn moe_grouped_decode_ok(&self) -> bool {
        match self {
            Self::Moe(m) => m.grouped_decode_ok(),
            _ => false,
        }
    }

    /// ATLAS_FP32_ROUTING active for this FFN (MoE only; false otherwise).
    pub fn fp32_routing_active(&self) -> bool {
        match self {
            Self::Moe(m) => m.fp32_routing_active(),
            _ => false,
        }
    }

    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match self {
            Self::Moe(m) => m.forward(input, ctx, stream),
            Self::Dense(d) => d.forward(input, ctx, stream),
            Self::None => Ok(input),
        }
    }

    /// [`Self::forward`] for a single-token mHC decode layer: the MoE may run
    /// the layer's low-rank `hc_post` itself (`hc_post`, fused with its last
    /// kernel under `ATLAS_QWEN4EXP_DECODE_FUSE`). Returns the output and
    /// whether the post ran; when it did not, the caller still runs it.
    pub fn forward_hc_post(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        hc_post: Option<ops::qwen4exp_decode_fuse::MoeHcPost>,
        stream: u64,
    ) -> Result<(DevicePtr, bool)> {
        match self {
            Self::Moe(m) => m.forward_hc_post(input, ctx, hc_post, stream),
            _ => Ok((self.forward(input, ctx, stream)?, false)),
        }
    }

    /// `rows` rows of `input` -> `moe_output` rows, each byte-identical to
    /// [`Self::forward`] on that row, with the MoE's per-row collectives
    /// batched (`MoeLayer::forward_rows`). `None` where that does not hold
    /// (a dense FFN, or a MoE whose decode takes a grouped prefill arm): the
    /// caller keeps its per-row `forward` loop. The flag: `hc_post` ran
    /// (`MoeLayer::forward_rows`); if not, the caller runs the post.
    pub fn forward_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        hc_post: Option<ops::qwen4exp_decode_fuse::MoeHcPost>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, bool)>> {
        match self {
            Self::Moe(m) => m.forward_rows(input, rows, hc_post, ctx, stream),
            _ => Ok(None),
        }
    }

    /// [`Self::forward_rows`] over the first `active` of `rows` rows of a
    /// padded batch, with the padding rows' outputs zeroed: a padding row
    /// (zero hidden, dummy state, discarded logits) then injects nothing
    /// into its highway, where `forward` on its zero input also produced
    /// zero but cost a full row of expert reads (`hc_post` only unpadded).
    pub fn forward_rows_padded(
        &self,
        input: DevicePtr,
        rows: usize,
        active: usize,
        hc_post: Option<ops::qwen4exp_decode_fuse::MoeHcPost>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, bool)>> {
        let active = active.min(rows);
        if active == 0 {
            return Ok(None);
        }
        let hc_post = hc_post.filter(|_| active == rows);
        let Some((out, posted)) = self.forward_rows(input, active, hc_post, ctx, stream)? else {
            return Ok(None);
        };
        if active < rows {
            let row = ctx.config.hidden_size * 2;
            ctx.gpu
                .memset_async(out.offset(active * row), 0, (rows - active) * row, stream)?;
        }
        Ok(Some((out, posted)))
    }

    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k2(input, ctx, stream),
            Self::Dense(d) => d.forward_k2(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_k3(input, ctx, stream),
            Self::Dense(d) => d.forward_k3(input, ctx, stream),
            Self::None => Ok(()),
        }
    }

    /// Whether the K=m (m<=8) batched-GEMV verify FFN is available (dense
    /// only — MoE / missing batch4/batch8 kernel / non-NVFP4 weights →
    /// false). Lets callers gate branch entry BEFORE computing the pre-FFN
    /// norm, so there is no half-done fallthrough to `forward_prefill`.
    pub fn can_forward_km(&self, m: u32) -> bool {
        matches!(self, Self::Dense(d) if d.can_forward_km(m))
    }

    /// K=m (m=4..8) verify FFN via batched GEMV (dense only). Returns
    /// `false` when the path is unavailable (MoE / missing batchm kernel /
    /// non-NVFP4 weights) so the caller can fall back to `forward_prefill`.
    pub fn try_forward_km(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        match self {
            Self::Dense(d) if d.can_forward_km(m) => {
                d.forward_km(input, m, ctx, stream)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_prefill(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_prefill(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_batched(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_token_major_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_token_major_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }

    pub fn forward_atomic_c4_decode(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_atomic_c4_decode(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_batched(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod glm_gpu_test_backend;
