// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};

use super::*;

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention::trait_impl::multi_seq::mla) fn glm_k3_query_stage(
        &self,
        c: &MultiSeqCtx<'_>,
        mla: &MlaWeights,
        long_verify: bool,
    ) -> Result<Option<QueryPlan>> {
        let batchm = enabled(&c.fwd.config.model_type)?;
        let compare = compare_enabled(&c.fwd.config.model_type, batchm)?;
        if !batchm || !long_verify {
            return Ok(None);
        }
        ensure!(
            !c.fwd.graph_capture && !c.fwd.gpu.stream_is_capturing(c.stream),
            "{FLAG} is eager-only and cannot run during graph capture"
        );
        ensure!(
            rows_supported(c.n)
                && c.h == HIDDEN
                && c.nq == 32
                && c.hd == 256
                && c.bf16 == BF16
                && c.bs > 0
                && c.bs.is_multiple_of(4)
                && mla.q_lora_rank == Q_LORA
                && mla.kv_lora_rank == 512
                && mla.nope == 256
                && mla.v_dim == 256
                && mla.rope == 0
                && mla.o_lora_rank == 0
                && mla.wq_a_nvfp4.is_none()
                && mla.wq_b_nvfp4.is_none()
                && mla.wq_a_fp8.is_none()
                && mla.wq_b_fp8.is_none()
                && mla.glm_indexer.is_some()
                && c.fwd.config.index_n_heads == 32
                && c.fwd.config.index_head_dim == 128,
            "{FLAG}: unsupported GLM K3 query geometry"
        );
        ensure!(
            c.seq_lens.len() == c.n
                && c.seq_lens
                    .windows(2)
                    .all(|p| p[0].checked_add(1) == Some(p[1])),
            "{FLAG}: query rows must retain the validated causal order"
        );
        // The exact dense/sparse threshold boundary is valid: index-Q for a
        // dense row is harmless and is ignored by the unchanged <=2048
        // fallback. A fully short verify simply keeps the scalar path.
        if !c.seq_lens.iter().any(|&position| position >= 2048) {
            return Ok(None);
        }
        aligned_nonnull(c.normed, 16, "normalized input")?;
        ensure!(
            self.dense_gemv_batchm_k.0 != 0 && self.dense_gemv_k.0 != 0 && self.rms_norm_w_k.0 != 0,
            "{FLAG}: scalar, batchm, or RMS query kernels are unavailable"
        );
        aligned_nonnull(mla.wq_a.weight, 16, "Qa weight")?;
        aligned_nonnull(mla.wq_b.weight, 16, "Qb weight")?;
        aligned_nonnull(mla.q_a_norm.weight, 16, "Qa RMS weight")?;
        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("{FLAG}: validated GLM indexer missing");
        aligned_nonnull(indexer.wq_b.weight, 16, "index-Q weight")?;

        let b = c.fwd.buffers;
        let s = b.sizes();
        let arenas = [
            (b.hidden_states(), s.hidden_states),
            (b.residual(), s.residual),
            (b.norm_output(), s.norm_output),
            (b.qkv_output(), s.qkv_output),
            (b.attn_output(), s.attn_output),
            (b.moe_output(), s.moe_output),
            (b.expert_gate_out(), s.expert_gate_out),
            (b.expert_up_out(), s.expert_up_out),
            (b.expert_down_out(), s.expert_down_out),
            (b.ssm_qkvz(), s.ssm_qkvz),
            (b.ssm_ba(), s.ssm_ba),
            (b.ssm_deinterleaved(), s.ssm_deinterleaved),
            (b.ssm_gates(), s.ssm_gates),
            (b.ssm_conv_out_f32(), s.ssm_conv_out_f32),
        ];
        aligned_nonnull(c.hidden, 16, "hidden input")?;
        aligned_nonnull(c.residual, 16, "residual")?;
        let live = [
            (
                c.hidden,
                arena_remaining_capacity(c.hidden, &arenas, "hidden input")?,
            ),
            (
                c.residual,
                arena_remaining_capacity(c.residual, &arenas, "residual")?,
            ),
            (b.expert_gate_out(), s.expert_gate_out),
            (b.expert_up_out(), s.expert_up_out),
            (b.expert_down_out(), s.expert_down_out),
            (b.qkv_output(), s.qkv_output),
            (b.attn_output(), s.attn_output),
            (b.moe_output(), s.moe_output),
            (b.ssm_gates(), s.ssm_gates),
            (b.ssm_conv_out_f32(), s.ssm_conv_out_f32),
            (b.ssm_qkvz(), s.ssm_qkvz),
            (mla.wq_a.weight, Q_LORA * HIDDEN * BF16),
            (mla.wq_b.weight, Q_DIM * Q_LORA * BF16),
            (mla.q_a_norm.weight, Q_LORA * BF16),
            (indexer.wq_b.weight, INDEX_DIM * Q_LORA * BF16),
        ];
        // ssm_qkvz is intentionally the short-lived diagnostic reference
        // arena. Query outputs remain disjoint from its whole allocation;
        // diagnostic mode borrows that exact range only before the causal
        // loop starts, as checked by QueryPlan::new.
        let diagnostic = if compare {
            Some((b.ssm_qkvz(), s.ssm_qkvz))
        } else {
            None
        };
        // Arenas sized for fewer rows keep the scalar per-row projections.
        let (latent_bytes, query_bytes) = (latent_bytes(c.n), query_bytes(c.n));
        if s.ssm_ba < latent_bytes || s.ssm_deinterleaved < query_bytes {
            tracing::debug!(
                rows = c.n,
                latent_capacity = s.ssm_ba,
                latent_bytes,
                query_capacity = s.ssm_deinterleaved,
                query_bytes,
                "{FLAG}: query staging arenas too small; scalar query projection"
            );
            return Ok(None);
        }
        QueryPlan::new(
            c.n,
            c.normed,
            arena_remaining_capacity(c.normed, &arenas, "normalized input")?,
            b.ssm_ba(),
            s.ssm_ba,
            b.ssm_deinterleaved(),
            s.ssm_deinterleaved,
            diagnostic,
            &live,
        )
        .map(Some)
    }
}
