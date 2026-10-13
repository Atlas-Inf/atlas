// SPDX-License-Identifier: AGPL-3.0-only
//! Per-row MLA chain; scratch is consumed before the next row.
use super::*;
impl Qwen3AttentionLayer {
    /// Single-sequence absorbed-MLA decode chain. Mirrors
    /// `decode::attention_forward_mla` 1:1 but takes an explicit
    /// per-sequence `normed` input and `o_out` destination so the caller
    /// can drive it once per sequence in a batched decode step.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_mla_decode_one(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: &AttnMetadataDev,
        normed: DevicePtr,
        o_out: DevicePtr,
        mla: &crate::layers::qwen3_attention::types::MlaWeights,
        d: MlaDims,
        stream: u64,
        position: usize,
        staged_v: Option<DevicePtr>,
        prepared_query: Option<super::k3_query::QueryRow>,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        let buffers = c.fwd.buffers;

        // ── Step 1: Q latent → norm → expand ──
        let (q_latent, q_full, index_query) = if let Some(query) = prepared_query {
            // The opt-in K3 query stage has already completed Qa, the exact
            // in-place RMS, Qb, and index-Q for every row.  Keep every causal
            // operation below on this row's retained slices.
            (query.q_latent, query.q_full, Some(query.index_query))
        } else {
            let q_latent = buffers.ssm_ba();
            if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
                self.nvfp4_decode_gemv(
                    gpu,
                    c.fwd.levers.gemv_sw,
                    normed,
                    wqa_nvfp4,
                    q_latent,
                    d.q_lora,
                    d.h,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wq_a,
                    q_latent,
                    d.q_lora,
                    d.h,
                    stream,
                )?;
            }
            ops::rms_norm(
                gpu,
                self.rms_norm_w_k,
                q_latent,
                &mla.q_a_norm,
                q_latent,
                1,
                d.q_lora,
                d.eps,
                stream,
            )?;
            let q_full = buffers.ssm_deinterleaved();
            if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
                self.nvfp4_decode_gemv(
                    gpu,
                    c.fwd.levers.gemv_sw,
                    q_latent,
                    wqb_nvfp4,
                    q_full,
                    d.q_dim,
                    d.q_lora,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    q_latent,
                    &mla.wq_b,
                    q_full,
                    d.q_dim,
                    d.q_lora,
                    stream,
                )?;
            }
            (q_latent, q_full, None)
        };

        // ── Step 2: Q_absorbed (Q_nope @ W_UK_T) ──
        let q_absorbed_buf = buffers.expert_up_out();
        self.ms_mla_q_absorb(c, mla, &d, q_full, q_absorbed_buf, stream)?;

        // Q_rope scatter (rope half of q_full → strided absorbed layout).
        let q_rope_direct = buffers.ssm_conv_out_f32();
        // GLM-5 full-attention layers use zero-width RoPE. Match the
        // single-sequence MLA path and skip the empty projection entirely:
        // launching any of these kernels with `mla_rope == 0` produces an
        // invalid zero-width CUDA grid during concurrent decode.
        if d.mla_rope == 0 {
            // No RoPE slice to scatter into the absorbed-Q layout.
        } else if self.mla_q_rope_scatter_k.0 != 0 {
            ops::mla_q_rope_scatter(
                gpu,
                self.mla_q_rope_scatter_k,
                q_full,
                q_absorbed_buf,
                q_rope_direct,
                d.nq,
                d.hd,
                d.mla_nope,
                d.mla_rope,
                d.kv_lora,
                d.mla_cache_dim,
                stream,
            )?;
        } else {
            for head_idx in 0..d.nq as usize {
                let src = q_full.offset((head_idx * d.hd as usize + mla.nope) * 2);
                gpu.copy_d2d_async(
                    src,
                    q_rope_direct.offset(head_idx * mla.rope * 2),
                    mla.rope * 2,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    src,
                    q_absorbed_buf
                        .offset((head_idx * d.mla_cache_dim as usize + mla.kv_lora_rank) * 2),
                    mla.rope * 2,
                    stream,
                )?;
            }
        }

        // ── Step 3: KV latent → norm ──
        let kv_latent = buffers.expert_gate_out();
        if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
            self.nvfp4_decode_gemv(
                gpu,
                c.fwd.levers.gemv_sw,
                normed,
                wkva_nvfp4,
                kv_latent,
                d.kv_lora,
                d.h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                normed,
                &mla.wkv_a,
                kv_latent,
                d.kv_lora,
                d.h,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            1,
            d.kv_lora,
            d.eps,
            stream,
        )?;

        // ── Step 4: K_rope + RoPE + writeback ──
        // `k_rope_single` reuses `ssm_ba` — safe: `q_latent` (the prior
        // `ssm_ba` user) was fully consumed by the `wq_b` GEMV above.
        let k_rope_single = buffers.ssm_ba();
        if d.mla_rope > 0 {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                normed,
                &mla.wkv_a_rope,
                k_rope_single,
                d.mla_rope,
                d.h,
                stream,
            )?;
            ops::rope_yarn(
                gpu,
                self.rope_yarn_k,
                q_rope_direct,
                k_rope_single,
                meta.positions,
                1,
                d.nq,
                1,
                d.mla_rope,
                d.mla_rope,
                mla.yarn_inv_freq,
                c.fwd.config.rope_theta as f32,
                stream,
            )?;
            if self.mla_q_rope_writeback_k.0 != 0 {
                ops::mla_q_rope_writeback(
                    gpu,
                    self.mla_q_rope_writeback_k,
                    q_rope_direct,
                    q_absorbed_buf,
                    d.nq,
                    d.mla_rope,
                    d.kv_lora,
                    d.mla_cache_dim,
                    stream,
                )?;
            } else {
                for head_idx in 0..d.nq as usize {
                    let src = q_rope_direct.offset(head_idx * mla.rope * 2);
                    let dst = q_absorbed_buf
                        .offset((head_idx * d.mla_cache_dim as usize + mla.kv_lora_rank) * 2);
                    gpu.copy_d2d_async(src, dst, mla.rope * 2, stream)?;
                }
            }
        }

        // ── Step 5: cache assemble + write (this seq's slot) ──
        // `k_out`/`v_out` use this layer's private QKV scratch region.
        let k_cache_entry = buffers.qkv_output();
        let v_cache_entry = k_cache_entry.offset(d.mla_cache_dim as usize * 2);
        if self.mla_cache_assemble_k.0 != 0 {
            ops::mla_cache_assemble(
                gpu,
                self.mla_cache_assemble_k,
                kv_latent,
                k_rope_single,
                k_cache_entry,
                v_cache_entry,
                d.kv_lora,
                d.mla_rope,
                d.mla_cache_dim,
                stream,
            )?;
        } else {
            gpu.copy_d2d_async(kv_latent, k_cache_entry, mla.kv_lora_rank * 2, stream)?;
            gpu.copy_d2d_async(
                k_rope_single,
                k_cache_entry.offset(mla.kv_lora_rank * 2),
                mla.rope * 2,
                stream,
            )?;
            gpu.copy_d2d_async(kv_latent, v_cache_entry, mla.kv_lora_rank * 2, stream)?;
            gpu.memset_async(
                v_cache_entry.offset(mla.kv_lora_rank * 2),
                0,
                mla.rope * 2,
                stream,
            )?;
        }
        self.write_kv_cache(
            gpu,
            k_cache_entry,
            v_cache_entry,
            kv_cache,
            meta.slot,
            1,
            1,
            d.mla_cache_dim,
            d.bs as u32,
            d.mla_cache_dim,
            d.mla_cache_dim,
            stream,
            c.fwd.graph_capture,
        )?;

        // ── Step 6: paged decode attention (this seq only) ──
        let attn_out = buffers.attn_output();
        if !self.glm_long_verify_attention(
            c,
            kv_cache,
            *meta,
            normed,
            q_latent,
            q_absorbed_buf,
            attn_out,
            position,
            &d,
            stream,
            index_query,
        )? {
            ops::paged_decode_attn_bf16(
                gpu,
                self.paged_decode_mla_k,
                q_absorbed_buf,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                1,
                d.nq,
                1,
                d.mla_cache_dim,
                d.bs as u32,
                d.inv_sqrt_d,
                d.nq * d.mla_cache_dim,
                0,
                stream,
            )?;
        }

        // ── Step 7: V extraction (attn_latent @ W_UV) ──
        // `ssm_qkvz` (not `norm_output`) — `norm_output` holds the `n`
        // per-sequence `normed` inputs that later loop iterations still
        // need; writing `v_extracted` there would clobber them.
        let v_extracted = staged_v.unwrap_or_else(|| buffers.ssm_qkvz());
        self.ms_mla_v_extract(c, mla, &d, attn_out, v_extracted, stream)?;

        // Retained K3 rows live beyond the indexer's keys/gates prefix. The
        // caller runs one O projection only after all causal rows finish.
        if staged_v.is_some() {
            return Ok(());
        }

        // ── Step 8: O projection → this seq's o_out slot ──
        if d.o_lora_rank > 0 {
            // DeepSeek-V4-Flash: low-rank O projection (wo_a → wo_b)
            let o_latent = buffers.attn_output();
            if let Some(ref woa_nvfp4) = mla.wo_a_nvfp4 {
                self.nvfp4_decode_gemv(
                    gpu,
                    c.fwd.levers.gemv_sw,
                    v_extracted,
                    woa_nvfp4,
                    o_latent,
                    d.o_lora_rank,
                    d.nq * d.mla_v_dim,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    v_extracted,
                    &mla.wo_a,
                    o_latent,
                    d.o_lora_rank,
                    d.nq * d.mla_v_dim,
                    stream,
                )?;
            }
            if let Some(ref wob_nvfp4) = mla.wo_b_nvfp4 {
                self.nvfp4_decode_gemv(
                    gpu,
                    c.fwd.levers.gemv_sw,
                    o_latent,
                    wob_nvfp4,
                    o_out,
                    d.h,
                    d.o_lora_rank,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    o_latent,
                    &mla.wo_b,
                    o_out,
                    d.h,
                    d.o_lora_rank,
                    stream,
                )?;
            }
        } else if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
            self.nvfp4_decode_gemv(
                gpu,
                c.fwd.levers.gemv_sw,
                v_extracted,
                wo_nvfp4,
                o_out,
                d.h,
                d.nq * d.mla_v_dim,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                v_extracted,
                &mla.wo,
                o_out,
                d.h,
                d.nq * d.mla_v_dim,
                stream,
            )?;
        }
        Ok(())
    }
}
