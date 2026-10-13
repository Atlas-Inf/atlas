// SPDX-License-Identifier: AGPL-3.0-only

//! MLA branch of `prefill_attention_with_cache_skip`. Mistral4-style
//! 2-step prefill with the unabsorbed/MHA fused fallback path that
//! expands K/V via `wkv_b` and runs HDIM=128 FlashAttention. Extracted
//! from `cache_skip.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

fn use_cublas_mla_prefill(cublas_enabled: bool, num_tokens: u32) -> bool {
    cublas_enabled && num_tokens > 1
}

#[allow(clippy::too_many_arguments)]
pub(super) struct CacheSkipMlaArgs {
    pub normed: DevicePtr,
    pub num_tokens: usize,
    pub n: u32,
    pub h: u32,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub kv_dim: usize,
    pub eps: f32,
    pub bf16: usize,
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_prefill_dense(
        &self,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let tc = crate::layers::w4a16_gemv_tiers::tc_kernel(m);
        if m <= 32
            && tc.0 != 0
            && let Some((_, q4)) = self.mla_q4.iter().find(|(w, _)| *w == weight.weight)
        {
            return ops::w4a16_gemv_batchm(ctx.gpu, tc, input, q4, output, m, n, k, stream);
        }
        if m <= 32
            && let Some((.., mx)) = self.mla_mx.iter().find(|(w, ..)| *w == weight.weight)
        {
            // ATLAS_GLM_DECODE_GEMV_BATCH: q_a and kv_a wait on the HC / norm
            // chain with the bus idle; their twin touches the weight meanwhile.
            // Not at 17..=32 rows, where the 32-row tier lost time to it
            // (scripts/dev/glm_decode_gemv_touch_bench.cu).
            if m <= 16
                && self
                    .mla
                    .as_ref()
                    .is_some_and(|mla| [mla.wq_a.weight, mla.wkv_a.weight].contains(&weight.weight))
                && let Some(touch) = ops::gemv_touch(ops::mxfp8_tc_twin(m))
            {
                return touch.mxfp8_tc(
                    ctx.gpu, input, mx.data, mx.scales, output, m, n, k, n, stream,
                );
            }
            let kernel = self.mxfp8_gemv_k[match m {
                0..=8 => 0,
                9..=16 => 1,
                _ => 2,
            }];
            return ops::mxfp8_gemv(
                ctx.gpu, kernel, input, mx.data, mx.scales, output, m, n, k, n, stream,
            );
        }
        if ctx.config.model_type == "glm5_next"
            && m <= crate::layers::canonical_verify::MAX_ROWS
            && crate::layers::canonical_verify::enabled()
        {
            // The indexer projections: one tensor-core family for every
            // verify width (cuBLASLt picks its algorithm per row count, and
            // one row took `dense_gemm_tc`).
            return crate::layers::canonical_verify::dense(
                ctx.gpu, input, weight, output, m, n, k, n, stream,
            );
        }
        if use_cublas_mla_prefill(ctx.dispatch.cublas_gemm, m) {
            return ops::cublas_bf16_proj_dense(input, weight.weight, output, m, n, k, stream);
        }
        if self.dense_gemm_tc_k.0 != 0 {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// Run the cache-skip MLA prefill chain. Always returns the output
    /// pointer — caller short-circuits with `return Ok(out)`.
    pub(super) fn prefill_attention_cache_skip_mla(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &CacheSkipMlaArgs,
    ) -> Result<DevicePtr> {
        let CacheSkipMlaArgs {
            normed,
            num_tokens,
            n,
            h,
            nq,
            nkv,
            hd,
            kv_dim,
            eps,
            bf16,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("prefill_attention_cache_skip_mla called without MLA config");

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        macro_rules! mprof {
            ($label:expr, $started:expr) => {
                if let Some(started) = $started {
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!(
                        "  MLA prefill [{}] N={}: {}µs",
                        $label,
                        n,
                        started.elapsed().as_micros()
                    );
                }
            };
        }
        let mut started = ctx.profile.then(std::time::Instant::now);

        // Q: latent → norm → expand
        let q_latent = ctx.buffers.ssm_ba();
        self.mla_prefill_dense(normed, &mla.wq_a, q_latent, n, q_lora, h, ctx, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            n,
            q_lora,
            eps,
            stream,
        )?;
        if mla.glm_indexer.is_some() {
            self.glm_index_prefill_cache_update(normed, n, 0, kv_cache, ctx, stream, None)?;
        }
        let qg_out = ctx.buffers.qkv_output();
        self.mla_prefill_dense(q_latent, &mla.wq_b, qg_out, n, nq * hd, q_lora, ctx, stream)?;
        mprof!("q_latent_expand", started);
        started = ctx.profile.then(std::time::Instant::now);

        // KV latent + K_rope
        let kv_latent = ctx.buffers.expert_gate_out();
        self.mla_prefill_dense(normed, &mla.wkv_a, kv_latent, n, kv_lora, h, ctx, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            n,
            kv_lora,
            eps,
            stream,
        )?;
        let k_rope_buf = ctx.buffers.ssm_ba();
        if mla_rope > 0 {
            self.mla_prefill_dense(
                normed,
                &mla.wkv_a_rope,
                k_rope_buf,
                n,
                mla_rope,
                h,
                ctx,
                stream,
            )?;
        }
        mprof!("kv_latent", started);
        started = ctx.profile.then(std::time::Instant::now);

        // Q rope extract → RoPE. GLM-5 has a zero-width RoPE partition.
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        if mla_rope > 0 {
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                qg_out,
                q_rope_tmp,
                n,
                nq,
                hd,
                mla_nope,
                mla_rope,
                nq * hd,
                stream,
            )?;
            let rope_meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_k,
                q_rope_tmp,
                k_rope_buf,
                rope_meta.positions,
                n,
                nq,
                1,
                mla_rope,
                mla_rope,
                mla.yarn_inv_freq,
                ctx.config.rope_theta as f32,
                stream,
            )?;
        }

        let mla_cache_dim = kv_lora + mla_rope;
        // Cache assembly (needed for decode regardless of path)
        let meta = ctx.attn_metadata.expect("MLA prefill requires metadata");
        let bs = kv_cache.block_size();
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            k_rope_buf,
            k_cache_assembled,
            v_cache_assembled,
            n,
            kv_lora,
            mla_rope,
            mla_cache_dim,
            stream,
        )?;
        self.write_kv_cache(
            ctx.gpu,
            k_cache_assembled,
            v_cache_assembled,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            bs as u32,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )?;
        mprof!("cache_write", started);
        started = ctx.profile.then(std::time::Instant::now);

        // Unabsorbed (MHA) prefill: expand K/V via wkv_b, use HDIM=128 FlashAttention
        let kv_expanded_dim = nkv * (mla_nope + mla_v_dim);
        let kv_expanded = ctx.buffers.ssm_deinterleaved();
        self.mla_prefill_dense(
            kv_latent,
            &mla.wkv_b,
            kv_expanded,
            n,
            kv_expanded_dim,
            kv_lora,
            ctx,
            stream,
        )?;
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        ops::mla_kv_assemble_batched(
            ctx.gpu,
            self.mla_kv_assemble_batched_k,
            kv_expanded,
            k_rope_buf,
            k_contiguous,
            v_contiguous,
            n,
            nkv,
            mla_nope,
            mla_v_dim,
            mla_rope,
            hd,
            nkv * (mla_nope + mla_v_dim),
            stream,
        )?;
        if mla_rope > 0 {
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                q_rope_tmp,
                qg_out,
                n,
                nq,
                hd,
                mla_nope,
                mla_rope,
                nq * hd,
                stream,
            )?;
        }
        mprof!("kv_expand", started);
        started = ctx.profile.then(std::time::Instant::now);
        // ATLAS_OP_DUMP hooks: the assembled V and the attention output, the
        // two tensors that decide an L=1 MLA result (softmax over one key is
        // 1.0, so the output reduces to V0 @ o_proj).
        if n > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                v_contiguous,
                (num_tokens - 1) * nkv as usize * hd as usize * 2,
                nkv as usize * hd as usize,
                self.attn_layer_idx,
                "mla_v",
                stream,
            )?;
        }
        let attn_out_fb = ctx.buffers.attn_output();
        ops::prefill_attention_64(
            ctx.gpu,
            self.prefill_attn_64_k,
            qg_out,
            k_contiguous,
            v_contiguous,
            attn_out_fb,
            n,
            1,
            nq,
            nkv,
            hd,
            self.effective_attn_scale(hd),
            true,
            0,
            stream,
        )
        .map_err(|e| anyhow::anyhow!("MLA flash_attn_64 fallback: {e}"))?;
        mprof!("flash_attention", started);
        started = ctx.profile.then(std::time::Instant::now);
        if n > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out_fb,
                (num_tokens - 1) * nq as usize * hd as usize * 2,
                nq as usize * hd as usize,
                self.attn_layer_idx,
                "mla_attn_out",
                stream,
            )?;
        }
        // wo projection — output to qkv_output (norm_output aliases downstream)
        let o_out = ctx.buffers.qkv_output();
        if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                attn_out_fb,
                wo_nvfp4,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else {
            self.mla_prefill_dense(attn_out_fb, &mla.wo, o_out, n, h, nq * hd, ctx, stream)?;
        }
        mprof!("o_proj", started);
        Ok(o_out)
    }
}

#[cfg(test)]
mod tests {
    use super::use_cublas_mla_prefill;

    #[test]
    fn cublas_mla_prefill_requires_batch_work_and_dispatch_support() {
        assert!(use_cublas_mla_prefill(true, 2));
        assert!(use_cublas_mla_prefill(true, 1000));
        assert!(!use_cublas_mla_prefill(true, 1));
        assert!(!use_cublas_mla_prefill(false, 1000));
    }
}
