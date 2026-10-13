// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 zero-RoPE MLA decode, including selected independent rows 2..8.
//!
//! The generic absorbed-MLA implementation is deliberately per-sequence to
//! accommodate several architectures. GLM-5 has a simpler fixed shape: no
//! RoPE arm, no output LoRA, and BF16 Q/KV/O projections. This first guarded
//! step batches the four stateless projections and, for exact K=5 verification,
//! Q absorption, cache mutation, causal paged attention, and value extraction.
//! Every row retains its own sequence length and block table.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use super::mla_gemv::MlaDims;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::types::MlaWeights;
use crate::layers::qwen3_attention::{
    Qwen3AttentionLayer, glm_mla_multi_seq_enabled as enabled, glm_multi_seq_sparse_enabled,
};
use crate::weight_map::{DenseWeight, QuantizedWeight};

fn batch4_kernel(
    rows: usize,
    requested: bool,
    c4: bool,
    kernel: KernelHandle,
) -> Result<KernelHandle> {
    if !requested {
        return Ok(KernelHandle(0));
    }
    ensure!(
        c4,
        "ATLAS_GLM_MLA_BATCH4=1 requires validated GLM C4 decode"
    );
    if rows != 4 {
        return Ok(KernelHandle(0));
    }
    ensure!(kernel.0 != 0, "GLM MLA batch4 kernel is unavailable");
    Ok(kernel)
}

#[cfg(test)]
mod batch4_tests {
    use super::*;

    #[test]
    fn batch4_is_opt_in_and_never_changes_other_widths() {
        for rows in 1..=5 {
            assert_eq!(
                batch4_kernel(rows, false, false, KernelHandle(0))
                    .unwrap()
                    .0,
                0
            );
            assert_eq!(
                batch4_kernel(rows, true, true, KernelHandle(42)).unwrap().0,
                if rows == 4 { 42 } else { 0 }
            );
        }
    }

    #[test]
    fn batch4_requires_c4_and_a_live_exact_row_kernel() {
        assert!(batch4_kernel(4, true, false, KernelHandle(42)).is_err());
        assert!(batch4_kernel(3, true, false, KernelHandle(42)).is_err());
        assert!(batch4_kernel(4, true, true, KernelHandle(0)).is_err());
    }
}

impl Qwen3AttentionLayer {
    pub(super) fn glm_mla_multi_seq_eligible(
        &self,
        c: &MultiSeqCtx<'_>,
        mla: &MlaWeights,
    ) -> Result<bool> {
        let independent = crate::model::glm_independent::selected(c.fwd, c.n)?;
        Ok(
            (independent || enabled() || glm_multi_seq_sparse_enabled(&c.fwd.config.model_type))
                && (2..=if independent { 8 } else { 5 }).contains(&c.n)
                && mla.rope == 0
                && mla.o_lora_rank == 0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn glm_mla_project(
        &self,
        c: &MultiSeqCtx<'_>,
        input: DevicePtr,
        nvfp4: Option<&QuantizedWeight>,
        dense: &DenseWeight,
        output: DevicePtr,
        n_out: u32,
        k: u32,
    ) -> Result<()> {
        if let Some(weight) = nvfp4 {
            match c.n {
                2 => ops::w4a16_gemv_batch2(
                    c.fwd.gpu,
                    self.w4a16_gemv_batch2_k,
                    input,
                    weight,
                    output,
                    n_out,
                    k,
                    c.stream,
                ),
                3 => ops::w4a16_gemv_batch3(
                    c.fwd.gpu,
                    self.w4a16_gemv_batch3_k,
                    input,
                    weight,
                    output,
                    n_out,
                    k,
                    c.stream,
                ),
                4 => {
                    let kernel = self.w4a16_batchm.kernel(4);
                    ensure!(
                        kernel.0 != 0,
                        "GLM MLA batch4 projection kernel is unavailable"
                    );
                    ops::w4a16_gemv_batchm(
                        c.fwd.gpu, kernel, input, weight, output, 4, n_out, k, c.stream,
                    )
                }
                5 => {
                    let kernel = self.w4a16_batchm.kernel(5);
                    ensure!(
                        kernel.0 != 0,
                        "GLM MLA batch5 projection kernel is unavailable"
                    );
                    ops::w4a16_gemv_batchm(
                        c.fwd.gpu, kernel, input, weight, output, 5, n_out, k, c.stream,
                    )
                }
                n @ 6..=8 if crate::model::glm_independent::selected(c.fwd, n)? => {
                    let kernel = self.w4a16_batchm.kernel(n as u32);
                    ensure!(
                        kernel.0 != 0,
                        "independent MLA projection kernel unavailable"
                    );
                    ops::w4a16_gemv_batchm(
                        c.fwd.gpu, kernel, input, weight, output, n as u32, n_out, k, c.stream,
                    )
                }
                n => anyhow::bail!("GLM MLA multi-sequence projection requires N=2..=5, got {n}"),
            }
        } else {
            let kernel = if c.n == 5 && self.dense_gemv_batch5_k.0 != 0 {
                self.dense_gemv_batch5_k
            } else {
                self.dense_gemv_batchm_k
            };
            ops::dense_gemv_batchm(
                c.fwd.gpu, kernel, input, dense, output, c.n as u32, n_out, k, n_out, c.stream,
            )
        }
    }

    pub(super) fn ms_glm_mla_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        mla: &MlaWeights,
        o_out: DevicePtr,
    ) -> Result<DevicePtr> {
        ensure!(
            self.glm_mla_multi_seq_eligible(c, mla)?,
            "GLM MLA batched path called for an unsupported MLA shape"
        );
        let independent = crate::model::glm_independent::selected(c.fwd, c.n)?;
        let independent_kernel =
            super::mla_independent::select(c.n, independent, self.independent_mla_handles())?;
        let batch4 = batch4_kernel(
            c.n,
            !independent && std::env::var("ATLAS_GLM_MLA_BATCH4").as_deref() == Ok("1"),
            crate::model::glm_c4::enabled(&c.fwd.config.model_type),
            self.mla_batched_gemv_batch4_k,
        )?;
        let sparse = glm_multi_seq_sparse_enabled(&c.fwd.config.model_type);
        if sparse {
            self.validate_glm_multi_seq_sparse(c, kv_cache, meta, mla)?;
        }

        let gpu = c.fwd.gpu;
        let buffers = c.fwd.buffers;
        let stream = c.stream;
        let rows = c.n as u32;
        let h = c.h as u32;
        let nq = c.nq;
        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let nope = mla.nope as u32;
        let v_dim = mla.v_dim as u32;
        let q_dim = nq * c.hd;
        let cache_dim = kv_lora;
        let bf16 = c.bf16;

        // Q down/up projections and latent normalization, all row-major.
        let q_latent = buffers.ssm_ba();
        self.glm_mla_project(
            c,
            c.normed,
            mla.wq_a_nvfp4.as_ref(),
            &mla.wq_a,
            q_latent,
            q_lora,
            h,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            rows,
            q_lora,
            c.eps,
            stream,
        )?;
        let q_full = buffers.ssm_deinterleaved();
        self.glm_mla_project(
            c,
            q_latent,
            mla.wq_b_nvfp4.as_ref(),
            &mla.wq_b,
            q_full,
            q_dim,
            q_lora,
        )?;

        // KV latent projection and norm share the same row batch.
        let kv_latent = buffers.expert_gate_out();
        self.glm_mla_project(
            c,
            c.normed,
            mla.wkv_a_nvfp4.as_ref(),
            &mla.wkv_a,
            kv_latent,
            kv_lora,
            h,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            rows,
            kv_lora,
            c.eps,
            stream,
        )?;

        let dims = MlaDims {
            h,
            nq,
            hd: c.hd,
            q_dim,
            q_lora,
            kv_lora,
            mla_nope: nope,
            mla_v_dim: v_dim,
            mla_rope: 0,
            mla_cache_dim: cache_dim,
            eps: c.eps,
            bs: c.bs as usize,
            inv_sqrt_d: self.effective_attn_scale(c.hd),
            o_lora_rank: 0,
        };

        // Exact-row kernels share Q-absorption and V-extraction weights,
        // assemble and write all cache entries, and submit independently masked
        // attention rows together. C2/C3 remains opt-in for controlled A/B;
        // K=5 retains its established default path.
        let q_absorbed = buffers.expert_up_out();
        let k_entries = buffers.qkv_output();
        let cache_row = cache_dim as usize * bf16;
        let attn_out = buffers.attn_output();
        let v_extracted = buffers.ssm_qkvz();
        let q_full_row = q_dim as usize * bf16;
        let q_absorbed_row = (nq * cache_dim) as usize * bf16;
        let attn_out_row = q_absorbed_row;
        let kv_row = kv_lora as usize * bf16;
        let v_row = (nq * v_dim) as usize * bf16;
        let batch23 = c.fwd.config.model_type == "glm5_next"
            && std::env::var("ATLAS_GLM_MLA_BATCH23").as_deref() == Ok("1");
        let batched_kernel = independent_kernel.unwrap_or(match c.n {
            2 if batch23 => self.mla_batched_gemv_batch2_k,
            3 if batch23 => self.mla_batched_gemv_batch3_k,
            4 => batch4,
            5 => self.mla_batched_gemv_batch5_k,
            _ => KernelHandle(0),
        });
        let batched_mla_gemv = batched_kernel.0 != 0;
        if batched_mla_gemv {
            ops::mla_batched_gemv_batchm(
                gpu,
                batched_kernel,
                q_full,
                mla.w_uk_t.weight,
                q_absorbed,
                kv_lora,
                nope,
                nq,
                c.hd,
                cache_dim,
                q_dim,
                nq * cache_dim,
                stream,
            )?;
        }
        if sparse {
            self.ms_glm_mla_sparse_attention(c, kv_cache, meta, mla, &dims, batched_kernel)?;
        } else if batched_mla_gemv {
            let v_entries = k_entries.offset(c.n * cache_row);
            ops::mla_cache_assemble_batched(
                gpu,
                self.mla_cache_assemble_batched_k,
                kv_latent,
                DevicePtr::NULL,
                k_entries,
                v_entries,
                rows,
                kv_lora,
                0,
                cache_dim,
                stream,
            )?;
            self.write_kv_cache(
                gpu,
                k_entries,
                v_entries,
                kv_cache,
                meta.slot,
                rows,
                1,
                cache_dim,
                c.bs,
                cache_dim,
                cache_dim,
                stream,
                c.fwd.graph_capture,
            )?;
            ops::paged_decode_attn_bf16(
                gpu,
                self.paged_decode_mla_k,
                q_absorbed,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                rows,
                nq,
                1,
                cache_dim,
                c.bs,
                dims.inv_sqrt_d,
                nq * cache_dim,
                0,
                stream,
            )?;
            ops::mla_batched_gemv_batchm(
                gpu,
                batched_kernel,
                attn_out,
                mla.w_uv.weight,
                v_extracted,
                v_dim,
                kv_lora,
                nq,
                cache_dim,
                v_dim,
                nq * cache_dim,
                nq * v_dim,
                stream,
            )?;
        } else {
            let v_entries = k_entries.offset(cache_row);
            for i in 0..c.n {
                let meta_i = AttnMetadataDev {
                    positions: meta.positions.offset(i * 4),
                    positions_h: meta.positions_h.offset(i * 4),
                    positions_w: meta.positions_w.offset(i * 4),
                    slot: meta.slot.offset(i * 8),
                    seq_len: meta.seq_len.offset(i * 4),
                    block_table: meta
                        .block_table
                        .offset(i * meta.max_blocks_per_seq as usize * 4),
                    max_blocks_per_seq: meta.max_blocks_per_seq,
                    num_seqs: 1,
                    seq_slot: DevicePtr::NULL,
                    moe_row_adapter: DevicePtr::NULL,
                };

                let q_absorbed_i = q_absorbed.offset(i * q_absorbed_row);
                self.ms_mla_q_absorb(
                    c,
                    mla,
                    &dims,
                    q_full.offset(i * q_full_row),
                    q_absorbed_i,
                    stream,
                )?;
                let attn_out_i = attn_out.offset(i * attn_out_row);
                ops::mla_cache_assemble(
                    gpu,
                    self.mla_cache_assemble_k,
                    kv_latent.offset(i * kv_row),
                    DevicePtr::NULL,
                    k_entries,
                    v_entries,
                    kv_lora,
                    0,
                    cache_dim,
                    stream,
                )?;
                self.write_kv_cache(
                    gpu,
                    k_entries,
                    v_entries,
                    kv_cache,
                    meta_i.slot,
                    1,
                    1,
                    cache_dim,
                    c.bs,
                    cache_dim,
                    cache_dim,
                    stream,
                    c.fwd.graph_capture,
                )?;
                ops::paged_decode_attn_bf16(
                    gpu,
                    self.paged_decode_mla_k,
                    q_absorbed_i,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out_i,
                    meta_i.block_table,
                    meta_i.seq_len,
                    meta_i.max_blocks_per_seq,
                    1,
                    nq,
                    1,
                    cache_dim,
                    c.bs,
                    dims.inv_sqrt_d,
                    nq * cache_dim,
                    0,
                    stream,
                )?;
                self.ms_mla_v_extract(
                    c,
                    mla,
                    &dims,
                    attn_out_i,
                    v_extracted.offset(i * v_row),
                    stream,
                )?;
            }
        }

        // Read the large O matrix once after all rows have been extracted.
        self.glm_mla_project(
            c,
            v_extracted,
            mla.wo_nvfp4.as_ref(),
            &mla.wo,
            o_out,
            h,
            nq * v_dim,
        )?;
        Ok(o_out)
    }
}

#[cfg(test)]
mod tests {
    use super::enabled;

    #[test]
    fn glm_mla_multiseq_is_gated() {
        if std::env::var_os("ATLAS_GLM_MLA_MULTI_SEQ").is_none() {
            assert!(!enabled());
        }
    }
}
