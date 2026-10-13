// SPDX-License-Identifier: AGPL-3.0-only

//! Exact dense stepping stone for chunked GLM-5 prefill.
//!
//! GLM-5 selects at most `index_topk` raw history tokens. At or below that
//! length the selection is the complete causal history, so dense attention is
//! exact. Unlike the old generic MLA branch, this path reads every preceding
//! chunk from the compressed paged cache. Longer sequences deliberately fail
//! closed until the native k-pool selector supplies sparse token indices.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use super::paged_mla::MlaPrefillArgs;
use crate::layer::ForwardContext;
use crate::layers::ops;

#[path = "paged_glm_projection.rs"]
mod projection;

#[path = "paged_glm_output.rs"]
mod output;
#[path = "paged_glm_owner.rs"]
mod owner;
#[path = "paged_glm_shard.rs"]
pub(in crate::layers::qwen3_attention) mod shard;
pub(crate) use owner::write_floor_legacy;
pub(in crate::layers::qwen3_attention) use owner::{GlmChunkOwner, glm_chunk_pieces};

fn dense_selection_is_exact(sequence_end: usize, index_topk: usize) -> bool {
    index_topk > 0 && sequence_end <= index_topk
}

impl Qwen3AttentionLayer {
    /// Chunked zero-RoPE MLA using the 512-wide absorbed cache.
    pub(super) fn prefill_attention_paged_glm_dense(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
        seq_len_start: usize,
        kv_write_floor: usize,
    ) -> Result<DevicePtr> {
        let meta = ctx
            .attn_metadata
            .expect("GLM paged prefill requires metadata");
        let owners = glm_chunk_pieces(meta, seq_len_start, args.num_tokens, ctx.config.index_topk);
        let floor = owner::honoured_write_floor(kv_write_floor);
        self.glm_chunk_attention(&owners, kv_cache, ctx, args, floor)
    }

    /// GLM MLA over the causal chunks of one or more sequences whose rows are
    /// stacked in `args.normed` (`ctx.attn_metadata` covers every row).
    /// Row-wise projections (q_a, kv_a, W_uv, o) and the KV cache write run
    /// once over all rows; the semantic index, q_b + W_uk absorb and sparse
    /// attention run per owner in the pinned native-sparse operand buffers.
    /// With several owners each owner's attention rows are parked in the
    /// (idle until the LM head) logits arena until the joint W_uv.
    ///
    /// The first `kv_write_floor` stacked rows replay positions whose latents
    /// and index keys already sit in shared prefix-cache blocks (Marconi warm
    /// hit, see `prefill_attention_paged` section 7). They are projected and
    /// attended like any row but written nowhere: every reader below takes
    /// those positions from the cache, as the sequences sharing them do.
    pub(super) fn glm_chunk_attention(
        &self,
        owners: &[GlmChunkOwner],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
        kv_write_floor: usize,
    ) -> Result<DevicePtr> {
        let MlaPrefillArgs {
            normed,
            num_tokens,
            n,
            h,
            nq,
            nkv: _,
            hd,
            kv_dim: _,
            eps,
            bf16,
            bs,
            stream,
            seq_len_start: _,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("GLM paged prefill called without MLA weights");
        ensure!(
            mla.glm_indexer.is_some() && mla.rope == 0,
            "GLM paged prefill requires the zero-RoPE semantic-index MLA shape"
        );
        ensure!(
            matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128),
            "GLM chunked prefill requires a BF16 or fp8_g128 KV cache; got {:?}",
            self.kv_dtype
        );
        ensure!(
            !owners.is_empty()
                && owners.iter().map(|o| o.rows).sum::<usize>() == num_tokens
                && owners
                    .windows(2)
                    .all(|w| w[0].row0 + w[0].rows == w[1].row0),
            "GLM chunk owners must tile the stacked rows"
        );
        let accelerated = projection::enabled(&ctx.config.model_type)?;
        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let nope = mla.nope as u32;
        let v_dim = mla.v_dim as u32;
        ensure!(
            kv_lora == 512 && nope == hd && v_dim == hd,
            "unsupported GLM MLA geometry: kv_lora={kv_lora}, nope={nope}, v={v_dim}, hd={hd}"
        );

        // Joint: q_a (+ norm), kv_a (+ norm) and the cache write.
        let q_latent = ctx.buffers.ssm_ba();
        self.paged_glm_projection(
            normed,
            &mla.wq_a,
            q_latent,
            n,
            q_lora,
            h,
            ctx,
            stream,
            accelerated,
        )?;
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
        let kv_latent = ctx.buffers.expert_gate_out();
        self.paged_glm_projection(
            normed,
            &mla.wkv_a,
            kv_latent,
            n,
            kv_lora,
            h,
            ctx,
            stream,
            accelerated,
        )?;
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
        let k_entries = ctx.buffers.ssm_qkvz();
        let v_entries = k_entries.offset(num_tokens * kv_lora as usize * bf16);
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            DevicePtr::NULL,
            k_entries,
            v_entries,
            n,
            kv_lora,
            0,
            kv_lora,
            stream,
        )?;
        let joint = ctx
            .attn_metadata
            .expect("GLM paged prefill requires metadata");
        let wf =
            owner::checked_write_floor(owners, kv_write_floor, num_tokens, ctx.config.index_kpool)?;
        if wf < num_tokens {
            let skip = wf * kv_lora as usize * bf16;
            self.write_kv_cache(
                ctx.gpu,
                k_entries.offset(skip),
                v_entries.offset(skip),
                kv_cache,
                // slot_mapping entries are int64 (8 bytes each)
                joint.slot.offset(wf * 8),
                n - wf as u32,
                1,
                kv_lora,
                bs,
                kv_lora,
                kv_lora,
                stream,
                ctx.graph_capture,
            )?;
        }
        // ATLAS_GLM_DET_TRACE: the latents written to the cache; below, each
        // piece's selected token ids (`x_` stages: see `det_trace`).
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        let lat = |width: u32| width as usize * bf16;
        det.tap("kv_lat", k_entries, (0, num_tokens), lat(kv_lora));
        det.tap("x_q_lat", q_latent, (0, num_tokens), lat(q_lora));

        // Per owner: semantic index, q_b + absorb, attention.
        let attn_latent = ctx.buffers.attn_output();
        let latent_row = nq as usize * kv_lora as usize * bf16;
        // Owners share the attention scratch, so each one's latent result is
        // parked for one joint W_uv + o_proj. When the chunk outgrows the
        // park (8K sub-chunked prefill) each owner projects its own rows.
        let o_out = ctx.buffers.norm_output();
        let per_owner = owners.len() > 1 && ctx.buffers.sizes().logits < num_tokens * latent_row;
        let parked = (owners.len() > 1 && !per_owner).then(|| ctx.buffers.logits());
        // Owner-batched verify: every row-wise projection (index keys, gates,
        // queries and weights, q_b and the W_uk absorb) runs once over all
        // owners' rows, so each weight is read once per layer rather than once
        // per owner; only cache writes, selection and attention stay per owner.
        let batched = self.glm_owner_projections(
            owners,
            normed,
            q_latent,
            num_tokens,
            nq as usize,
            self.glm_shard_lane(kv_cache, ctx).is_some(),
            ctx,
            stream,
        )?;
        for o in owners {
            let on = o.rows as u32;
            let octx = ForwardContext {
                attn_metadata: Some(o.meta),
                midchunk_capture: None,
                ..*ctx
            };
            let sequence_end = o
                .seq_len_start
                .checked_add(o.rows)
                .ok_or_else(|| anyhow::anyhow!("GLM prefill sequence length overflow"))?;
            let use_dense = o.dense_is_exact(ctx.config.index_topk);
            let o_normed = normed.offset(o.row0 * h as usize * bf16);
            let o_latent = q_latent.offset(o.row0 * q_lora as usize * bf16);
            let rows_of = |base: DevicePtr, row_bytes: usize| base.offset(o.row0 * row_bytes);
            let select = |octx: &ForwardContext| {
                self.glm_index_prefill_cache_update(
                    o_normed,
                    on,
                    o.write_skip(wf),
                    kv_cache,
                    octx,
                    stream,
                    batched.map(|b| (rows_of(b.keys, b.key_row), rows_of(b.gates, b.key_row))),
                )?;
                if use_dense {
                    return Ok(None);
                }
                let projected = batched.map(|b| b.index_rows(o.row0)).transpose()?;
                let start = o.seq_len_start;
                self.glm_index_prefill_select(
                    o_latent, o_normed, on, start, kv_cache, octx, stream, projected,
                )
                .map(Some)
            };
            // Queries projected ahead of the selection (owner-batched) can be
            // swapped with the peer of a sharded cache while it runs.
            let ready = batched.map(|b| rows_of(b.q_absorbed, latent_row));
            let (sparse_indices, queries_swapped) =
                self.glm_shard_swap_queries_during(kv_cache, &octx, o.rows, ready, stream, select)?;
            // The BF16 dense and native kernels read an fp8_g128 owner through
            // a dequantized view; long owners and verify rows read FP8 directly,
            // as do, under the pipelined kernel, sparse pieces native declines.
            let sparse_view = ops::glm_sparse_owner_needs_view(&ctx.config.model_type, on, || {
                ops::glm_sparse_native_admits(&octx, on, o.seq_len_start, false, stream)
            })?;
            // Sharded latents (`ATLAS_GLM_KV_SHARD=1`): few rows merge each rank's
            // attention over its own tokens; larger owners read an assembled view.
            let sharded = kv_cache.latent_shard().is_some();
            let latents = self.glm_owner_latents(kv_cache, &octx, o, sequence_end, stream)?;
            let merge_form = latents.is_none();
            let [k_source, v_source, table_source] = latents.unwrap_or([DevicePtr::NULL; 3]);
            let view = self.glm_owner_bf16_view(
                k_source,
                table_source,
                sequence_end,
                !merge_form && (use_dense || sparse_view),
                bs,
                ctx,
                stream,
            )?;
            ensure!(
                merge_form || !use_dense || self.kv_dtype == KvCacheDtype::Bf16 || view.is_some(),
                "GLM dense prefill has no BF16 view of the fp8_g128 cache"
            );
            let (k_cache, v_cache, block_table, cache_dtype) = match view {
                Some(v) => (v.latents, v.latents, v.identity_table, KvCacheDtype::Bf16),
                None => (k_source, v_source, table_source, self.kv_dtype),
            };
            let q_absorbed = match batched {
                Some(b) => rows_of(b.q_absorbed, latent_row),
                None => {
                    let q_full = ctx.buffers.qkv_output();
                    self.paged_glm_projection(
                        o_latent,
                        &mla.wq_b,
                        q_full,
                        on,
                        nq * hd,
                        q_lora,
                        &octx,
                        stream,
                        accelerated,
                    )?;
                    let q_absorbed = ctx.buffers.ssm_deinterleaved();
                    self.glm_absorb_queries(q_full, q_absorbed, on, nq, ctx, stream)?;
                    q_absorbed
                }
            };
            det.tap("x_q_abs", q_absorbed, (o.row0, o.rows), latent_row);
            if merge_form {
                self.glm_shard_merge_owner(
                    kv_cache,
                    &octx,
                    o,
                    (q_absorbed, queries_swapped),
                    sparse_indices,
                    attn_latent,
                    stream,
                )?;
            } else if let Some((indices, index_width)) = sparse_indices {
                det.tap("sel", indices, (o.row0, o.rows), index_width as usize * 4);
                let mut profile = super::glm_index::profile_start(&octx, stream)?;
                let sparse_args = ops::GlmSparsePrefillTc {
                    config: ctx.config,
                    dtype: cache_dtype,
                    // mla_cache_assemble_batched above writes the same normalized
                    // NoPE latent to both conventional paged cache sides.
                    identical_kv_latent: true,
                    query: q_absorbed,
                    k_cache,
                    v_cache,
                    indices,
                    output: attn_latent,
                    block_table,
                    rows: on,
                    heads: nq,
                    head_dim: kv_lora,
                    index_width,
                    block_size: bs,
                    scale: self.effective_attn_scale(hd),
                };
                let (physical_blocks, table_blocks, block_bytes) = match view {
                    Some(v) => (v.blocks, v.blocks, 16 * 512 * 2),
                    None => (
                        kv_cache.num_blocks(),
                        o.meta.max_blocks_per_seq as usize,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx),
                    ),
                };
                // SparkGLM-only: the out-of-tree NVIDIA sparse-MLA prefill,
                // when loaded (`ATLAS_GLM_SPARSE_NATIVE=1`). Sharded latents
                // without a BF16 view have no geometry the bridge understands.
                let native = cache_dtype == KvCacheDtype::Bf16
                    && (view.is_some() || !sharded)
                    && ops::try_glm_sparse_native(
                        &octx,
                        &sparse_args,
                        o.seq_len_start,
                        physical_blocks,
                        table_blocks,
                        block_bytes,
                        // This caller is ordinary continued prefill. Repaired K3
                        // verification has a separate multi-sequence attention path.
                        false,
                        stream,
                    )?;
                // Few-row owners (verify) split over the selected IDs; the MoE
                // expert scratch is dead until this layer's FFN.
                let accelerated = native
                    || ops::try_glm_sparse_prefill_tc_split(
                        ctx.gpu,
                        &sparse_args,
                        ctx.buffers.expert_gate_out(),
                        ctx.buffers.sizes().expert_gate_out,
                        stream,
                    )?;
                if !accelerated {
                    ensure!(
                        cache_dtype == KvCacheDtype::Bf16,
                        "fp8_g128 GLM sparse attention requires ATLAS_GLM_SPARSE_PREFILL_TC=1"
                    );
                    ops::glm_sparse_mla_prefill(
                        ctx.gpu,
                        self.glm_sparse_attn_k,
                        q_absorbed,
                        k_cache,
                        v_cache,
                        indices,
                        attn_latent,
                        block_table,
                        on,
                        nq,
                        kv_lora,
                        index_width,
                        bs,
                        self.glm_sparse_attn_heads_per_cta,
                        self.effective_attn_scale(hd),
                        stream,
                    )?;
                }
                let sparse_attention_us =
                    super::glm_index::profile_lap(&octx, stream, &mut profile)?;
                if profile.is_some() {
                    tracing::info!(
                        "ATLAS_GLM_INDEX_PROFILE phase=attention layer={} rows={} seq_end={} selected={} sparse_attention_us={}",
                        self.attn_layer_idx,
                        on,
                        sequence_end,
                        index_width,
                        sparse_attention_us,
                    );
                }
            } else {
                ensure!(
                    self.prefill_attn_paged_512_k.0 != 0,
                    "GLM paged prefill kernel inferspark_prefill_paged_512 is unavailable"
                );
                ops::prefill_attention_paged_512(
                    ctx.gpu,
                    self.prefill_attn_paged_512_k,
                    q_absorbed,
                    k_cache,
                    v_cache,
                    attn_latent,
                    block_table,
                    on,
                    sequence_end as u32,
                    o.seq_len_start as u32,
                    nq,
                    1,
                    kv_lora,
                    bs,
                    0,
                    self.effective_attn_scale(hd),
                    stream,
                )?;
            }

            det.tap("x_attn_lat", attn_latent, (o.row0, o.rows), latent_row);
            // Convert the latent attention result back to each head's value
            // width, then apply the row-parallel output projection.
            if per_owner {
                self.paged_glm_output(
                    mla,
                    attn_latent,
                    o_out.offset(o.row0 * h as usize * bf16),
                    [o.rows as u32, nq, h],
                    &octx,
                    stream,
                    accelerated,
                )?;
            } else if let Some(park) = parked {
                ctx.gpu.copy_d2d_async(
                    attn_latent,
                    park.offset(o.row0 * latent_row),
                    o.rows * latent_row,
                    stream,
                )?;
            }
        }

        if !per_owner {
            self.paged_glm_output(
                mla,
                parked.unwrap_or(attn_latent),
                o_out,
                [n, nq, h],
                ctx,
                stream,
                accelerated,
            )?;
        }
        Ok(o_out)
    }
}

#[cfg(test)]
#[path = "paged_glm_tests.rs"]
mod tests;
