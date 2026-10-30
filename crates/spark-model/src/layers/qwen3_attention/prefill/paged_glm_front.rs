// SPDX-License-Identifier: AGPL-3.0-only

//! The joint front of GLM chunk attention, run once over every stacked row:
//! q_a and kv_a with their norms and the latent cache write. Under
//! `ATLAS_GLM_MLA_FUSE=1` ([`ops::MlaFuse`]) a canonical verify block also
//! gets its semantic-index cache update and, for a sharded causal owner, the
//! merge form's localization here, all in two launches with the same bytes.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache, SparseIndexCacheDtype};

use super::super::super::Qwen3AttentionLayer;
use super::MlaPrefillArgs;
use super::{GlmChunkOwner, owner};
use crate::layer::ForwardContext;
use crate::layers::glm_kv_shard::{MERGE_MAX_ROWS, WIDTH};
use crate::layers::ops;
use crate::layers::qwen3_attention::types::MlaWeights;
use crate::weight_map::DenseWeight;

/// What the front leaves for the owners.
#[derive(Clone, Copy)]
pub(super) struct GlmFront {
    /// The normalized q_a latents `[rows, q_lora]`.
    pub q_latent: DevicePtr,
    /// Stacked rows below the KV write floor.
    pub write_floor: usize,
    /// Every row's semantic-index cache update is done.
    pub indexed: bool,
    /// The single owner's causal IDs are localized in the shard scratch.
    pub localized: bool,
}

impl Qwen3AttentionLayer {
    /// q_a (+ norm), kv_a (+ norm) and the cache write of every stacked row.
    pub(super) fn glm_chunk_front(
        &self,
        owners: &[GlmChunkOwner],
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
        kv_write_floor: usize,
        mla: &MlaWeights,
    ) -> Result<GlmFront> {
        let MlaPrefillArgs {
            normed,
            num_tokens,
            n,
            h,
            eps,
            bf16,
            bs,
            stream,
            ..
        } = *args;
        let (q_lora, kv_lora) = (mla.q_lora_rank as u32, mla.kv_lora_rank as u32);
        let q_latent = ctx.buffers.ssm_ba();
        let kv_latent = ctx.buffers.expert_gate_out();
        let wf =
            owner::checked_write_floor(owners, kv_write_floor, num_tokens, ctx.config.index_kpool)?;
        // ATLAS_GLM_DET_TRACE: the latents written to the cache.
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        let lat = |width: u32| width as usize * bf16;
        if let Some(localized) = self.glm_front_fused(owners, kv_cache, ctx, args, mla, wf)? {
            det.tap("kv_lat", kv_latent, (0, num_tokens), lat(kv_lora));
            det.tap("x_q_lat", q_latent, (0, num_tokens), lat(q_lora));
            return Ok(GlmFront {
                q_latent,
                write_floor: wf,
                indexed: true,
                localized,
            });
        }
        let accelerated = super::projection::enabled(&ctx.config.model_type)?;
        let project = |weight: &DenseWeight, out: DevicePtr, width: u32| {
            self.paged_glm_projection(normed, weight, out, n, width, h, ctx, stream, accelerated)
        };
        // ATLAS_GLM_FP4L_MLA_FP8: q_a and kv_a in one launch (`mla_native`).
        let paired = self.mla_native_pair(
            ctx.gpu,
            [
                (&mla.wq_a, q_latent, q_lora),
                (&mla.wkv_a, kv_latent, kv_lora),
            ],
            normed,
            n,
            h,
            stream,
        )?;
        if !paired {
            project(&mla.wq_a, q_latent, q_lora)?;
        }
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
        if !paired {
            project(&mla.wkv_a, kv_latent, kv_lora)?;
        }
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
        det.tap("kv_lat", k_entries, (0, num_tokens), lat(kv_lora));
        det.tap("x_q_lat", q_latent, (0, num_tokens), lat(q_lora));
        Ok(GlmFront {
            q_latent,
            write_floor: wf,
            indexed: false,
            localized: false,
        })
    }

    /// The fused front ([`ops::MlaFuse`]) where it applies: a canonical
    /// verify block of at most `MLA_FUSE_MAX_ROWS` rows with nothing below
    /// the write floor, the MXFP8 q_a / kv_a and BF16 index projections the
    /// canonical path would run, an fp8_g128 latent cache and the BF16
    /// four-token semantic index. Returns whether it localized the owner.
    fn glm_front_fused(
        &self,
        owners: &[GlmChunkOwner],
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        args: &MlaPrefillArgs,
        mla: &MlaWeights,
        wf: usize,
    ) -> Result<Option<bool>> {
        let Some(fuse) = ops::MlaFuse::get(ctx.gpu)? else {
            return Ok(None);
        };
        let (Some(indexer), Some(meta)) = (mla.glm_indexer.as_ref(), ctx.attn_metadata) else {
            return Ok(None);
        };
        let mx = |w: &DenseWeight| {
            let quad = self.mla_q4.iter().any(|(p, _)| *p == w.weight);
            let mx = self.mla_mx.iter().find(|(p, ..)| *p == w.weight);
            (!quad).then_some(mx.map(|(.., m)| *m))
        };
        let index_dim = ctx.config.index_head_dim as u32;
        let index_ok = kv_cache.sparse_index_config().is_some_and(|s| {
            s.dtype == SparseIndexCacheDtype::Bf16
                && s.tokens_per_pool == 4
                && s.tokens_per_pool == ctx.config.index_kpool
                && s.head_dim == ctx.config.index_head_dim
                && index_dim == 128
        });
        let (Some(Some(q_a)), Some(Some(kv_a)), Some(None), Some(None)) = (
            mx(&mla.wq_a),
            mx(&mla.wkv_a),
            mx(&indexer.wk),
            mx(&indexer.kpool_gate),
        ) else {
            return Ok(None);
        };
        let n = args.n;
        if !(ctx.config.model_type == "glm5_next"
            && crate::layers::canonical_verify::enabled()
            && (1..=ops::MLA_FUSE_MAX_ROWS).contains(&n)
            && wf == 0
            && index_ok
            && self.kv_dtype == KvCacheDtype::Fp8G128
            && mla.rope == 0
            && mla.kv_lora_rank == 512
            && args.h.is_multiple_of(64))
        {
            return Ok(None);
        }
        let layer = self.attn_layer_idx;
        let (q_latent, kv_latent) = (ctx.buffers.ssm_ba(), ctx.buffers.expert_gate_out());
        let keys = ctx.buffers.ssm_qkvz();
        let gates = keys.offset(n as usize * index_dim as usize * 2);
        let proj = |w: crate::weight_map::Mxfp8Weight, out, rows: usize| {
            debug_assert!(!w.tiled, "the fused front reads row-major q_a / kv_a twins");
            ops::MxProj {
                data: w.data,
                scales: w.scales,
                out,
                n: rows as u32,
            }
        };
        // The q_a / kv_a touch twins' geometry, where they ran (<= 16 rows).
        let touch = (n <= 16)
            .then(|| ops::gemv_touch(ops::mxfp8_tc_twin(n)))
            .flatten()
            .map(|t| (t.bytes, t.ctas));
        let down = ops::DownArgs {
            input: args.normed,
            rows: n,
            k: args.h,
            wk: indexer.wk.weight,
            gate: indexer.kpool_gate.weight,
            keys,
            gates,
            ni: index_dim,
            q_a: proj(q_a, q_latent, mla.q_lora_rank),
            kv_a: proj(kv_a, kv_latent, mla.kv_lora_rank),
            touch,
        };
        let shard = kv_cache.latent_shard();
        let (latent_cache, owner) = match shard {
            Some(s) => (
                kv_cache.latent_pool_ptr(layer),
                ops::ShardRank {
                    rank: s.spec.rank as u32,
                    world: s.spec.world as u32,
                },
            ),
            None => (
                kv_cache.k_pool_ptr(layer),
                ops::ShardRank { rank: 0, world: 1 },
            ),
        };
        // A lone sharded causal owner (the merge form) localizes here.
        let localize = match owners {
            [o] if shard.is_some()
                && o.rows <= MERGE_MAX_ROWS
                && o.dense_is_exact(ctx.config.index_topk) =>
            {
                let (ids, counts) = self.glm_shard_localize_target(kv_cache, o.rows as u32)?;
                Some((
                    o.meta.block_table,
                    ids,
                    counts,
                    WIDTH,
                    o.seq_len_start as u32,
                ))
            }
            _ => None,
        };
        let mid = ops::MidArgs {
            rows: n,
            q_latent,
            q_norm: mla.q_a_norm.weight,
            q_dim: mla.q_lora_rank as u32,
            kv_latent,
            kv_norm: mla.kv_a_norm.weight,
            kv_dim: mla.kv_lora_rank as u32,
            eps: args.eps,
            latent_cache,
            slots: meta.slot,
            block_size: args.bs,
            owner,
            index: ops::IndexCache {
                keys,
                gates,
                norm_weight: indexer.k_norm_weight.weight,
                norm_bias: indexer.k_norm_bias.weight,
                eps: 1e-6,
                dim: index_dim,
                tail: kv_cache.sparse_index_tail_pool_ptr(layer),
                tail_map: kv_cache.sparse_index_tail_map_ptr(),
                tail_stride: kv_cache.sparse_index_tail_block_stride_bytes(layer) as u64,
                ape: indexer.kpool_ape.weight,
                pool: kv_cache.sparse_index_pool_ptr(layer),
                pool_stride: kv_cache.sparse_index_block_stride_bytes(layer) as u64,
                tokens_per_pool: 4,
            },
            localize,
        };
        fuse.launch_down(ctx.gpu, &down, args.stream)?;
        fuse.launch_mid(ctx.gpu, &mid, args.stream)?;
        // ATLAS_GLM_DET_TRACE: the keys and gates the cache update wrote.
        let det = crate::det_trace::on_stream(ctx.gpu, args.stream);
        det.tap("x_ikeys", keys, (0, n as usize), index_dim as usize * 2);
        det.tap("x_igates", gates, (0, n as usize), index_dim as usize * 2);
        Ok(Some(localize.is_some()))
    }
}
