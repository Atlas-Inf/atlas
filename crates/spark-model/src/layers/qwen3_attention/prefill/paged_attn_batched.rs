// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 Path B: batched paged-prefill attention dispatch.
//!
//! Mirror of [`prefill_attention_paged_attn`](super::paged_attn) for the
//! same-chunk-len batched case. Handles only the standard paths
//! (BF16/FP8/NVFP4 paged attention, BR=32 or BR=64). Unsupported paths
//! (HDIM=512 Gemma-4, HSS streaming, Turbo3/4/8 variants) bail with
//! `Err` — the caller falls back to per-stream `prefill_attention_paged`
//! for those layers.
//!
//! Constraint: scheduler-enforced same-chunk-len and same-q_offset
//! (= `seq_len_start`) across all batched streams. Each stream's KV
//! pages are accessed via `block_table_ptrs[b]` — built by
//! `stage_batched_attn_metadata` in commit 4fc6d30.

#![allow(unused_imports, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layer::{BatchedAttnMetadata, ForwardContext};
use crate::layers::ops;

#[allow(dead_code)]
pub(in crate::layers::qwen3_attention) struct PagedAttnBatchedArgs<'a> {
    pub q_contiguous: DevicePtr,
    pub attn_out: DevicePtr,
    pub seq_len_start: usize,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub bs: usize,
    pub inv_sqrt_d: f32,
    pub kv_len: u32,
    pub batched_meta: &'a BatchedAttnMetadata,
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    /// [`crate::layer::TransformerLayer::honours_state_end`]: the batched
    /// paged kernels below (rows128 included) apply `state_end` when no
    /// sliding window is set. MLA, mHC and the QSA indexer take other routes,
    /// and HSS, HDIM > 256 and other KV dtypes refuse batched mode outright.
    pub(in crate::layers::qwen3_attention) fn honours_state_end_inner(&self) -> bool {
        self.mla.is_none()
            && self.hc.is_none()
            && self.qsa.is_none()
            && self.sliding_window.is_none()
    }

    /// Run batched paged Flash Attention across N streams. Uses the
    /// `prefill_attention_paged_*_batched` kernels shipped in commit
    /// 4ec2cf2 + ops bindings from commit a96fc67.
    #[allow(dead_code)]
    pub(in crate::layers::qwen3_attention) fn prefill_attention_paged_attn_batched(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &PagedAttnBatchedArgs<'_>,
    ) -> Result<()> {
        let PagedAttnBatchedArgs {
            q_contiguous,
            attn_out,
            seq_len_start,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            kv_len,
            batched_meta,
            stream,
        } = *args;
        let bs_u = bs as u32;
        let chunk_len = batched_meta.chunk_len;
        let batch_size = batched_meta.batch_size;
        let block_table_ptrs = batched_meta.block_table_ptrs;
        // VARLEN: per-stream Q offsets/lengths and KV extents. NULL under the
        // legacy uniform path, where the kernels keep using `b * chunk_len`
        // and the scalar `kv_len`. `chunk_len` is the MAX and bounds the grid.
        let cu_seqlens = batched_meta.cu_seqlens;
        let kv_lens = batched_meta.kv_lens;

        // Q12 batched mode only supports the standard paths. HSS, MLA,
        // and HDIM=512 layers fall back to per-stream at the dispatch
        // level.
        if hd > 256 {
            anyhow::bail!(
                "prefill_attention_paged_attn_batched: HDIM={} (>256) not supported \
                 in batched mode (layer {}). Caller should fall back to per-stream.",
                hd,
                self.attn_layer_idx
            );
        }
        // Must mirror `check_kernel_batched_eligible`'s `allow_chunk_zero`
        // exactly — a batch admitted there and rejected here bails after the
        // streams have already been mutated.
        let allow_first_chunk = crate::layers::ops::prefill_batched_first_chunk_enabled()
            || crate::layers::ops::prefill_varlen_enabled();
        if seq_len_start == 0 && !allow_first_chunk {
            anyhow::bail!(
                "prefill_attention_paged_attn_batched: seq_len_start=0 not supported \
                 (batched kernels are paged-only; non-paged BR=32 batched kernel is \
                 not yet shipped). Caller should fall back to per-stream."
            );
        }

        let use_br64 = chunk_len >= 256;
        let q_offset_u32 = seq_len_start as u32;
        // 128-row tiles: a score pass's GQA tiles at any length (no sliding
        // window: the split entry has none), else the lever's per-head tiles
        // in place of BR=64.
        let rows128 = if batched_meta.gqa_tiles && self.sliding_window.is_none() {
            Some(super::rows128::Tiles::gqa(
                &batched_meta.kv_lens_host,
                &batched_meta.cu_seqlens_host,
            ))
        } else {
            use_br64.then_some(super::rows128::Tiles::PerHead)
        }
        .filter(|&t| hd == 256 && super::rows128::available(ctx.gpu, t));
        if let (KvCacheDtype::Bf16, Some(tiles)) = (self.kv_dtype, rows128) {
            // Same arguments and bit-identical output, 128-row tiles (split
            // GQA tiles: within bf16 rounding).
            let pools = super::rows128::Pools {
                k: kv_cache.k_pool_ptr(self.attn_layer_idx),
                v: kv_cache.v_pool_ptr(self.attn_layer_idx),
                blocks: kv_cache.num_blocks(),
                block_stride: kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx),
            };
            return super::rows128::launch(
                ctx.gpu,
                tiles,
                pools,
                [q_contiguous, attn_out, block_table_ptrs],
                batch_size,
                [cu_seqlens, kv_lens, batched_meta.state_end],
                [chunk_len, kv_len, q_offset_u32],
                [nq, nkv, hd, bs_u, self.sliding_window.unwrap_or(0)],
                inv_sqrt_d,
                stream,
            );
        }
        let (fp8_k_scale, fp8_v_scale) = self.effective_fp8_scales();

        match (self.kv_dtype, use_br64) {
            (KvCacheDtype::Nvfp4, false) => {
                if self.prefill_attn_paged_nvfp4_batched_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_nvfp4_batched kernel not loaded — \
                         rebuild atlas-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_nvfp4_batched(
                    ctx.gpu,
                    self.prefill_attn_paged_nvfp4_batched_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    batched_meta.state_end,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )?;
            }
            (KvCacheDtype::Nvfp4, true) => {
                if self.prefill_attn_paged_nvfp4_batched_64_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_nvfp4_batched_64 kernel not loaded — \
                         rebuild atlas-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_nvfp4_batched_64(
                    ctx.gpu,
                    self.prefill_attn_paged_nvfp4_batched_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    batched_meta.state_end,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )?;
            }
            (KvCacheDtype::Fp8, false) => {
                if self.prefill_attn_paged_fp8_batched_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_fp8_batched kernel not loaded — \
                         rebuild atlas-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_fp8_batched(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8_batched_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    batched_meta.state_end,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    fp8_k_scale,
                    fp8_v_scale,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    stream,
                )?;
            }
            (KvCacheDtype::Fp8, true) => {
                if self.prefill_attn_paged_fp8_batched_64_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_fp8_batched_64 kernel not loaded — \
                         rebuild atlas-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_fp8_batched_64(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8_batched_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    batched_meta.state_end,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    fp8_k_scale,
                    fp8_v_scale,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    stream,
                )?;
            }
            (KvCacheDtype::Bf16, false) => {
                if self.prefill_attn_paged_batched_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_batched kernel not loaded — \
                         rebuild atlas-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_batched(
                    ctx.gpu,
                    self.prefill_attn_paged_batched_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    batched_meta.state_end,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    stream,
                )?;
            }
            (KvCacheDtype::Bf16, true) => {
                if self.prefill_attn_paged_batched_64_k.0 == 0 {
                    anyhow::bail!(
                        "prefill_attn_paged_batched_64 kernel not loaded — \
                         rebuild atlas-kernels (commit 4ec2cf2)."
                    );
                }
                ops::prefill_attention_paged_batched_64(
                    ctx.gpu,
                    self.prefill_attn_paged_batched_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    block_table_ptrs,
                    batch_size,
                    cu_seqlens,
                    kv_lens,
                    batched_meta.state_end,
                    chunk_len,
                    kv_len,
                    q_offset_u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    stream,
                )?;
            }
            (dtype, _) => {
                anyhow::bail!(
                    "prefill_attention_paged_attn_batched: kv_dtype {:?} not yet supported \
                     in batched mode (layer {}). Falls back to per-stream.",
                    dtype,
                    self.attn_layer_idx
                );
            }
        }

        // Note: q_contiguous and attn_out should be DevicePtr's pointing
        // at the stacked buffer (N*chunk_len tokens worth of data each).
        // The kernel reads/writes at per-batch offsets internally via
        // blockIdx.z = b.
        let _ = nq;
        Ok(())
    }
}
