// SPDX-License-Identifier: AGPL-3.0-only

//! Ling gated-MLA decoder layer (the 7 full-attention layers): absorbed
//! multi-latent attention with a head-wise sigmoid output gate, per
//! `kernels/gb10/ling-3.2-air/nvfp4/gated_mla.cu`.
//!
//! K cache: the house `PagedKvCache` configured 1 head x 576 wide (the
//! factory's absorbed-MLA sizing), written by
//! `ling_mla_kv_a_norm_rope_cache`. V cache: the 512-wide normed latent has
//! no compressed-tail equal to K's row width, so each layer owns its
//! `[num_blocks, block_size, 512]` latent pool, sized to the paged pool on
//! first use and indexed by the SAME slots — admission is the house pool's.
//!
//! Decode reads the paged path. Prefill: a first-position sub-chunk runs
//! the single-pass chunk-local `ling_mla_prefill_attn` over the contiguous
//! rows (the fast path the kernel documents for ISL<=4096 single-pass
//! serving); continuing chunks loop 8-row groups over the paged kernel.

use std::sync::Arc;

use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::kv_cache::PagedKvCache;

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;

use super::spec::{MLA_K_ROW_BF16, MLA_V_ROW_BF16};
use super::{LingFfn, LingKernels, LingScratch, MLA_MAX_ROWS_PAGED};
use crate::layer::{ForwardContext, LayerState, TransformerLayer};
use crate::layers::ops;
use crate::weight_map::DenseWeight;

pub(crate) struct LingMlaWeights {
    /// q_proj `[32 * 192, hidden]` (per head [nope 128 | rope 64], full rank).
    pub q_proj: DenseWeight,
    /// kv_a_proj_with_mqa `[576, hidden]`.
    pub kv_a: DenseWeight,
    /// kv_a_layernorm `[512]` weight.
    pub kv_a_norm: DenseWeight,
    /// g_proj `[32, hidden]` — the head-wise attention-output gate.
    pub g_proj: DenseWeight,
    /// dense (o_proj) `[hidden, 32 * 128]`.
    pub o_proj: DenseWeight,
    /// Load-time transposed `w_uk` `[heads=32, 512, 128]` BF16
    /// (w_uk[h][n][j] == kv_b_proj[h * 256 + j][n]).
    pub w_uk: DenseWeight,
    /// kv_b_proj `[32 * 256, 512]` base pointer — the V extractor reads its
    /// v slice in place (W_UV[h][n][k] == kv_b_proj[h * 256 + 128 + n][k]).
    pub kv_b: DenseWeight,
}

pub(crate) struct LingMlaLayer {
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    weights: LingMlaWeights,
    ffn: LingFfn,
    kernels: Arc<LingKernels>,
    scratch: Arc<LingScratch>,
    /// Attention-layer index in the paged cache (and this layer's V pool).
    attn_idx: usize,
    heads: u32,
    kv_lora: u32,
    rope: u32,
    nope: u32,
    v_dim: u32,
    eps: f32,
    rope_theta: f64,
    /// Layer-owned V latent pools, lazily sized to the paged pool.
    v_pool: Mutex<Option<DevicePtr>>,
}

impl LingMlaLayer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_norm: DenseWeight,
        post_attn_norm: DenseWeight,
        weights: LingMlaWeights,
        ffn: LingFfn,
        kernels: Arc<LingKernels>,
        scratch: Arc<LingScratch>,
        attn_idx: usize,
        config: &ModelConfig,
    ) -> Result<Self> {
        ensure!(
            config.kv_lora_rank == 512
                && config.qk_nope_head_dim == 128
                && config.qk_rope_head_dim == 64
                && config.v_head_dim == 128
                && config.num_attention_heads == 32,
            "ling MLA: unsupported geometry for the gated-MLA kernels"
        );
        Ok(Self {
            input_norm,
            post_attn_norm,
            weights,
            ffn,
            kernels,
            scratch,
            attn_idx,
            heads: config.num_attention_heads as u32,
            kv_lora: config.kv_lora_rank as u32,
            rope: config.qk_rope_head_dim as u32,
            nope: config.qk_nope_head_dim as u32,
            v_dim: config.v_head_dim as u32,
            eps: config.rms_norm_eps as f32,
            rope_theta: config.rope_theta,
            v_pool: Mutex::new(None),
        })
    }

    fn v_cache_pool(&self, gpu: &dyn GpuBackend, kv_cache: &PagedKvCache) -> Result<DevicePtr> {
        let mut pool = self.v_pool.lock();
        if let Some(p) = *pool {
            return Ok(p);
        }
        let blocks = kv_cache.num_blocks();
        let block_size = kv_cache.block_size();
        let bytes = blocks * block_size * MLA_V_ROW_BF16 * 2;
        let p = gpu.alloc(bytes)?;
        gpu.memset(p, 0, bytes)?;
        *pool = Some(p);
        tracing::info!(
            "ling MLA layer {}: allocated the V latent pool ({} blocks x {} x {} B)",
            self.attn_idx,
            blocks,
            block_size,
            MLA_V_ROW_BF16 * 2
        );
        Ok(p)
    }

    fn kernel_checks(&self, kv_cache: &PagedKvCache) -> Result<(u32, u32)> {
        let block_size = kv_cache.block_size() as u32;
        let cfg_k = kv_cache.config();
        ensure!(
            cfg_k.head_dim == MLA_K_ROW_BF16 && cfg_k.num_kv_heads == 1,
            "ling MLA: the paged pool must be the absorbed 1 x 576 layout, got {} x {}",
            cfg_k.num_kv_heads,
            cfg_k.head_dim
        );
        Ok((block_size, cfg_k.num_layers as u32))
    }

    /// Projections for `rows` rows into the scratch lanes.
    #[allow(clippy::too_many_arguments)]
    fn project(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let w = &self.weights;
        let hidden = self.scratch.hidden as u32;
        let q_row = self.heads * (self.nope + self.rope);
        let kv_dim = self.kv_lora + self.rope;
        if rows == 1 {
            ops::dense_gemv(
                gpu, k.gemv_k, normed, &w.q_proj, s.mla_q, q_row, hidden, stream,
            )?;
            ops::dense_gemv(
                gpu, k.gemv_k, normed, &w.kv_a, s.mla_kv_a, kv_dim, hidden, stream,
            )?;
            ops::dense_gemv(
                gpu, k.gemv_k, normed, &w.g_proj, s.mla_g, self.heads, hidden, stream,
            )?;
        } else {
            ops::dense_gemm(
                gpu, k.gemm_k, normed, &w.q_proj, s.mla_q, rows, q_row, hidden, stream,
            )?;
            ops::dense_gemm(
                gpu, k.gemm_k, normed, &w.kv_a, s.mla_kv_a, rows, kv_dim, hidden, stream,
            )?;
            ops::dense_gemm(
                gpu, k.gemm_k, normed, &w.g_proj, s.mla_g, rows, self.heads, hidden, stream,
            )?;
        }
        Ok(())
    }

    /// kv_a norm + partial rope + cache write for `rows` rows.
    fn write_cache(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        kv_cache: &mut PagedKvCache,
        k_contig: Option<(DevicePtr, DevicePtr)>,
        slots: DevicePtr,
        positions: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let s = &self.scratch;
        let k = &self.kernels;
        let w = &self.weights;
        let (block_size, _) = self.kernel_checks(kv_cache)?;
        let v_pool = self.v_cache_pool(gpu, kv_cache)?;
        KernelLaunch::new(gpu, k.mla_kv_a_cache_k)
            .grid([rows, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(s.mla_kv_a)
            .arg_ptr(w.kv_a_norm.weight)
            .arg_f32(self.eps)
            .arg_ptr(k_contig.map_or(DevicePtr::NULL, |(kk, _)| kk))
            .arg_ptr(k_contig.map_or(DevicePtr::NULL, |(_, vv)| vv))
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_idx))
            .arg_ptr(v_pool)
            .arg_ptr(slots)
            .arg_ptr(positions)
            .arg_u32(block_size)
            .arg_u32(self.kv_lora + self.rope)
            .arg_u32(self.kv_lora)
            .arg_words(&[self.rope_theta.to_bits()])
            .launch(stream)?;
        Ok(v_pool)
    }

    /// q absorb + rope for `rows` rows (q_proj rows -> q_final).
    fn q_absorb(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        positions: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        KernelLaunch::new(gpu, k.mla_q_absorb_k)
            .grid([self.heads, div_ceil(self.kv_lora, 256), 1])
            .block([256, 1, 1])
            .arg_ptr(s.mla_q)
            .arg_ptr(self.weights.w_uk.weight)
            .arg_ptr(s.mla_q_final)
            .arg_ptr(positions)
            .arg_u32(rows)
            .arg_u32(self.heads)
            .arg_u32(self.nope)
            .arg_u32(self.rope)
            .arg_u32(self.kv_lora)
            .arg_words(&[self.rope_theta.to_bits()])
            .launch(stream)
    }

    /// Paged absorbed attention for `rows` (1..=8) query rows of one
    /// sequence whose table is `block_table` (device i32) and whose cached
    /// entries before this block number `base_len`.
    fn paged_attn(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        block_table: DevicePtr,
        max_blocks: u32,
        base_len: DevicePtr,
        v_pool: DevicePtr,
        kv_cache: &PagedKvCache,
        start_row: usize,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let (block_size, _) = self.kernel_checks(kv_cache)?;
        let head_stride_q = (self.heads * (self.kv_lora + self.rope) * 2) as usize;
        let head_stride_o = (self.heads * self.kv_lora * 2) as usize;
        KernelLaunch::new(gpu, k.mla_paged_attn_k)
            .grid([self.heads, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.mla_q_final.offset(start_row * head_stride_q))
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_idx))
            .arg_ptr(v_pool)
            .arg_ptr(s.mla_latent.offset(start_row * head_stride_o))
            .arg_ptr(block_table)
            .arg_ptr(base_len)
            .arg_u32(max_blocks)
            .arg_u32(rows)
            .arg_u32(self.kv_lora + self.rope)
            .arg_u32(self.kv_lora)
            // Softmax scale: qk_head_dim ** -0.5 (the kernel's contract).
            .arg_f32(1.0 / ((self.nope + self.rope) as f32).sqrt())
            .arg_u32(block_size)
            .launch(stream)
    }

    /// V extraction with the head-wise gate, writing the T-wide rows.
    fn v_extract(&self, gpu: &dyn GpuBackend, rows: u32, stream: u64) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        KernelLaunch::new(gpu, k.mla_v_gated_k)
            .grid([rows * self.heads * (self.v_dim / 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(s.mla_latent)
            .arg_ptr(self.weights.kv_b.weight)
            .arg_ptr(s.mla_g)
            .arg_ptr(s.attn_out)
            .arg_u32(rows)
            .arg_u32(self.heads)
            .arg_u32(self.nope)
            .arg_u32(self.v_dim)
            .arg_u32(self.kv_lora)
            .launch(stream)
    }

    /// The per-call host uploads (eager path): positions, slots, base_len.
    fn upload_meta(
        &self,
        gpu: &dyn GpuBackend,
        rows_start: usize,
        rows: usize,
        seq_len_start: usize,
        block_table: &[u32],
        block_size: usize,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, u32)> {
        let s = &self.scratch;
        ensure!(
            rows_start + rows <= s.rows,
            "ling MLA: sub-chunk exceeds the scratch row cap"
        );
        let mut pos: Vec<u32> = Vec::with_capacity(rows);
        let mut slots: Vec<i64> = Vec::with_capacity(rows);
        for t in 0..rows {
            let position = seq_len_start + rows_start + t;
            let block = position / block_size;
            let slot = block_table.get(block).ok_or_else(|| {
                anyhow::anyhow!("ling MLA: block table too short for position {position}")
            })?;
            pos.push(position as u32);
            slots.push((*slot as i64) * (block_size as i64) + (position % block_size) as i64);
        }
        gpu.copy_h2d_async(
            unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, pos.len() * 4) },
            s.mla_pos.offset(rows_start * 4),
            stream,
        )?;
        gpu.copy_h2d_async(
            unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, slots.len() * 8) },
            s.mla_slots.offset(rows_start * 8),
            stream,
        )?;
        Ok((s.mla_pos, s.mla_slots, block_table.len() as u32))
    }

    /// Fast-path single-pass prefill attention over the contiguous rows
    /// (`ling_mla_prefill_attn`, chunk-local causal from position 0).
    fn prefill_attn_fast(&self, gpu: &dyn GpuBackend, seq_len: u32, stream: u64) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        KernelLaunch::new(gpu, k.mla_prefill_attn_k)
            .grid([self.heads, div_ceil(seq_len, 16), 1])
            .block([256, 1, 1])
            .arg_ptr(s.mla_q_final)
            .arg_ptr(s.mla_k_contig)
            .arg_ptr(s.mla_v_contig)
            .arg_ptr(s.mla_latent)
            .arg_u32(seq_len)
            .arg_u32(1)
            .arg_f32(1.0 / ((self.nope + self.rope) as f32).sqrt())
            .launch(stream)
    }

    fn attention_half_decode(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &[u32],
        stream: u64,
    ) -> Result<()> {        let s = &self.scratch;
        let k = &self.kernels;
        let hidden_w = s.hidden as u32;
        ops::rms_norm(
            gpu,
            k.rms_norm_k,
            hidden,
            &self.input_norm,
            s.normed,
            1,
            hidden_w,
            self.eps,
            stream,
        )?;
        self.project(gpu, 1, s.normed, stream)?;
        let (positions, slots, max_blocks) = self.upload_meta(
            gpu,
            0,
            1,
            seq_len,
            block_table,
            kv_cache.block_size(),
            stream,
        )?;
        let v_pool = self.write_cache(gpu, 1, kv_cache, None, slots, positions, stream)?;
        self.q_absorb(gpu, 1, positions, stream)?;
        let base = (seq_len as u32).to_le_bytes();
        gpu.copy_h2d_async(&base, s.mla_base_len, stream)?;
        let table_i32: Vec<i32> = block_table.iter().map(|&b| b as i32).collect();
        ensure!(
            table_i32.len() <= s.max_blocks,
            "ling MLA: block table ({} entries) exceeds the staging cap {}",
            table_i32.len(),
            s.max_blocks
        );
        gpu.copy_h2d_async(
            unsafe {
                std::slice::from_raw_parts(table_i32.as_ptr() as *const u8, table_i32.len() * 4)
            },
            s.mla_block_table,
            stream,
        )?;
        self.paged_attn(
            gpu,
            1,
            s.mla_block_table,
            max_blocks,
            s.mla_base_len,
            v_pool,
            kv_cache,
            0,
            stream,
        )?;
        self.v_extract(gpu, 1, stream)?;
        ops::dense_gemv(
            gpu,
            k.gemv_k,
            s.attn_out,
            &self.weights.o_proj,
            s.ffn_in,
            hidden_w,
            self.heads * self.v_dim,
            stream,
        )?;
        ops::residual_add(gpu, k.residual_add_k, hidden, s.ffn_in, hidden_w, stream)
    }

    fn ffn_half(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm_k,
            hidden,
            &self.post_attn_norm,
            s.ffn_in,
            rows,
            s.hidden as u32,
            self.eps,
            stream,
        )?;
        self.ffn
            .forward(gpu, s.ffn_in, s.ffn_out, rows as usize, stream)?;
        ops::residual_add(
            gpu,
            self.kernels.residual_add_k,
            hidden,
            s.ffn_out,
            rows * s.hidden as u32,
            stream,
        )
    }

    /// The drafter-context prefill of this layer's compressed KV ONLY:
    /// `rows` input rows at consecutive positions `seq_len_start..` write
    /// their paged K row and V latent and nothing else — no Q projection,
    /// no attention, no FFN (a single autoregressive layer's cached K/V
    /// depends only on its own normed input row). This is the Ling body of
    /// the `prefill_mla_kv_only` shape the house MTP drafters use to build
    /// the NEXTN drafter's context over the prompt pairs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn kv_only_rows(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        kv_cache: &mut PagedKvCache,
        rows: usize,
        seq_len_start: usize,
        block_table: &[u32],
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let hidden_w = s.hidden as u32;
        let rows_u32 = rows as u32;
        ensure!(
            rows <= s.rows && seq_len_start + rows <= s.rows,
            "ling MLA kv-only: {rows} rows at {seq_len_start} exceed the scratch cap"
        );
        ops::rms_norm(
            gpu,
            k.rms_norm_k,
            hidden,
            &self.input_norm,
            s.normed,
            rows_u32,
            hidden_w,
            self.eps,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            k.gemm_k,
            s.normed,
            &self.weights.kv_a,
            s.mla_kv_a,
            rows_u32,
            self.kv_lora + self.rope,
            hidden_w,
            stream,
        )?;
        let (positions, slots, _) = self.upload_meta(
            gpu,
            0,
            rows,
            seq_len_start,
            block_table,
            kv_cache.block_size(),
            stream,
        )?;
        self.write_cache(gpu, rows_u32, kv_cache, None, slots, positions, stream)?;
        Ok(())
    }

    /// Prefill attention over `rows` rows starting at `seq_len_start`.
    /// Single-pass fast path only when this sub-chunk IS the whole prompt.
    #[allow(clippy::too_many_arguments)]
    fn attention_half_prefill(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        kv_cache: &mut PagedKvCache,
        rows: usize,
        seq_len_start: usize,
        whole_prompt: bool,
        block_table: &[u32],
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let hidden_w = s.hidden as u32;
        let rows_u32 = rows as u32;
        ops::rms_norm(
            gpu,
            k.rms_norm_k,
            hidden,
            &self.input_norm,
            s.normed,
            rows_u32,
            hidden_w,
            self.eps,
            stream,
        )?;
        self.project(gpu, rows_u32, s.normed, stream)?;
        let (positions, slots, max_blocks) = self.upload_meta(
            gpu,
            0,
            rows,
            seq_len_start,
            block_table,
            kv_cache.block_size(),
            stream,
        )?;
        let v_pool = self.write_cache(
            gpu,
            rows_u32,
            kv_cache,
            if whole_prompt && seq_len_start == 0 {
                Some((s.mla_k_contig, s.mla_v_contig))
            } else {
                None
            },
            slots,
            positions,
            stream,
        )?;
        self.q_absorb(gpu, rows_u32, positions, stream)?;
        if whole_prompt && seq_len_start == 0 {
            self.prefill_attn_fast(gpu, rows_u32, stream)?;
        } else {
            // Continuing chunk: 8-row groups over the paged cache.
            let base = (seq_len_start as u32).to_le_bytes();
            gpu.copy_h2d_async(&base, s.mla_base_len, stream)?;
            let table_i32: Vec<i32> = block_table.iter().map(|&b| b as i32).collect();
            ensure!(
                table_i32.len() <= s.max_blocks,
                "ling MLA: block table ({} entries) exceeds the staging cap {}",
                table_i32.len(),
                s.max_blocks
            );
            gpu.copy_h2d_async(
                unsafe {
                    std::slice::from_raw_parts(table_i32.as_ptr() as *const u8, table_i32.len() * 4)
                },
                s.mla_block_table,
                stream,
            )?;
            for start in (0..rows).step_by(MLA_MAX_ROWS_PAGED) {
                let batch = (rows - start).min(MLA_MAX_ROWS_PAGED) as u32;
                // base_len must name the CACHE ENTRIES BEFORE this group —
                // the paged kernel adds the row offset itself.
                if start > 0 {
                    let b = (seq_len_start + start) as u32;
                    gpu.copy_h2d_async(&b.to_le_bytes(), s.mla_base_len, stream)?;
                }
                self.paged_attn(
                    gpu,
                    batch,
                    s.mla_block_table,
                    max_blocks,
                    s.mla_base_len,
                    v_pool,
                    kv_cache,
                    start,
                    stream,
                )?;
            }
        }
        self.v_extract(gpu, rows_u32, stream)?;
        ops::dense_gemm(
            gpu,
            k.gemm_k,
            s.attn_out,
            &self.weights.o_proj,
            s.ffn_in,
            rows_u32,
            hidden_w,
            self.heads * self.v_dim,
            stream,
        )?;
        ops::residual_add(
            gpu,
            k.residual_add_k,
            hidden,
            s.ffn_in,
            rows_u32 * hidden_w,
            stream,
        )
    }
}

impl TransformerLayer for LingMlaLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.attention_half_decode(ctx.gpu, hidden, kv_cache, seq_len, block_table, stream)?;
        self.ffn_half(ctx.gpu, hidden, 1, stream)
    }

    /// The MTP/K-token verify of one sequence: `num_tokens` consecutive
    /// rows ride the paged kernel's multi-row body (`ling_mla_paged_attn`
    /// serves rows 1..=8 with per-row causal over the just-written block —
    /// PR #112's verify-row support), base_len naming the entries before
    /// the window. A single row runs the plain decode path unchanged.
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if num_tokens <= 1 {
            self.attention_half_decode(ctx.gpu, hidden, kv_cache, seq_len, block_table, stream)?;
            return self.ffn_half(ctx.gpu, hidden, 1, stream);
        }
        anyhow::ensure!(
            num_tokens <= MLA_MAX_ROWS_PAGED,
            "ling MLA: verify {num_tokens} rows exceeds the paged kernel's {}",
            MLA_MAX_ROWS_PAGED
        );
        self.attention_half_prefill(
            ctx.gpu,
            hidden,
            kv_cache,
            num_tokens,
            seq_len,
            false,
            block_table,
            stream,
        )?;
        self.ffn_half(ctx.gpu, hidden, num_tokens as u32, stream)
    }

    /// One NEXTN draft: this layer serves the K = 2 verify (the paged
    /// kernel's row cap aside, the family's agreed verify width is 2).
    fn verify_max_drafts(&self) -> Option<usize> {
        Some(1)
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Sub-chunk the prompt: every sub-chunk runs the paged attention
        // path (correct under any chunking); the single-pass contiguous
        // fast path stays opt-in (ATLAS_LING_MLA_FAST_PREFILL=1, valid
        // only for a from-position-0 whole prompt that fits one chunk).
        let fast = std::env::var("ATLAS_LING_MLA_FAST_PREFILL").as_deref() == Ok("1")
            && seq_len_start == 0;
        let cap = self.scratch.rows;
        for (n, chunk) in (0..num_tokens).step_by(cap).enumerate() {
            let chunk_rows = (num_tokens - chunk).min(cap);
            self.attention_half_prefill(
                ctx.gpu,
                hidden.offset(chunk * self.scratch.hidden * 2),
                kv_cache,
                chunk_rows,
                seq_len_start + chunk,
                fast && n == 0 && num_tokens <= cap,
                block_table,
                stream,
            )?;
            self.ffn_half(
                ctx.gpu,
                hidden.offset(chunk * self.scratch.hidden * 2),
                chunk_rows as u32,
                stream,
            )?;
        }
        Ok(())
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(crate::layer::EmptyLayerState))
    }

    /// Eager only in this PR: the per-step uploads (positions / slots /
    /// block table) are host-side, and the layer-owned V pool binds late.
    fn decode_graph_unsupported(&self) -> bool {
        true
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
#[path = "mla_gpu_parity_tests.rs"]
mod mla_gpu_parity_tests;

#[cfg(test)]
#[path = "e2e_gpu_parity_tests.rs"]
mod e2e_gpu_parity_tests;
