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
//! Decode reads the paged path. Prefill (`ATLAS_LING_MLA_PACKED_PREFILL`,
//! default on): every chunk runs `ling_mla_prefill_attn` over the layer's
//! contiguous context — the prefix rows `ling_mla_ctx_gather` refills from
//! the paged pools plus the chunk's own rows — so a continuation chunk
//! carries the same fast path a whole prompt does (`kv_base > 0`);
//! `=0` restores 8-row paged groups for continuations plus the
//! `ATLAS_LING_MLA_FAST_PREFILL=1` position-0 special case. Verify rows
//! always read the paged path.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Mutex, MutexGuard};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::kv_cache::PagedKvCache;

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;

use super::spec::{MAX_PREFILL_ROWS, MLA_K_ROW_BF16, MLA_V_ROW_BF16};
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

/// `ATLAS_LING_MLA_PACKED_PREFILL`: unset or any value other than "0"
/// prefills each chunk over the packed context — the layer's contiguous
/// K/V mirror, refilled from the paged pools when a chunk continues a
/// cached prefix — so every chunk runs `ling_mla_prefill_attn` (chunk-local
/// causal against the prefix) instead of <=8-row paged steps, and the
/// position-0 whole-prompt special case is just `kv_base == 0`. "0"
/// restores the legacy behavior: paged 8-row groups per continuation chunk
/// plus the `ATLAS_LING_MLA_FAST_PREFILL=1` whole-prompt fast path. Read
/// once.
fn packed_prefill() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_LING_MLA_PACKED_PREFILL").as_deref() != Ok("0"))
}

/// `ATLAS_LING_PROJ_GEMM_MIN_ROWS` (shared with the KDA half): the smallest
/// chunk the prefill projections take the BF16 tensor-core lane for. Default
/// 16; `0` restores the scalar-only floor. The lane never touches cuBLASLt's
/// heuristic first pick — the split-K it picked for these small-M shapes
/// corrupted every projection from a serve's second request on (PR #131) —
/// it takes the heuristic's in-order k-chain kernel instead, byte-identical
/// to the scalar path at any row count, and falls back to the scalar floor
/// when the runtime offers no k-chain for the shape or the library is absent.
/// Read once.
fn proj_gemm_min_rows() -> u32 {
    static MIN: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        proj_gemm_min_rows_from(
            std::env::var("ATLAS_LING_PROJ_GEMM_MIN_ROWS")
                .ok()
                .as_deref(),
        )
    })
}

/// The threshold resolution, split out for the unit test. `None` = env
/// absent (default 16); `Some("0")` = the scalar-only floor (u32::MAX);
/// `Some(other)` = that row count, clamped to >= 1.
fn proj_gemm_min_rows_from(v: Option<&str>) -> u32 {
    match v {
        None => 16,
        Some("0") => u32::MAX,
        Some(s) => s.parse::<u32>().ok().unwrap_or(16).max(1),
    }
}

/// Whether a `decode_multi_seq_rows` window is `num_rows` rows of ONE
/// sequence: exactly one state and every row names it (the MTP verify's
/// `row_owner = [0; k]` with the sequence's own layer state). Distinct
/// sequences per row (the a2 multi-request lane) keep the trait's default
/// per-row loop.
fn single_owner_window(states_len: usize, row_owner: &[usize]) -> bool {
    states_len == 1 && !row_owner.is_empty() && row_owner.iter().all(|&o| o == 0)
}

/// One prefill-shaped projection: the BF16 tensor-core lane from the M
/// threshold up, the scalar GEMM below it.
#[allow(clippy::too_many_arguments)]
fn proj_gemm(
    gpu: &dyn GpuBackend,
    k: &LingKernels,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k_dim: u32,
    stream: u64,
) -> Result<()> {
    if rows >= proj_gemm_min_rows() {
        // The in-order k-chain (`_any`: the heuristic's first in-order
        // m16n8k16 kernel, tiles 256x128 / 128x256 / 128x64 / 64x256 / 64x64).
        // Byte-identical to the scalar path over the same operands and
        // never a split-K. A declined or unavailable lane falls back to the
        // scalar floor below — the heuristic's first pick is not a valid
        // option for these shapes (PR #131).
        let kchain_ok = spark_runtime::cublaslt::bf16_gemm_act_weight_t_kchain_any(
            input.0,
            weight.weight.0,
            output.0,
            [rows, n, k_dim],
            stream,
        )
        .unwrap_or_else(|e| {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "ling prefill TC lane unavailable ({e}); projections fall back to the scalar path"
                );
            }
            false
        });
        if kchain_ok {
            return Ok(());
        }
    }
    ops::dense_gemm(gpu, k.gemm_k, input, weight, output, rows, n, k_dim, stream)
}

/// The layer's contiguous prefill context: the sequence's K/V rows AS THE
/// PREFILL KERNEL READS THEM (normed latent + roped tail), used as the
/// `kv_base` prefix of `ling_mla_prefill_attn`. Bound once at
/// [`LingMlaLayer::ctx_pool`] to the whole max window and never grown;
/// what an earlier call left in the mirror is deliberately untrusted —
/// the elide stamp decides whether the mirror contents ARE the current
/// call's prefix or must be re-gathered from the paged pools.
struct CtxPools {
    k: DevicePtr,
    v: DevicePtr,
    /// Mirror capacity in rows: bound at first use to
    /// `max(config.max_seq_len, MAX_PREFILL_ROWS)`.
    rows: usize,
    /// Elide stamp: mirror rows `[0, stamp_rows)` hold the prefix rows of
    /// the prefill call whose per-call epoch is `stamp_tag`.
    stamp_tag: u64,
    stamp_rows: usize,
}

/// True when the context mirror already holds exactly the chunk's prefix
/// rows `[0, chunk_start)` for this prefill call: the call's earlier
/// sub-chunks wrote them contiguously. One-shot by design — a foreign
/// call's epoch never matches, so a stale mirror can never elide a
/// foreign sequence's prefix.
fn ctx_elides_prefix(mirror_tag: u64, mirror_rows: usize, tag: u64, chunk_start: usize) -> bool {
    // Epoch 0 (the pristine stamp) never elides, even at (0, 0).
    tag != 0 && tag == mirror_tag && chunk_start == mirror_rows
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
    /// Layer-owned packed prefill context, bound once to the whole max
    /// window (see `ctx_pool`): the contiguous mirror + the elide stamp.
    ctx: Mutex<Option<CtxPools>>,
    /// Whole-window context-mirror size in rows:
    /// `max(config.max_seq_len, MAX_PREFILL_ROWS)`.
    max_window: usize,
    /// Per-prefill-call epoch for the elide stamp (chunks of one
    /// TransformerLayer::prefill call share a tag; every new call — a new
    /// request or a continued request's next step — gets a fresh one, so
    /// a stale mirror can never falsely elide a foreign prefix).
    ctx_epoch: AtomicU64,
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
            ctx: Mutex::new(None),
            max_window: config.max_seq_len.max(MAX_PREFILL_ROWS),
            ctx_epoch: AtomicU64::new(0),
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

    /// The layer's packed prefill context: `window` rows of the sequence's
    /// own K/V in the packed form the attention kernel reads, bound ONCE
    /// to the whole max window and never grown. The old grow path
    /// re-allocated the mirror while the previous chunk's in-flight writer
    /// kernels still targeted it — CUDA_ERROR_ILLEGAL_ADDRESS, sticky
    /// ("prefill chunk layer 7", 16.4k request) — so growth is deleted,
    /// not fenced: a pool that never reallocs cannot race.
    fn ctx_pool(
        &self,
        gpu: &dyn GpuBackend,
        need_rows: usize,
    ) -> Result<MutexGuard<'_, Option<CtxPools>>> {
        ensure!(
            need_rows <= self.max_window,
            "ling MLA layer {}: prefill chunk ({} rows) exceeds the whole-window context mirror ({}) — max_seq_len violated",
            self.attn_idx,
            need_rows,
            self.max_window
        );
        let mut cell = self.ctx.lock();
        if cell.is_none() {
            let rows = self.max_window;
            let k = gpu.alloc(rows * MLA_K_ROW_BF16 * 2)?;
            gpu.memset(k, 0, rows * MLA_K_ROW_BF16 * 2)?;
            let v = gpu.alloc(rows * MLA_V_ROW_BF16 * 2)?;
            gpu.memset(v, 0, rows * MLA_V_ROW_BF16 * 2)?;
            tracing::info!(
                "ling MLA layer {}: bound the packed prefill context at the full {}-row window ({} MiB)",
                self.attn_idx,
                rows,
                rows * (MLA_K_ROW_BF16 + MLA_V_ROW_BF16) * 2 / (1024 * 1024)
            );
            *cell = Some(CtxPools {
                k,
                v,
                rows,
                stamp_tag: 0,
                stamp_rows: 0,
            });
        }
        Ok(cell)
    }

    fn next_ctx_tag(&self) -> u64 {
        // Epoch 0 is reserved for the pristine stamp, so a fresh mirror
        // never elides a gather.
        self.ctx_epoch.fetch_add(1, Ordering::Relaxed) + 1
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
            proj_gemm(
                gpu, k, normed, &w.q_proj, s.mla_q, rows, q_row, hidden, stream,
            )?;
            proj_gemm(
                gpu, k, normed, &w.kv_a, s.mla_kv_a, rows, kv_dim, hidden, stream,
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

    /// q absorb + rope for `rows` rows (q_proj rows -> q_final). The row
    /// loop tiles across the grid's third dimension (`row_tile` rows a
    /// CTA) so wide chunks occupy the SMs; a 64-row tile at 512 rows
    /// exposes heads x 2 x 8 CTAs.
    fn q_absorb(
        &self,
        gpu: &dyn GpuBackend,
        rows: u32,
        positions: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let row_tile = rows.min(64);
        KernelLaunch::new(gpu, k.mla_q_absorb_k)
            .grid([
                self.heads,
                div_ceil(self.kv_lora, 256),
                div_ceil(rows, row_tile),
            ])
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
            .arg_u32(row_tile)
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

    /// The pre-graph form of the same uploads (`TransformerLayer::
    /// decode_meta_prestage`): ONE sequence's `rows` rows at positions
    /// `seq_len_start..`, plus the `base_len` word (cache entries before the
    /// block) and the i32 block table, all into the FIXED scratch staging the
    /// captured kernels then read. Byte-identical to the in-layer uploads of
    /// the decode and K=2 verify arms it replaces, issued from the
    /// dispatch's Phase 1 instead — pageable H2D inside a CUDA-graph capture
    /// is illegal (status 900); `docs/ling-decode-graphs.md`.
    fn prestage_meta(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        seq_len_start: usize,
        block_table: &[u32],
        block_size: usize,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        self.upload_meta(gpu, 0, rows, seq_len_start, block_table, block_size, stream)?;
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
        Ok(())
    }

    /// Fast-path causal prefill attention over the packed context
    /// (`ling_mla_prefill_attn`): the chunk's `seq_len` query rows against
    /// the context rows [0, kv_base + seq_len), causal inside the chunk —
    /// a position-0 whole prompt is just `kv_base == 0`.
    fn prefill_attn_contig(
        &self,
        gpu: &dyn GpuBackend,
        k_ctx: DevicePtr,
        v_ctx: DevicePtr,
        seq_len: u32,
        kv_base: u32,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        KernelLaunch::new(gpu, k.mla_prefill_attn_k)
            .grid([self.heads, div_ceil(seq_len, 16), 1])
            .block([256, 1, 1])
            .arg_ptr(s.mla_q_final)
            .arg_ptr(k_ctx)
            .arg_ptr(v_ctx)
            .arg_ptr(s.mla_latent)
            .arg_u32(seq_len)
            .arg_u32(1)
            .arg_f32(1.0 / ((self.nope + self.rope) as f32).sqrt())
            .arg_u32(kv_base)
            .launch(stream)
    }

    /// Refill the packed context's prefix rows [0, kv_base) from the paged
    /// pools (`ling_mla_ctx_gather`): that prefix is exactly what the paged
    /// path walks, so the packed path's correctness envelope is unchanged
    /// for every continuation source — earlier prompt chunks, earlier
    /// turns' decode, or a restored prefix cache.
    #[allow(clippy::too_many_arguments)]
    fn ctx_gather(
        &self,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
        ctx: &CtxPools,
        block_table: DevicePtr,
        max_blocks: u32,
        kv_base: u32,
        v_pool: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (block_size, _) = self.kernel_checks(kv_cache)?;
        KernelLaunch::new(gpu, self.kernels.mla_ctx_gather_k)
            .grid([
                div_ceil(
                    kv_base as u32 * (MLA_K_ROW_BF16 + MLA_V_ROW_BF16) as u32,
                    256,
                ),
                1,
                1,
            ])
            .block([256, 1, 1])
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_idx))
            .arg_ptr(v_pool)
            .arg_ptr(ctx.k)
            .arg_ptr(ctx.v)
            .arg_ptr(block_table)
            .arg_u32(max_blocks)
            .arg_u32(kv_base)
            .arg_u32(block_size)
            .launch(stream)
    }

    fn attention_half_decode(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &[u32],
        staged: bool,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
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
        // Staged steps (decode_meta_prestage ran in Phase 1) read the fixed
        // staging — including `mla_block_table`/`mla_base_len` below — and
        // skip the pageable uploads, which a capture would refuse.
        let (positions, slots, max_blocks) = if staged {
            (s.mla_pos, s.mla_slots, s.max_blocks as u32)
        } else {
            self.upload_meta(
                gpu,
                0,
                1,
                seq_len,
                block_table,
                kv_cache.block_size(),
                stream,
            )?
        };
        let v_pool = self.write_cache(gpu, 1, kv_cache, None, slots, positions, stream)?;
        self.q_absorb(gpu, 1, positions, stream)?;
        if !staged {
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
        }
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

    pub(crate) fn ffn_half(
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
        // `seq_len_start` is an ABSOLUTE sequence position — the NEXTN
        // drafter's context prefill walks the whole prompt in chunks, so any
        // position-constrained bound here refuses every chunk past the
        // scratch row count ("64 rows at 128 exceed the scratch cap" —
        // speculation silently died for real-context prompts and the serve
        // fell to serial decode). Only the CHUNK width occupies scratch;
        // positions beyond the table fail properly in `upload_meta` when
        // the block table is shorter than the chunk's last position.
        ensure!(
            rows <= s.rows,
            "ling MLA kv-only: {rows} rows exceed the scratch cap {cap}",
            cap = s.rows
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
    /// `packed` runs the packed-context path (default,
    /// [`packed_prefill`]): prefix rows are gathered from the paged pools
    /// into the layer's ctx and the chunk runs `ling_mla_prefill_attn`
    /// with `kv_base = seq_len_start`. `whole_prompt` is the legacy
    /// single-pass fast path, valid only for a position-0 whole prompt
    /// (assembled here when the packed path is off). `staged`
    /// (decode/verify captures only): the step metadata was prestaged in
    /// Phase 1 and the in-layer uploads are skipped — reachable with
    /// `staged` only from `decode_batched`, i.e. `rows <= 8` = one group
    /// of the paged loop.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_half_prefill(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        kv_cache: &mut PagedKvCache,
        rows: usize,
        seq_len_start: usize,
        whole_prompt: bool,
        packed: bool,
        ctx_tag: u64,
        staged: bool,
        block_table: &[u32],
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let hidden_w = s.hidden as u32;
        let rows_u32 = rows as u32;
        if packed {
            let mut cell = self.ctx_pool(gpu, seq_len_start + rows)?;
            let mirror = cell
                .as_mut()
                .expect("packed prefill context pool binds at first use");
            // Elide the prefix re-gather when this prefill call's earlier
            // sub-chunks already left exactly the chunk's prefix rows in
            // the mirror: same per-call tag (one TransformerLayer::prefill
            // call over one sequence; every other call carries a fresh
            // tag, so a foreign sequence's rows can never falsely elide)
            // and exactly contiguous extent. Otherwise the ctx_gather
            // below refills the prefix from the paged pools — the
            // correctness envelope is unchanged.
            let elide =
                ctx_elides_prefix(mirror.stamp_tag, mirror.stamp_rows, ctx_tag, seq_len_start);
            let v_pool = self.v_cache_pool(gpu, kv_cache)?;
            let result = self.attention_half_packed(
                gpu,
                hidden,
                kv_cache,
                rows,
                seq_len_start,
                mirror,
                v_pool,
                elide,
                block_table,
                stream,
            );
            // The mirror now holds [0, seq_len_start + rows) of this call's
            // sequence (gathered prefix + chunk rows): arm the next
            // sub-chunk's elision. On Err the stamp stays stale — worst
            // case one redundant gather.
            if result.is_ok() {
                mirror.stamp_tag = ctx_tag;
                mirror.stamp_rows = seq_len_start + rows;
            }
            result?;
            return Ok(());
        }
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
        let (positions, slots, max_blocks) = if staged {
            (s.mla_pos, s.mla_slots, s.max_blocks as u32)
        } else {
            self.upload_meta(
                gpu,
                0,
                rows,
                seq_len_start,
                block_table,
                kv_cache.block_size(),
                stream,
            )?
        };
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
            self.prefill_attn_contig(gpu, s.mla_k_contig, s.mla_v_contig, rows_u32, 0, stream)?;
        } else {
            // Continuing chunk: 8-row groups over the paged cache. A staged
            // step reads the prestaged base_len/table instead (`staged`
            // never reaches the start > 0 group re-uploads: rows <= 8).
            if !staged {
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
                        std::slice::from_raw_parts(
                            table_i32.as_ptr() as *const u8,
                            table_i32.len() * 4,
                        )
                    },
                    s.mla_block_table,
                    stream,
                )?;
            }
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
        proj_gemm(
            gpu,
            k,
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

    /// The packed-context prefill attention half: chunk rows continue the
    /// sequence's context mirror, `ling_mla_ctx_gather` refills the prefix
    /// the paged pools hold, and one `ling_mla_prefill_attn` launch covers
    /// prefix + chunk-local causal.
    #[allow(clippy::too_many_arguments)]
    fn attention_half_packed(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        kv_cache: &mut PagedKvCache,
        rows: usize,
        seq_len_start: usize,
        ctx: &CtxPools,
        v_pool: DevicePtr,
        elide: bool,
        block_table: &[u32],
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let hidden_w = s.hidden as u32;
        let rows_u32 = rows as u32;
        let kv_row = (self.kv_lora + self.rope) as usize;
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
        // Elide the prefix re-gather when the elide stamp proves this
        // prefill call's earlier sub-chunks already wrote exactly
        // [0, seq_len_start) of this sequence into the mirror (see the
        // caller's ctx_elides_prefix check); the fallback ctx_gather
        // refills the prefix from the paged pools.Skipped or not, the
        // chunk's own rows land right after the prefix below.
        if seq_len_start > 0 && !elide {
            self.ctx_gather(
                gpu,
                kv_cache,
                ctx,
                s.mla_block_table,
                max_blocks,
                seq_len_start as u32,
                v_pool,
                stream,
            )?;
        }
        // The chunk's own rows land right after the gathered prefix, so the
        // context is one contiguous run of positions [0, seq_len_start + rows).
        self.write_cache(
            gpu,
            rows_u32,
            kv_cache,
            Some((
                ctx.k.offset(seq_len_start * kv_row * 2),
                ctx.v.offset(seq_len_start * self.kv_lora as usize * 2),
            )),
            slots,
            positions,
            stream,
        )?;
        self.q_absorb(gpu, rows_u32, positions, stream)?;
        self.prefill_attn_contig(gpu, ctx.k, ctx.v, rows_u32, seq_len_start as u32, stream)?;
        self.v_extract(gpu, rows_u32, stream)?;
        proj_gemm(
            gpu,
            k,
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
        let staged = crate::model::graph_flags::ling_meta_staged(ctx.graph_capture);
        self.attention_half_decode(
            ctx.gpu,
            hidden,
            kv_cache,
            seq_len,
            block_table,
            staged,
            stream,
        )?;
        self.ffn_half(ctx.gpu, hidden, 1, stream)
    }

    /// The MTP/K-token verify of one sequence: `num_tokens` consecutive
    /// rows ride the paged kernel's multi-row body (`ling_mla_paged_attn`
    /// serves rows 1..=8 with per-row causal over the just-written block —
    /// PR #112's verify-row support), base_len naming the entries before
    /// the window. A single row runs the plain decode path unchanged.
    /// The K=2 MTP verify arrives here as `num_rows` rows of ONE sequence:
    /// `row_owner` all-zero, one shared layer state. The trait default
    /// would decode each row separately — correct eagerly (each row
    /// uploads its own position), exactly wrong under a captured graph:
    /// the staged arm keys every row to the ROW-0 entries of the
    /// prestaged metadata (`ling_meta_staged` carries no row index), so
    /// draft/bonus rows computed at the row-0 position. That was the
    /// ARM-after replay nondeterminism — the first divergence lands at
    /// the FIRST gated-MLA layer's verify (`ATLAS_LING_CAP_TRACE` staged
    /// dumps read `pos=[base, 1]`: row 1 keyed to warmup-era staging).
    /// Fan the single-owner window through the multi-row arm instead —
    /// the same `decode_batched` route the verify's SSM branch already
    /// takes — so an eager or captured step alike reads the `rows`-wide
    /// prestage Phase 1 wrote (`decode_meta_prestage(rows = k)`):
    /// positions/slots/base_len/block table as a K-row window.
    fn decode_multi_seq_rows<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_rows: usize,
        seq_states: &'a mut [&'b mut (dyn LayerState + 'static)],
        row_owner: &[usize],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if single_owner_window(seq_states.len(), row_owner) {
            let mut block_table = block_tables.first().cloned().unwrap_or_default();
            let mut disk_block_ids = Vec::new();
            let mut disk_last = Vec::new();
            return self.decode_batched(
                hidden,
                residual,
                num_rows,
                seq_states[0],
                kv_cache,
                seq_lens.first().copied().unwrap_or(0),
                &mut block_table,
                &mut disk_block_ids,
                &mut disk_last,
                ctx,
                stream,
            );
        }
        let mut owned: Vec<Box<dyn LayerState>> = (0..num_rows)
            .map(|_| self.alloc_state(ctx.gpu))
            .collect::<Result<_>>()?;
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            owned.iter_mut().map(|s| s.as_mut()).collect();
        self.decode_multi_seq(
            hidden,
            residual,
            num_rows,
            num_rows,
            &mut refs,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

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
            let staged = crate::model::graph_flags::ling_meta_staged(ctx.graph_capture);
            self.attention_half_decode(
                ctx.gpu,
                hidden,
                kv_cache,
                seq_len,
                block_table,
                staged,
                stream,
            )?;
            return self.ffn_half(ctx.gpu, hidden, 1, stream);
        }
        anyhow::ensure!(
            num_tokens <= MLA_MAX_ROWS_PAGED,
            "ling MLA: verify {num_tokens} rows exceeds the paged kernel's {}",
            MLA_MAX_ROWS_PAGED
        );
        let staged = crate::model::graph_flags::ling_meta_staged(ctx.graph_capture);
        self.attention_half_prefill(
            ctx.gpu,
            hidden,
            kv_cache,
            num_tokens,
            seq_len,
            false,
            // The packed path is a prefill shape: a per-verify gather of
            // the prompt prefix would run on every decode step.
            false,
            0, // ctx_tag (packed=false: unused)
            staged,
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
        // Sub-chunk the prompt; every chunk runs the packed context path
        // (the default; over-capacity chunks fall back inside). The legacy
        // behavior — paged 8-row groups plus the ATLAS_LING_MLA_FAST_
        // PREFILL=1 position-0 whole-prompt fast path — is restored with
        // ATLAS_LING_MLA_PACKED_PREFILL=0.
        let packed = packed_prefill();
        let fast = std::env::var("ATLAS_LING_MLA_FAST_PREFILL").as_deref() == Ok("1")
            && seq_len_start == 0;
        let cap = self.scratch.rows;
        // One epoch per TransformerLayer::prefill call: this call's
        // sub-chunks share the tag (the elide stamp recognizes exactly
        // this call's contiguously-written prefix).
        let ctx_tag = self.next_ctx_tag();
        for (n, chunk) in (0..num_tokens).step_by(cap).enumerate() {
            let chunk_rows = (num_tokens - chunk).min(cap);
            self.attention_half_prefill(
                ctx.gpu,
                hidden.offset(chunk * self.scratch.hidden * 2),
                kv_cache,
                chunk_rows,
                seq_len_start + chunk,
                fast && n == 0 && num_tokens <= cap,
                packed,
                ctx_tag,
                false,
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

    /// The Ling family's step-metadata staging (`decode_meta_prestage`):
    /// positions, paged slots, `base_len` and the block table for ONE
    /// sequence's `rows` rows, into the fixed scratch a captured step reads
    /// (`docs/ling-decode-graphs.md`). Called from the dispatch's Phase 1,
    /// pre-graph; single-sequence decode (`rows = 1`) and the K=2 MTP verify
    /// (`rows = k`) are the capture targets.
    fn decode_meta_prestage(
        &self,
        rows: usize,
        seq_len_start: usize,
        block_table: &[u32],
        block_size: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.prestage_meta(gpu, rows, seq_len_start, block_table, block_size, stream)
    }

    /// Graphable under the family switch
    /// (`crate::model::graph_flags::ling_decode_graph`, the graph slice's
    /// flip): the per-step metadata is prestaged pre-graph
    /// (`decode_meta_prestage`, Phase 1 of decode_dispatch and the K=2
    /// verify), the captured arms read the fixed scratch staging, the V
    /// latent pool binds at the first prefill and is allocation-stable per
    /// serve, and `ling_mla_paged_attn` walks `base_len` from a device
    /// word (its `max_blocks_per_seq` is only the `seq` stride, 0 at
    /// single-seq). With the switch off the family stays eager, as before
    /// (the per-seq KDA state no longer blocks it either — the slot pool
    /// fixed that).
    fn decode_graph_unsupported(&self) -> bool {
        !crate::model::graph_flags::ling_decode_graph()
    }

    /// `ATLAS_LING_CAP_TRACE` readset dump: the staged metadata values
    /// (positions/slots/base_len plus a slice of the staged block table,
    /// tail garbage included) and one FNV per physical block of the
    /// layer-owned V-latent pool (block size 64, the served config).
    fn ling_cap_dump_meta(&self, gpu: &dyn GpuBackend, blocks: &[u32], stream: u64, tag: &str) {
        let s = &self.scratch;
        let mut pos = [0u8; 32];
        let mut slots = [0u8; 32];
        let mut base = [0u8; 4];
        let mut tbl = [0u8; 132 * 4];
        let _ = gpu.synchronize(stream);
        let has = gpu.copy_d2h(s.mla_pos, &mut pos).is_ok()
            && gpu.copy_d2h(s.mla_slots, &mut slots).is_ok()
            && gpu.copy_d2h(s.mla_base_len, &mut base).is_ok()
            && gpu.copy_d2h(s.mla_block_table, &mut tbl).is_ok();
        if !has {
            return;
        }
        let posv: Vec<u32> = pos
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let slotv: Vec<i64> = slots
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let tblv: Vec<i32> = tbl
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let basev = u32::from_le_bytes(base);
        let mut vhs = String::new();
        let v_pool = *self.v_pool.lock();
        if let Some(v_pool) = v_pool {
            let bbytes = 64 * MLA_V_ROW_BF16 * 2;
            let mut buf = vec![0u8; bbytes];
            for &b in blocks {
                if gpu
                    .copy_d2h(v_pool.offset((b as usize) * bbytes), &mut buf)
                    .is_err()
                {
                    vhs.push('?');
                    continue;
                }
                if !vhs.is_empty() {
                    vhs.push(',');
                }
                vhs.push_str(&format!(
                    "{}:{:016x}",
                    b,
                    crate::model::trait_impl::ling_cap_trace::ling_cap_fnv1a(&buf)
                ));
            }
        } else {
            vhs.push_str("NOPOOL");
        }
        tracing::warn!(
            "LING_CAP_META[{tag}] pos={:?} slots={:?} base={} tbl[0..10]={:?} vpool=[{}]",
            &posv[..2],
            &slotv[..2],
            basev,
            &tblv[..10],
            vhs
        );
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
mod verify_rows_routing_tests {
    use super::single_owner_window;

    #[test]
    fn mtp_verify_window_is_single_owner() {
        // The K=2 verify: k rows, one shared state, every row owner 0.
        assert!(single_owner_window(1, &[0]));
        assert!(single_owner_window(1, &[0, 0]));
    }

    #[test]
    fn distinct_owners_and_degenerate_windows_are_not() {
        // The a2 multi-request lane names different states per row.
        assert!(!single_owner_window(1, &[0, 1]));
        assert!(!single_owner_window(2, &[0, 1]));
        assert!(!single_owner_window(2, &[0, 0]));
        // No rows to own: keep the default loop.
        assert!(!single_owner_window(1, &[]));
    }

    /// The verify must keep fanning single-owner windows through the
    /// multi-row arm — if this file ever loses the override, the captured
    /// draft rows key to row-0 staging and the replay diverges (see the
    /// override's doc comment).
    #[test]
    fn decode_multi_seq_rows_overrides_the_default_loop() {
        let src = include_str!("mla.rs");
        assert!(src.contains("fn decode_multi_seq_rows"));
        assert!(src.contains("single_owner_window(seq_states.len()"));
        assert!(
            src.contains("return self.decode_batched("),
            "single-owner windows must fan through decode_batched"
        );
    }
}

#[cfg(test)]
#[path = "mla_gpu_parity_tests.rs"]
mod mla_gpu_parity_tests;

#[cfg(test)]
#[path = "e2e_gpu_parity_tests.rs"]
mod e2e_gpu_parity_tests;

#[cfg(test)]
mod ctx_pool_tests {
    use super::ctx_elides_prefix;

    #[test]
    fn elides_exactly_contiguous_same_call() {
        // chunk 0 wrote [0, 512); chunk 1 starts at 512 of the same call.
        assert!(ctx_elides_prefix(7, 512, 7, 512));
        // chunk 2 continues the chain.
        assert!(ctx_elides_prefix(7, 1024, 7, 1024));
    }

    #[test]
    fn gathers_on_tag_mismatch() {
        // A new prefill call (fresh epoch) must re-gather: the mirror may
        // hold a DIFFERENT sequence's rows.
        assert!(!ctx_elides_prefix(7, 512, 8, 512));
        // The pristine stamp (epoch 0) never matches a real call's tag
        // (epochs start at 1).
        assert!(!ctx_elides_prefix(0, 0, 0, 0));
        assert!(!ctx_elides_prefix(0, 512, 0, 512));
    }

    #[test]
    fn gathers_on_non_contiguous_chunk_start() {
        // Same tag but chunk_start past the stamp (a gap — the paged path
        // ran a sub-chunk in between): the mirror does not cover
        // [0, chunk_start) contiguously.
        assert!(!ctx_elides_prefix(7, 512, 7, 1024));
        assert!(!ctx_elides_prefix(7, 1024, 7, 512));
    }
}

#[cfg(test)]
mod proj_gemm_threshold_tests {
    use super::proj_gemm_min_rows_from;

    #[test]
    fn default_is_16_and_zero_restores_the_floor() {
        assert_eq!(proj_gemm_min_rows_from(None), 16);
        assert_eq!(proj_gemm_min_rows_from(Some("0")), u32::MAX);
    }

    #[test]
    fn explicit_values_engage_at_their_row_count() {
        assert_eq!(proj_gemm_min_rows_from(Some("16")), 16);
        assert_eq!(proj_gemm_min_rows_from(Some("3")), 3);
        assert_eq!(proj_gemm_min_rows_from(Some("8192")), 8192);
    }

    #[test]
    fn unparsable_falls_back_to_default() {
        assert_eq!(proj_gemm_min_rows_from(Some("not-a-number")), 16);
    }

    /// The lane must never route back through the heuristic first pick
    /// (the split-K path that corrupted small-M projections, PR #131).
    #[test]
    fn lane_is_kchain_never_the_heuristic_first_pick() {
        let src = include_str!("mla.rs");
        assert!(src.contains("bf16_gemm_act_weight_t_kchain_any"));
        let body = src
            .splitn(2, "fn proj_gemm(")
            .nth(1)
            .unwrap()
            .split("\n}\n")
            .next()
            .unwrap();
        assert!(
            !body.contains("ops::bf16_gemm("),
            "proj_gemm must never call the heuristic first-pick lane"
        );
    }
}
