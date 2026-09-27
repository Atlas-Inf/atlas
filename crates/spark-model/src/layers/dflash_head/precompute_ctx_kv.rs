// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash Option B — once-per-propose precompute of drafter ctx K/V.
//!
//! Lifts ctx-side work out of the γ-block layer loop so the per-layer
//! attention path runs over `q_len = γ` rows instead of `n_attn = γ + ctx`.
//! The drafter's paged BF16 KV cache holds the ctx K/V across propose
//! calls; this module appends the *new* ctx slots (delta since the last
//! propose) once, fused across all L drafter layers.
//!
//! Pipeline per call — matches vLLM `precompute_and_store_context_kv`
//! (`qwen3_dflash.py:342-434`) op-for-op:
//!   1. Batched `fc` projection: `[n, L_t * h_t] → [n, h]`.
//!   2. `hidden_norm` (RMS) over `[n, h]`.
//!   3. Single fused KV GEMM: `[n, h] × [h, L * 2 * kv_dim]
//!      → [n, L * 2 * kv_dim]`.  (py:381–392)
//!   4. Compact all L layers' K into contiguous `[L*n, kv_dim]` staging.
//!      (py:386–391 permute+contiguous)
//!   5. Per-layer k_norm over `[n, kv_dim]` blocks.  (py:393–401)
//!   6. **Single** fused RoPE over `[L*n, kv_dim]` at repeated positions.
//!      (py:403–418 — one `ops.rotary_embedding` call for all layers)
//!   7. Per-layer `reshape_and_cache` writing K/V into the layer's paged
//!      cache at the appropriate slot mapping.  (py:420–434)
//!
//! Scratch buffers borrowed (all available before the γ-block layer loop):
//!   `mlp_intermediate` → all_k_stage `[L*n, kv_dim]` BF16
//!   `norm_buf`         → extended_positions `[L*n]` i32 (first L*n*4 bytes)
//!   `k_buf` / `v_buf`  → per-layer K/V staging for cache write

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

use super::{BlockDiffusionDraftHead, DflashScratch};
use crate::layer::ForwardContext;
use crate::weight_map::DenseWeight;

impl BlockDiffusionDraftHead {
    /// Project new ctx hidden states through `fc + hidden_norm`, derive
    /// fused K/V across all drafter layers, apply per-layer k_norm + fused
    /// RoPE, and write the results into the per-layer paged KV cache slots.
    ///
    /// Mirrors `precompute_and_store_context_kv` (qwen3_dflash.py:342).
    ///
    /// `ctx_base_ptr`: base of the captured-target-hidden accumulator
    ///   (`[max_ctx_len, L_t * h_t]` BF16).
    /// `start_slot`: first ctx slot to project (inclusive).
    /// `new_ctx_count`: number of contiguous ctx slots starting at
    ///   `start_slot` to feed through.
    /// `slot_positions`: `&[i32]` of length `new_ctx_count` — the TRUE
    ///   fixed RoPE position of each row being computed (stamped at append
    ///   time, vLLM convention).
    /// `slot_mapping_dev`: device pointer to an `i32[new_ctx_count]`
    ///   array of paged-cache slot indices.
    /// `commit`: when `true`, write the computed K/V into the paged cache.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn precompute_ctx_kv(
        &self,
        ctx_base_ptr: DevicePtr,
        start_slot: usize,
        new_ctx_count: usize,
        slot_positions: &[i32],
        slot_mapping_dev: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        commit: bool,
        scratch: &DflashScratch,
    ) -> Result<()> {
        if new_ctx_count == 0 {
            return Ok(());
        }

        if self.fused_kv_weight.is_none() {
            anyhow::bail!(
                "DFlash precompute_ctx_kv called without fused_kv_weight — build-order bug"
            );
        }

        let gpu = ctx.gpu;
        let h = self.hidden_size as u32;
        let n = new_ctx_count as u32;
        let target_hidden_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let ctx_slot_bytes = target_hidden_dim * 2;

        // One-shot diagnostic dump (ATLAS_DFLASH_PRECOMPUTE_DUMP=1).
        // Per-model latch (see `ModelStats::dumped`) rather than a static: an
        // operator who sets the flag and then swaps models must still get the
        // dump, instead of it being swallowed by the previous model's shot.
        let dump =
            self.startup.diagnostics.precompute_dump && ctx.stats.dumped.keyed("dflash_precompute");
        let dump_buf = |label: &str, ptr: DevicePtr, bytes: usize| -> Result<()> {
            if !dump {
                return Ok(());
            }
            let mut buf = vec![0u8; bytes];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let path = format!("/tmp/atlas_precompute_{label}.bin");
            if let Err(e) = std::fs::write(&path, &buf) {
                tracing::warn!("precompute dump {label} write failed: {e}");
            } else {
                tracing::info!("precompute dump {label}: {} bytes → {}", bytes, path);
            }
            Ok(())
        };

        // ── Steps 1–3: fc projection + hidden_norm + fused KV GEMM ────
        let src = ctx_base_ptr.offset(start_slot * ctx_slot_bytes);
        self.ctx_kv_project(
            gpu,
            src,
            n,
            h,
            target_hidden_dim as u32,
            scratch.fc_proj,
            scratch.fused_kv_out,
            stream,
            Some(&dump_buf),
        )?;
        // ── Steps 4–8: per-sequence scatter (positions, compaction,
        // k_norm, rope, reshape_and_cache).
        self.ctx_kv_scatter(
            gpu,
            scratch.fused_kv_out,
            new_ctx_count,
            slot_positions,
            slot_mapping_dev,
            ctx,
            stream,
            commit,
            scratch,
            Some(&dump_buf),
        )
    }

    /// Steps 1–3 of `precompute_ctx_kv`: fc → hidden_norm → fused KV GEMM
    /// over `n` rows starting at `src`, outputs into the caller's buffers.
    /// Factored out so the batched propose can run ONE call over the
    /// concatenated Σn_i rows of all sequences (#58/#116).
    ///
    /// **Row identity:** every arm inside is row-independent — rms_norm
    /// reads only the row it writes, and `drafter_dense_gemm` resolves to
    /// `dense_gemm_bf16_pipelined` (mma m16n8k16, K-dim accumulation only)
    /// for every M above the GEMV bound, so row i's bytes are identical
    /// to the n_i-row launch the serial path would run. The batch planner
    /// excludes chunks whose per-seq arm would be the GEMV (small-m
    /// lever on AND n_i ≤ DENSE_GEMV_BATCHM_MAX_M), so included rows
    /// always compare equal on the same kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ctx_kv_project(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        n: u32,
        h: u32,
        target_hidden_dim: u32,
        fc_dst: DevicePtr,
        fused_dst: DevicePtr,
        stream: u64,
        dump: Option<&dyn Fn(&str, DevicePtr, usize) -> Result<()>>,
    ) -> Result<()> {
        let Some(fused_kv) = self.fused_kv_weight else {
            anyhow::bail!("DFlash ctx_kv_project called without fused_kv_weight — build-order bug");
        };
        let bf16 = 2usize;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let l_total = self.num_layers;

        // Step 1: fc projection [n, L_t*h_t] → [n, h].
        self.drafter_dense_gemm(gpu, src, &self.fc, fc_dst, n, h, target_hidden_dim, stream)?;
        if let Some(d) = dump {
            d("fc_proj", fc_dst, n as usize * h as usize * bf16)?;
        }
        // Step 2: hidden_norm RMS in-place.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            fc_dst,
            &self.hidden_norm,
            fc_dst,
            n,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        if let Some(d) = dump {
            d("fc_proj_normed", fc_dst, n as usize * h as usize * bf16)?;
        }
        // Step 3: fused KV GEMM → [n, L * 2 * kv_dim], row layout
        // [K_0 | V_0 | K_1 | V_1 | …].
        let fused_w = DenseWeight { weight: fused_kv };
        self.drafter_dense_gemm(
            gpu,
            fc_dst,
            &fused_w,
            fused_dst,
            n,
            (l_total as u32) * 2 * kv_dim,
            h,
            stream,
        )?;
        if let Some(d) = dump {
            d(
                "fused_kv_out",
                fused_dst,
                n as usize * l_total * 2 * kv_dim as usize * bf16,
            )?;
        }
        Ok(())
    }

    /// Steps 4–8 of `precompute_ctx_kv`: repeated positions, K/V
    /// compaction, per-layer k_norm, fused rope, and `reshape_and_cache`
    /// into the drafter paged cache — the per-sequence stage of the
    /// batched-ctx flow. `fused_src` points at this sequence's rows in
    /// the fused GEMM output (own scratch serially, the batch buffer
    /// when batched).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ctx_kv_scatter(
        &self,
        gpu: &dyn GpuBackend,
        fused_src: DevicePtr,
        new_ctx_count: usize,
        slot_positions: &[i32],
        slot_mapping_dev: DevicePtr,
        _ctx: &ForwardContext,
        stream: u64,
        commit: bool,
        scratch: &DflashScratch,
        dump_opt: Option<&dyn Fn(&str, DevicePtr, usize) -> Result<()>>,
    ) -> Result<()> {
        let bf16 = 2usize;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let n = new_ctx_count as u32;
        let l_total = self.num_layers;
        let kv_slab_bytes = (kv_dim as usize) * bf16;
        let row_stride = l_total * 2 * kv_slab_bytes;
        let dump = |label: &str, ptr: DevicePtr, bytes: usize| -> Result<()> {
            match dump_opt {
                Some(d) => d(label, ptr, bytes),
                None => Ok(()),
            }
        };

        // ── Step 4: build extended position array for fused RoPE ─────
        // py:407  `positions_repeated = context_positions.repeat(L)`
        //   = slot_positions ×L = [p0..p_{n-1}, p0..p_{n-1}, …] (L copies).
        // Stored in norm_buf (first L*n*4 bytes; norm_buf = 2 MB >>  this).
        // norm_buf is not needed until step 3a of the γ-block layer loop
        // (forward_block_layer_pre_attn), which runs after precompute returns.
        debug_assert_eq!(slot_positions.len(), new_ctx_count);
        {
            // #58: stage into the lane's pinned region and ship with
            // copy_h2d_async_retained — the old pageable copy_h2d was a
            // per-sequence cuStreamSynchronize (a ~full drain per call).
            // Overflow of the carve region falls back to that same sync
            // path — correctness identical, only the drain returns.
            let bytes = l_total * new_ctx_count * 4;
            match Self::ctx_positions_region(scratch, bytes) {
                Some(region) => {
                    // SAFETY: `ctx_positions_region` returned `bytes` of the
                    // page-locked buffer; regions are exclusive per carve so
                    // no other writer races this slice before the next
                    // stream sync (the step's readback `synchronize`).
                    let staging = unsafe { std::slice::from_raw_parts_mut(region, bytes) };
                    for (chunk, p) in staging
                        .chunks_mut(4)
                        .zip(slot_positions.iter().cloned().cycle())
                    {
                        chunk.copy_from_slice(&p.to_le_bytes());
                    }
                    gpu.copy_h2d_async_retained(staging, scratch.norm_buf, stream)?;
                }
                None => {
                    let repeated_bytes: Vec<u8> = slot_positions
                        .iter()
                        .cloned()
                        .cycle()
                        .take(l_total * new_ctx_count)
                        .flat_map(|p: i32| p.to_le_bytes())
                        .collect();
                    gpu.copy_h2d(&repeated_bytes, scratch.norm_buf)?;
                }
            }
        }

        // ── Step 5: compact all L layers' K → all_k_stage ────────────
        // py:386–391  `all_kv = all_kv_flat.view(n,L,2,nkv,hd)
        //                          .permute(2,1,0,3,4).contiguous()`
        //              `all_k = all_kv[0]`  → [L, n, nkv, hd] contiguous.
        // Atlas: copy_d2d row-by-row to build the same [L, n, kv_dim] layout
        // in mlp_intermediate (borrowed; not used until step 3j of the
        // γ-block layer loop). Capacity: n_attn × inter × 2 >> L×n×kv_dim×2.
        let all_k_stage = scratch.mlp_intermediate;
        for l in 0..l_total {
            for row in 0..new_ctx_count {
                let k_src = fused_src.offset(row * row_stride + l * 2 * kv_slab_bytes);
                let k_dst = all_k_stage.offset((l * new_ctx_count + row) * kv_slab_bytes);
                // #58: stream-ordered async — downstream k_norm/rope/
                // reshape_and_cache run on this same stream, so the drain
                // the sync copy_d2d paid bought nothing.
                gpu.copy_d2d_async(k_src, k_dst, kv_slab_bytes, stream)?;
            }
        }

        // ── Step 6: per-layer k_norm ──────────────────────────────────
        // py:393–401  `for i in range(L): ops.rms_norm(all_k_normed[i],
        //               all_k[i], self._k_norm_weights[i], eps)`
        // Each block: all_k_stage[l*n .. (l+1)*n] shape [n, kv_dim]
        //   → treated as [n * num_kv_heads, head_dim] for per-head norm.
        for l in 0..l_total {
            let k_l = all_k_stage.offset(l * new_ctx_count * kv_slab_bytes);
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                k_l,
                &self.layers[l].k_norm,
                k_l,
                n * self.num_kv_heads as u32,
                self.head_dim as u32,
                self.rms_norm_eps,
                stream,
            )?;
        }

        // ── Step 7: single fused RoPE across all L layers ─────────────
        // py:403–418  `all_k_flat = all_k_normed.view(L * n, kv)`
        //              `ops.rotary_embedding(positions_repeated, all_k_flat,
        //                None, head_size, cos_sin_cache, is_neox)`
        // Atlas: rope_yarn with seq_len=L*n, num_q_heads=0 (K-only).
        //   K buffer = all_k_stage[0..L*n*kv_dim].
        //   positions = norm_buf (L*n i32 written in step 4).
        // Grid: [num_kv_heads, ceil(L*n / pos_per_block), 1] — all CTAs
        //   process K heads across the full L*n row range.
        ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            all_k_stage, // Q — unread when num_q_heads=0
            all_k_stage, // K = all_k_flat [L*n, kv_dim]
            scratch.norm_buf,
            l_total as u32 * n,
            0, // num_q_heads=0 → K-only (rope.cu:46-48)
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )?;

        if dump_opt.is_some() {
            dump(
                "layer0_k_post_rope",
                all_k_stage,
                new_ctx_count * kv_slab_bytes,
            )?;
        }

        // ── Step 8: per-layer V compaction + reshape_and_cache ────────
        // py:420–434  per-layer `attn.impl.do_kv_cache_update(...)`.
        // Compact V_l inline (no norm/RoPE applied to V — oracle matches).
        // K is read from all_k_stage[l*n..]; V from fused_kv_out (GEMM output).
        let v_stage = scratch.v_buf;
        for l in 0..l_total {
            let k_l = all_k_stage.offset(l * new_ctx_count * kv_slab_bytes);

            // Compact V_l from the fused GEMM output.
            for row in 0..new_ctx_count {
                let v_src =
                    fused_src.offset(row * row_stride + l * 2 * kv_slab_bytes + kv_slab_bytes);
                let v_dst = v_stage.offset(row * kv_slab_bytes);
                // #58: stream-ordered async — reshape_and_cache consumes
                // v_stage on this same stream.
                gpu.copy_d2d_async(v_src, v_dst, kv_slab_bytes, stream)?;
            }

            if dump_opt.is_some() && l == 0 {
                dump("layer0_v", v_stage, new_ctx_count * kv_slab_bytes)?;
            }

            if commit {
                let (k_pool, v_pool) = {
                    let cache = self.kv_cache.lock();
                    (cache.k_pool_ptr(l), cache.v_pool_ptr(l))
                };
                ops::reshape_and_cache(
                    gpu,
                    self.kernels.reshape_cache_bf16,
                    k_l,
                    v_stage,
                    k_pool,
                    v_pool,
                    slot_mapping_dev,
                    n,
                    self.num_kv_heads as u32,
                    self.head_dim as u32,
                    16, // block_size — matches from_weights.rs
                    kv_dim,
                    kv_dim,
                    0,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}

impl BlockDiffusionDraftHead {
    /// #58: carve `bytes` from the lane scratch's pinned position-staging
    /// region. Returns the region start pointer, or `None` on overflow —
    /// callers fall back to the synchronous `copy_h2d` in that case.
    ///
    /// The cursor is reset at each public propose entry (`propose_drafts`,
    /// `propose_on_lanes`, `propose_batch`). Reuse across calls is safe
    /// because every propose ends in the step's readback
    /// (`synchronize` + `copy_d2h` of the drafted tokens), which completes
    /// all enqueued `copy_h2d_async_retained` copies before the next call
    /// overwrites the region.
    pub(super) fn ctx_positions_region(scratch: &DflashScratch, bytes: usize) -> Option<*mut u8> {
        let base = scratch
            .ctx_positions_host_pinned
            .load(std::sync::atomic::Ordering::Relaxed);
        if base.is_null() {
            return None;
        }
        carve_region(
            &scratch.ctx_positions_cursor,
            scratch.ctx_positions_pinned_bytes,
            bytes,
        )
        .map(|off| unsafe { base.add(off) })
    }

    /// Reset this lane scratch's carve cursor — call at each public
    /// propose entry (see `ctx_positions_region` for the lifetime rule).
    pub(super) fn ctx_positions_reset(scratch: &DflashScratch) {
        scratch
            .ctx_positions_cursor
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Pure bookkeeping for the pinned carve: returns the offset of a fresh
/// `bytes`-sized region, or `None` when the buffer is exhausted. Atomic so
/// propose lanes sharing a scratch's staging can carve without a lock.
pub(super) fn carve_region(
    cursor: &std::sync::atomic::AtomicUsize,
    capacity: usize,
    bytes: usize,
) -> Option<usize> {
    let mut cur = cursor.load(std::sync::atomic::Ordering::Relaxed);
    loop {
        let next = cur.checked_add(bytes)?;
        if next > capacity {
            return None;
        }
        match cursor.compare_exchange_weak(
            cur,
            next,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        ) {
            Ok(_) => return Some(cur),
            Err(c) => cur = c,
        }
    }
}
