// SPDX-License-Identifier: AGPL-3.0-only

//! set_down_transpose_scratch.

use super::*;

impl MoeLayer {
    /// Wire a shared per-prefill down_proj scratch + transposed pointer table.
    ///
    /// Called by the factory after the persistent MoE transpose pass falls
    /// back to gate+up only. The scratch and pointer tables are shared
    /// across all MoE layers — one allocation reused layer-by-layer during
    /// the sequential forward. The same `scale2_vals` buffer is reused
    /// from the existing untransposed `down_ptrs` (transpose preserves
    /// per-tensor scales).
    pub fn set_down_transpose_scratch(
        &mut self,
        scratch_packed: DevicePtr,
        scratch_scale: DevicePtr,
        packed_ptrs_t: DevicePtr,
        scale_ptrs_t: DevicePtr,
    ) {
        self.down_t_scratch_packed = Some(scratch_packed);
        self.down_t_scratch_scale = Some(scratch_scale);
        self.down_ptrs_t = Some(ExpertPtrTable {
            packed_ptrs: packed_ptrs_t,
            scale_ptrs: scale_ptrs_t,
            scale2_vals: self.down_ptrs.scale2_vals,
        });
    }

    /// Run the batched transpose kernel to populate `down_t_scratch_*` from
    /// the untransposed `down_ptrs` source. Must be called once at the
    /// start of every layer's prefill, before the silu_down GEMM. No-op
    /// when scratch isn't wired (decode-only / persistent-full-transpose
    /// paths).
    pub(crate) fn transpose_down_into_scratch(
        &self,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(dpt) = self.down_ptrs_t.as_ref() else {
            return Ok(());
        };
        // Only run the transpose when scratch is wired (vs persistent
        // transpose_for_prefill_impl path which sets down_ptrs_t to its
        // own allocations and leaves scratch fields None).
        if self.down_t_scratch_packed.is_none() {
            return Ok(());
        }
        let num_experts = ctx.config.num_experts as u32;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        // Packed: [N=hidden, K/2=inter/2] → [K/2, N] per expert.
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.packed_ptrs,
            dpt.packed_ptrs,
            h,
            inter / 2,
            num_experts,
            stream,
        )?;
        // Scale: [N, K/GROUP_SIZE=inter/16] → [K/16, N] per expert.
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.scale_ptrs,
            dpt.scale_ptrs,
            h,
            inter / 16,
            num_experts,
            stream,
        )?;
        Ok(())
    }

    /// **NOT CURRENTLY WIRED IN** — this helper attempted to overlap the
    /// lazy down_proj transpose with the TP attention allreduce by
    /// kicking it off on `prefill_stream` right after attention. It
    /// regressed cold TTFT by ~30 % on GB10 — both when scheduled
    /// against compute-bound MoE GEMMs AND when scheduled against the
    /// (RDMA-dominated) TP allreduce window. Either GB10's SM scheduling
    /// has hidden contention costs across streams, or the per-call
    /// event-sync overhead exceeds the ~4 ms transpose savings.
    ///
    /// Kept in source for future reference — future work that figures
    /// out the GB10 stream-scheduling pattern can re-wire it from
    /// `qwen3_attention::trait_impl::prefill` after attention but before
    /// the TP allreduce, then have silu_down stall via
    /// `lazy_transpose_done_event()`.
    #[allow(dead_code)]
    pub(crate) fn kick_off_lazy_transpose(
        &self,
        ctx: &crate::layer::ForwardContext,
        compute_stream: u64,
    ) -> Result<()> {
        let Some(dpt) = self.down_ptrs_t.as_ref() else {
            return Ok(());
        };
        if self.down_t_scratch_packed.is_none() {
            return Ok(());
        }
        // prefill_stream waits for compute_stream's "attention done" point.
        ctx.gpu.record_event(self.event_a, compute_stream)?;
        ctx.gpu
            .stream_wait_event(self.prefill_stream, self.event_a)?;

        let num_experts = ctx.config.num_experts as u32;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.packed_ptrs,
            dpt.packed_ptrs,
            h,
            inter / 2,
            num_experts,
            self.prefill_stream,
        )?;
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.scale_ptrs,
            dpt.scale_ptrs,
            h,
            inter / 16,
            num_experts,
            self.prefill_stream,
        )?;
        // Record transpose-done event on the secondary stream. The
        // silu_down call site stalls compute_stream on this event before
        // reading from scratch.
        ctx.gpu.record_event(self.event_b, self.prefill_stream)?;
        Ok(())
    }

    /// Companion to the (currently-unwired) `kick_off_lazy_transpose`
    /// — silu_down would call this to know whether to stall on the
    /// secondary-stream event.
    #[allow(dead_code)]
    pub(crate) fn has_overlapped_transpose(&self) -> bool {
        self.down_t_scratch_packed.is_some()
    }

    /// Companion to `kick_off_lazy_transpose`.
    #[allow(dead_code)]
    pub(crate) fn lazy_transpose_done_event(&self) -> u64 {
        self.event_b
    }

    /// True when prefill dispatch (forward_batched) should route to
    /// `_t` transposed-layout kernels.
    ///
    /// Fires for both unified mode (Phase 8a — originals freed) and hybrid
    /// mode (Block C Path 2 — originals retained alongside transposed).
    /// Both build the same persistent `*_ptrs_t` device-side pointer tables.
    ///
    /// Requires:
    /// 1. `ATLAS_UNIFIED_MOE_LAYOUT=1` OR `ATLAS_HYBRID_MOE_LAYOUT=1`
    ///    (read at construction).
    /// 2. Persistent transposed pointer tables for all three projections.
    /// 3. NOT the lazy-scratch path — scratch-backed `down_ptrs_t` only
    ///    holds one layer at a time, so multi-layer dispatch would read
    ///    stale data. Persistent transpose pass must have populated down_t.
    #[inline]
    pub(crate) fn use_t_layout_for_prefill(&self) -> bool {
        (self.unified_layout || self.hybrid_layout)
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.down_t_scratch_packed.is_none()
    }

    /// True when decode dispatch (forward, forward_k2, forward_k3) should
    /// route to `_t` transposed-layout kernels.
    ///
    /// Only fires in unified mode — hybrid mode keeps the originals so
    /// decode + MTP verify (small N, warp-reduction wins) can preserve
    /// the ~35 tok/s throughput that pure unified layout regresses by 15 %.
    #[inline]
    /// Eligible to route DECODE through the grouped read-once GEMM
    /// (forward_prefill). Native-NVFP4 routed only: bf16/fp8-dequant gate ptrs
    /// absent, no DeepSeek-V4 hash routing, single-GPU (no EP). Deliberately
    /// ALLOWS the mixed BF16 shared expert (forward_prefill handles it as a
    /// separate batched pass), unlike forward_token_major_decode which bails.
    pub(crate) fn grouped_decode_ok(&self) -> bool {
        self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && self.tid2eid_dev.is_none()
    }

    pub(crate) fn use_t_layout_for_decode(&self) -> bool {
        self.unified_layout
            && !self.hybrid_layout
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.down_t_scratch_packed.is_none()
    }
}

// `build_cutlass_grouped_sfb` moved here from helpers_a.rs for the 500-LoC cap.
impl MoeLayer {
    /// Build per-expert swizzled SFB weight-scale tables for the CUTLASS grouped
    /// NVFP4 path (`ATLAS_HOLO_MOE_GROUPED_CUTLASS`). For each expert, swizzle the
    /// `[K/16,N]` `gate_ptrs_t`/`up_ptrs_t` scale into the CUTLASS SFB atom via
    /// `pack_weight_sfb`, then upload the per-expert pointer arrays. The grouped
    /// kernel pairs these with `gate_ptrs.packed` (`[N,K/2]`) + the real per-expert
    /// `scale2`. Requires FAST_MOE=full (gate_ptrs_t/up_ptrs_t present); no-op else.
    pub fn build_cutlass_grouped_sfb(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let num = self.weights.experts.len();
        // Swizzled SFB atom size (bytes): round_up(N,128) * round_up(K/16,4).
        let sfb_len = |n: usize, k: usize| n.div_ceil(128) * 128 * (k / 16).div_ceil(4) * 4;
        // Prefer the Atlas-transposed [K/16,N] scales when they exist. Without
        // them (a checkpoint served straight from its native tables, e.g.
        // Laguna with the unified transpose disabled) fall back to the
        // ORIGINAL [N,K/16] scales and tell the packer to read N-major — the
        // SFB output is identical, so this avoids materialising a transposed
        // copy purely to feed the swizzle.
        let (gate_scale_dev, up_scale_dev, src_n_major) =
            match (self.gate_ptrs_t.as_ref(), self.up_ptrs_t.as_ref()) {
                (Some(g), Some(u)) => (g.scale_ptrs, u.scale_ptrs, false),
                _ => (self.gate_ptrs.scale_ptrs, self.up_ptrs.scale_ptrs, true),
            };
        if gate_scale_dev.is_null() || up_scale_dev.is_null() {
            return Ok(());
        }
        let down_scale_dev = match self.down_ptrs_t.as_ref() {
            Some(d) => Some(d.scale_ptrs),
            None if !self.down_ptrs.scale_ptrs.is_null() => Some(self.down_ptrs.scale_ptrs),
            None => None,
        };
        let mut owned: Vec<DevicePtr> = Vec::new();
        // Swizzle each expert's [K/16,N] scale into the CUTLASS SFB atom. `n`/`k`
        // are the projection's GEMM dims: gate/up = (inter, hidden); down = (hidden, inter).
        // Returns the HOST vector of per-expert SFB pointers: the grouped C entry
        // consumes pointer values host-side, so no device copy of this table is
        // ever made — the values go straight into the layer-owned snapshot below.
        let mut build_one = |scale_ptrs_dev: DevicePtr, n: usize, k: usize| -> Result<Vec<u64>> {
            let len = sfb_len(n, k);
            let sp = crate::layers::ops::read_expert_ptrs_u64(gpu, scale_ptrs_dev, num)?;
            let mut sfb_ptrs = vec![0u64; num];
            for (e, &sptr) in sp.iter().enumerate() {
                if sptr == 0 {
                    continue; // remote/placeholder expert
                }
                let sfb = gpu.alloc(len)?;
                spark_runtime::cutlass::pack_weight_sfb(
                    sptr,
                    sfb.0,
                    n as u32,
                    k as u32,
                    src_n_major,
                    stream,
                )?;
                sfb_ptrs[e] = sfb.0;
                owned.push(sfb);
            }
            gpu.synchronize(stream)?;
            Ok(sfb_ptrs)
        };
        let gate_sfb = build_one(gate_scale_dev, inter, h)?;
        let up_sfb = build_one(up_scale_dev, inter, h)?;
        let down = match down_scale_dev {
            Some(ds) => Some((
                self.down_ptrs.packed_ptrs,
                build_one(ds, h, inter)?,
                self.down_ptrs.scale2_vals,
            )),
            None => None,
        };
        self.cutlass_grouped_host = Some(crate::layers::ops::MoeCutlassHostTables::snapshot(
            gpu,
            num,
            self.gate_ptrs.packed_ptrs,
            gate_sfb,
            self.gate_ptrs.scale2_vals,
            self.up_ptrs.packed_ptrs,
            up_sfb,
            self.up_ptrs.scale2_vals,
            down,
        )?);
        self._cutlass_sfb_owned = owned;
        tracing::info!(
            "CUTLASS grouped SFB: built {num} experts gate/up (N={inter} K={h}) + down (N={h} K={inter})"
        );
        Ok(())
    }
}
