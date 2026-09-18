// SPDX-License-Identifier: AGPL-3.0-only

//! FLA chunked GDN prefill dispatch for the `atlas_scale` (gfx1151) target.
//!
//! Hoisted so `trait_prefill_recur.rs` / `trait_prefill_gdn.rs` stay under the
//! 500-LoC cap — both `cfg!(atlas_scale)` arms call this one gate + launch.
//! The ported kernels live in `kernels/strix-hip/common/gated_delta_rule_fla.cu`
//! (scalar Gram / register-S spine / slim fwd_o — all under the 64KB LDS cap).

use super::*;

impl Qwen3SsmLayer {
    /// Try the FLA chunked path (recompute_wu → chunk_delta_h_vfused → fwd_o).
    /// `Ok(true)` = dispatched; `Ok(false)` = a gate missed, caller falls back
    /// to split4. Gates: not exact-replay (Marconi restores keep split4's
    /// token-sequential scan — FLA's 64-token regrouping drifts vs a
    /// snapshot-anchored pass), 128-dim heads, allocated scratch, and the three
    /// ported kernels resolved. The spine handle required is the FUSED one —
    /// on gfx1151 the ksplit fallback's ~99KB double-buffer cannot launch under
    /// the LDS cap, so `use_fused` is unconditional in `ops::gdn_prefill_fla`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fla_gdn_prefill_atlas_scale(
        &self,
        ctx: &ForwardContext,
        h_state: DevicePtr,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gate_ptr: DevicePtr,
        beta_ptr: DevicePtr,
        out_ptr: DevicePtr,
        total: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        conv_dim: usize,
        gb_stride: u32,
        stream: u64,
    ) -> Result<bool> {
        // Runtime off-switch for the new path: ATLAS_GDN_FLA_GFX=0 keeps the
        // proven split4 kernel. A bring-up on real silicon needs a same-binary
        // A/B; a recompile cannot distinguish kernel from harness.
        if std::env::var("ATLAS_GDN_FLA_GFX").ok().as_deref() == Some("0") {
            return Ok(false);
        }
        let fla_scratch = ctx.buffers.gdn_fla_scratch();
        if ctx.gdn_exact_replay
            || kd != 128
            || vd != 128
            || fla_scratch.0 == 0
            || self.gdn_prefill_fla_recompute_wu_k.0 == 0
            || self.gdn_prefill_fla_chunk_delta_h_fused_k.0 == 0
            || self.gdn_prefill_fla_chunk_fwd_o_k.0 == 0
        {
            return Ok(false);
        }
        // One-time positive signal — the spine name is the most consequential
        // fact in a GDN measurement (see init.rs); a silent fallback here would
        // make an A/B unfalsifiable.
        if ctx.stats.once("log:gdn_fla_chunked") {
            tracing::info!(
                "GDN prefill: FLA chunked path ACTIVE (atlas_scale port: recompute_wu → chunk_delta_h_vfused → chunk_fwd_o)"
            );
        }
        let num_chunks = total.div_ceil(64);
        let nt = num_chunks as usize;
        let w_out = fla_scratch;
        let u_out = w_out.offset(nt * nv * 64 * kd * 2);
        let s_out = u_out.offset(nt * nv * 64 * vd * 2);
        let uc_out = s_out.offset(nt * nv * kd * vd * 2);
        let gc_out = uc_out.offset(nt * nv * 64 * vd * 2);
        ops::gdn_prefill_fla(
            ctx.gpu,
            self.gdn_prefill_fla_recompute_wu_k,
            self.gdn_prefill_fla_chunk_delta_h_k,
            self.gdn_prefill_fla_chunk_delta_h_tc_vblock_k,
            self.gdn_prefill_fla_chunk_delta_h_fused_k,
            self.gdn_prefill_fla_chunk_delta_h_tma_k,
            self.gdn_prefill_fla_chunk_fwd_o_k,
            h_state,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            out_ptr,
            w_out,
            u_out,
            s_out,
            uc_out,
            gc_out,
            1,
            total,
            num_chunks,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32,
            conv_dim as u32,
            gb_stride,
            false, // single-stream: contiguous h_state (not a pointer table)
            spark_runtime::gpu::DevicePtr::NULL, // cu_seqlens (unused)
            spark_runtime::gpu::DevicePtr::NULL, // cu_chunks (unused)
            false, // not varlen
            ctx.profile,
            stream,
        )?;
        Ok(true)
    }
}
