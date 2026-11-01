// SPDX-License-Identifier: AGPL-3.0-only

//! The GDN recurrent state of the qwen4_exp mid-chunk prefix-cache checkpoint
//! (`layers::qwen4exp_ckpt`) on NVIDIA. The conv window rides the existing
//! mid-chunk conv split (`conv1d_prefill_capture`).

use super::*;
use crate::layers::qwen4exp_ckpt as ckpt;

impl Qwen3SsmLayer {
    /// Whether the GDN prefill takes the FLA chunked path (the predicate of
    /// that branch in `prefill_gdn_recurrence`).
    pub(super) fn fla_route(&self, ctx: &ForwardContext, kd: usize, vd: usize) -> bool {
        static NO_FLA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let no_fla =
            *NO_FLA.get_or_init(|| std::env::var("ATLAS_NO_GDN_FLA").as_deref() == Ok("1"));
        !no_fla
            && !ctx.gdn_exact_replay
            && kd == 128
            && vd == 128
            && ctx.buffers.gdn_fla_scratch().0 != 0
            && self.gdn_prefill_fla_recompute_wu_k.0 != 0
            && self.gdn_prefill_fla_chunk_delta_h_k.0 != 0
            && self.gdn_prefill_fla_chunk_fwd_o_k.0 != 0
    }

    /// Capture this layer's recurrent state at the checkpoint rows, when a
    /// checkpoint pass is running:
    /// * FLA (cold passes): arm the state spine's `_cap` twin when the row is
    ///   a 64-token chunk boundary (the `_capn` twin for more than one row);
    ///   the FLA launch takes it. `Ok(false)`.
    /// * Token-sequential warm-replay recurrence (`gdn_regresident`): run it
    ///   up to each row, copy the state out, run on from it -- the state
    ///   chains across a split token by token, as across chunks.
    ///   `Ok(true)`: the recurrence ran here.
    ///
    /// Anything else captures nothing and the model does not register the
    /// checkpoint. `ptrs` = [h_state, q, k, v, gates, out].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_recurrence_ckpt(
        &self,
        ctx: &ForwardContext,
        midcap_idx: Option<usize>,
        ptrs: [DevicePtr; 6],
        rows: u32,
        [nk, nv, kd, vd, conv_dim]: [usize; 5],
        stream: u64,
    ) -> Result<bool> {
        let (Some(cap), Some(idx)) = (ctx.midchunk_capture.as_ref(), midcap_idx) else {
            return Ok(false);
        };
        // Every capture point of the pass (the tail row, dense and
        // branch-point rows), ascending.
        let points = cap.points(idx, rows as usize);
        if !ckpt::active() || points.is_empty() {
            return Ok(false);
        }
        // A stale arm (a spine that did not take it) must not reach this
        // layer's launch.
        let _ = ckpt::take_spine();
        if self.fla_route(ctx, kd, vd) {
            // The opt-in FlashInfer scan runs ahead of FLA and has no twin.
            // All points or none: a partial capture is never registered.
            let on_grid = points.iter().all(|(cl, _)| cl.is_multiple_of(ckpt::CHUNK));
            if on_grid && !ops::gdn_flashinfer::available() {
                let arms: Vec<_> = points
                    .iter()
                    .map(|&(cl, (h, _))| ((cl / ckpt::CHUNK) as u32, h))
                    .collect();
                ckpt::arm_spines(&arms);
            }
            return Ok(false);
        }
        let [h_state, q, k, v, gates, out] = ptrs;
        let (bf16, fp32, gb) = (2usize, 4usize, nv * 2);
        let seg = |start: usize, len: u32| -> Result<()> {
            let gate = gates.offset(start * gb * fp32);
            ops::gdn_prefill_regresident(
                ctx.gpu,
                self.gdn_prefill_regresident_k,
                h_state,
                q.offset(start * conv_dim * bf16),
                k.offset(start * conv_dim * bf16),
                v.offset(start * conv_dim * bf16),
                gate,
                gate.offset(nv * fp32),
                out.offset(start * nv * vd * bf16),
                1,
                len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb as u32,
                stream,
            )
        };
        let mut start = 0usize;
        for (cl, (h_dst, _)) in points {
            seg(start, (cl - start) as u32)?;
            ctx.gpu
                .copy_d2d_async(h_state, h_dst, cap.h_bytes, stream)?;
            start = cl;
        }
        ckpt::h_captured();
        seg(start, rows - start as u32)?;
        Ok(true)
    }
}
