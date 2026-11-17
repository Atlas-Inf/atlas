// SPDX-License-Identifier: AGPL-3.0-only

//! gfx1151 (`atlas_scale`) mid-chunk prefix-cache capture on the FLA spine.
//!
//! A capture pass used to run the whole GDN recurrence as split4 segments so it
//! could copy `h_state` out at the capture row (`prefill_gdn_recurrence`).
//! When that row sits on the 64-token chunk grid and is the pass's only point,
//! the cold pass's own FLA spine can store it instead: its `_vfused_cap` twin
//! writes chunk `row / 64`'s FP32 entry state into the same slot. Strix, six 2k
//! prompts: split4 3.4 s vs the FLA spine ~1.75 s.

use super::*;
use crate::layers::qwen4exp_ckpt as ckpt;

impl Qwen3SsmLayer {
    /// `Ok(true)`: the pass ran on FLA with the capture stored; `Ok(false)`:
    /// nothing launched (no capture here, off the chunk grid, two points, the
    /// twin missing, or FLA declined) and the caller's split4 capture runs.
    /// `ptrs` = [h_state, q, k, v, gates, out].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fla_midchunk_capture(
        &self,
        ctx: &ForwardContext,
        midcap_idx: Option<usize>,
        ptrs: [DevicePtr; 6],
        k: u32,
        [nk, nv, kd, vd, conv_dim]: [usize; 5],
        gb_stride: u32,
        stream: u64,
    ) -> Result<bool> {
        let (Some(cap), Some(idx)) = (ctx.midchunk_capture.as_ref(), midcap_idx) else {
            return Ok(false);
        };
        let cl = cap.cap_local;
        if cl == 0
            || cl as u32 >= k
            || cl % ckpt::CHUNK != 0
            || cap.cap_local_early.is_some()
            // The pipe spine has no gfx1151 twin.
            || std::env::var("ATLAS_GDN_PIPE").as_deref() == Ok("1")
            || crate::layers::try_kernel(
                ctx.gpu,
                "gated_delta_rule_fla",
                "gated_delta_rule_chunk_delta_h_vfused_cap",
            )
            .0 == 0
        {
            return Ok(false);
        }
        let [h_state, q, kp, v, gates, out] = ptrs;
        ckpt::arm_spines(&[((cl / ckpt::CHUNK) as u32, cap.h_dsts[idx])]);
        let ran = self.fla_gdn_prefill_atlas_scale(
            ctx,
            h_state,
            q,
            kp,
            v,
            gates,
            gates.offset(nv * 4),
            out,
            k,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            gb_stride,
            stream,
        )?;
        if !ran {
            // A stale arm must not reach a later launch.
            let _ = ckpt::take_spine();
        }
        Ok(ran)
    }
}
