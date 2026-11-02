// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_BATCH_SMALL` (`model/qwen4exp_batch_fast.rs`): the GDN
//! mixer's per-sequence small-kernel chains of a batched step as one launch
//! over the rows, every byte what the chain writes
//! (`scripts/dev/qwen4exp_batch_small_bench.cu`):
//!
//! * batched decode, one token a sequence: BA gates, conv + L2 norm,
//!   recurrence, sigmoid gated norm, four launches a sequence
//!   (`decode_ms_ssm_recurrent`'s loop) -> `qwen4exp_gdn_decode_fused_rows`;
//! * exact MTP verify, `k` tokens a sequence (one or several sequences): the
//!   exact arm's per-token conv, conv rollback copy, recurrence, gated norm
//!   and H rollback copy (`decode_batched_conv_gdn_exact`, up to `5k - 2`
//!   launches a sequence) -> `qwen4exp_gdn_verify_fused_rows`, H and the conv
//!   windows held in registers across the tokens.
//!
//! Both only where the chain is the four-kernel FP32 arm the fused step
//! reproduces ([`Qwen3SsmLayer::four_kernel_f32_arm`]); otherwise nothing is
//! launched and the caller runs its chain.

use super::trait_decode_batched::GdnStates;
use super::*;
use crate::layers::ops::qwen4exp_gdn_rows::{
    GDN_VERIFY_KMAX, GdnDecodeRows, GdnVerifyRows, GdnVerifySeq, gdn_decode_rows, gdn_verify_rows,
};

fn ssm_state(state: &mut dyn LayerState, i: usize) -> Result<&mut SsmLayerState> {
    state
        .as_any_mut()
        .downcast_mut::<SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))
}

impl Qwen3SsmLayer {
    /// The batched multi-sequence decode's per-sequence recurrent inner as
    /// `qwen4exp_gdn_decode_fused_rows`: row `i` of `normed` (BA input) and
    /// of `qkvz` (`qkvz_size` apart) against sequence `i`'s state, gates into
    /// `ssm_gates` rows and the normed output into `normed_out` rows
    /// (`value_dim` apart), as the loop leaves them. Not on the loop's
    /// `ATLAS_GDN_FUSED_CONV` arm. Returns whether it ran.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_decode_fused_rows(
        &self,
        states: &mut [&mut (dyn LayerState + 'static)],
        n: usize,
        normed: DevicePtr,
        qkvz: DevicePtr,
        qkvz_size: usize,
        normed_out: DevicePtr,
        use_fused_conv: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !ctx.levers.qwen4exp_batch_small || use_fused_conv || !self.four_kernel_f32_arm(ctx) {
            return Ok(false);
        }
        let mut seqs = Vec::with_capacity(n);
        for (i, state) in states.iter_mut().enumerate().take(n) {
            let s = ssm_state(&mut **state, i)?;
            seqs.push((s.h_state, s.conv_state));
        }
        let c = ctx.config;
        gdn_decode_rows(
            ctx.gpu,
            &GdnDecodeRows {
                states: &seqs,
                qkvz,
                qkvz_stride: qkvz_size as u32,
                conv_w: self.ssm.conv1d.weight,
                ba_in: normed,
                ba_w: self.ssm.in_proj_ba.weight,
                a_log: self.ssm.a_log.weight,
                dt_bias: self.ssm.dt_bias.weight,
                gates: ctx.buffers.ssm_gates(),
                norm_w: self.ssm.norm.weight,
                out: normed_out,
            },
            c.linear_num_key_heads as u32,
            c.linear_num_value_heads as u32,
            c.linear_key_head_dim as u32,
            c.linear_value_head_dim as u32,
            c.linear_conv_kernel_dim as u32,
            c.hidden_size as u32,
            1e-6,
            c.rms_norm_eps as f32,
            stream,
        )
    }

    /// The exact verify's conv + GDN + norm (phases 5-7 of
    /// `decode_batched_inner`, `verify_exact_for`) for every sequence of the
    /// step as `qwen4exp_gdn_verify_fused_rows`: rows of `qkvz` (`qkvz_size`
    /// apart) and `gates` (`[gate | beta]`), normed rows into `normed_out`
    /// (`value_dim` apart), each sequence's rollback slots for its tokens
    /// `0..k-1`, as the exact arm leaves them. Returns whether it ran; on
    /// `false` (lever off, another arm, a depth past the kernel's, a sequence
    /// without its slots) nothing was launched and the caller's arm runs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_verify_fused_rows(
        &self,
        gdn: &mut GdnStates<'_, '_>,
        num_tokens: usize,
        qkvz: DevicePtr,
        qkvz_size: usize,
        gates: DevicePtr,
        normed_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        // ATLAS_QWEN4EXP_EXACT_DEFER: an exact step marked pending stores no
        // state (`gdn_defer.rs`); it takes the deferred kernel or fails. (The
        // wyN deferral marks the same flag, never under the exact verify.)
        if super::verify_exact_for(ctx.levers) && Self::gdn_step_pending(gdn, num_tokens)? {
            self.gdn_verify_deferred(
                gdn, num_tokens, qkvz, qkvz_size, gates, normed_out, ctx, stream,
            )?;
            return Ok(true);
        }
        if !ctx.levers.qwen4exp_batch_small
            || !super::verify_exact_for(ctx.levers)
            || !self.four_kernel_f32_arm(ctx)
        {
            return Ok(false);
        }
        let mut seqs = Vec::new();
        let mut push = |s: &SsmLayerState, row0: usize, k: usize| {
            if !(1..=GDN_VERIFY_KMAX).contains(&k)
                || s.h_state_intermediates.len() + 1 < k
                || s.conv_state_intermediates.len() + 1 < k
            {
                return false;
            }
            let mut q = GdnVerifySeq {
                h: s.h_state,
                conv: s.conv_state,
                h_snap: [DevicePtr::NULL; GDN_VERIFY_KMAX - 1],
                conv_snap: [DevicePtr::NULL; GDN_VERIFY_KMAX - 1],
                row0: row0 as u32,
                k: k as u32,
            };
            q.h_snap[..k - 1].copy_from_slice(&s.h_state_intermediates[..k - 1]);
            q.conv_snap[..k - 1].copy_from_slice(&s.conv_state_intermediates[..k - 1]);
            seqs.push(q);
            true
        };
        let all = match gdn {
            GdnStates::Single(state) => push(ssm_state(&mut **state, 0)?, 0, num_tokens),
            GdnStates::Multi { states, ks, .. } => {
                anyhow::ensure!(
                    ks.len() == states.len() && ks.iter().sum::<usize>() == num_tokens,
                    "gdn_verify_fused_rows: num_tokens {num_tokens} != sum of ks {ks:?}"
                );
                let mut row0 = 0;
                let mut all = true;
                for (i, (state, &k)) in states.iter_mut().zip(ks.iter()).enumerate() {
                    all &= push(ssm_state(&mut **state, i)?, row0, k);
                    row0 += k;
                }
                all
            }
        };
        if !all {
            return Ok(false);
        }
        let c = ctx.config;
        gdn_verify_rows(
            ctx.gpu,
            &GdnVerifyRows {
                seqs: &seqs,
                qkvz,
                qkvz_stride: qkvz_size as u32,
                conv_w: self.ssm.conv1d.weight,
                gates,
                norm_w: self.ssm.norm.weight,
                out: normed_out,
            },
            c.linear_num_key_heads as u32,
            c.linear_num_value_heads as u32,
            c.linear_key_head_dim as u32,
            c.linear_value_head_dim as u32,
            c.linear_conv_kernel_dim as u32,
            1e-6,
            c.rms_norm_eps as f32,
            stream,
        )
    }
}
