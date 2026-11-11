// SPDX-License-Identifier: AGPL-3.0-only

//! The layer side of `ATLAS_QWEN4EXP_EXACT_DEFER`
//! (`layers/ops/qwen4exp_gdn_defer.rs`): which verifies defer, and the
//! deferred launch itself.
//!
//! Whether a sequence's verify defers is decided ONCE, on the host, before
//! the step (`mark_gdn_deferred_commit` sets `gdn_commit_pending` from
//! [`Qwen3SsmLayer::exact_defer_arm`]), because a captured verify graph
//! replays without running this code. The forward then follows the flag: a
//! pending batch takes the deferred kernel or fails, so a sequence can never
//! be marked pending while the storing kernel ran (its commit would replay
//! from a state that already moved), nor the reverse.

use super::trait_decode_batched::GdnStates;
use super::*;
use crate::layers::ops::qwen4exp_gdn_defer::{
    GdnDeferRows, GdnDeferSeq, defer_kernels, exact_defer_requested, gdn_verify_defer_rows,
};
use crate::layers::ops::qwen4exp_gdn_rows::GDN_VERIFY_KMAX;

fn ssm_ref(state: &dyn LayerState, i: usize) -> Result<&SsmLayerState> {
    state
        .as_any()
        .downcast_ref::<SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))
}

/// `(state, first row, rows)` of every sequence of the step.
fn step_seqs<'s>(
    gdn: &'s GdnStates<'_, '_>,
    num_tokens: usize,
) -> Result<Vec<(&'s SsmLayerState, usize, usize)>> {
    match gdn {
        GdnStates::Single(state) => Ok(vec![(ssm_ref(&**state, 0)?, 0, num_tokens)]),
        GdnStates::Multi { states, ks, .. } => {
            anyhow::ensure!(
                ks.len() == states.len() && ks.iter().sum::<usize>() == num_tokens,
                "deferred GDN verify: num_tokens {num_tokens} != sum of ks {ks:?}"
            );
            let mut row0 = 0;
            let mut out = Vec::with_capacity(ks.len());
            for (i, (state, &k)) in states.iter().zip(ks.iter()).enumerate() {
                out.push((ssm_ref(&**state, i)?, row0, k));
                row0 += k;
            }
            Ok(out)
        }
    }
}

impl Qwen3SsmLayer {
    /// The exact verify of a sequence at `num_tokens` rows defers: the switch,
    /// and every condition the fused-rows exact arm needs
    /// (`gdn_verify_fused_rows`), and the deferred kernels. The per-sequence
    /// staging is checked by the caller (`mark_gdn_deferred_commit`).
    pub(super) fn exact_defer_arm(
        &self,
        gpu: &dyn GpuBackend,
        levers: &crate::layers::ops::ModelLevers,
        config: &atlas_core::config::ModelConfig,
        num_tokens: usize,
    ) -> bool {
        exact_defer_requested() && self.exact_defer_capable(gpu, levers, config, num_tokens)
    }

    /// [`Self::exact_defer_arm`] without the switch: what the forward of a
    /// pending step needs (the flag carries the switch there).
    fn exact_defer_capable(
        &self,
        gpu: &dyn GpuBackend,
        levers: &crate::layers::ops::ModelLevers,
        config: &atlas_core::config::ModelConfig,
        num_tokens: usize,
    ) -> bool {
        levers.qwen4exp_batch_small
            && super::verify_exact_for(levers)
            && self.four_kernel_f32_arm_for(config)
            && (1..=GDN_VERIFY_KMAX).contains(&num_tokens)
            && {
                let (v, c) = defer_kernels(gpu);
                v.0 != 0 && c.0 != 0
            }
    }

    /// `Ok(true)`: every sequence of the step is pending a deferred commit,
    /// `Ok(false)`: none is. A mix is a bug in the marking and an error.
    pub(super) fn gdn_step_pending(gdn: &GdnStates<'_, '_>, num_tokens: usize) -> Result<bool> {
        let seqs = step_seqs(gdn, num_tokens)?;
        let pending = seqs.iter().filter(|(s, ..)| s.gdn_commit_pending).count();
        anyhow::ensure!(
            pending == 0 || pending == seqs.len(),
            "deferred GDN verify: {pending} of {} sequences pending a commit",
            seqs.len()
        );
        Ok(pending > 0)
    }

    /// The deferred exact verify of a pending step. Errors when it cannot
    /// launch: the sequences are pending, so the storing arms must not run.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_verify_deferred(
        &self,
        gdn: &GdnStates<'_, '_>,
        num_tokens: usize,
        qkvz: DevicePtr,
        qkvz_size: usize,
        gates: DevicePtr,
        normed_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.exact_defer_capable(ctx.gpu, ctx.levers, ctx.config, 1),
            "deferred GDN verify: sequences pending but the exact deferred arm is off"
        );
        let seqs: Vec<GdnDeferSeq> = step_seqs(gdn, num_tokens)?
            .into_iter()
            .map(|(s, row0, k)| GdnDeferSeq {
                h: s.h_state,
                conv: s.conv_state,
                stage_qkv: s.gdn_commit_qkv,
                stage_gb: s.gdn_commit_gb,
                row0: row0 as u32,
                k: k as u32,
                fuse_n: s.gdn_fuse_n,
            })
            .collect();
        let c = ctx.config;
        let launched = gdn_verify_defer_rows(
            ctx.gpu,
            &GdnDeferRows {
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
        )?;
        anyhow::ensure!(
            launched,
            "deferred GDN verify: qwen4exp_gdn_verify_defer_rows refused the step \
             (shape, alignment or staging) with its sequences pending"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "gdn_defer_tests.rs"]
mod tests;
