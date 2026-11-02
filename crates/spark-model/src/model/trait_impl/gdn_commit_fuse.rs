// SPDX-License-Identifier: AGPL-3.0-only

//! The deferred GDN commit fused into the next verify
//! (`ATLAS_QWEN4EXP_GDN_COMMIT_FUSE=1`, default off; needs
//! `ATLAS_QWEN4EXP_EXACT_DEFER`).
//!
//! Under the exact deferral (`layers/ops/qwen4exp_gdn_defer.rs`) a verify
//! reads H0 and stages its rows' inputs, and after the verdict
//! `qwen4exp_gdn_commit_layers` replays the accepted prefix from H0: a second
//! read of every sequence's state and its write, ~4.5 ms a C8 step on the
//! secondary stream, where it takes the SMs from the next propose (nsys,
//! 2026-10-07). Fused, the verdict launches nothing: it leaves the commit
//! PENDING (`n` accepted tokens, the layers' state and staging), and the
//! sequence's next deferred verify replays those `n` tokens first, stores the
//! committed state once and verifies its own rows from the registers
//! (`qwen4exp_gdn_verify_defer_rows`, `fuse_n`). The replay is the commit
//! kernel's loop verbatim on the same inputs, so the stored bytes are the
//! commit's and the verify reads the floats it would have reloaded:
//! `scripts/dev/qwen4exp_gdn_defer_bench.cu` (defer-check) holds both.
//!
//! The kernel learns `n` from the slot's pending-commit word
//! (`SsmStatePool::fuse_word`, a pointer the verify graphs bake). Every
//! deferred verify sets it ([`TransformerModel::gdn_fuse_arm`], at the
//! dispatch entry where the pending flags are set): `n` when the step defers
//! exactly the layers the pending commit holds, else the commit runs on its
//! own first and the word is 0.
//!
//! Anything else that reads or writes a pending sequence's GDN state first
//! FLUSHES (the standalone commit, then a stream sync so any stream sees the
//! state): every forward that is not a deferred verify (prefill, decode,
//! mixed, rollback, checkpoints), the snapshot / leaf / prefix-cache saves
//! and slot moves (`sync_secondary_dispatch`, the point they already order
//! after the commit at), state save/restore, and on the EP worker every
//! command but the batched verify and the draft-assist walk. A freed
//! sequence's pending commit is dropped. Ranks decide independently: GDN
//! state is rank-local and both forms land the same bytes.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use atlas_core::config::LayerType;
use parking_lot::Mutex;

use super::super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::layers::ops::qwen4exp_gdn_defer::GdnCommitLayer;
use crate::traits::SequenceState;

/// `ATLAS_QWEN4EXP_GDN_COMMIT_FUSE=1`, read once. Off under the serial
/// checks (`ATLAS_QWEN4EXP_EXACT_VERIFY_CHECK`, `..._BATCH_FAST_CHECK`): they
/// decode a verify's rows from the live state inside its dispatch, after the
/// pending commit was handed to the verify, and the fused verify is not
/// idempotent (it stores the commit).
pub(crate) fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4EXP_GDN_COMMIT_FUSE").as_deref() == Ok("1");
        let checks = crate::model::qwen4exp_exact_verify::check_requested()
            || crate::model::qwen4exp_batch_fast::check_requested();
        if on && checks {
            tracing::warn!("ATLAS_QWEN4EXP_GDN_COMMIT_FUSE off: a serial verify check is on");
        }
        on && !checks
    })
}

/// A slot's commit left for its next verify.
struct Pending {
    n: usize,
    /// The model layer indices it commits, ascending.
    idx: Vec<usize>,
    layers: Vec<GdnCommitLayer>,
}

/// The model's pending commits, by SSM pool slot.
#[derive(Default)]
pub(crate) struct GdnFuse {
    pending: Mutex<HashMap<usize, Pending>>,
    fused: AtomicU64,
    flushed: AtomicU64,
}

impl GdnFuse {
    fn count(&self, fused: bool) {
        let (f, s) = if fused {
            (
                self.fused.fetch_add(1, Ordering::Relaxed) + 1,
                self.flushed.load(Ordering::Relaxed),
            )
        } else {
            (
                self.fused.load(Ordering::Relaxed),
                self.flushed.fetch_add(1, Ordering::Relaxed) + 1,
            )
        };
        if (f + s).is_power_of_two() || (f + s).is_multiple_of(4096) {
            tracing::info!(
                "qwen4_exp GDN commit fuse: {f} commits fused into a verify, {s} flushed"
            );
        }
    }
}

/// What a pending commit must match to fuse: the layers the step defers.
fn fuses(pending_idx: &[usize], deferred_idx: &[usize], exact: bool) -> bool {
    exact && !deferred_idx.is_empty() && pending_idx == deferred_idx
}

impl TransformerModel {
    /// Set `gdn_commit_pending` on every GDN layer state of `seq` for a
    /// verify about to run at `num_tokens` rows. MUST be called at the
    /// `decode_verify*` dispatch entry — outside any `begin_capture` —
    /// because on graph replay the layer's forward host code never runs.
    /// The predicate is the same one the layer uses to pick the `_defer`
    /// kernel (`TransformerLayer::gdn_deferred_wyn`), plus the wyN arm's
    /// per-sequence intermediates-layout precondition, so the flag can
    /// never disagree with the kernel the capture baked.
    pub(super) fn mark_gdn_deferred_commit(
        &self,
        seq: &mut SequenceState,
        num_tokens: usize,
    ) -> Result<()> {
        if !self.levers.gdn_deferred_commit {
            return Ok(());
        }
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let defer = self.layers[i].gdn_deferred_wyn(&self.levers, num_tokens);
            // ATLAS_QWEN4EXP_EXACT_DEFER rides the same staging pools.
            let exact = self.layers[i].gdn_exact_defer(
                self.gpu.as_ref(),
                &self.levers,
                &self.config,
                num_tokens,
            );
            if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                ssm.gdn_commit_pending = (defer
                    && ssm.h_inter_pool_layout(num_tokens, self.ssm_pool.h_stored_bytes))
                    || (exact && !ssm.gdn_commit_qkv.is_null() && !ssm.gdn_commit_gb.is_null());
            }
        }
        // ATLAS_QWEN4EXP_GDN_COMMIT_FUSE: this verify's pending-commit word.
        self.gdn_fuse_arm(seq)
    }

    /// The verdict's exact commit of `n` tokens over `layers` (model indices
    /// `idx`) of `seq`: left pending when the fuse is on and every layer has
    /// its slot word, else launched now on `stream` as before.
    pub(super) fn gdn_commit_or_defer(
        &self,
        seq: &SequenceState,
        idx: Vec<usize>,
        layers: Vec<GdnCommitLayer>,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        let worded = idx.iter().all(|&i| {
            seq.layer_states[i]
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .is_some_and(|s| !s.gdn_fuse_n.is_null())
        });
        if !(requested() && worded) || layers.is_empty() {
            return self.commit_gdn_exact(&layers, n, stream);
        }
        let mut map = self.gdn_fuse.pending.lock();
        // A verify consumes its pending commit before its verdict; a stale
        // one here would be overwritten, so land it first.
        if let Some(old) = map.remove(&seq.slot_idx) {
            self.gdn_fuse_land(&old)?;
        }
        map.insert(seq.slot_idx, Pending { n, idx, layers });
        Ok(())
    }

    /// At a verify's dispatch entry, after `mark_gdn_deferred_commit` set the
    /// pending flags: hand the slot's pending commit to this verify when it
    /// defers exactly those layers, else land it first; set the slot word.
    pub(super) fn gdn_fuse_arm(&self, seq: &SequenceState) -> Result<()> {
        if !requested() {
            return Ok(());
        }
        let mut word = None;
        let mut deferred = Vec::new();
        for (i, state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            if let Some(s) = state.as_any().downcast_ref::<SsmLayerState>()
                && s.gdn_commit_pending
                && !s.gdn_fuse_n.is_null()
            {
                deferred.push(i);
                word.get_or_insert(s.gdn_fuse_n);
            }
        }
        let exact = self.gdn_pending_is_exact();
        let stream = self.gpu.default_stream();
        // Held across the launches: a flush on another thread cannot land a
        // commit after this verify has been ordered behind it.
        let mut map = self.gdn_fuse.pending.lock();
        let n = match map.remove(&seq.slot_idx) {
            Some(p) if fuses(&p.idx, &deferred, exact) => {
                self.gdn_fuse.count(true);
                p.n
            }
            // Landed on this verify's stream, ahead of it.
            Some(p) => {
                self.gdn_fuse_land(&p)?;
                0
            }
            None => 0,
        };
        if let Some(w) = word {
            self.gpu.memset_32_async(w, n as u32, 1, stream)?;
        }
        Ok(())
    }

    fn gdn_fuse_land(&self, p: &Pending) -> Result<()> {
        self.gdn_fuse.count(false);
        self.commit_gdn_exact(&p.layers, p.n, self.gpu.default_stream())
    }

    /// Land every pending commit (module docs), then sync the default
    /// stream so a reader on any stream sees the committed state.
    pub(crate) fn gdn_fuse_flush_all(&self) -> Result<()> {
        if !requested() {
            return Ok(());
        }
        let mut map = self.gdn_fuse.pending.lock();
        if map.is_empty() {
            return Ok(());
        }
        // Each entry leaves the map only once landed: a failed land keeps it
        // and every later one pending (a drain would drop them).
        let slots: Vec<usize> = map.keys().copied().collect();
        for slot in slots {
            if let Some(p) = map.get(&slot) {
                self.gdn_fuse_land(p)?;
            }
            map.remove(&slot);
        }
        self.gpu.synchronize(self.gpu.default_stream())
    }

    /// [`Self::gdn_fuse_flush_all`] where the caller cannot fail.
    pub(crate) fn gdn_fuse_flush_all_logged(&self) {
        if let Err(e) = self.gdn_fuse_flush_all() {
            tracing::error!("qwen4_exp GDN commit fuse: flush failed: {e:#}");
        }
    }

    /// A freed slot's pending commit is moot.
    pub(crate) fn gdn_fuse_drop_slot(&self, slot: usize) {
        if requested() {
            self.gdn_fuse.pending.lock().remove(&slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fuses;

    #[test]
    fn only_the_same_deferred_layers_fuse() {
        let all = [0, 2, 4, 5];
        assert!(fuses(&all, &all, true));
        // The wyN deferral, a different layer set, or nothing deferred: land it.
        assert!(!fuses(&all, &all, false));
        assert!(!fuses(&all, &[0, 2, 4], true));
        assert!(!fuses(&all, &[0, 2, 4, 5, 6], true));
        assert!(!fuses(&[], &[], true));
    }
}
