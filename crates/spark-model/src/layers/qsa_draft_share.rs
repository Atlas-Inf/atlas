// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_MTP_INDEX_SHARE=1` (default off; drafts only, NOT exact):
//! within one MTP propose the drafter selects its blocks once, at the first
//! draft position, and the later draft positions attend those blocks plus
//! every token from that selection's tail start on (blocks the earlier drafts
//! closed included) — vLLM's `index_share_for_mtp_iteration`. Skips the
//! drafter's q prep, ~19K-block score and top-k per later draft. The verify
//! decides every token, so output is unchanged; acceptance may move.

use std::sync::OnceLock;
use std::sync::atomic::Ordering;

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::{QsaIndexer, QsaSeqState};
use super::{SHARE_MARGIN, env_on};
use crate::layer::{AttnLayerState, LayerState};

/// The drafter's selection reuse on one sequence's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftShare {
    Off,
    /// The next fresh selection is kept for reuse.
    Armed,
    /// `sel_dev` holds this state's selection `generation`: its blocks in the
    /// first `block_topk * ratio` slots, its tail from `tail_start`.
    Holding {
        generation: u64,
        tail_start: usize,
    },
}

fn index_share_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_on("ATLAS_QWEN4EXP_MTP_INDEX_SHARE"))
}

/// Arm (`on`, start of a propose) or disarm (end of it) the drafter's
/// selection reuse on its body layer's state. A no-op unless
/// `ATLAS_QWEN4EXP_MTP_INDEX_SHARE=1`, and for a state with no QSA carry yet.
pub fn draft_share_set(state: &mut dyn LayerState, on: bool) {
    if !index_share_requested() {
        return;
    }
    if let Some(attn) = state.as_any_mut().downcast_mut::<AttnLayerState>()
        && let Some(st) = attn.qsa.as_mut()
    {
        st.share = if on {
            DraftShare::Armed
        } else {
            DraftShare::Off
        };
    }
}

impl QsaIndexer {
    /// A fresh selection was written to `sel_dev` for `st`: keep it for the
    /// rest of the propose if reuse is armed.
    pub(in crate::layers::qsa) fn draft_share_record(
        &self,
        st: &mut QsaSeqState,
        tail_start: usize,
    ) {
        let generation = self.rows.sel_generation.fetch_add(1, Ordering::Relaxed) + 1;
        if st.share != DraftShare::Off {
            st.share = DraftShare::Holding {
                generation,
                tail_start,
            };
        }
    }

    /// The reused selection for a later draft at `visible` tokens: rewrite
    /// the tail slots (`tail_start..visible`) and return the new `n_sel`.
    /// `None`: nothing held, another selection overwrote `sel_dev` since, or
    /// the tail outgrew the margin — select afresh.
    pub(in crate::layers::qsa) fn draft_share_tail(
        &self,
        st: &QsaSeqState,
        visible: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<u32>> {
        let DraftShare::Holding {
            generation,
            tail_start,
        } = st.share
        else {
            return Ok(None);
        };
        let base = (self.block_topk * self.ratio) as usize;
        let cap = (self.budget + self.ratio) as usize + SHARE_MARGIN;
        if generation != self.rows.sel_generation.load(Ordering::Relaxed)
            || visible < tail_start
            || base + (visible - tail_start) > cap
        {
            return Ok(None);
        }
        let ids: Vec<u8> = (tail_start..visible)
            .flat_map(|t| (t as i32).to_le_bytes())
            .collect();
        gpu.copy_h2d_async(&ids, self.sel_dev.offset(base * 4), stream)?;
        Ok(Some((base + visible - tail_start) as u32))
    }
}

#[cfg(test)]
mod share_tests {
    use super::*;
    use spark_runtime::gpu::DevicePtr;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn indexer(gpu: &MockGpuBackend) -> QsaIndexer {
        // ratio 4, budget 64 -> block_topk 16; selection cap 68 + margin.
        QsaIndexer::new(
            DevicePtr::NULL,
            DevicePtr::NULL,
            DevicePtr::NULL,
            2,
            8,
            4,
            64,
            256,
            8,
            1e5,
            1e-5,
            128,
            2,
            16,
            gpu,
        )
        .unwrap()
    }

    /// Armed -> the first fresh selection is held -> later drafts reuse it,
    /// their tail growing from the held tail start; any other selection on
    /// the layer in between (another sequence) invalidates it.
    #[test]
    fn a_held_selection_serves_later_drafts_until_overwritten() {
        let gpu = MockGpuBackend::new();
        let qsa = indexer(&gpu);
        let mut st = qsa.new_seq_state(&gpu).unwrap();
        assert_eq!(qsa.draft_share_tail(&st, 70, &gpu, 0).unwrap(), None);
        qsa.draft_share_record(&mut st, 68); // Off: nothing kept
        assert_eq!(st.share, DraftShare::Off);
        st.share = DraftShare::Armed;
        qsa.draft_share_record(&mut st, 68); // first draft, visible 70
        assert!(matches!(
            st.share,
            DraftShare::Holding { tail_start: 68, .. }
        ));
        // Next drafts: 64 block slots + tokens 68..visible.
        assert_eq!(qsa.draft_share_tail(&st, 71, &gpu, 0).unwrap(), Some(67));
        assert_eq!(qsa.draft_share_tail(&st, 73, &gpu, 0).unwrap(), Some(69));
        // Past the margin: select afresh.
        assert_eq!(qsa.draft_share_tail(&st, 68 + 21, &gpu, 0).unwrap(), None);
        // Another sequence's selection overwrote sel_dev.
        let mut other = qsa.new_seq_state(&gpu).unwrap();
        qsa.draft_share_record(&mut other, 100);
        assert_eq!(qsa.draft_share_tail(&st, 71, &gpu, 0).unwrap(), None);
    }
}
