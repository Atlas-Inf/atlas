// SPDX-License-Identifier: AGPL-3.0-only

//! Draft TP for a lone sequence (`ATLAS_QWEN4EXP_MTP_DRAFT_TP_ONE`): the
//! switch, the announce word of its per-sequence propose and that propose's
//! run. See `qwen4exp_mtp_tp.rs`; split out for the 500-LoC cap.

use super::*;

/// `ATLAS_QWEN4EXP_MTP_DRAFT_TP_ONE`: a lone sequence's propose splits too.
/// Parsed as the switch it rides on, and refused without it.
pub fn one_requested() -> Result<bool> {
    let one = parse(
        std::env::var("ATLAS_QWEN4EXP_MTP_DRAFT_TP_ONE")
            .ok()
            .as_deref(),
    )
    .context("ATLAS_QWEN4EXP_MTP_DRAFT_TP_ONE")?;
    ensure!(
        !one || requested()?,
        "ATLAS_QWEN4EXP_MTP_DRAFT_TP_ONE=1 needs ATLAS_QWEN4EXP_MTP_DRAFT_TP=1"
    );
    Ok(one)
}

impl Qwen4ExpMtpHead {
    /// The announce word of a lone sequence's per-sequence propose of
    /// `drafts` positions over `comm`, when it splits
    /// (`ATLAS_QWEN4EXP_MTP_DRAFT_TP_ONE`): a one-row plan.
    pub(super) fn tp_one_word(
        &self,
        comm: &dyn CommBackend,
        drafts: usize,
        grammar: bool,
    ) -> Option<u32> {
        let pair = comm.rank() == 0 && comm.world_size() == 2;
        (self.tp_one
            && self.tp.is_some()
            && pair
            && !grammar
            && comm.supports_peer_exchange_async()
            && self.draft.rows_batchable(1, self.dense_gemv_batchm_k))
        .then(|| Plan { n: 1, drafts }.word().ok())
        .flatten()
    }

    /// The run of a per-sequence propose of `drafts` positions handed
    /// `comm`: the model announced it as a one-row plan ([`Self::tp_one_word`]).
    /// A grammar-masked propose is never announced and never swaps.
    pub(in super::super) fn draft_tp_one_run<'a>(
        &'a self,
        comm: Option<&'a dyn CommBackend>,
        gpu: &'a dyn GpuBackend,
        drafts: usize,
        grammar: bool,
        stream: u64,
    ) -> Result<Option<TpRun<'a>>> {
        if grammar {
            return Ok(None);
        }
        self.draft_tp_run(comm, gpu, Plan { n: 1, drafts }, stream)
    }
}
