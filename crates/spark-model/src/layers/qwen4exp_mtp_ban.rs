// SPDX-License-Identifier: AGPL-3.0-only

//! The MTP drafter's half of the qwen4_exp min_tokens end-token ban
//! (`ATLAS_QWEN4EXP_EOS_BAN=1`, `EosBan::target`), and the batched propose's
//! draft head + argmax step it hooks into.
//!
//! Below a request's floor the target never picks a model end token
//! (`spark-server` `min_tokens_ban`), so a draft of one there could only be
//! rejected and would cut the rest of its chain. A draft at depth `d` of a
//! window anchored at position `p` sits at `p + d`; below the floor
//! (`EosBan::banned_draft_depth`) the end tokens' draft-head rows get a logit
//! far below any real one before the argmax: BF16 `0xFEFE` (-1.7e38), a byte
//! memset on the stream. Every draft head takes it: the BF16 prefix or id
//! list, its NVFP4 copy, and under `ATLAS_QWEN4EXP_MTP_DRAFT_TP` the rows the
//! head rank assembles (the worker's argmax is discarded). The grammar-masked
//! full-vocabulary path clears the ids from its host bitmask instead.
//!
//! Drafts only move acceptance: every token is the target's pick. The
//! default draft head (`--mtp-vocab` 100k) holds no end token, so there the
//! ban costs nothing.

use super::*;
use crate::traits::EosBan;

/// Byte whose BF16 pair `0xFEFE` is -1.7e38.
const BANNED_BYTE: u8 = 0xFE;

impl Qwen4ExpMtpProposerState {
    /// Whether the draft for position `input_pos + 1` may not be an end token.
    pub(super) fn bans_draft_after(&self, input_pos: usize) -> bool {
        EosBan::banned_draft_depth(self.end_floor, input_pos, 1) > 0
    }
}

/// The floor a qwen4_exp drafter keeps: the request's under the target ban,
/// else none (`ProposerState::set_end_floor`).
pub(super) fn drafter_floor(floor: usize, target_ban: bool) -> usize {
    if target_ban { floor } else { 0 }
}

/// Clear the model end tokens from a grammar bitmask (bit set = allowed).
pub(super) fn ban_in_bitmask(bitmask: &[i32]) -> Vec<i32> {
    let mut out = bitmask.to_vec();
    for id in EosBan::model_end_ids() {
        if let Some(word) = out.get_mut(id as usize / 32) {
            *word &= !(1i32 << (id % 32));
        }
    }
    out
}

impl Qwen4ExpMtpHead {
    /// Mask the end tokens' draft-head rows in the logits rows `rows` (each
    /// `self.draft.rows()` wide) on `stream`.
    pub(super) fn ban_draft_rows(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        rows: impl IntoIterator<Item = usize>,
        stream: u64,
    ) -> Result<()> {
        let width = self.draft.rows() as usize;
        let cols: Vec<usize> = EosBan::model_end_ids()
            .into_iter()
            .filter_map(|id| self.draft.row_of(id))
            .map(|c| c as usize)
            .collect();
        if cols.is_empty() {
            return Ok(());
        }
        for r in rows {
            for &c in &cols {
                gpu.memset_async(logits.offset((r * width + c) * 2), BANNED_BYTE, 2, stream)?;
            }
        }
        Ok(())
    }

    /// Batched propose step 6 at position `j`: the draft head over the `n`
    /// rows, the end-token ban on the rows whose depth `j + 1` is below their
    /// floor (`banned[i]` banned depths), and the argmax into slab row `j + 1`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draft_argmax_rows(
        &self,
        j: usize,
        n: usize,
        num_drafts: usize,
        banned: &[u32],
        ctx: &ForwardContext,
        stream: u64,
        want_lp: bool,
        tp: Option<&draft_tp::TpRun<'_>>,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let (h32, n32) = (ctx.config.hidden_size as u32, n as u32);
        // Under draft TP the worker projects half of the rows (`draft_tp`).
        let logits = ctx.buffers.logits();
        let rows = self.draft.rows();
        match tp {
            Some(run) => run.head(&self.draft, self.dense_gemv_batchm_k, self.h_out, logits)?,
            None => self.draft.project_rows(
                gpu,
                self.dense_gemv_batchm_k,
                self.h_out,
                logits,
                n32,
                h32,
                stream,
            )?,
        }
        let below = (0..n).filter(|&i| banned.get(i).is_some_and(|&d| (j as u32) < d));
        self.ban_draft_rows(gpu, logits, below, stream)?;
        let ids = self.batch_slab.offset((j + 1) * n * 4);
        if want_lp {
            let lp = self
                .batch_slab
                .offset(qwen4exp_mtp_batch::lp_off(n, num_drafts) + j * n * 4);
            ops::argmax_bf16_batch_lp(
                gpu,
                self.argmax_batch_lp_k,
                logits,
                ids,
                lp,
                rows,
                n32,
                rows,
                stream,
            )
        } else {
            ops::argmax_bf16_batch(
                gpu,
                self.argmax_batch_k,
                logits,
                ids,
                rows,
                n32,
                rows,
                stream,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_target_ban_gives_the_drafter_a_floor() {
        assert_eq!(drafter_floor(110, true), 110);
        // A GLM-style ban (or the switch off) leaves qwen4_exp drafts alone.
        assert_eq!(drafter_floor(110, false), 0);
    }

    #[test]
    fn the_bitmask_ban_clears_only_the_end_tokens() {
        // The model end tokens of this test process (`split_head_bans_model_
        // end_tokens_on_every_rank` may install GLM's first; either way the
        // installed set is what the drafter bans).
        let ids = EosBan::model_end_ids();
        let mask = vec![-1i32; 248_320usize.div_ceil(32)];
        let out = ban_in_bitmask(&mask);
        for (w, (&a, &b)) in mask.iter().zip(&out).enumerate() {
            for bit in 0..32 {
                let id = (w * 32 + bit) as u32;
                let cleared = (a >> bit) & 1 == 1 && (b >> bit) & 1 == 0;
                assert_eq!(cleared, ids.contains(&id), "id {id}");
            }
        }
    }
}
