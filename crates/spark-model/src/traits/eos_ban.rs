// SPDX-License-Identifier: AGPL-3.0-only

//! `EosBan`: the per-sequence `min_tokens` end-token ban for greedy verify
//! heads, plus the rank-global model end tokens a split head excludes.

/// End tokens a greedy verify head may not pick below `floor` (the sequence
/// position `prompt_len + min_tokens`). Unused id slots are `u32::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EosBan {
    pub floor: usize,
    pub ids: [u32; 4],
}

static MODEL_END_TOKENS: std::sync::OnceLock<[u32; 4]> = std::sync::OnceLock::new();

impl Default for EosBan {
    fn default() -> Self {
        Self {
            floor: 0,
            ids: [u32::MAX; 4],
        }
    }
}

impl EosBan {
    /// The ban for a request's `min_tokens` floor. A sequence without end
    /// tokens (`ignore_eos`) has nothing to ban: no floor, so no verify head
    /// or drafter excludes the model's end tokens.
    pub fn new(prompt_len: usize, min_tokens: usize, eos_tokens: &[u32]) -> Self {
        let mut ids = [u32::MAX; 4];
        for (slot, &id) in ids.iter_mut().zip(eos_tokens) {
            *slot = id;
        }
        Self {
            floor: if min_tokens == 0 || eos_tokens.is_empty() {
                0
            } else {
                prompt_len + min_tokens
            },
            ids,
        }
    }

    /// Install this ban on `seq`: the greedy verify heads mask end tokens
    /// below the floor, and the drafter skips them there, so a banned end
    /// token never truncates an otherwise acceptable draft chain.
    pub fn arm(self, seq: &mut super::SequenceState) {
        if self.floor > 0 {
            let (floor, prompt) = (self.floor, seq.prompt_len);
            tracing::info!(
                "min_tokens: end tokens banned below position {floor} (prompt {prompt})"
            );
        }
        seq.eos_ban = self;
        if let Some(proposer) = seq.proposer_state.as_mut() {
            proposer.set_end_floor(self.floor);
        }
    }

    /// Install the model's end tokens on every rank. A TP2 vocabulary-split
    /// head bans them in each rank's half, and only the head rank carries the
    /// per-request `EosBan`: without this the worker's half (which holds GLM's
    /// `<|user|>`) returns its end token as the "unbanned" pick.
    pub fn install_model_end_tokens(eos_tokens: &[u32]) {
        let _ = MODEL_END_TOKENS.set(Self::new(0, 0, eos_tokens).ids);
    }

    /// End ids a split verify head excludes from its unbanned pick: the
    /// installed model end tokens, else this sequence's own.
    pub fn head_ids(&self) -> [u32; 4] {
        MODEL_END_TOKENS.get().copied().unwrap_or(self.ids)
    }

    /// The installed model end tokens (unused slots `u32::MAX`).
    pub fn model_end_ids() -> [u32; 4] {
        Self::default().head_ids()
    }

    /// Leading draft depths (depth `d` drafts position `anchor_pos + d`) that
    /// may not be an end token: those below `floor` (0 = no min_tokens).
    pub fn banned_draft_depth(floor: usize, anchor_pos: usize, max_depth: usize) -> u32 {
        floor.saturating_sub(anchor_pos + 1).min(max_depth) as u32
    }

    /// Bit `j` is set when verify row `j` of a pass whose first input sits at
    /// `base_pos` predicts a position below the floor (row `j` predicts
    /// `base_pos + j + 1`). `rows` is at most 64.
    pub fn row_mask(&self, base_pos: usize, rows: usize) -> u64 {
        let banned = self.floor.saturating_sub(base_pos + 1).min(rows).min(64);
        if banned == 64 {
            u64::MAX
        } else {
            (1u64 << banned) - 1
        }
    }
}

#[cfg(test)]
mod eos_ban_tests {
    use super::EosBan;

    #[test]
    fn rows_below_the_min_tokens_floor_are_masked() {
        // prompt 100, min_tokens 10: output index i sits at position 100 + i,
        // so positions below 110 may not end the turn.
        let ban = EosBan::new(100, 10, &[7, 9]);
        assert_eq!(ban.ids, [7, 9, u32::MAX, u32::MAX]);
        assert_eq!(ban.row_mask(100, 8), 0xFF); // predicts 101..=108
        assert_eq!(ban.row_mask(105, 8), 0b1111); // 106..=109 banned, 110.. free
        assert_eq!(ban.row_mask(109, 8), 0);
        assert_eq!(ban.row_mask(0, 64), u64::MAX); // all 64 rows below 110
        assert_eq!(EosBan::new(100, 0, &[7]).row_mask(0, 8), 0);
        assert_eq!(EosBan::default().row_mask(0, 8), 0);
    }

    #[test]
    fn no_end_tokens_means_no_floor() {
        // `ignore_eos`: the end tokens are ordinary tokens, min_tokens or not.
        assert_eq!(EosBan::new(100, 10, &[]), EosBan::default());
    }

    #[test]
    fn drafts_below_the_floor_skip_end_tokens() {
        // Floor 110; depth d drafts anchor + d, so an anchor at 105 bans
        // depths 1..=4 (positions 106..=109).
        assert_eq!(EosBan::banned_draft_depth(110, 105, 7), 4);
        assert_eq!(EosBan::banned_draft_depth(110, 109, 7), 0);
        assert_eq!(EosBan::banned_draft_depth(110, 50, 7), 7);
        assert_eq!(EosBan::banned_draft_depth(0, 50, 7), 0);
    }

    #[test]
    fn split_head_bans_model_end_tokens_on_every_rank() {
        // A worker rank's sequence carries the default ban (no ids), yet GLM's
        // <|user|> lives in its vocabulary half.
        EosBan::install_model_end_tokens(&[154820, 154827, 154829]);
        assert_eq!(
            EosBan::default().head_ids(),
            [154820, 154827, 154829, u32::MAX]
        );
    }
}
