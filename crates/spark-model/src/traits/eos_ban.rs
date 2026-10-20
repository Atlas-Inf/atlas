// SPDX-License-Identifier: AGPL-3.0-only

//! `EosBan`: the per-sequence `min_tokens` end-token ban for greedy verify
//! heads, plus the rank-global model end tokens a split head excludes.
//!
//! qwen4_exp (`ATLAS_QWEN4EXP_EOS_BAN=1`, default off) bans at the pick
//! instead: a [`EosBan::target`] ban makes the scheduler's target pick (serial
//! decode and every verify row alike) exclude `ids` while the output is below
//! `min_tokens`, as vLLM's min_tokens processor masks its stop ids, even
//! under `ignore_eos`. Without it an end token below the floor is picked,
//! discarded (or, under `ignore_eos`, emitted) and fed back, and the MTP
//! draft head (the first 100k ids) can never propose it, so every such pick
//! rejects the rest of its verify window.

/// End tokens a greedy verify head may not pick below `floor` (the sequence
/// position `prompt_len + min_tokens`). Unused id slots are `u32::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EosBan {
    pub floor: usize,
    pub ids: [u32; 4],
    /// The target's own picks exclude `ids` while fewer than `min_tokens`
    /// tokens are out (`ATLAS_QWEN4EXP_EOS_BAN`, [`EosBan::for_request`]).
    pub target: bool,
}

static MODEL_END_TOKENS: std::sync::OnceLock<[u32; 4]> = std::sync::OnceLock::new();
static TARGET_BAN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

impl Default for EosBan {
    fn default() -> Self {
        Self {
            floor: 0,
            ids: [u32::MAX; 4],
            target: false,
        }
    }
}

impl EosBan {
    /// The ban for a request's `min_tokens` floor. A sequence without end
    /// tokens (`ignore_eos`) has nothing to ban: no floor, so no verify head
    /// or drafter excludes the model's end tokens. `min_tokens` is the
    /// request's, unvalidated: the floor saturates (a floor past any position
    /// bans every row, as the request asks).
    pub fn new(prompt_len: usize, min_tokens: usize, eos_tokens: &[u32]) -> Self {
        let mut ids = [u32::MAX; 4];
        for (slot, &id) in ids.iter_mut().zip(eos_tokens) {
            *slot = id;
        }
        Self {
            floor: if min_tokens == 0 || eos_tokens.is_empty() {
                0
            } else {
                prompt_len.saturating_add(min_tokens)
            },
            ids,
            target: false,
        }
    }

    /// `ATLAS_QWEN4EXP_EOS_BAN=1`, read once.
    pub fn qwen4exp_requested() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_EOS_BAN").as_deref() == Ok("1"))
    }

    /// Arm the target ban for every request this process admits: qwen4_exp
    /// under `ATLAS_QWEN4EXP_EOS_BAN=1`. After
    /// [`EosBan::install_model_end_tokens`]. Returns whether it is on.
    pub fn install_target_ban(model_type: &str) -> bool {
        let on = model_type == "qwen4_exp" && Self::qwen4exp_requested();
        let on = *TARGET_BAN.get_or_init(|| on);
        if on {
            tracing::info!(
                "qwen4_exp min_tokens end-token ban ON (ATLAS_QWEN4EXP_EOS_BAN=1): below a \
                 request's min_tokens the target and the MTP drafter never pick {:?}",
                Self::model_end_ids()
            );
        }
        on
    }

    /// Whether [`EosBan::install_target_ban`] armed the target ban.
    pub fn target_ban_installed() -> bool {
        TARGET_BAN.get().copied().unwrap_or(false)
    }

    /// The ban a request is admitted with: [`EosBan::new`], or under the
    /// installed target ban [`EosBan::targeted`] over the model end tokens.
    pub fn for_request(prompt_len: usize, min_tokens: usize, eos_tokens: &[u32]) -> Self {
        if Self::target_ban_installed() {
            Self::targeted(prompt_len, min_tokens, &Self::model_end_ids(), eos_tokens)
        } else {
            Self::new(prompt_len, min_tokens, eos_tokens)
        }
    }

    /// A target ban: the model end tokens `model_end` (whether or not they end
    /// this request: vLLM bans its stop ids below min_tokens under
    /// `ignore_eos` too), then the request's own end tokens, in four slots.
    pub fn targeted(
        prompt_len: usize,
        min_tokens: usize,
        model_end: &[u32],
        eos_tokens: &[u32],
    ) -> Self {
        let mut merged: Vec<u32> = Vec::with_capacity(model_end.len() + eos_tokens.len());
        for &id in model_end.iter().chain(eos_tokens) {
            if id != u32::MAX && !merged.contains(&id) {
                merged.push(id);
            }
        }
        let ban = Self::new(prompt_len, min_tokens, &merged);
        Self {
            target: ban.floor > 0,
            ..ban
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
        floor
            .saturating_sub(anchor_pos.saturating_add(1))
            .min(max_depth) as u32
    }

    /// Bit `j` is set when verify row `j` of a pass whose first input sits at
    /// `base_pos` predicts a position below the floor (row `j` predicts
    /// `base_pos + j + 1`). `rows` is at most 64.
    pub fn row_mask(&self, base_pos: usize, rows: usize) -> u64 {
        let banned = self
            .floor
            .saturating_sub(base_pos.saturating_add(1))
            .min(rows)
            .min(64);
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
    fn a_target_ban_bans_the_model_end_tokens_under_ignore_eos() {
        // `ignore_eos` leaves the request no end tokens; the target ban still
        // bans the model's, and merges a request's own after them.
        let ban = EosBan::targeted(100, 10, &[7, 9, u32::MAX, u32::MAX], &[]);
        assert_eq!(ban.ids, [7, 9, u32::MAX, u32::MAX]);
        assert_eq!((ban.floor, ban.target), (110, true));
        let ban = EosBan::targeted(100, 10, &[7, 9], &[9, 11, 13, 15]);
        assert_eq!(ban.ids, [7, 9, 11, 13]);
        // Without min_tokens nothing is banned, and the plain ban never targets.
        let none = EosBan::targeted(100, 0, &[7, 9], &[]);
        assert_eq!((none.floor, none.target), (0, false));
        assert!(!EosBan::new(100, 10, &[7]).target);
    }

    #[test]
    fn an_unbounded_min_tokens_saturates_the_floor() {
        // `min_tokens` reaches here unvalidated; `prompt_len + min_tokens`
        // wrapped in release to a floor below the prompt (no ban at all).
        let ban = EosBan::new(100, usize::MAX, &[7]);
        assert_eq!(ban.floor, usize::MAX);
        assert_eq!(ban.row_mask(usize::MAX - 1, 8), 0);
        assert_eq!(ban.row_mask(1 << 40, 64), u64::MAX);
        assert_eq!(EosBan::banned_draft_depth(ban.floor, usize::MAX, 7), 0);
        assert_eq!(EosBan::banned_draft_depth(ban.floor, 1 << 40, 7), 7);
        let t = EosBan::targeted(100, usize::MAX, &[7, 9], &[]);
        assert_eq!((t.floor, t.target), (usize::MAX, true));
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
