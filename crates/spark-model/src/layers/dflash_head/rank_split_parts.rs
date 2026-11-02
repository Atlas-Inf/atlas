// SPDX-License-Identifier: AGPL-3.0-only

//! What a rank split shares (`rank_split`): its switches and their parsing.

use anyhow::{Result, bail};

/// What the worker shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parts {
    /// Each layer's gate/up and down projections.
    pub mlp: bool,
    /// The vocabulary head.
    pub head: bool,
    /// Each layer's q/k/v and o projections (needs `mlp`).
    pub attn: bool,
}

/// Bytes per row of a `Swap::Candidates` payload: one row's top-K list, f32
/// values and u32 ids (`ops::dflash2_selector_lists_bytes`).
pub(super) const CANDIDATE_ROW_BYTES: usize = 8 * crate::layers::ops::DFLASH2_SELECTOR_MAX_TOP_K;

impl Parts {
    /// `1` = the MLP and the head, else a comma list of `mlp`, `head` and
    /// `attn`; unset, empty
    /// or `0` = off. Anything else is refused: a typo must not run unsplit
    /// on one rank.
    pub(crate) fn parse(value: Option<&str>) -> Result<Option<Self>> {
        let all = Self {
            mlp: true,
            head: true,
            attn: false,
        };
        match value.map(str::trim) {
            None | Some("") | Some("0") => Ok(None),
            Some("1") => Ok(Some(all)),
            Some(list) => {
                let mut parts = Self {
                    mlp: false,
                    head: false,
                    attn: false,
                };
                for part in list.split(',') {
                    match part.trim() {
                        "mlp" => parts.mlp = true,
                        "head" => parts.head = true,
                        "attn" => parts.attn = true,
                        other => bail!("ATLAS_GLM_DRAFT_TP: unknown part '{other}' in '{list}'"),
                    }
                }
                Ok(Some(parts))
            }
        }
    }

    /// The value both ranks must agree on (0 = off).
    pub(crate) fn word(parts: Option<Self>) -> u64 {
        parts.map_or(0, |p| {
            p.mlp as u64 | (p.head as u64) << 1 | (p.attn as u64) << 5
        })
    }
}

/// `ATLAS_GLM_DRAFT_TP`, from the profile both ranks share.
pub fn requested() -> Result<Option<Parts>> {
    Parts::parse(std::env::var("ATLAS_GLM_DRAFT_TP").ok().as_deref())
}

/// `ATLAS_GLM_DRAFT_TP_BATCH`: `1` splits batched proposes too; unset, empty
/// or `0` = off; anything else is refused.
pub(crate) fn parse_batch(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_GLM_DRAFT_TP_BATCH must be 0 or 1, got '{other}'"),
    }
}

/// `ATLAS_GLM_DRAFT_TP_BATCH`, from the profile both ranks share.
pub fn batch_requested() -> Result<bool> {
    parse_batch(std::env::var("ATLAS_GLM_DRAFT_TP_BATCH").ok().as_deref())
}

/// `ATLAS_GLM_DRAFT_TP_TOPK`: `1` swaps candidate lists instead of logits;
/// unset, empty or `0` = off; anything else is refused.
pub(crate) fn parse_topk(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_GLM_DRAFT_TP_TOPK must be 0 or 1, got '{other}'"),
    }
}

/// `ATLAS_GLM_DRAFT_TP_TOPK`, from the profile both ranks share.
pub fn topk_requested() -> Result<bool> {
    parse_topk(std::env::var("ATLAS_GLM_DRAFT_TP_TOPK").ok().as_deref())
}

/// The value both ranks must agree on: [`Parts::word`], plus bit 2 for the
/// batched split, bit 3 for the context split and bit 4 for the candidate
/// swap (so the word is unchanged with them off).
pub(crate) fn parity_word(parts: Option<Parts>, batch: bool, ctx: bool, topk: bool) -> u64 {
    Parts::word(parts) | (batch as u64) << 2 | (ctx as u64) << 3 | (topk as u64) << 4
}
