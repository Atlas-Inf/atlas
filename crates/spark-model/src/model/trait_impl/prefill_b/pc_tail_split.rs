// SPDX-License-Identifier: AGPL-3.0-only
//! Whether a short prompt skips the tail-checkpoint split.

/// `ATLAS_TAIL_SPLIT_MIN_TOKENS=N` (default 0 = every prompt): a prompt of
/// fewer than `N` tokens runs its last chunk as one pass, without the
/// tail-checkpoint split.
///
/// The split's second pass sweeps the activated experts again (~150 ms on
/// GLM-5.3-Flash), which on a 60-150-token chat prompt doubles its prefill,
/// and every short prompt of a burst pays it in series. What the split buys
/// is a checkpoint at most two blocks under the prompt end for the next
/// turn; without it that turn restores from an older snapshot (or none) and
/// replays at most `N` more rows inside its own pass, which on this
/// weight-bound MoE costs little next to a pass. The decision reads only the
/// prompt length and this setting, so every rank, cold and warm, takes the
/// same pass shapes for a given prompt (`ATLAS_STARTUP_PARITY` checks the
/// setting agrees across ranks).
pub(in crate::model) fn tail_split_min_tokens() -> usize {
    parse_tail_split_min_tokens(std::env::var("ATLAS_TAIL_SPLIT_MIN_TOKENS").ok().as_deref())
}

pub(in crate::model) fn parse_tail_split_min_tokens(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

/// Whether the last chunk `[chunk_start, total)` splits at `cut`
/// (`ATLAS_NO_TAIL_SPLIT` = `disabled`, [`tail_split_min_tokens`]).
pub(in crate::model) fn tail_split_at(
    total: usize,
    chunk_start: usize,
    cut: usize,
    disabled: bool,
    min_tokens: usize,
) -> bool {
    !disabled && total >= min_tokens && cut > chunk_start && cut < total
}
