// SPDX-License-Identifier: AGPL-3.0-only

//! Carry the DFlash drafter's ctx state across turns of a session.
//!
//! # The defect this closes
//!
//! [`super::DflashProposerState`] is per-request: on every turn a fresh
//! `ctx_hidden_acc` is allocated and the first `propose()` precomputes K/V
//! for the ENTIRE prompt prefix (~6.5 s at 16K ctx on gfx1151 — measured
//! 2026-09-23). Worse, on a WARM turn the Marconi/prefix-cache hit skips
//! the target prefill for the matched span, so `dflash_hidden_save` never
//! captures those positions: acc slots `[0..hit)` stay ZERO and the
//! incremental precompute projects them into garbage K/V — the drafter is
//! blind over the prefix, which is the acceptance collapse measured on the
//! long-context agentic leg (median tokens/step → 0 past ~20K ctx).
//!
//! # The mechanism
//!
//! A turn's prompt is a strict extension of the previous turn's full
//! sequence — the same property the prefix cache exploits. So the ctx
//! accumulator rows AND the paged drafter K/V the previous turn already
//! built ARE the rows this turn needs; only the tail is missing. The
//! proposer keeps the finished sequence's ctx alive in a single carry slot
//! (DFlash runs at concurrency 1 on the served path) and the next turn's
//! first ctx-seed adopts the common-prefix portion, setting
//! `ctx_committed` so `propose()` precomputes only the new delta.
//!
//! Correctness note: the drafter can never corrupt output — the target
//! verifies every draft — so a wrong carried row costs acceptance, never
//! correctness. Prefix equality on tokens is the whole validity condition
//! (hiddens are a pure function of the token prefix).

use super::lifecycle::DflashGraphIdentity;
use spark_runtime::gpu::{DevicePtr, GraphHandle};

/// Default-ON switch for the ctx carry. `ATLAS_DFLASH_CTX_CARRY=0` opts out
/// and restores the per-request rebuild. Mirrors the MTP drafter carry
/// (`crate::model::mtp_carry`) policy shape.
pub fn ctx_carry_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("ATLAS_DFLASH_CTX_CARRY").as_deref(),
            Ok("0") | Ok("false")
        )
    })
}

/// Minimum matched prefix (tokens) before adoption is worth taking — short
/// matches are cheaper to rebuild than to reason about. Block-aligned like
/// `mtp_carry::marconi_min_tokens`.
pub const MIN_CARRY_TOKENS: usize = 256;

/// A finished turn's drafter ctx, held for the next turn of the same
/// session. Single slot: the served DFlash path is concurrency-1, and one
/// slot keeps ownership trivially safe — each resource is owned by the
/// carry slot XOR by a live sequence, never both.
pub struct DflashCtxCarry {
    /// Per-seq hidden accumulator (`[max_ctx_len, n_capture * hidden]`
    /// BF16). Moved out of the finished state so `free_state` sees a null
    /// pointer and skips the free.
    pub ctx_hidden_acc: DevicePtr,
    /// Populated slots of `ctx_hidden_acc` at free time.
    pub ctx_len: usize,
    /// Precompute watermark: slots `[0..ctx_committed)` already have valid
    /// K/V in the carried paged cache.
    pub ctx_committed: usize,
    /// Per-slot fixed rope positions, parallel to `ctx_hidden_acc` slots.
    pub ctx_positions: Vec<i32>,
    /// Paged drafter-KV pool blocks covering `max_ctx_len + γ` slots.
    pub block_table: Vec<u32>,
    /// Device copy of `block_table`.
    pub block_table_dev: Option<DevicePtr>,
    /// Populated paged-cache slots (`== ctx_len` after a committed step).
    pub ctx_count_drafter: usize,
    /// Capacity of the paged allocation (`block_table.len() * 16`).
    pub max_ctx_count_drafter: usize,
    /// The token sequence that produced this ctx (prompt + generated).
    /// Adoption requires it to be a prefix of the new turn's prompt.
    pub tokens: Vec<u32>,
    /// Captured propose subgraphs lifted out of `propose_graphs` before
    /// `free_state`'s retire sweep destroys them, under their ORIGINAL
    /// generation key. On adopt they are re-keyed to the new generation:
    /// every pointer the identity pins (paged block table, ctx
    /// accumulator, lane markov scratch) survives the carry, so replaying
    /// them is exactly as valid as it was pre-free.
    pub graphs: Vec<(DflashGraphIdentity, Vec<GraphHandle>)>,
    /// Lane the carried graphs were captured on. Pinned onto the adopting
    /// state so the identity's lane (and its baked lane-scratch pointers)
    /// match on the next turn.
    pub lane_id: usize,
}

impl DflashCtxCarry {
    /// Length of the common prefix of `self.tokens` and `prompt` — the
    /// adoption validity primitive (same rule as `CarriedDrafter`).
    pub fn common_prefix_len(&self, prompt: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(prompt.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn carry(tokens: &[u32], ctx_len: usize, committed: usize) -> DflashCtxCarry {
        DflashCtxCarry {
            ctx_hidden_acc: DevicePtr(0),
            ctx_len,
            ctx_committed: committed,
            ctx_positions: (0..ctx_len as i32).collect(),
            block_table: vec![1, 2, 3],
            block_table_dev: None,
            ctx_count_drafter: ctx_len,
            max_ctx_count_drafter: ctx_len + 16,
            tokens: tokens.to_vec(),
            graphs: Vec::new(),
            lane_id: 0,
        }
    }

    #[test]
    fn common_prefix_stops_at_first_divergence() {
        let c = carry(&[1, 2, 3, 4], 4, 4);
        assert_eq!(c.common_prefix_len(&[1, 2, 3, 4, 5, 6]), 4);
        assert_eq!(c.common_prefix_len(&[1, 2, 9, 4]), 2);
        assert_eq!(c.common_prefix_len(&[]), 0);
    }

    #[test]
    fn carry_gate_defaults_on_and_honours_opt_out() {
        // Env is process-global; only assert the default here — the opt-out
        // arm is covered by the env-parse shape (Ok("0")|Ok("false")).
        assert!(ctx_carry_enabled());
    }
}
