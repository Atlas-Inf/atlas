// SPDX-License-Identifier: AGPL-3.0-only

//! The row-pure half of a verify span's host picks, fanned out
//! (`ATLAS_VERIFY_PICK_PAR=1`, default off).
//!
//! A host-picked verify span (every row inside `<think>`) runs its K rows in
//! order, because each pick commits think state, history and counters the
//! next row's stages read (`SpanShadow`). But two pieces of a row's work read
//! nothing except that row's logits: the BF16 -> F32 dequant, and the F2
//! confidence stage's `top1_confident` — a ~248k-term `exp` sum, ~0.5 ms on
//! GB10 against ~0.08 ms for the dequant, and the bulk of a thinking row's
//! host pick once the sequence has thought 400 tokens. Both are computed
//! here for all K rows at once on the rayon pool; the serial loop then
//! consumes row `i` exactly where it would have dequantized it, and F2 takes
//! the precomputed answer for the same untouched logits (`with_hint`).
//!
//! Same functions on the same bytes, so every row's F32 logits and every F2
//! decision are the serial path's; only which thread computes them changes.
//!
//! The same switch fans the batched verify's host-picked SEQUENCES out
//! across the pool (`verify_pick_batch`): each sequence's picks read and
//! write only its own state.

use rayon::prelude::*;

use super::{ActiveSeq, LogitsContext, dequant_into};
use crate::scheduler::logit_processors::f2_confidence::{top1_confident, with_hint};

/// `ATLAS_VERIFY_PICK_PAR=1` (read once).
pub(in crate::scheduler) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_VERIFY_PICK_PAR").ok().as_deref() == Some("1"))
}

thread_local! {
    /// Reused per-row F32 buffers (first-touch pages cost more than the
    /// dequant on GB10; see `scratch.rs`).
    static ROWS: std::cell::RefCell<Vec<Vec<f32>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The K rows of one span, dequantized, with each row's F2 answer when F2
/// can run on it. Hands its buffers back to `ROWS` on drop.
pub(super) struct Prepicked {
    rows: Vec<Vec<f32>>,
    f2: Vec<Option<bool>>,
}

/// Whether F2 can compute on some row of this span: inside `<think>`, armed
/// by the watchdog config, not already forcing the end, and within K picks of
/// the 400-token gate. A superset of the stage's own condition at every row;
/// a hint the stage does not consume is simply dropped. Not under a
/// min_tokens end-token ban: F2 reads the masked row there (`min_tokens_ban`).
pub(super) fn f2_may_run(a: &ActiveSeq, ctx: &LogitsContext, k: usize) -> bool {
    crate::scheduler::min_tokens_ban::banned_ids(a).is_none()
        && !ctx.sampling.disable_watchdogs
        && ctx.watchdog.confidence_early_stop
        && a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_tokens as usize + k >= 400
}

impl Prepicked {
    /// Dequantize the `k` rows of `buf` (BF16, `vocab` wide) in parallel,
    /// and compute each row's [`top1_confident`] when `want_f2` is set.
    pub(super) fn build(buf: &[u8], k: usize, vocab: usize, want_f2: bool) -> Self {
        let mut rows = ROWS.with(|r| std::mem::take(&mut *r.borrow_mut()));
        rows.resize_with(k.max(rows.len()), Vec::new);
        let row_bytes = vocab * 2;
        let f2: Vec<Option<bool>> = rows[..k]
            .par_iter_mut()
            .enumerate()
            .map(|(i, row)| {
                dequant_into(&buf[i * row_bytes..(i + 1) * row_bytes], false, vocab, row);
                want_f2.then(|| top1_confident(row))
            })
            .collect();
        Self { rows, f2 }
    }

    /// Row `i`'s pick: the pipeline on the precomputed row, with F2's answer.
    pub(super) fn pick(
        &mut self,
        i: usize,
        vocab: usize,
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> u32 {
        let (row, hint) = (&mut self.rows[i], self.f2[i]);
        with_hint(hint, || super::pick_dequantized(row, vocab, a, ctx))
    }
}

impl Drop for Prepicked {
    fn drop(&mut self) {
        let rows = std::mem::take(&mut self.rows);
        ROWS.with(|r| *r.borrow_mut() = rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parallel dequant + F2 equal the serial functions row for row.
    #[test]
    fn prepicked_rows_match_serial() {
        let (k, vocab) = (4usize, 4099usize);
        let mut x: u32 = 7;
        let buf: Vec<u8> = (0..k * vocab)
            .flat_map(|j| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // One dominant logit per row 1 and 3, flat elsewhere, so the
                // F2 answer differs between rows.
                let v = if j % vocab == 17 && (j / vocab) % 2 == 1 {
                    40.0f32
                } else {
                    (x >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
                };
                ((v.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let pre = Prepicked::build(&buf, k, vocab, true);
        for i in 0..k {
            let mut serial = Vec::new();
            dequant_into(
                &buf[i * vocab * 2..(i + 1) * vocab * 2],
                false,
                vocab,
                &mut serial,
            );
            assert_eq!(
                pre.rows[i].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row {i}"
            );
            assert_eq!(pre.f2[i], Some(top1_confident(&serial)), "row {i}");
        }
        assert_eq!(
            pre.f2,
            vec![Some(false), Some(true), Some(false), Some(true)]
        );
    }
}
