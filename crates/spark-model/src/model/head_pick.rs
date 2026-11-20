// SPDX-License-Identifier: AGPL-3.0-only
//! A step's largest logit without its logits, for a model whose ranks
//! project the head in shares (`head_shares`) and hold an FP8 copy of their
//! share (DeepSeek-V4.1 with its draft head; `ATLAS_DSV41_HEAD_PICK=0` for the
//! head in shares at every step).
//!
//! A greedy step asks its logits for their argmax and nothing else, and the
//! pick (`layers::deepseek_v41::head`) gives exactly that token for half
//! the head's bytes. So where the ranks pick, a decode step or a verify
//! leaves the token of each row and no logits, and marks the rows of the
//! logits buffer so: the argmax of a marked row is its token, and whoever
//! reads a marked row's bytes (a sampled request, logprobs, a penalty, a
//! masked pick) first has the head run on the rows of that step, here, on
//! the rank that serves the request. That is the whole head on one rank,
//! four times what a share costs, so a step whose logits are read is a
//! reason to stop picking for a while: the ranks then run the head in
//! shares as before, and try picking again after a number of steps that
//! doubles each time the try is read again. A request that samples costs a
//! whole head a few times in thousands of steps; a greedy one whose logits
//! are read once (a banned first token, the end of thinking) costs one.
//!
//! Whether the ranks pick is every rank's, since a pick and a head in
//! shares are different collectives: the rank that serves requests says so
//! before the next command it starts ([`EP_CMD_HEAD_PICK`]), which is the
//! one point every rank is known to be between steps.
//!
//! Exactness: the pick is `argmax_bf16` of the row the head in shares would
//! have left (`dsv41_head.cu`), and the head run for a reader is
//! `head_project`'s GEMV over the whole vocabulary, whose columns are the
//! shares' bytes.
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::DevicePtr;

use super::qwen4exp_lmhead_split::HeadArith;
use super::types::TransformerModel;
use crate::layers::deepseek_v41::head::{HeadPick, Mark, ROWS};

/// Head -> worker: whether the ranks pick from here on (a u32 follows, 0 or
/// 1).
pub(in crate::model) const EP_CMD_HEAD_PICK: u32 = 0xFFFF_FFE4;

/// Steps without picking after a step whose logits were read, the first
/// time and at most; and the picks in a row after which the count starts
/// over.
const HOLD_FIRST: u32 = 8;
const HOLD_MOST: u32 = 4096;
const SETTLED: u32 = 64;

/// Unless `ATLAS_DSV41_HEAD_PICK=0`. Every rank must say the same
/// (`startup_parity`): a pick and a head in shares are different
/// collectives.
pub fn allowed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DSV41_HEAD_PICK").as_deref() != Ok("0"))
}

/// When the rank that serves requests would have the ranks pick.
#[derive(Debug)]
struct Hold {
    want: bool,
    /// Steps left before it would pick again.
    left: u32,
    /// The hold the next read costs, and the picks since the last read.
    next: u32,
    unread: u32,
}

impl Default for Hold {
    fn default() -> Self {
        Self {
            want: true,
            left: 0,
            next: HOLD_FIRST,
            unread: 0,
        }
    }
}

impl Hold {
    /// A step that ran the head in shares: one fewer to hold for.
    fn held_a_step(&mut self) {
        self.left = self.left.saturating_sub(1);
        self.want = self.left == 0;
    }

    /// A step that picked.
    fn picked_a_step(&mut self) {
        self.unread += 1;
        if self.unread >= SETTLED {
            self.next = HOLD_FIRST;
        }
    }

    /// A picked step's logits were read: no picks for a while, and for
    /// twice as long the next time.
    fn was_read(&mut self) {
        self.left = self.next;
        self.next = (self.next * 2).min(HOLD_MOST);
        self.unread = 0;
        self.want = false;
    }
}

/// `ATLAS_DSV41_HEAD_PICK_CHECK=1`: every pick beside the head in shares,
/// on every rank, and an error where the head's argmax is another token.
/// The step then has its logits too, and costs both.
pub fn check_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DSV41_HEAD_PICK_CHECK").as_deref() == Ok("1"))
}

pub struct HeadPicks {
    pick: HeadPick,
    /// Whether the ranks pick: the same on every rank at every step.
    on: AtomicBool,
    hold: Mutex<Hold>,
    /// The last pick: its step, its rows, the hidden rows it read, and
    /// whether its rows of the logits may still be marked.
    step: AtomicU32,
    rows: AtomicUsize,
    normed: AtomicU64,
    marked: AtomicBool,
}

impl HeadPicks {
    pub fn new(pick: HeadPick) -> Self {
        Self {
            pick,
            on: AtomicBool::new(true),
            hold: Mutex::default(),
            step: AtomicU32::new(0),
            rows: AtomicUsize::new(0),
            normed: AtomicU64::new(0),
            marked: AtomicBool::new(false),
        }
    }
}

impl TransformerModel {
    /// Install the pick: every rank's, built from its share of the head.
    pub fn set_head_pick(&mut self, pick: HeadPick) {
        self.head_pick = Some(HeadPicks::new(pick));
    }

    /// The tokens of the largest logits of the first `rows` rows of
    /// `normed`, in place of their logits: `[rows]` u32 on the device, and
    /// the rows of `self.buffers.logits()` marked. `None`, nothing launched:
    /// the caller runs the head, as every rank's does.
    pub(in crate::model) fn head_pick_rows(
        &self,
        normed: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let Some(picks) = self.head_pick.as_ref() else {
            return Ok(None);
        };
        if !picks.on.load(Relaxed) || rows > ROWS {
            picks.hold.lock().held_a_step();
            return Ok(None);
        }
        let step = picks.step.load(Relaxed).wrapping_add(1);
        let mark = Mark {
            logits: self.buffers.logits(),
            vocab: self.config.vocab_size,
            step,
        };
        let comm = self.comm_ref();
        let picked = picks
            .pick
            .pick(self.gpu.as_ref(), comm, normed, rows, Some(mark), stream)?;
        if check_requested() {
            self.head_pick_check(picks, normed, rows, picked, stream)?;
        }
        picks.step.store(step, Relaxed);
        picks.rows.store(rows, Relaxed);
        picks.normed.store(normed.0, Relaxed);
        picks.marked.store(true, Relaxed);
        picks.hold.lock().picked_a_step();
        Ok(Some(picked))
    }

    /// [`check_requested`]: the head in shares over the rows just picked
    /// from, and its argmax of each against the pick.
    fn head_pick_check(
        &self,
        picks: &HeadPicks,
        normed: DevicePtr,
        rows: usize,
        picked: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let mut bytes = vec![0u8; rows * 4];
        self.gpu.copy_d2h(picked, &mut bytes)?;
        let picked: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let kept = picks.pick.kept(self.gpu.as_ref(), rows)?;
        ensure!(
            self.head_rows_in_shares(normed, rows, stream)?,
            "deepseek_v41: the head in shares declined beside a pick"
        );
        // The rows are logits now, so this is the head's own argmax.
        let whole = crate::traits::Model::argmax_batch(self, self.buffers.logits(), rows, stream)?;
        ensure!(
            whole == picked,
            "deepseek_v41: picked {picked:?} where the head's argmax is {whole:?} (kept {kept:?} of this rank's tokens)"
        );
        static CHECKED: AtomicU64 = AtomicU64::new(0);
        static KEPT: AtomicU64 = AtomicU64::new(0);
        static MOST: AtomicU32 = AtomicU32::new(0);
        let checked = CHECKED.fetch_add(rows as u64, Relaxed) + rows as u64;
        let total = KEPT.fetch_add(kept.iter().map(|&k| u64::from(k)).sum(), Relaxed);
        let most = MOST.fetch_max(kept.iter().copied().max().unwrap_or(0), Relaxed);
        if checked / 256 != (checked - rows as u64) / 256 {
            tracing::info!(
                "deepseek_v41 pick check: {checked} rows equal the head's argmax; this rank kept {total} tokens of them in all, {most} of one row at most"
            );
        }
        Ok(())
    }

    /// The last pick's row that `logits_ptr` lies in and what its mark
    /// says, if that row may still be marked: `(row, token)`.
    fn head_marked_row(&self, logits_ptr: DevicePtr) -> Result<Option<(usize, u32)>> {
        let Some(picks) = self.head_pick.as_ref() else {
            return Ok(None);
        };
        if !picks.marked.load(Relaxed) {
            return Ok(None);
        }
        let row_bytes = self.config.vocab_size * 2;
        let Some(offset) = logits_ptr.0.checked_sub(self.buffers.logits().0) else {
            return Ok(None);
        };
        let row = offset as usize / row_bytes;
        if row >= picks.rows.load(Relaxed) {
            return Ok(None);
        }
        let mut bytes = [0u8; Mark::BYTES];
        self.gpu
            .copy_d2h(self.buffers.logits().offset(row * row_bytes), &mut bytes)?;
        let token = Mark::read(&bytes, picks.step.load(Relaxed));
        if token.is_none() {
            // A head has run over the row since.
            picks.marked.store(false, Relaxed);
        }
        Ok(token.map(|token| (row, token)))
    }

    /// The argmax of each of the `n` rows of logits at `logits_ptr`, where
    /// every one of them is a row the last pick marked. `None`: they are
    /// logits, and the caller's argmax reads them; rows of which only some
    /// are marked are made logits first.
    pub(in crate::model) fn head_picked(
        &self,
        logits_ptr: DevicePtr,
        n: usize,
    ) -> Result<Option<Vec<u32>>> {
        let row_bytes = self.config.vocab_size * 2;
        let mut picked = Vec::with_capacity(n);
        for i in 0..n {
            match self.head_marked_row(logits_ptr.offset(i * row_bytes))? {
                Some((_, token)) => picked.push(token),
                None if picked.is_empty() => return Ok(None),
                None => {
                    self.head_whole_for_read(logits_ptr)?;
                    return Ok(None);
                }
            }
        }
        Ok(Some(picked))
    }

    /// Before the bytes at `logits_ptr` are read: if they lie in a row the
    /// last pick marked, the head on that step's rows, here, so that they
    /// are logits; and no picks for a while.
    pub(in crate::model) fn head_whole_for_read(&self, logits_ptr: DevicePtr) -> Result<()> {
        if self.head_marked_row(logits_ptr)?.is_none() {
            return Ok(());
        }
        let picks = self.head_pick.as_ref().expect("a marked row");
        let (rows, vocab) = (picks.rows.load(Relaxed), self.config.vocab_size);
        self.head_project(
            HeadArith::Gemv,
            DevicePtr(picks.normed.load(Relaxed)),
            0,
            vocab,
            rows,
            self.buffers.logits(),
            vocab,
            self.gpu.default_stream(),
        )?;
        picks.marked.store(false, Relaxed);
        let mut hold = picks.hold.lock();
        hold.was_read();
        tracing::debug!(
            rows,
            hold = hold.left,
            "deepseek_v41: a picked step's logits were read; the head in shares for a while"
        );
        Ok(())
    }

    /// Head, before a command it starts: tell the ranks if whether they
    /// pick changes, and change with them. A command of its own, with the
    /// preamble the workers' idle receive expects.
    pub(super) fn head_pick_agree(&self) -> Result<()> {
        let Some(picks) = self.head_pick.as_ref() else {
            return Ok(());
        };
        let want = picks.hold.lock().want;
        if want == picks.on.load(Relaxed) {
            return Ok(());
        }
        if self.ep_protocol_v2 {
            self.ep_broadcast_u32(0)?;
        }
        self.ep_broadcast_u32(EP_CMD_HEAD_PICK)?;
        self.ep_broadcast_u32(u32::from(want))?;
        picks.on.store(want, Relaxed);
        Ok(())
    }

    /// Worker side of [`EP_CMD_HEAD_PICK`].
    pub(super) fn head_pick_receive(&self) -> Result<()> {
        let on = self.ep_broadcast_u32(0)? != 0;
        let picks = self.head_pick.as_ref();
        let picks = picks.ok_or_else(|| {
            anyhow::anyhow!("the head said whether the ranks pick, and this rank holds no pick")
        })?;
        picks.on.store(on, Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{HOLD_FIRST, HOLD_MOST, Hold, SETTLED};

    fn steps(hold: &mut Hold, held: u32) {
        (0..held).for_each(|_| hold.held_a_step());
    }

    #[test]
    fn a_read_step_holds_the_picks_off_twice_as_long_each_time() {
        let mut hold = Hold::default();
        assert!(hold.want);
        // Read: no picks for the first hold, then picks again.
        hold.was_read();
        assert!(!hold.want);
        steps(&mut hold, HOLD_FIRST - 1);
        assert!(!hold.want);
        steps(&mut hold, 1);
        assert!(hold.want);
        // Read on the first try again: twice as long, up to the most.
        hold.was_read();
        steps(&mut hold, 2 * HOLD_FIRST - 1);
        assert!(!hold.want);
        steps(&mut hold, 1);
        assert!(hold.want);
        (0..20).for_each(|_| hold.was_read());
        assert_eq!(hold.left, HOLD_MOST);
        // Picks that nobody reads: the next read costs the first hold.
        steps(&mut hold, HOLD_MOST);
        (0..SETTLED).for_each(|_| hold.picked_a_step());
        hold.was_read();
        assert_eq!(hold.left, HOLD_FIRST);
    }
}
