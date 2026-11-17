// SPDX-License-Identifier: AGPL-3.0-only

//! What the draft head keeps of one sequence, and how the layers it reads
//! hand it over.
//!
//! The head reads the main model at three layers: the mean of the stream's
//! copies at each one's input, for the last 128 positions. Those layers
//! leave them here as they run, a ring a sequence, and a draft pass turns
//! the positions it has not yet seen into a window key for each of its
//! stages. A pass runs right after the forward of the sequence it drafts
//! for, on every rank, so the sequence a pass reads is the one whose layers
//! ran last ([`Tap`]).
//!
//! A verify writes rows that may be refused, and a prefill only its last
//! rows. Neither needs taking back: a pass reads only positions before the
//! sequence's end, which are final, and the ranges below say which of them
//! the rings hold.

use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::cache_recipe::WINDOW;
use super::ops::HEAD_DIM;

/// `ATLAS_DSV41_DRAFT`: serve the release's draft head. Every rank of a
/// model must agree, for each holds its part of the head.
pub fn requested() -> bool {
    std::env::var("ATLAS_DSV41_DRAFT").is_ok_and(|v| v != "0")
}

/// Where the layers the head reads leave the sequence they last ran. A
/// process serves one model.
pub fn tap() -> &'static Tap {
    static TAP: OnceLock<Tap> = OnceLock::new();
    TAP.get_or_init(Tap::default)
}

/// The sequence whose target layers ran last.
#[derive(Default)]
pub struct Tap {
    current: Mutex<Option<Arc<Mutex<DraftSeq>>>>,
}

impl Tap {
    pub fn set(&self, seq: &Arc<Mutex<DraftSeq>>) {
        *self.current.lock().expect("draft tap") = Some(seq.clone());
    }

    pub fn current(&self) -> Option<Arc<Mutex<DraftSeq>>> {
        self.current.lock().expect("draft tap").clone()
    }

    /// `seq` is ending: no pass may read it again.
    pub fn forget(&self, seq: &Arc<Mutex<DraftSeq>>) {
        let mut current = self.current.lock().expect("draft tap");
        if current.as_ref().is_some_and(|held| Arc::ptr_eq(held, seq)) {
            *current = None;
        }
    }
}

/// What a pass after `len` tokens does first: the positions it makes window
/// keys for, and the positions whose keys its rows then read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub ingest: Range<usize>,
    pub seen: Range<usize>,
}

impl Plan {
    /// The ring slots the block's rows read, a window's worth: the seen
    /// positions' in order, then none.
    pub fn slots(&self) -> Vec<i32> {
        let mut row: Vec<i32> = self.seen.clone().map(|at| (at % WINDOW) as i32).collect();
        row.resize(WINDOW, -1);
        row
    }
}

/// The run of positions a layer has written, after it writes `rows` from
/// `first`: one that starts inside the run or at its end carries it on, and
/// anything else starts another.
pub fn written(run: &Range<usize>, first: usize, rows: usize) -> Range<usize> {
    match (run.start..=run.end).contains(&first) {
        true => run.start..first + rows,
        false => first..first + rows,
    }
}

/// The runs of ring slots that hold `positions`, at most a window of them,
/// in order: (slot, count).
pub fn slot_runs(positions: &Range<usize>) -> Vec<(usize, usize)> {
    let (slot, count) = (positions.start % WINDOW, positions.len().min(WINDOW));
    let first = count.min(WINDOW - slot);
    [(slot, first), (0, count - first)]
        .into_iter()
        .filter(|&(_, count)| count > 0)
        .collect()
}

/// The plan of a pass after `len` tokens, given the positions every target
/// layer has written (`have`) and those the stages hold keys for (`held`).
/// None when the layers have not written up to the sequence's end, or the
/// sequence is too short to draft for.
pub fn plan(have: &Range<usize>, held: &Range<usize>, len: usize) -> Option<Plan> {
    if len < 2 || have.end < len || have.start >= len {
        return None;
    }
    let oldest = len.saturating_sub(WINDOW);
    let floor = oldest.max(have.start);
    // The keys held carry on when the positions after them are all here.
    let carries = !held.is_empty() && held.end >= floor && held.end <= len;
    let (start, from) = match carries {
        true => (held.start.max(oldest), held.end),
        false => (floor, floor),
    };
    Some(Plan {
        ingest: from..len,
        seen: start..len,
    })
}

/// One sequence's part of the draft head on one rank.
pub struct DraftSeq {
    /// `[128, targets * hidden]` BF16: slot `position % 128` holds the
    /// position's stream means at the target layers, side by side.
    pub means: DevicePtr,
    /// A stage's window: `[128, 512]` BF16, slot `position % 128` the key of
    /// the main model's position.
    pub windows: Vec<DevicePtr>,
    /// The positions each target layer has written.
    written: Vec<Range<usize>>,
    /// The positions the windows hold keys for.
    held: Range<usize>,
}

impl DraftSeq {
    pub fn alloc(
        gpu: &dyn GpuBackend,
        hidden: usize,
        targets: usize,
        stages: usize,
    ) -> Result<Self> {
        let window = || -> Result<DevicePtr> {
            let keys = gpu.alloc(WINDOW * HEAD_DIM * 2)?;
            gpu.memset(keys, 0, WINDOW * HEAD_DIM * 2)?;
            Ok(keys)
        };
        Ok(Self {
            means: gpu.alloc(WINDOW * targets * hidden * 2)?,
            windows: (0..stages).map(|_| window()).collect::<Result<_>>()?,
            written: vec![0..0; targets],
            held: 0..0,
        })
    }

    pub fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.means)?;
        self.windows.iter().try_for_each(|&keys| gpu.free(keys))
    }

    /// Target layer `target` is about to run `rows` tokens from `position`:
    /// the means of their streams (`[rows, copies, hidden]` BF16) into the
    /// ring, the last window of them when there are more
    /// (`dsv41_stream_mean`).
    #[allow(clippy::too_many_arguments)]
    pub fn take(
        &mut self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        target: usize,
        streams: DevicePtr,
        rows: usize,
        position: usize,
        hidden: usize,
        copies: usize,
        stream: u64,
    ) -> Result<()> {
        let wide = self.written.len() * hidden;
        let mut done = rows.saturating_sub(WINDOW);
        let first = position + done;
        for (slot, count) in slot_runs(&(first..position + rows)) {
            KernelLaunch::new(gpu, kernel)
                .grid([div_ceil((count * hidden) as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(streams.offset(done * copies * hidden * 2))
                .arg_ptr(self.means.offset((slot * wide + target * hidden) * 2))
                .arg_u32(hidden as u32)
                .arg_u32(copies as u32)
                .arg_u32(wide as u32)
                .arg_u32(count as u32)
                .launch(stream)?;
            done += count;
        }
        self.wrote(target, first, position + rows - first);
        Ok(())
    }

    /// Target layer `target` wrote the means of `rows` positions from
    /// `first`. Keys made from the rows they replace are gone.
    pub fn wrote(&mut self, target: usize, first: usize, rows: usize) {
        self.written[target] = written(&self.written[target], first, rows);
        self.held.end = self.held.end.min(first);
        self.held.start = self.held.start.min(self.held.end);
    }

    pub fn plan(&self, len: usize) -> Option<Plan> {
        let have = self.written.iter().fold(0..usize::MAX, |all, run| {
            all.start.max(run.start)..all.end.min(run.end)
        });
        plan(&have, &self.held, len)
    }

    /// The pass `plan` was for has run.
    pub fn ran(&mut self, plan: &Plan) {
        self.held = plan.seen.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::{Plan, plan, slot_runs, written};

    #[test]
    fn a_layers_rows_carry_its_run_on_or_start_another() {
        // A prompt's last rows, then decode steps after them.
        let run = written(&(0..0), 5000, 128);
        assert_eq!(run, 5000..5128);
        assert_eq!(written(&run, 5128, 1), 5000..5129);
        // A verify's rows, and the next from where its accepted rows end.
        let run = written(&(5000..5129), 5129, 6);
        assert_eq!(written(&run, 5131, 6), 5000..5137);
        // A new prompt in the same sequence starts over.
        assert_eq!(written(&run, 0, 40), 0..40);
        assert_eq!(written(&run, 9000, 128), 9000..9128);
    }

    #[test]
    fn a_ring_run_wraps_once() {
        assert_eq!(slot_runs(&(0..5)), vec![(0, 5)]);
        assert_eq!(slot_runs(&(126..130)), vec![(126, 2), (0, 2)]);
        assert_eq!(slot_runs(&(256..384)), vec![(0, 128)]);
        assert_eq!(slot_runs(&(300..428)), vec![(44, 84), (0, 44)]);
        assert!(slot_runs(&(7..7)).is_empty());
    }

    #[test]
    fn a_pass_makes_the_keys_it_lacks_and_reads_a_window() {
        // After a prompt: the window whole, all of it new.
        let first = plan(&(4872..5000), &(0..0), 5000).expect("a plan");
        assert_eq!(
            first,
            Plan {
                ingest: 4872..5000,
                seen: 4872..5000
            }
        );
        // After a verify that kept three rows: those three.
        let next = plan(&(4872..5006), &first.seen, 5003).expect("a plan");
        assert_eq!(
            next,
            Plan {
                ingest: 5000..5003,
                seen: 4875..5003
            }
        );
        // A short sequence reads what there is.
        let short = plan(&(0..9), &(0..0), 9).expect("a plan");
        assert_eq!(short.seen, 0..9);
        assert_eq!(short.slots()[..10], [0, 1, 2, 3, 4, 5, 6, 7, 8, -1]);
        assert_eq!(short.slots().len(), 128);
    }

    #[test]
    fn keys_do_not_carry_over_a_gap() {
        // Steps the head did not draft for, more than a window of them.
        let late = plan(&(0..900), &(100..300), 900).expect("a plan");
        assert_eq!(late.ingest, 772..900);
        assert_eq!(late.seen, 772..900);
        // Fewer: the keys held carry on.
        let soon = plan(&(0..350), &(172..300), 350).expect("a plan");
        assert_eq!(soon.ingest, 300..350);
        assert_eq!(soon.seen, 222..350);
        // The layers' rows start after the keys held end: start over there.
        let apart = plan(&(820..900), &(700..800), 900).expect("a plan");
        assert_eq!(apart.ingest, 820..900);
        assert_eq!(apart.seen, 820..900);
    }

    #[test]
    fn no_pass_before_the_layers_have_written_the_end() {
        assert_eq!(plan(&(0..10), &(0..0), 11), None);
        assert_eq!(plan(&(0..1), &(0..0), 1), None);
        assert_eq!(plan(&(0..0), &(0..0), 5), None);
    }
}
