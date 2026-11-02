// SPDX-License-Identifier: AGPL-3.0-only

//! In-flight shared-prefix plants (`pc_inflight`): where a plant lands, and
//! that the head and the worker split and save alike when the plant travels
//! through the command stream.

use super::super::pc_policy::{branch_split_at, tail_cut};
use super::plant_target;

const BS: usize = 16;

#[test]
fn a_plant_ahead_of_the_chunk_is_the_target() {
    // A 22,600-token agentic prompt whose first 22,496 tokens are shared.
    assert_eq!(plant_target(Some(22_496), 0, 0, 22_600, BS), Some(22_496));
    assert_eq!(
        plant_target(Some(22_496), 16_384, 0, 22_600, BS),
        Some(22_496)
    );
    // Restored at a shallower checkpoint: still worth planting.
    assert_eq!(
        plant_target(Some(22_496), 16_384, 16_384, 22_600, BS),
        Some(22_496)
    );
}

#[test]
fn a_plant_that_was_passed_restored_or_covered_by_the_tail_plants_nothing() {
    let total = 22_600;
    // The chunk starts at it: the previous chunk's end decided it.
    assert_eq!(plant_target(Some(16_384), 16_384, 0, total, BS), None);
    // Already computed.
    assert_eq!(plant_target(Some(8_192), 16_384, 0, total, BS), None);
    // At or under the restore depth.
    assert_eq!(plant_target(Some(20_000), 0, 20_000, total, BS), None);
    assert_eq!(plant_target(Some(20_000), 0, 20_016, total, BS), None);
    // At or past the tail cut, which gets its own checkpoint.
    let cut = tail_cut(total, BS);
    assert_eq!(plant_target(Some(cut), 0, 0, total, BS), None);
    assert_eq!(
        plant_target(Some(cut - BS), 0, 0, total, BS),
        Some(cut - BS)
    );
    // Not block-aligned, zero, or no request.
    assert_eq!(plant_target(Some(20_001), 0, 0, total, BS), None);
    assert_eq!(plant_target(Some(0), 0, 0, total, BS), None);
    assert_eq!(plant_target(None, 0, 0, total, BS), None);
}

/// One rank's prefill state in the executable model of
/// `prefill_chunk_dispatch`: the passes it runs and the checkpoints it saves.
#[derive(Default, Debug, PartialEq, Eq)]
struct Rank {
    plant: Option<usize>,
    branch: Option<usize>,
    skip_to: usize,
    passes: Vec<(usize, usize)>,
    saves: Vec<usize>,
}

impl Rank {
    /// `prefill_chunk_dispatch`: tail split, then the plant and the branch
    /// split, then one pass and the non-last chunk's checkpoint decision.
    fn chunk(&mut self, total: usize, start: usize, len: usize, last: bool) {
        let cut = tail_cut(total, BS);
        if last && cut > start && cut < total {
            self.chunk(total, start, cut - start, false);
            return self.chunk(total, cut, total - cut, true);
        }
        if let Some(at) = plant_target(self.plant, start, self.skip_to, total, BS) {
            self.branch = Some(at);
        }
        if let Some(at) = branch_split_at(self.branch, (start, len), false) {
            self.chunk(total, start, at - start, false);
            return self.chunk(total, at, start + len - at, last);
        }
        self.passes.push((start, len));
        let end = start + len;
        if !last && (self.branch == Some(end) || end == cut) {
            self.saves.push(end);
        }
    }
}

/// What the head puts on the wire for one sequence, in order.
#[derive(Clone, Copy)]
enum Word {
    Plant(usize),
    Chunk(usize, usize),
}

/// The head runs `schedule` and sends each step; the worker runs what it
/// reads, in order. Returns (head, worker).
fn lockstep(total: usize, skip_to: usize, schedule: &[Word]) -> (Rank, Rank) {
    let mut head = Rank {
        skip_to,
        ..Default::default()
    };
    let mut wire = Vec::new();
    for &w in schedule {
        wire.push(w);
        match w {
            Word::Plant(at) => head.plant = Some(at),
            Word::Chunk(s, l) => head.chunk(total, s, l, s + l >= total),
        }
    }
    let mut worker = Rank {
        skip_to,
        ..Default::default()
    };
    for w in wire {
        match w {
            Word::Plant(at) => worker.plant = Some(at),
            Word::Chunk(s, l) => worker.chunk(total, s, l, s + l >= total),
        }
    }
    (head, worker)
}

#[test]
fn a_plant_before_chunk_zero_splits_and_saves_alike_on_both_ranks() {
    use Word::*;
    let total = 22_600;
    let sched = [Plant(22_496), Chunk(0, 16_384), Chunk(16_384, 6_216)];
    let (head, worker) = lockstep(total, 0, &sched);
    assert_eq!(head, worker);
    let cut = tail_cut(total, BS);
    assert_eq!(
        head.passes,
        vec![
            (0, 16_384),
            (16_384, 22_496 - 16_384),
            (22_496, cut - 22_496),
            (cut, total - cut)
        ]
    );
    assert_eq!(head.saves, vec![22_496, cut]);
}

/// The leader is already prefilling when a follower arrives: the plant goes
/// between its chunk commands and lands in the next chunk on both ranks.
#[test]
fn a_plant_between_chunks_lands_in_the_next_chunk_on_both_ranks() {
    use Word::*;
    let total = 40_000;
    let sched = [
        Chunk(0, 8_192),
        Plant(30_000),
        Chunk(8_192, 8_192),
        Chunk(16_384, 8_192),
        Chunk(24_576, 8_192),
        Chunk(32_768, 7_232),
    ];
    let (head, worker) = lockstep(total, 0, &sched);
    assert_eq!(head, worker);
    assert!(head.saves.contains(&30_000));
    assert!(head.passes.contains(&(24_576, 30_000 - 24_576)));
}

/// A plant that arrives after its position was computed, or under the
/// restore depth, changes nothing on either rank.
#[test]
fn a_late_or_restored_plant_changes_no_chunk_shape() {
    use Word::*;
    let total = 22_600;
    let plain = lockstep(total, 0, &[Chunk(0, 16_384), Chunk(16_384, 6_216)]).0;
    let late = lockstep(
        total,
        0,
        &[Chunk(0, 16_384), Plant(8_192), Chunk(16_384, 6_216)],
    );
    assert_eq!(late.0, late.1);
    assert_eq!(late.0.passes, plain.passes);
    let restored = lockstep(
        total,
        20_000,
        &[Plant(19_984), Chunk(0, 16_384), Chunk(16_384, 6_216)],
    );
    assert_eq!(restored.0, restored.1);
    assert_eq!(restored.0.passes, plain.passes);
}

/// The model catches a plant applied out of order: had the worker taken it
/// one chunk later than the head, the ranks would run different passes. The
/// command stream rules that out by construction.
#[test]
fn a_plant_out_of_order_would_split_differently() {
    use Word::*;
    let total = 22_600;
    let (head, _) = lockstep(
        total,
        0,
        &[Chunk(0, 8_192), Plant(12_288), Chunk(8_192, 8_192)],
    );
    let (late_worker, _) = lockstep(
        total,
        0,
        &[Chunk(0, 8_192), Chunk(8_192, 8_192), Plant(12_288)],
    );
    assert_ne!(head.passes, late_worker.passes);
}
