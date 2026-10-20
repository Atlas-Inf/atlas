// SPDX-License-Identifier: AGPL-3.0-only

//! The prefix-cache policy (`pc_policy`): placement arithmetic, and the
//! two-rank restore-depth agreement.

use super::{
    Agreed, agree_restore, branch_checkpoint_at, branch_split_at, layer_write_floor, replay_floor,
    tail_cut,
};

const BS: usize = 16;

/// The warm-turn tail checkpoint sits one block below the last block
/// boundary strictly under the prompt end: for a prompt of `n` tokens,
/// `floor((n-1)/16)*16 - 16`. The next turn's block-floored match lands on
/// `floor((n-1)/16)*16` or one block below it, so the checkpoint is always
/// eligible and the anchor-to-match gap is 0-31 tokens.
#[test]
fn tail_cut_is_one_block_below_the_last_boundary() {
    assert_eq!(tail_cut(40_000, BS), 39_968);
    assert_eq!(tail_cut(40_001, BS), 39_984);
    assert_eq!(tail_cut(40_016, BS), 39_984);
    assert_eq!(tail_cut(40_017, BS), 40_000);
    assert_eq!(tail_cut(17, BS), 0);
    assert_eq!(tail_cut(0, BS), 0);
    for n in 33..2_000 {
        let floor = (n - 1) / BS * BS;
        assert_eq!(tail_cut(n, BS), floor - BS, "n={n}");
        assert!(n - tail_cut(n, BS) > BS && n - tail_cut(n, BS) <= 2 * BS);
    }
}

#[test]
fn branch_checkpoint_needs_min_tokens_above_the_restore() {
    // A new session sharing a 30K system prompt, nothing restorable.
    assert_eq!(
        branch_checkpoint_at(30_000, 0, 32_000, BS, 2048),
        Some(30_000)
    );
    // The conversation's own tail restored (0-31 tokens below the match):
    // it continues, no extra pass.
    assert_eq!(branch_checkpoint_at(30_000, 29_968, 32_000, BS, 2048), None);
    assert_eq!(branch_checkpoint_at(30_000, 27_968, 32_000, BS, 2048), None);
    // Only a shallow snapshot (an older, shorter branch point) is below a
    // deep shared prefix: plant the deeper one once.
    assert_eq!(
        branch_checkpoint_at(30_000, 8_000, 32_000, BS, 2048),
        Some(30_000)
    );
    assert_eq!(
        branch_checkpoint_at(30_000, 27_952, 32_000, BS, 2048),
        Some(30_000)
    );
    // Restored at (or, defensively, past) the match: nothing to plant.
    assert_eq!(branch_checkpoint_at(30_000, 30_000, 32_000, BS, 0), None);
    assert_eq!(branch_checkpoint_at(30_000, 30_016, 32_000, BS, 0), None);
}

#[test]
fn branch_checkpoint_respects_the_minimum_and_the_tail() {
    assert_eq!(branch_checkpoint_at(2_032, 0, 10_000, BS, 2048), None);
    assert_eq!(
        branch_checkpoint_at(2_048, 0, 10_000, BS, 2048),
        Some(2_048)
    );
    assert_eq!(branch_checkpoint_at(0, 0, 10_000, BS, 0), None);
    // At or past the tail cut the tail checkpoint already covers it: an
    // identical retried prompt (matched == floor) never pays a second split.
    let total = 30_000;
    let cut = tail_cut(total, BS);
    assert_eq!(branch_checkpoint_at(cut, 0, total, BS, 2048), None);
    assert_eq!(branch_checkpoint_at(cut + BS, 0, total, BS, 2048), None);
    assert_eq!(
        branch_checkpoint_at(cut - BS, 0, total, BS, 2048),
        Some(cut - BS)
    );
}

#[test]
fn split_only_strictly_inside_the_chunk() {
    let at = Some(20_000);
    assert_eq!(branch_split_at(at, (16_384, 8_192), false), Some(20_000));
    // On a chunk boundary: the chunk end saves it, no split.
    assert_eq!(branch_split_at(at, (20_000, 8_192), false), None);
    assert_eq!(branch_split_at(at, (11_808, 8_192), false), None);
    // Outside the chunk, no plan, or verify passengers aboard: no split.
    assert_eq!(branch_split_at(at, (0, 8_192), false), None);
    assert_eq!(branch_split_at(at, (24_576, 8_192), false), None);
    assert_eq!(branch_split_at(None, (16_384, 8_192), false), None);
    assert_eq!(branch_split_at(at, (16_384, 8_192), true), None);
}

/// Splitting a chunk at the planned point and re-checking each half never
/// splits again, so the recursion in `pc_branch_split` is exactly one level.
#[test]
fn a_split_chunk_does_not_split_again() {
    let at = Some(20_000);
    let (start, len) = (16_384, 8_192);
    let a = branch_split_at(at, (start, len), false).unwrap();
    assert_eq!(branch_split_at(at, (start, a - start), false), None);
    assert_eq!(branch_split_at(at, (a, start + len - a), false), None);
}

/// Two ranks running [`agree_restore`] in lockstep over a real min-reduction
/// (a channel pair; a missing peer call times out instead of hanging). Each
/// rank has its local restorable depth and the depths at which it holds an
/// exact-prefix snapshot (id = depth + rank). Returns each rank's result and
/// reduction count.
fn two_ranks(a: (usize, &[usize]), b: (usize, &[usize])) -> [((usize, Agreed), usize); 2] {
    use std::sync::mpsc::channel;
    use std::time::Duration;
    let (to_b, from_a) = channel::<u32>();
    let (to_a, from_b) = channel::<u32>();
    let run = |rank: usize,
               (depth, held): (usize, &[usize]),
               tx: std::sync::mpsc::Sender<u32>,
               rx: std::sync::mpsc::Receiver<u32>| {
        let held = held.to_vec();
        std::thread::spawn(move || {
            let mut calls = 0;
            let min = |v: u32| -> anyhow::Result<u32> {
                calls += 1;
                tx.send(v)?;
                Ok(v.min(rx.recv_timeout(Duration::from_secs(5))?))
            };
            let probe = |at: usize| held.contains(&at).then_some(at + rank);
            let r = agree_restore(depth, min, probe).expect("no collective mismatch");
            (r, calls)
        })
    };
    let ta = run(0, a, to_b, from_b);
    let tb = run(1, b, to_a, from_a);
    [ta.join().unwrap(), tb.join().unwrap()]
}

#[test]
fn equal_depths_restore_locally_on_both_ranks() {
    let [a, b] = two_ranks((30_000, &[30_000]), (30_000, &[30_000]));
    assert_eq!(a, ((30_000, Agreed::Local), 2));
    assert_eq!(b, ((30_000, Agreed::Local), 2));
}

#[test]
fn deeper_rank_restores_at_the_shallower_depth_when_it_holds_it() {
    // Rank A kept a deeper tail; B's pool lost it and restores at 20K. A
    // still holds an exact snapshot at 20K.
    let [a, b] = two_ranks((30_000, &[20_000, 30_000]), (20_000, &[20_000]));
    assert_eq!(a, ((20_000, Agreed::At(20_000)), 2));
    assert_eq!(b, ((20_000, Agreed::Local), 2));
}

/// Eviction victims diverged: A evicted the 20K snapshot B restores from.
/// Neither rank restores, and both issue the same reductions.
#[test]
fn diverged_victims_fall_back_to_recompute_everywhere() {
    let [a, b] = two_ranks((30_000, &[30_000]), (20_000, &[20_000]));
    assert_eq!(a, ((0, Agreed::None), 2));
    assert_eq!(b, ((0, Agreed::None), 2));
}

#[test]
fn a_rank_with_nothing_to_restore_forces_a_full_recompute() {
    for (x, y) in [
        ((30_000, &[30_000][..]), (0, &[][..])),
        ((0, &[][..]), (0, &[][..])),
    ] {
        let [a, b] = two_ranks(x, y);
        assert_eq!(a, ((0, Agreed::None), 1));
        assert_eq!(b, ((0, Agreed::None), 1));
    }
}

/// `ATLAS_GLM_PC_WRITE_FLOOR`: a recompute-all prefix hit (base floor 0)
/// floors each pass at the rows it spends under the radix match. Off, on any
/// model but GLM, and for every Marconi replay and cold pass, the base floor
/// is unchanged.
#[test]
fn write_floor_covers_the_matched_rows_of_a_recompute() {
    let glm = |base, matched, start, rows| {
        layer_write_floor(true, "glm5_next", base, matched, start, rows)
    };
    // Off, or another hybrid: whatever the base chose.
    for (base, matched) in [(0, 0), (0, 4096), (32, 4096)] {
        for (flag, model) in [(false, "glm5_next"), (true, "qwen3_next")] {
            let floor = layer_write_floor(flag, model, base, matched, 0, 8192);
            assert_eq!(floor, base, "{flag} {model}");
        }
    }
    // Cold (no match): nothing to protect.
    assert_eq!(glm(0, 0, 0, 4096), 0);
    // A 20 000-token match recomputed in 8192-row chunks: the first two are
    // wholly shared, the third is shared up to the match, later ones are new.
    assert_eq!(glm(0, 20_000, 0, 8192), 8192);
    assert_eq!(glm(0, 20_000, 8192, 8192), 8192);
    assert_eq!(glm(0, 20_000, 16_384, 8192), 3616);
    assert_eq!(glm(0, 20_000, 24_576, 100), 0);
    // A Marconi replay already floors at the match: the flag adds nothing.
    assert_eq!(glm(32, 4096, 4064, 500), 32);
}

// A Marconi replay keeps its rows under the match unwritten; a pass without a
// restore takes the lookup's floor (0 for a full recompute).
#[test]
fn replay_floor_covers_the_replayed_matched_rows() {
    assert_eq!(replay_floor(true, 320, 256, 256, 100), 64);
    assert_eq!(replay_floor(true, 320, 256, 256, 40), 40);
    assert_eq!(replay_floor(true, 256, 256, 256, 100), 0);
    assert_eq!(replay_floor(false, 64, 0, 0, 70), 0);
}
