// SPDX-License-Identifier: AGPL-3.0-only

//! Scratch layout, policy and LSE-merge math of the GLM KV shard. The merge
//! tests run a CPU mirror of the split kernel's partial semantics and of
//! `glm_sparse_decode_split_merge` against plain softmax attention.

use super::*;

#[path = "glm_kv_shard_settings_tests.rs"]
mod settings;

fn disjoint(regions: &[(usize, usize)]) -> bool {
    let mut sorted = regions.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0)
}

#[test]
fn merge_layout_regions_are_aligned_and_disjoint() {
    for rows in 1..=MERGE_MAX_ROWS as u32 {
        let splits = MERGE_SPLITS;
        let m = MergeLayout::new(rows, splits);
        let (r, s) = (rows as usize, splits as usize);
        let part = r * 32 * 512 * 4;
        let lse = r * 32 * 4;
        let regions = [
            (m.q_peer, r * 32 * 512 * 2),
            (m.ids, r * 2051 * 4),
            (m.own_o, (s + 1) * part),
            (m.own_lse, (s + 1) * lse),
            (m.out_lse, lse),
            (m.peer_o, s * part),
            (m.peer_lse, s * lse),
            (m.send, MergeLayout::partial_bytes(rows)),
            (m.recv, MergeLayout::partial_bytes(rows)),
        ];
        assert!(disjoint(&regions), "rows {rows}");
        assert!(regions.iter().all(|&(o, _)| o % 256 == 0));
        assert!(regions.iter().all(|&(o, n)| o + n <= m.total));
        // The compact form's row counts sit in the own partition the
        // in-place merge leaves free, past the `splits` it fills.
        let counts = m.counts(rows, splits);
        assert_eq!(counts, m.own_o + s * part);
        assert!(counts + r * 4 <= m.own_o + (s + 1) * part && counts.is_multiple_of(4));
    }
}

#[test]
fn scratch_layout_holds_views_pieces_and_the_widest_owner() {
    let bytes = 16 * 528;
    let l = ScratchLayout::new(32772, 8256, bytes);
    let widest = (1..=MERGE_MAX_ROWS as u32)
        .map(|r| MergeLayout::new(r, MERGE_SPLITS).total)
        .max()
        .unwrap();
    let work = (2 * (PIECE_BLOCKS * bytes).next_multiple_of(256)).max(widest);
    let regions = [
        (l.view, 32772 * bytes),
        (l.mine_slot, 32772 * 4),
        (l.mine_dst, 32772 * 4),
        (l.peer_dst, 32772 * 4),
        (l.check, 16),
        (l.slots, 8256 * 8),
        (l.work, work),
    ];
    assert!(disjoint(&regions));
    assert_eq!(l.total, l.work + work.next_multiple_of(256));
    // 512K tokens of fp8_g128 latents: a 264 MiB view + ~47 MiB of work
    // (64 rows in MERGE_SPLITS partitions per group).
    assert!(l.total < 320 << 20, "{}", l.total);
    assert!(widest < 48 << 20, "{widest}");
}

/// One partition's normalized output and natural LSE (`-inf`, zeros when
/// it selects nothing): the `*_split` kernel's contract.
fn partial(q: &[f32], keys: &[Vec<f32>], ids: &[Option<usize>], scale: f32) -> (Vec<f32>, f32) {
    let scores: Vec<(usize, f32)> = ids
        .iter()
        .flatten()
        .map(|&t| {
            (
                t,
                scale * q.iter().zip(&keys[t]).map(|(a, b)| a * b).sum::<f32>(),
            )
        })
        .collect();
    let Some(m) = scores.iter().map(|s| s.1).reduce(f32::max) else {
        return (vec![0.0; q.len()], f32::NEG_INFINITY);
    };
    let mut out = vec![0.0; q.len()];
    let mut l = 0.0f32;
    for &(t, s) in &scores {
        let p = (s - m).exp();
        l += p;
        out.iter_mut().zip(&keys[t]).for_each(|(o, v)| *o += p * v);
    }
    out.iter_mut().for_each(|o| *o /= l);
    (out, m + l.ln())
}

/// CPU mirror of `glm_sparse_decode_split_merge` for one (row, head).
fn merge(parts: &[(Vec<f32>, f32)]) -> (Vec<f32>, f32) {
    let dim = parts[0].0.len();
    let live: Vec<&(Vec<f32>, f32)> = parts.iter().filter(|p| p.1 != f32::NEG_INFINITY).collect();
    let Some(mx) = live.iter().map(|p| p.1).reduce(f32::max) else {
        return (vec![0.0; dim], f32::NEG_INFINITY);
    };
    let weights: Vec<f32> = live.iter().map(|p| (p.1 - mx).exp()).collect();
    let sum: f32 = weights.iter().sum();
    let mut out = vec![0.0; dim];
    for (p, w) in live.iter().zip(&weights) {
        out.iter_mut()
            .zip(&p.0)
            .for_each(|(o, v)| *o += w / sum * v);
    }
    (out, mx + sum.ln())
}

/// Deterministic pseudo-random values in [-1, 1).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// A row's sharded attention: each rank's `splits` partitions over the
/// selected IDs it owns; the peer's heads get one FP32-merged partial, which
/// joins the own partitions in the final merge.
fn sharded(
    q: &[f32],
    keys: &[Vec<f32>],
    selected: &[i32],
    owner: impl Fn(usize) -> usize,
    rank: usize,
    splits: usize,
    scale: f32,
) -> Vec<f32> {
    let local = |r: usize| -> Vec<Option<usize>> {
        selected
            .iter()
            .map(|&t| (t >= 0 && owner(t as usize) == r).then_some(t as usize))
            .collect()
    };
    let parts_of = |r: usize| -> Vec<(Vec<f32>, f32)> {
        let ids = local(r);
        let per = ids.len().div_ceil(splits);
        ids.chunks(per.max(1))
            .map(|c| partial(q, keys, c, scale))
            .collect()
    };
    let mut own = parts_of(rank);
    own.push(merge(&parts_of(1 - rank)));
    merge(&own).0
}

#[test]
fn owner_split_lse_merge_equals_full_softmax_attention() {
    let (dim, tokens) = (64usize, 300usize);
    let keys: Vec<Vec<f32>> = (0..tokens).map(|t| values(t as u64 + 7, dim)).collect();
    let q = values(99, dim);
    let scale = 0.0625;
    // A sparse selection with padding, blocks of 16 owned by parity.
    let mut selected: Vec<i32> = (0..tokens as i32).filter(|t| t % 3 != 1).collect();
    selected.extend([-1, -1, -1]);
    let all: Vec<Option<usize>> = selected
        .iter()
        .map(|&t| (t >= 0).then_some(t as usize))
        .collect();
    let (want, _) = partial(&q, &keys, &all, scale);
    let owners: [fn(usize) -> usize; 2] = [|t| (t / 16) % 2, |t| usize::from(t >= 250)];
    for block_owner in owners {
        for rank in 0..2 {
            for splits in [1, 4, 15] {
                let got = sharded(&q, &keys, &selected, block_owner, rank, splits, scale);
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "{g} vs {w}");
                }
            }
        }
    }
}

#[test]
fn a_rank_owning_nothing_contributes_nothing() {
    let keys: Vec<Vec<f32>> = (0..20).map(|t| values(t + 1, 8)).collect();
    let q = values(5, 8);
    let selected: Vec<i32> = (0..20).collect();
    let all: Vec<Option<usize>> = (0..20).map(Some).collect();
    let (want, _) = partial(&q, &keys, &all, 0.5);
    for rank in 0..2 {
        let got = sharded(&q, &keys, &selected, |_| 1, rank, 3, 0.5);
        assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-6));
    }
    // Nothing selected at all: zeros, as the unsharded kernel writes.
    let none = sharded(&q, &keys, &[-1, -1], |_| 0, 0, 2, 0.5);
    assert!(none.iter().all(|&v| v == 0.0));
}

/// CPU mirror of `glm_kv_shard_localize_compact` for one row: 256 threads
/// of `ceil(width / 256)` consecutive IDs, kept IDs packed by an exclusive
/// scan of the threads' counts, dropped ones behind them. Returns the row
/// and its count.
fn compact_row(selected: &[i32], owner: impl Fn(usize) -> usize, rank: usize) -> (Vec<i32>, usize) {
    let width = selected.len();
    let chunk = width.div_ceil(256);
    let kept: Vec<Vec<i32>> = (0..256)
        .map(|t| {
            let (begin, end) = ((t * chunk).min(width), ((t + 1) * chunk).min(width));
            selected[begin..end]
                .iter()
                .copied()
                .filter(|&id| id >= 0 && owner(id as usize) == rank)
                .collect()
        })
        .collect();
    let total: usize = kept.iter().map(Vec::len).sum();
    let mut out = vec![i32::MIN; width];
    let mut before = 0;
    for (t, ids) in kept.iter().enumerate() {
        let (begin, end) = ((t * chunk).min(width), ((t + 1) * chunk).min(width));
        out[before..before + ids.len()].copy_from_slice(ids);
        let dropped = begin - before;
        let n = (end - begin) - ids.len();
        out[total + dropped..total + dropped + n].fill(-1);
        before += ids.len();
    }
    (out, total)
}

/// The counted split kernel's partition `z` of a row of `count` IDs over
/// `splits`: whole 32-ID tiles, `ceil(tiles / splits)` each.
fn counted_partition(count: usize, splits: usize, z: usize) -> std::ops::Range<usize> {
    let per = count.div_ceil(32).div_ceil(splits);
    let begin = (z * per * 32).min(count);
    begin..(begin + per * 32).min(count)
}

#[test]
fn compaction_is_a_stable_partition_with_its_count() {
    let owner = |t: usize| (t / 16) % 2;
    for width in [1usize, 31, 256, 257, 2051, 4096] {
        let selected: Vec<i32> = (0..width as i32)
            .map(|i| if i % 7 == 3 { -1 } else { (i * 37) % 5000 })
            .collect();
        for rank in 0..2 {
            let (row, count) = compact_row(&selected, owner, rank);
            let want: Vec<i32> = selected
                .iter()
                .copied()
                .filter(|&t| t >= 0 && owner(t as usize) == rank)
                .collect();
            assert_eq!(count, want.len(), "width {width}");
            assert_eq!(row[..count], want[..], "width {width}");
            assert!(row[count..].iter().all(|&t| t == -1), "width {width}");
        }
    }
}

#[test]
fn counted_partitions_tile_the_compact_prefix() {
    for count in (0..=WIDTH as usize).step_by(7).chain([1, 32, 33, 2051]) {
        for splits in 1..=MAX_SPLITS as usize {
            let mut next = 0;
            for z in 0..splits {
                let range = counted_partition(count, splits, z);
                assert_eq!(
                    range.start,
                    next.min(count),
                    "count {count} splits {splits}"
                );
                next = range.end;
            }
            assert_eq!(next, count, "count {count} splits {splits}");
        }
    }
}

#[test]
fn compact_owner_split_lse_merge_equals_full_softmax_attention() {
    let (dim, tokens) = (64usize, 600usize);
    let keys: Vec<Vec<f32>> = (0..tokens).map(|t| values(t as u64 + 11, dim)).collect();
    let q = values(3, dim);
    let scale = 0.0625;
    let mut selected: Vec<i32> = (0..tokens as i32).rev().filter(|t| t % 5 != 2).collect();
    selected.extend([-1, -1]);
    let all: Vec<Option<usize>> = selected
        .iter()
        .map(|&t| (t >= 0).then_some(t as usize))
        .collect();
    let (want, _) = partial(&q, &keys, &all, scale);
    // Parity blocks, and a lopsided shard (rank 1 stores only a few tokens).
    let owners: [fn(usize) -> usize; 2] = [|t| (t / 16) % 2, |t| usize::from(t >= 560)];
    for owner in owners {
        for rank in 0..2 {
            for splits in [1usize, 6, 15] {
                // Each rank's compact row, split as the counted kernel does.
                let parts_of = |r: usize| -> Vec<(Vec<f32>, f32)> {
                    let (row, count) = compact_row(&selected, owner, r);
                    (0..splits)
                        .map(|z| {
                            let ids: Vec<Option<usize>> = row[counted_partition(count, splits, z)]
                                .iter()
                                .map(|&t| Some(t as usize))
                                .collect();
                            partial(&q, &keys, &ids, scale)
                        })
                        .collect()
                };
                // The peer's partial is merged where it landed: last.
                let mut own = parts_of(rank);
                own.push(merge(&parts_of(1 - rank)));
                let (got, _) = merge(&own);
                for (g, w) in got.iter().zip(&want) {
                    assert!((g - w).abs() <= 1e-5 * w.abs().max(1.0), "{g} vs {w}");
                }
            }
        }
    }
}

use crate::layers::ops::shard_test_gpu::{ShardGpu, ShardPair};

const LANE: ExchangeLane = ExchangeLane {
    stream: 7,
    begun: 21,
    landed: 22,
};

#[test]
fn an_overlapped_exchange_runs_on_the_lane_fenced_around_the_compute() {
    let gpu = ShardGpu::default();
    let pair = ShardPair { gpu: &gpu, rank: 0 };
    let payload = (DevicePtr(0x100), DevicePtr(0x200), 4096);
    let during = |_: &WindowComm| {
        gpu.record_event(99, 3)?; // stands for the compute-stream launches
        Ok(5)
    };
    assert_eq!(
        overlapped_exchange(&gpu, &pair, LANE, payload, 3, during).unwrap(),
        5
    );
    assert_eq!(
        gpu.order(),
        [
            // The lane sees everything the compute stream produced so far...
            "record e21 s3",
            "wait s7 e21",
            "exchange 4096 s7",
            "record e22 s7",
            // ...the compute continues meanwhile...
            "record e99 s3",
            // ...and meets the landed payload before it goes on.
            "wait s3 e22",
        ]
    );
}

#[test]
fn a_failed_overlap_window_still_waits_for_the_exchange() {
    let gpu = ShardGpu::default();
    let pair = ShardPair { gpu: &gpu, rank: 1 };
    let payload = (DevicePtr(0x100), DevicePtr(0x200), 64);
    let failed = overlapped_exchange(&gpu, &pair, LANE, payload, 3, |_| -> Result<()> {
        bail!("selection failed")
    });
    assert!(failed.is_err());
    assert_eq!(gpu.order().last().map(String::as_str), Some("wait s3 e22"));
}

#[test]
fn the_pair_is_refused_inside_an_overlap_window() {
    use spark_comm::CommBackend;
    let gpu = ShardGpu::default();
    let pair = ShardPair { gpu: &gpu, rank: 1 };
    let payload = (DevicePtr(0x100), DevicePtr(0x200), 64);
    let (a, b) = (DevicePtr(0x300), DevicePtr(0x400));
    overlapped_exchange(&gpu, &pair, LANE, payload, 3, |fenced| {
        assert_eq!((fenced.rank(), fenced.world_size()), (1, 2));
        // What an index split would ask before exchanging its rows.
        assert!(!fenced.supports_exchange_async(64));
        let refused = [
            pair_exchange(fenced, a, b, 64, 3),
            fenced.all_reduce(a.0, 64),
            fenced.all_reduce_async(a.0, 64, 3),
            fenced.all_gather(a.0, b.0, 64),
            fenced.reduce_scatter(a.0, b.0, 64),
            fenced.broadcast(a.0, 64, 0),
            fenced.barrier(),
            fenced.peer_exchange_async(a.0, b.0, 64, 3),
            fenced.send_to(a.0, 64, 0, 3),
            fenced.recv_from(a.0, 64, 0, 3),
        ];
        for result in refused {
            let err = result.unwrap_err().to_string();
            assert!(err.contains("overlap window on rank 1"), "{err}");
        }
        Ok(())
    })
    .unwrap();
    // Nothing but the lane's own exchange reached the pair.
    let exchanges = gpu
        .order()
        .iter()
        .filter(|s| s.starts_with("exchange"))
        .count();
    assert_eq!(exchanges, 1);
}
