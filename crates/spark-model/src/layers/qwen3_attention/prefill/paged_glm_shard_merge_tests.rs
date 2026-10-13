// SPDX-License-Identifier: AGPL-3.0-only

//! Launch and exchange order of the merge form in each tuning, on a
//! recording GPU and pair.

use super::*;
use crate::layers::glm_kv_shard::MergeTuning;
use crate::layers::ops::shard_test_gpu::{Op, ShardGpu, ShardPair, ptr};
use spark_runtime::kv_cache::LatentShardSpec;

const SCRATCH: u64 = 0x4000_0000;
const WORK: usize = 0x10_0000;
const STREAM: u64 = 3;
const LANE: ExchangeLane = ExchangeLane {
    stream: 7,
    begun: 21,
    landed: 22,
};
const QUERY: u64 = 0x100;
const OUTPUT: u64 = 0x200;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c
}

/// Rank 1's merge of one layer under `tuning`, recording on `gpu`.
fn merge<'a>(
    gpu: &'a ShardGpu,
    pair: &'a ShardPair<'a>,
    config: &'a ModelConfig,
    tuning: MergeTuning,
) -> ShardMerge<'a> {
    ShardMerge {
        gpu,
        comm: pair,
        config,
        shard: LatentShard {
            spec: LatentShardSpec {
                rank: 1,
                world: 2,
                scratch_bytes: 0,
                view_blocks: 0,
                write_rows: 0,
                lane: tuning.overlap,
            },
            local_blocks: 0,
            scratch: DevicePtr(SCRATCH),
            identity: DevicePtr(0x300),
            lane: tuning.overlap.then_some(LANE),
        },
        work: WORK,
        work_bytes: 64 << 20,
        dtype: KvCacheDtype::Fp8G128,
        pool: DevicePtr(0x400),
        scale: 0.0625,
        compact: tuning.compact,
        lane: tuning.overlap.then_some(LANE),
    }
}

fn owner(query: u64, rows: u32, queries_swapped: bool) -> ShardRows {
    ShardRows {
        query: DevicePtr(query),
        selected: Some(DevicePtr(0x500)),
        causal_start: 0,
        block_table: DevicePtr(0x600),
        rows,
        end: None,
        queries_swapped,
    }
}

/// Run one owner of `rows` rows under `tuning`; the recorded trace.
fn run(rows: u32, tuning: MergeTuning, queries_swapped: bool) -> ShardGpu {
    let gpu = ShardGpu::default();
    let (pair, config) = (ShardPair { gpu: &gpu, rank: 1 }, config());
    merge(&gpu, &pair, &config, tuning)
        .run(
            owner(QUERY, rows, queries_swapped),
            DevicePtr(OUTPUT),
            STREAM,
        )
        .unwrap();
    gpu
}

const SPLIT: &str = "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split";
const COUNTED: &str = "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted";
const Q_BYTES: usize = 8 * 32 * 512 * 2;
const P_BYTES: usize = 8 * 32 * 513 * 4;

fn tuning(compact: bool, overlap: bool) -> MergeTuning {
    MergeTuning {
        compact,
        overlap,
        check: false,
    }
}

#[test]
fn untuned_merge_is_the_original_sequence_on_the_compute_stream() {
    let gpu = run(8, tuning(false, false), false);
    let (q, p) = (
        format!("exchange {Q_BYTES} s3"),
        format!("exchange {P_BYTES} s3"),
    );
    let part = 8 * 32 * 512 * 4;
    assert_eq!(
        gpu.order(),
        [
            q.as_str(),
            "glm_kv_shard_localize",
            SPLIT,
            "glm_sparse_decode_split_merge_f32",
            p.as_str(),
            SPLIT,
            &format!("copy {part}"),
            &format!("copy {}", 8 * 32 * 4),
            "glm_sparse_decode_split_merge",
        ]
    );
    // Everything stays on the compute stream; the last merge takes the
    // peer's partial as one more partition.
    let ops = gpu.ops();
    assert!(ops.iter().all(|op| match op {
        Op::Launch { stream, .. } | Op::Exchange { stream, .. } => *stream == STREAM,
        _ => true,
    }));
    let Some(Op::Launch { args, grid, .. }) = ops.last() else {
        panic!("merge launch");
    };
    let splits = shard::merge_splits(8);
    assert_eq!(
        (*grid, &args[7]),
        ([8 * 32, 1, 1], &(splits + 1).to_ne_bytes().to_vec())
    );
}

#[test]
fn compact_merge_counts_rows_and_merges_the_partial_where_it_landed() {
    let gpu = run(8, tuning(true, false), false);
    let (q, p) = (
        format!("exchange {Q_BYTES} s3"),
        format!("exchange {P_BYTES} s3"),
    );
    assert_eq!(
        gpu.order(),
        [
            q.as_str(),
            "glm_kv_shard_localize_compact",
            COUNTED,
            "glm_sparse_decode_split_merge_f32",
            p.as_str(),
            COUNTED,
            "glm_sparse_decode_split_merge_extra",
        ]
    );
    let splits = shard::merge_splits(8);
    let m = MergeLayout::new(8, splits);
    let at = |offset: usize| SCRATCH + (WORK + offset) as u64;
    let counts = ptr(at(m.counts(8, splits)));
    let ops = gpu.ops();
    let launches: Vec<&Vec<Vec<u8>>> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Launch { args, .. } => Some(args),
            _ => None,
        })
        .collect();
    // The localize writes the counts both attention launches read...
    assert_eq!(launches[0][2], counts);
    assert_eq!(launches[1].last(), Some(&counts));
    assert_eq!(launches[3].last(), Some(&counts));
    // ...and the merge reads the peer's partial from the receive buffer.
    assert_eq!(launches[4].last(), Some(&ptr(at(m.recv))));
    assert_eq!(launches[4][7], splits.to_ne_bytes().to_vec());
    let exchanged: Vec<(u64, u64)> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Exchange { send, recv, .. } => Some((*send, *recv)),
            _ => None,
        })
        .collect();
    assert_eq!(exchanged, [(QUERY, at(m.q_peer)), (at(m.send), at(m.recv))]);
}

#[test]
fn overlapped_merge_swaps_on_the_lane_beside_the_launches_it_does_not_feed() {
    let gpu = run(8, tuning(true, true), false);
    assert_eq!(
        gpu.order(),
        [
            // Queries fly while the selection is localized.
            "record e21 s3",
            "wait s7 e21",
            &format!("exchange {Q_BYTES} s7"),
            "record e22 s7",
            "glm_kv_shard_localize_compact",
            "wait s3 e22",
            // The peer's heads need the peer's queries.
            COUNTED,
            "glm_sparse_decode_split_merge_f32",
            // Partials fly while this rank's own heads attend.
            "record e21 s3",
            "wait s7 e21",
            &format!("exchange {P_BYTES} s7"),
            "record e22 s7",
            COUNTED,
            "wait s3 e22",
            "glm_sparse_decode_split_merge_extra",
        ]
    );
    // The overlap alone keeps the original kernels.
    let plain = run(8, tuning(false, true), false).order();
    assert_eq!(plain.iter().filter(|s| *s == SPLIT).count(), 2);
    assert_eq!(
        plain.last().map(String::as_str),
        Some("glm_sparse_decode_split_merge")
    );
}

#[test]
fn queries_swapped_ahead_are_not_swapped_again() {
    for t in [tuning(false, true), tuning(true, true)] {
        let order = run(8, t, true).order();
        assert_eq!(
            order.iter().filter(|s| s.starts_with("exchange")).count(),
            1,
            "{order:?}"
        );
        assert!(order[0].starts_with("glm_kv_shard_localize"), "{order:?}");
    }
}

/// The verify path under the overlap: each owner's queries fly beside its
/// selection (a marker copy here), then its merge runs without a second swap.
fn verify_owners(queries: &[u64]) -> ShardGpu {
    let gpu = ShardGpu::default();
    let (pair, config) = (ShardPair { gpu: &gpu, rank: 1 }, config());
    let m = merge(&gpu, &pair, &config, tuning(true, true));
    let work = DevicePtr(SCRATCH).offset(WORK);
    for &query in queries {
        let select = |fenced: &WindowComm| {
            assert!(shard::pair_exchange(fenced, work, work, 8, STREAM).is_err());
            gpu.copy_d2d_async(DevicePtr(1), DevicePtr(2), 1, STREAM)
        };
        let q = (DevicePtr(query), 8);
        swap_queries_during(&gpu, &pair, LANE, work, q, STREAM, select).unwrap();
        m.run(owner(query, 8, true), DevicePtr(OUTPUT), STREAM)
            .unwrap();
    }
    gpu
}

#[test]
fn verify_owners_swap_queries_beside_selection_one_owner_at_a_time() {
    let one_owner = [
        "record e21 s3",
        "wait s7 e21",
        &format!("exchange {Q_BYTES} s7"),
        "record e22 s7",
        "copy 1", // the selection
        "wait s3 e22",
        "glm_kv_shard_localize_compact",
        COUNTED,
        "glm_sparse_decode_split_merge_f32",
        "record e21 s3",
        "wait s7 e21",
        &format!("exchange {P_BYTES} s7"),
        "record e22 s7",
        COUNTED,
        "wait s3 e22",
        "glm_sparse_decode_split_merge_extra",
    ]
    .map(String::from);
    assert_eq!(verify_owners(&[QUERY]).order(), one_owner);
    // Two owners share the lane and the scratch: the second owner's swap
    // starts only after the first owner's merge was enqueued.
    let gpu = verify_owners(&[QUERY, QUERY + 0x1000]);
    assert_eq!(gpu.order(), [one_owner.clone(), one_owner].concat());
    let ops = gpu.ops();
    let q_peer = SCRATCH + (WORK + MergeLayout::new(8, shard::merge_splits(8)).q_peer) as u64;
    let swaps: Vec<(u64, u64)> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Exchange {
                send, recv, bytes, ..
            } if *bytes == Q_BYTES => Some((*send, *recv)),
            _ => None,
        })
        .collect();
    assert_eq!(swaps, [(QUERY, q_peer), (QUERY + 0x1000, q_peer)]);
    // Each merge reads the peer's queries where that swap landed them, then
    // its own.
    let queries: Vec<&Vec<u8>> = ops
        .iter()
        .filter_map(|op| match op {
            Op::Launch { symbol, args, .. } if symbol == COUNTED => Some(&args[0]),
            _ => None,
        })
        .collect();
    assert_eq!(
        queries,
        [
            &ptr(q_peer),
            &ptr(QUERY),
            &ptr(q_peer),
            &ptr(QUERY + 0x1000)
        ]
    );
}
