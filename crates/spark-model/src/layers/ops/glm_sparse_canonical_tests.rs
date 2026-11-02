// SPDX-License-Identifier: AGPL-3.0-only
//! Gate, scratch layout and launch sequence of the canonical form. Its bits
//! against the shard's are the GPU test `canonical_matches_the_shard_bitwise`.

use super::*;
use crate::layers::ops::shard_test_gpu::{Op, ShardGpu, ptr, word};

fn config(tp_world_size: usize, tp_rank: usize) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c.tp_world_size = tp_world_size;
    c.tp_rank = tp_rank;
    c
}

#[test]
fn the_variable_is_on_unless_zero() {
    assert!(parse(None).unwrap());
    assert!(parse(Some("1")).unwrap());
    assert!(!parse(Some("0")).unwrap());
    assert!(parse(Some("yes")).is_err());
}

#[test]
fn only_owners_a_shard_merges_take_the_canonical_form() {
    // The test environment leaves ATLAS_GLM_KV_CANONICAL unset (on).
    for rank in 0..2 {
        let c = config(2, rank);
        for rows in [1, 8, 64] {
            assert_eq!(
                glm_kv_canonical_rank(&c, 32, rows).unwrap(),
                Some(rank as u32)
            );
        }
        // Owners past the merge form's rows, a 64-head body, no rows.
        for (heads, rows) in [(32, 65), (64, 8), (32, 0)] {
            assert_eq!(glm_kv_canonical_rank(&c, heads, rows).unwrap(), None);
        }
    }
    // No pair, or not GLM.
    assert_eq!(glm_kv_canonical_rank(&config(1, 0), 32, 8).unwrap(), None);
    let mut other = config(2, 0);
    other.model_type = "qwen3_next".into();
    assert_eq!(glm_kv_canonical_rank(&other, 32, 8).unwrap(), None);
}

#[test]
fn layout_regions_are_aligned_disjoint_and_bounded() {
    for rows in 1..=MERGE_MAX_ROWS as u32 {
        let splits = MERGE_SPLITS;
        let m = CanonicalLayout::new(rows, splits);
        let (r, s) = (rows as usize, splits as usize);
        let (part, lse) = (r * 32 * 512 * 4, r * 32 * 4);
        let mut regions = [
            (m.own_ids, r * 2051 * 4),
            (m.peer_ids, r * 2051 * 4),
            (m.own_counts, r * 4),
            (m.peer_counts, r * 4),
            (m.own_o, s * part),
            (m.own_lse, s * lse),
            (m.peer_o, s * part),
            (m.peer_lse, s * lse),
            (m.extra, part + lse),
            (m.out_lse, lse),
        ];
        regions.sort_unstable();
        assert!(
            regions.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0),
            "rows {rows}"
        );
        assert!(
            regions
                .iter()
                .all(|&(o, n)| o % 256 == 0 && o + n <= m.total)
        );
        assert_eq!(CanonicalLayout::bytes(rows), m.total);
        // The MoE expert scratch of a GLM arena holds this many times over.
        assert!(m.total < 40 << 20, "rows {rows}: {}", m.total);
    }
}

const SCRATCH: u64 = 0x1000_0000;

fn args<'a>(c: &'a ModelConfig, rows: u32) -> GlmSparsePrefillTc<'a> {
    let p = DevicePtr;
    GlmSparsePrefillTc {
        config: c,
        dtype: KvCacheDtype::Fp8G128,
        identical_kv_latent: true,
        query: p(0x100),
        k_cache: p(0x200),
        v_cache: p(0x200),
        indices: p(0x300),
        output: p(0x400),
        block_table: p(0x500),
        rows,
        heads: 32,
        head_dim: 512,
        index_width: 2051,
        block_size: 16,
        scale: 0.0625,
    }
}

fn run(rows: u32, selected: Option<u64>) -> Vec<Op> {
    let (c, gpu) = (config(2, 1), ShardGpu::default());
    let region = (DevicePtr(SCRATCH), CanonicalLayout::bytes(rows));
    let selected = selected.map(DevicePtr);
    glm_sparse_canonical(&gpu, &args(&c, rows), selected, 7, 1, region, 19).unwrap();
    gpu.ops()
}

fn launch_args(op: &Op) -> (&str, [u32; 3], &Vec<Vec<u8>>) {
    let Op::Launch {
        symbol, grid, args, ..
    } = op
    else {
        panic!("a launch, got {op:?}");
    };
    (symbol, *grid, args)
}

#[test]
fn partition_paired_split_and_paired_merge_in_three_launches() {
    let rows = 8;
    let ops = run(rows, Some(0x300));
    let splits = MERGE_SPLITS;
    let m = CanonicalLayout::new(rows, splits);
    let at = |o: usize| ptr(SCRATCH + o as u64);
    let [partition, pair, merge] = &ops[..] else {
        panic!("three launches, got {ops:?}");
    };
    let (symbol, grid, a) = launch_args(partition);
    assert_eq!((symbol, grid), ("glm_kv_canonical_partition", [8, 1, 1]));
    let mut want = vec![ptr(0x300), at(m.own_ids), at(m.own_counts)];
    want.extend([at(m.peer_ids), at(m.peer_counts)]);
    want.extend([8u32, 2051, 16, 1, 7].map(word));
    assert_eq!(a, &want);
    // Both groups take this rank's queries; the peer group's partials are
    // the ones merged to the partial.
    let (symbol, grid, a) = launch_args(pair);
    assert_eq!(
        (symbol, grid),
        (
            "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted_pair",
            [1, 8, 2 * splits]
        )
    );
    assert_eq!(
        [&a[0], &a[3], &a[5]],
        [&ptr(0x100), &at(m.own_ids), &ptr(0x500)]
    );
    assert_eq!(a[12..15], [at(m.own_o), at(m.own_lse), at(m.own_counts)]);
    assert_eq!(
        a[15..],
        [
            at(m.peer_ids),
            at(m.peer_o),
            at(m.peer_lse),
            at(m.peer_counts),
            ptr(0x100)
        ]
    );
    let (symbol, grid, a) = launch_args(merge);
    assert_eq!(
        (symbol, grid),
        ("glm_sparse_decode_split_merge_pair", [8 * 32, 1, 1])
    );
    let mut want = vec![at(m.own_o), at(m.own_lse), ptr(0x400), at(m.out_lse)];
    want.extend([8u32, 32, 512, splits].map(word));
    want.extend([at(m.peer_o), at(m.peer_lse), at(m.extra)]);
    assert_eq!(a, &want);
}

#[test]
fn causal_rows_select_nothing() {
    let ops = run(8, None);
    let (_, _, partition) = launch_args(&ops[0]);
    assert_eq!(partition[0], ptr(0));
}

#[test]
fn scratch_must_fit_and_stay_clear_of_the_inputs() {
    let (c, gpu) = (config(2, 0), ShardGpu::default());
    let need = CanonicalLayout::bytes(8);
    let a = args(&c, 8);
    let p = DevicePtr;
    let ok = |region| glm_sparse_canonical(&gpu, &a, Some(p(0x300)), 0, 0, region, 19).is_ok();
    assert!(ok((p(SCRATCH), need)));
    assert!(!ok((p(SCRATCH), need - 1)));
    assert!(!ok((p(SCRATCH + 16), need)));
    // Over the query, the output and the selection.
    assert!(!ok((p(0x100), need)));
    assert!(!ok((p(0), need)));
    let wide = args(&c, 65);
    assert!(glm_sparse_canonical(&gpu, &wide, None, 0, 0, (p(SCRATCH), 1 << 30), 19).is_err());
}
