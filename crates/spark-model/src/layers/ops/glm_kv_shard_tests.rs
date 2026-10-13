// SPDX-License-Identifier: AGPL-3.0-only
//! Launch ABI of the GLM KV shard kernels, the canonical form's partition,
//! and the counted / paired / extra-partition attention entry points of the
//! merge form.
use super::*;
use crate::layers::ops::{
    GlmSparsePrefillTc, PartialGroup, launch_merge_extra, launch_merge_f32, launch_merge_pair,
    launch_sparse_partial_pair, launch_sparse_partials, merge_kernel,
};
use spark_runtime::kv_cache::KvCacheDtype;

use shard_test_gpu::{Op, ShardGpu, ptr, word};

const OWNER: ShardRank = ShardRank { rank: 1, world: 2 };
const SELECTED: u64 = 0x1000;
const OUT: u64 = 0x2000;
const COUNTS: u64 = 0x3000;
const TABLE: u64 = 0x4000;

fn localize(gpu: &ShardGpu, width: u32, out: u64) -> Result<()> {
    glm_kv_shard_localize(
        gpu,
        Some(DevicePtr(SELECTED)),
        DevicePtr(out),
        DevicePtr(COUNTS),
        DevicePtr(TABLE),
        8,
        width,
        16,
        OWNER,
        5,
        19,
    )
}

#[test]
fn localize_packs_each_row_in_one_cta() {
    let gpu = ShardGpu::default();
    localize(&gpu, 2051, OUT).unwrap();
    let mut args = vec![ptr(SELECTED), ptr(OUT), ptr(COUNTS), ptr(TABLE)];
    args.extend([8u32, 2051, 16, 1, 2, 5].map(word));
    assert_eq!(
        gpu.ops(),
        [Op::Launch {
            symbol: "glm_kv_shard_localize_compact".into(),
            grid: [8, 1, 1],
            shared: 0,
            stream: 19,
            args,
        }]
    );
}

#[test]
fn packing_refuses_in_place_and_rows_wider_than_one_cta() {
    let gpu = ShardGpu::default();
    assert!(localize(&gpu, 2051, SELECTED).is_err());
    assert!(localize(&gpu, 4097, OUT).is_err());
    localize(&gpu, 4096, OUT).unwrap();
}

fn partition(gpu: &ShardGpu, selected: Option<u64>, peer: u64, rank: u32) -> Result<()> {
    let p = DevicePtr;
    glm_kv_canonical_partition(
        gpu,
        selected.map(p),
        [p(OUT), p(COUNTS)],
        [p(peer), p(0x5000)],
        8,
        2051,
        16,
        rank,
        5,
        19,
    )
}

#[test]
fn canonical_partition_packs_both_groups_of_each_row_in_one_cta() {
    let gpu = ShardGpu::default();
    partition(&gpu, Some(SELECTED), 0x6000, 1).unwrap();
    partition(&gpu, None, 0x6000, 0).unwrap();
    let args = |selected: u64, rank: u32| {
        let mut a = [selected, OUT, COUNTS, 0x6000, 0x5000].map(ptr).to_vec();
        a.extend([8u32, 2051, 16, rank, 5].map(word));
        a
    };
    let launch = |selected, rank| Op::Launch {
        symbol: "glm_kv_canonical_partition".into(),
        grid: [8, 1, 1],
        shared: 0,
        stream: 19,
        args: args(selected, rank),
    };
    assert_eq!(gpu.ops(), [launch(SELECTED, 1), launch(0, 0)]);
    // Out of place, one CTA per row, a pair's rank.
    assert!(partition(&gpu, Some(SELECTED), SELECTED, 0).is_err());
    assert!(partition(&gpu, Some(SELECTED), 0x6000, 2).is_err());
}

fn config() -> atlas_core::config::ModelConfig {
    let mut c = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c
}

#[test]
fn counted_partials_append_the_row_counts_to_the_split_abi() {
    let c = config();
    for (dtype, name) in [
        (KvCacheDtype::Fp8G128, "fp8g128"),
        (KvCacheDtype::Bf16, "bf16"),
    ] {
        let gpu = ShardGpu::default();
        let a = GlmSparsePrefillTc {
            config: &c,
            dtype,
            identical_kv_latent: true,
            query: DevicePtr(0x100),
            k_cache: DevicePtr(0x200),
            v_cache: DevicePtr(0x200),
            indices: DevicePtr(0x300),
            output: DevicePtr(0x400),
            block_table: DevicePtr(0x500),
            rows: 8,
            heads: 32,
            head_dim: 512,
            index_width: 2051,
            block_size: 16,
            scale: 0.0625,
        };
        let (po, pl) = (DevicePtr(0x600), DevicePtr(0x700));
        launch_sparse_partials(&gpu, &a, 6, None, po, pl, 19).unwrap();
        launch_sparse_partials(&gpu, &a, 6, Some(DevicePtr(COUNTS)), po, pl, 19).unwrap();
        let ops = gpu.ops();
        let [
            Op::Launch {
                symbol: plain,
                grid,
                shared,
                args: plain_args,
                ..
            },
            Op::Launch {
                symbol: counted,
                grid: counted_grid,
                shared: counted_shared,
                args: counted_args,
                ..
            },
        ] = &ops[..]
        else {
            panic!("two launches, got {ops:?}");
        };
        let split = format!("glm_sparse_mla_prefill_{name}_head32_tc_kv_pad_split");
        assert_eq!((plain, counted), (&split, &format!("{split}_counted")));
        assert_eq!((*grid, *shared), ([1, 8, 6], 69376));
        assert_eq!((counted_grid, counted_shared), (grid, shared));
        assert_eq!(plain_args.len(), 14);
        assert_eq!(counted_args[..14], plain_args[..]);
        assert_eq!(counted_args[14..], [ptr(COUNTS)]);
    }
}

#[test]
fn partial_pair_appends_the_second_group_to_the_counted_abi() {
    let c = config();
    for (dtype, name) in [
        (KvCacheDtype::Fp8G128, "fp8g128"),
        (KvCacheDtype::Bf16, "bf16"),
    ] {
        let gpu = ShardGpu::default();
        let p = DevicePtr;
        let a = GlmSparsePrefillTc {
            config: &c,
            dtype,
            identical_kv_latent: true,
            query: p(0x100),
            k_cache: p(0x200),
            v_cache: p(0x200),
            indices: p(0x300),
            output: p(0x400),
            block_table: p(0x500),
            rows: 8,
            heads: 32,
            head_dim: 512,
            index_width: 2051,
            block_size: 16,
            scale: 0.0625,
        };
        let group = |base: u64| PartialGroup {
            query: p(base),
            indices: p(base + 1),
            counts: p(base + 2),
            part_o: p(base + 3),
            part_lse: p(base + 4),
        };
        launch_sparse_partials(&gpu, &a, 3, Some(p(COUNTS)), p(0x600), p(0x700), 19).unwrap();
        launch_sparse_partial_pair(&gpu, &a, 3, [group(0x10), group(0x20)], 19).unwrap();
        let ops = gpu.ops();
        let [
            Op::Launch {
                grid: counted_grid,
                args: counted,
                ..
            },
            Op::Launch {
                symbol,
                grid,
                shared,
                args,
                ..
            },
        ] = &ops[..]
        else {
            panic!("two launches, got {ops:?}");
        };
        let pair = format!("glm_sparse_mla_prefill_{name}_head32_tc_kv_pad_split_counted_pair");
        assert_eq!(symbol, &pair);
        // Both groups' partitions in one grid, the counted launch's CTAs.
        assert_eq!((*grid, *shared), ([1, 8, 6], 69376));
        assert_eq!(*counted_grid, [1, 8, 3]);
        // The counted ABI with the first group's query, IDs, partials and
        // counts, then the second group's IDs, partials, counts and query.
        let mut want = counted.clone();
        want[0] = ptr(0x10);
        want[3] = ptr(0x11);
        want[12..15].clone_from_slice(&[ptr(0x13), ptr(0x14), ptr(0x12)]);
        want.extend([0x21, 0x23, 0x24, 0x22, 0x20].map(ptr));
        assert_eq!(args, &want);
    }
}

#[test]
fn merge_pair_takes_both_groups_and_the_partial() {
    let gpu = ShardGpu::default();
    let p = DevicePtr;
    let (own, out, peer) = ([p(0x10), p(0x20)], [p(0x30), p(0x40)], [p(0x60), p(0x70)]);
    launch_merge_pair(&gpu, own, out, 8, 6, peer, p(0x50), 19).unwrap();
    let mut args: Vec<Vec<u8>> = [0x10u64, 0x20, 0x30, 0x40].map(ptr).to_vec();
    args.extend([8u32, 32, 512, 6].map(word));
    args.extend([0x60u64, 0x70, 0x50].map(ptr));
    assert_eq!(
        gpu.ops(),
        [Op::Launch {
            symbol: "glm_sparse_decode_split_merge_pair".into(),
            grid: [8 * 32, 1, 1],
            shared: 0,
            stream: 19,
            args,
        }]
    );
    // 15 partitions + the partial fill the merge's 16 weights.
    assert!(launch_merge_pair(&gpu, own, out, 8, 16, peer, p(0x50), 19).is_err());
    assert!(launch_merge_pair(&gpu, own, out, 8, 6, peer, DevicePtr::NULL, 19).is_err());
}

#[test]
fn only_the_merge_forms_entry_points_come_from_the_shard_module() {
    // The earlier unsharded kernels (the plain split and the BF16 merge) keep
    // their modules; the merge form's (sharded and canonical) are the shard
    // module's.
    let (c, gpu) = (config(), ShardGpu::default());
    let p = |v: u64| DevicePtr(v);
    let a = GlmSparsePrefillTc {
        config: &c,
        dtype: KvCacheDtype::Fp8G128,
        identical_kv_latent: true,
        query: p(0x100),
        k_cache: p(0x200),
        v_cache: p(0x200),
        indices: p(0x300),
        output: p(0x400),
        block_table: p(0x500),
        rows: 8,
        heads: 32,
        head_dim: 512,
        index_width: 2051,
        block_size: 16,
        scale: 0.0625,
    };
    launch_sparse_partials(&gpu, &a, 6, None, p(0x600), p(0x700), 19).unwrap();
    merge_kernel(&gpu).unwrap();
    launch_sparse_partials(&gpu, &a, 6, Some(p(COUNTS)), p(0x600), p(0x700), 19).unwrap();
    launch_merge_f32(&gpu, p(0x10), p(0x20), p(0x30), p(0x40), 8, 6, 19).unwrap();
    launch_merge_extra(&gpu, p(0x10), p(0x20), p(0x30), p(0x40), 8, 6, p(0x50), 19).unwrap();
    let g = PartialGroup {
        query: p(1),
        indices: p(2),
        counts: p(3),
        part_o: p(4),
        part_lse: p(5),
    };
    launch_sparse_partial_pair(&gpu, &a, 6, [g, g], 19).unwrap();
    launch_merge_pair(&gpu, [p(1); 2], [p(2); 2], 8, 6, [p(3); 2], p(4), 19).unwrap();
    partition(&gpu, Some(SELECTED), 0x6000, 0).unwrap();
    let modules = gpu.modules();
    let modules: Vec<(&str, &str)> = modules
        .iter()
        .map(|(module, symbol)| (module.as_str(), symbol.as_str()))
        .collect();
    assert_eq!(
        modules,
        [
            (
                "glm_sparse_prefill_kv_reuse",
                "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split"
            ),
            (
                "glm_sparse_decode_split_merge",
                "glm_sparse_decode_split_merge"
            ),
            (
                MODULE,
                "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted"
            ),
            (MODULE, "glm_sparse_decode_split_merge_f32"),
            (MODULE, "glm_sparse_decode_split_merge_extra"),
            (
                MODULE,
                "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted_pair"
            ),
            (MODULE, "glm_sparse_decode_split_merge_pair"),
            (MODULE, "glm_kv_canonical_partition"),
        ]
    );
}

#[test]
fn extra_partition_merge_takes_the_local_splits_and_the_landed_partial() {
    let gpu = ShardGpu::default();
    let p = |v: u64| DevicePtr(v);
    launch_merge_extra(&gpu, p(0x10), p(0x20), p(0x30), p(0x40), 8, 6, p(0x50), 19).unwrap();
    let mut args: Vec<Vec<u8>> = [0x10u64, 0x20, 0x30, 0x40].map(ptr).to_vec();
    args.extend([8u32, 32, 512, 6].map(word));
    args.push(ptr(0x50));
    assert_eq!(
        gpu.ops(),
        [Op::Launch {
            symbol: "glm_sparse_decode_split_merge_extra".into(),
            grid: [8 * 32, 1, 1],
            shared: 0,
            stream: 19,
            args,
        }]
    );
    // 15 local partitions + the extra one fill the merge's 16 weights.
    assert!(launch_merge_extra(&gpu, p(1), p(2), p(3), p(4), 8, 16, p(5), 19).is_err());
    assert!(launch_merge_extra(&gpu, p(1), p(2), p(3), p(4), 8, 6, DevicePtr::NULL, 19).is_err());
}
