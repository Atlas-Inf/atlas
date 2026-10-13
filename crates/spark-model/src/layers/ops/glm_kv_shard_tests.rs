// SPDX-License-Identifier: AGPL-3.0-only
//! Launch ABI of the GLM KV shard kernels and of the counted / extra-partition
//! attention entry points `ATLAS_GLM_KV_SHARD_COMPACT=1` uses.
use super::*;
use crate::layers::ops::{
    GlmSparsePrefillTc, launch_merge_extra, launch_merge_f32, launch_sparse_partials, merge_kernel,
};
use spark_runtime::kv_cache::KvCacheDtype;

use shard_test_gpu::{Op, ShardGpu, ptr, word};

const OWNER: ShardRank = ShardRank { rank: 1, world: 2 };
const SELECTED: u64 = 0x1000;
const OUT: u64 = 0x2000;
const COUNTS: u64 = 0x3000;
const TABLE: u64 = 0x4000;

fn localize(gpu: &ShardGpu, counts: Option<u64>, width: u32, out: u64) -> Result<()> {
    glm_kv_shard_localize(
        gpu,
        Some(DevicePtr(SELECTED)),
        DevicePtr(out),
        counts.map(DevicePtr),
        DevicePtr(TABLE),
        8,
        width,
        16,
        OWNER,
        5,
        19,
    )
}

fn tail() -> Vec<Vec<u8>> {
    [8u32, 2051, 16, 1, 2, 5].map(word).to_vec()
}

#[test]
fn localize_without_counts_is_the_per_id_launch() {
    let gpu = ShardGpu::default();
    localize(&gpu, None, 2051, OUT).unwrap();
    let mut args = vec![ptr(SELECTED), ptr(OUT), ptr(TABLE)];
    args.extend(tail());
    assert_eq!(
        gpu.ops(),
        [Op::Launch {
            symbol: "glm_kv_shard_localize".into(),
            grid: [9, 8, 1],
            shared: 0,
            stream: 19,
            args,
        }]
    );
}

#[test]
fn localize_with_counts_packs_each_row_in_one_cta() {
    let gpu = ShardGpu::default();
    localize(&gpu, Some(COUNTS), 2051, OUT).unwrap();
    let mut args = vec![ptr(SELECTED), ptr(OUT), ptr(COUNTS), ptr(TABLE)];
    args.extend(tail());
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
fn compaction_refuses_in_place_and_rows_wider_than_one_cta() {
    let gpu = ShardGpu::default();
    assert!(localize(&gpu, Some(COUNTS), 2051, SELECTED).is_err());
    assert!(localize(&gpu, Some(COUNTS), 4097, OUT).is_err());
    localize(&gpu, Some(COUNTS), 4096, OUT).unwrap();
    // The per-id kernel has neither limit.
    localize(&gpu, None, 4097, OUT).unwrap();
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
fn only_the_shards_entry_points_come_from_the_shard_module() {
    // An unsharded server launches the plain split and the BF16 merge: their
    // modules must stay the ones it always loaded, with nothing added.
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
