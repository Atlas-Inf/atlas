// SPDX-License-Identifier: AGPL-3.0-only
//! The GLM paged prefill over a latent-sharded cache (`ATLAS_GLM_KV_SHARD=1`):
//! which pool, slots and block table each launch is handed, and what the pair
//! exchanges. Recorded launches, not CUDA numerics.
use super::*;
use crate::layers::glm_kv_shard::{self as shard, ScratchLayout};
use spark_runtime::kv_cache::LatentShard;
use std::sync::Mutex;

const MAP_SLOTS: u64 = 830;
const LOCALIZE: u64 = 831;
const COPY_BLOCKS: u64 = 832;
const LATENT_WRITE: u64 = 821;
const SPARSE_ATTN: u64 = 840;
const PIPE_ATTN: u64 = 841;
const RANK: usize = 1;
const BLOCKS: usize = 512;

/// This rank of a pair whose peer is in step (it sends what this rank
/// sends), or answers the 8-byte verdict swap with `verdict`: the byte count
/// of every exchange, in order.
struct Pair<'a> {
    gpu: &'a TestGpu,
    verdict: Option<u64>,
    exchanges: Mutex<Vec<usize>>,
}

impl spark_comm::CommBackend for Pair<'_> {
    fn rank(&self) -> usize {
        RANK
    }
    fn world_size(&self) -> usize {
        2
    }
    fn supports_exchange_async(&self, _: usize) -> bool {
        true
    }
    fn exchange_async(&self, send: u64, recv: u64, n: usize, add: bool, _: u64) -> Result<bool> {
        assert!(!add, "shard exchanges copy, never add");
        self.exchanges.lock().unwrap().push(n);
        match self.verdict.filter(|_| n == 8) {
            Some(word) => self.gpu.copy_h2d(&word.to_le_bytes(), DevicePtr(recv))?,
            None => self.gpu.copy_d2d(DevicePtr(send), DevicePtr(recv), n)?,
        }
        Ok(true)
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        anyhow::bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        anyhow::bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected receive")
    }
}

/// What one sharded `prefill_attention_paged` call did.
struct Pass {
    /// Typed launches `(kernel, arguments)` in order.
    launches: Vec<(u64, Vec<Vec<u8>>)>,
    exchanges: Vec<usize>,
    shard: LatentShard,
    layout: ScratchLayout,
    /// This rank's latent pool, the sequence's slots and its block table.
    pool: DevicePtr,
    slot: DevicePtr,
    block_table: DevicePtr,
}

impl Pass {
    fn of(&self, kernel: u64) -> Vec<&Vec<Vec<u8>>> {
        let launches = self.launches.iter();
        launches.filter(|l| l.0 == kernel).map(|l| &l.1).collect()
    }
    fn scratch(&self, offset: usize) -> Vec<u8> {
        ptr(self.shard.scratch.offset(offset))
    }
}

fn ptr(p: DevicePtr) -> Vec<u8> {
    p.0.to_ne_bytes().to_vec()
}

fn word(v: u32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

/// `rows` rows from `seq_len_start` through the paged prefill of rank
/// [`RANK`] on a sharded `dtype` cache, the first `floor` rows unwritten.
fn pass(dtype: KvCacheDtype, seq_len_start: usize, rows: usize, floor: usize) -> Pass {
    let (result, pass) = pass_with(dtype, seq_len_start, rows, floor, None, None);
    result.unwrap();
    pass
}

/// [`pass`] and its result, with logical block `misplaced` of this rank's
/// table on the wrong residue and the peer answering the verdict swap with
/// `verdict`.
fn pass_with(
    dtype: KvCacheDtype,
    seq_len_start: usize,
    rows: usize,
    floor: usize,
    misplaced: Option<usize>,
    verdict: Option<u64>,
) -> (Result<()>, Pass) {
    let mut out = None;
    fixture_with(dtype, |gpu, config, layer| {
        layer.glm_sparse_attn_k = KernelHandle(SPARSE_ATTN);
        let layer = &*layer;
        let mut arena = BufferArena::new(config, rows.max(256), 32768, 16, 1, gpu).unwrap();
        if dtype == KvCacheDtype::Fp8G128 {
            arena.attach_glm_latent_scratch(4096, gpu).unwrap();
        }
        let mut dispatch = ops::GemmDispatch::defaults();
        dispatch.cublas_gemm = false;
        let derived = ops::DerivedWeights::new();
        let levers = ops::ModelLevers::defaults();
        let stats = ops::ModelStats::new();
        let base = gpu.alloc(4 * 4096).unwrap();
        let meta = AttnMetadataDev {
            positions: base,
            positions_h: base,
            positions_w: base,
            slot: base.offset(4096),
            seq_len: base.offset(2 * 4096),
            block_table: base.offset(3 * 4096),
            max_blocks_per_seq: BLOCKS as u32,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        // Logical block `l` is physical block `l`: every residue matches,
        // but for the misplaced one.
        let block = |l: u32| l + u32::from(misplaced == Some(l as usize));
        let table: Vec<u8> = (0..BLOCKS as u32)
            .map(block)
            .flat_map(u32::to_le_bytes)
            .collect();
        gpu.copy_h2d(&table, meta.block_table).unwrap();
        let pair = Pair {
            gpu,
            verdict,
            exchanges: Mutex::default(),
        };
        let ctx = ForwardContext {
            buffers: &arena,
            gpu,
            config,
            dispatch: &dispatch,
            derived: &derived,
            levers: &levers,
            stats: &stats,
            ssm_batch: None,
            attn_metadata: Some(meta),
            profile: false,
            comm: Some(&pair),
            graph_capture: false,
            gdn_exact_replay: true,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 512,
            num_layers: 1,
            dtype,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let spec = shard::spec(RANK, &kv_config, BLOCKS * 16 - 64, 256.max(rows));
        let mut cache = PagedKvCache::new_latent_sharded(kv_config, BLOCKS, gpu, spec).unwrap();
        cache
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
            .unwrap();
        let before = gpu.1.lock().unwrap().len();
        let result = layer.prefill_attention_paged(
            &mut crate::layer::EmptyLayerState,
            arena.norm_output(),
            rows,
            seq_len_start,
            &mut cache,
            &Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
            None,
            floor,
            &ctx,
            0,
        );
        let shard = cache.latent_shard().unwrap();
        let pass = Pass {
            launches: gpu.1.lock().unwrap()[before..].to_vec(),
            exchanges: pair.exchanges.lock().unwrap().clone(),
            shard,
            layout: ScratchLayout::of(&shard, cache.config()).unwrap(),
            pool: cache.latent_pool_ptr(0),
            slot: meta.slot,
            block_table: meta.block_table,
        };
        out = Some((result.map(|_| ()), pass));
    });
    out.unwrap()
}

#[test]
fn a_floored_write_maps_only_the_written_rows_to_this_ranks_slots() {
    // 8 verify-sized rows past the top-k boundary, the first 4 replaying
    // cached positions: the slots of rows [4, 8) are localized and written
    // to this rank's pool, never through the unsharded pool accessors.
    for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Fp8G128] {
        let p = pass(dtype, 4096, 8, 4);
        let local = p.scratch(p.layout.slots);
        let [map] = p.of(MAP_SLOTS)[..] else {
            panic!("{dtype:?}: one slot mapping");
        };
        let owner = [word(RANK as u32), word(2)];
        assert_eq!(
            map[..3],
            [ptr(p.slot.offset(4 * 8)), local.clone(), word(4)]
        );
        assert_eq!(map[4..], owner, "{dtype:?}");
        let sides = if dtype == KvCacheDtype::Bf16 { 2 } else { 1 };
        let [write] = p.of(LATENT_WRITE)[..] else {
            panic!("{dtype:?}: one latent write");
        };
        assert_eq!(write[sides], ptr(p.pool), "{dtype:?} pool");
        assert_eq!(write[2 * sides], local, "{dtype:?} slots");
    }
}

#[test]
fn few_rows_take_the_merge_form_over_this_ranks_tokens() {
    for dtype in [KvCacheDtype::Bf16, KvCacheDtype::Fp8G128] {
        let p = pass(dtype, 4096, 8, 0);
        // Queries, then the peer's heads' partial: the merge form's two swaps.
        assert_eq!(
            p.exchanges,
            [8 * 32 * 512 * 2, 8 * 32 * 513 * 4],
            "{dtype:?}"
        );
        let [localize] = p.of(LOCALIZE)[..] else {
            panic!("{dtype:?}: one localize");
        };
        // The selection is read through the sequence's table; the attention
        // reads the localized ids through the identity table over the pool.
        assert_eq!(localize[2], ptr(p.block_table), "{dtype:?}");
        assert!(p.of(COPY_BLOCKS).is_empty(), "{dtype:?}: no view");
        assert!(
            p.of(SPARSE_ATTN).is_empty(),
            "{dtype:?}: no unsharded reader"
        );
        let reads_pool =
            |a: &&Vec<Vec<u8>>| a.len() > 5 && a[1] == ptr(p.pool) && a[5] == ptr(p.shard.identity);
        let partials = p.launches.iter().map(|(_, a)| a).filter(reads_pool).count();
        assert_eq!(partials, 2, "{dtype:?}: peer heads, then own heads");
    }
}

#[test]
fn a_chunk_reads_its_history_assembled_from_both_ranks() {
    // 128 rows (a view-form owner) ending at token 4224 = 264 blocks, of
    // which this rank stores the 132 odd ones.
    let p = pass(KvCacheDtype::Bf16, 4096, 128, 0);
    let block = 16 * 512 * 2;
    assert_eq!(
        p.exchanges,
        [8, 132 * block],
        "the table verdicts, then one round of the peer's blocks"
    );
    let view = p.scratch(p.layout.view);
    let copies = p.of(COPY_BLOCKS);
    let blocks: Vec<_> = copies.iter().map(|a| a[4].clone()).collect();
    assert_eq!(blocks, [word(132), word(132), word(132)], "own, send, land");
    assert_eq!((&copies[0][0], &copies[0][2]), (&ptr(p.pool), &view));
    assert_eq!(copies[2][2], view, "the peer's blocks land in the view");
    // The unchanged sparse kernel reads the view through the identity table.
    let [attn] = p.of(SPARSE_ATTN)[..] else {
        panic!("one sparse attention launch");
    };
    assert_eq!(attn[1..3], [view.clone(), view]);
    assert_eq!(attn[5], ptr(p.shard.identity));
    assert!(p.of(LOCALIZE).is_empty(), "the view form localizes nothing");
}

#[test]
fn a_table_off_its_residues_fails_on_both_ranks_after_the_verdict_swap() {
    // This rank's table is broken: it still swaps its verdict, so the peer
    // fails too instead of waiting for blocks that never come.
    let (result, p) = pass_with(KvCacheDtype::Bf16, 4096, 128, 0, Some(7), None);
    let err = format!("{:#}", result.unwrap_err());
    assert!(err.contains("logical block 7"), "{err}");
    assert_eq!(p.exchanges, [8], "the verdict, and no block after it");
    assert!(p.of(COPY_BLOCKS).is_empty());
    // The peer's table is broken, or it attends another number of blocks:
    // this rank's sound table does not carry it past the swap.
    for (verdict, why) in [
        (264 | 1 << 63, "breaks the ownership rule"),
        (263, "its peer 263"),
    ] {
        let (result, p) = pass_with(KvCacheDtype::Bf16, 4096, 128, 0, None, Some(verdict));
        let err = format!("{:#}", result.unwrap_err());
        assert!(
            err.contains("attends 264 blocks") && err.contains(why),
            "{err}"
        );
        assert_eq!(p.exchanges, [8]);
        assert!(p.of(COPY_BLOCKS).is_empty());
    }
}

/// The production flags the view form has to work under, in a child process
/// (they are read from the environment): an `fp8_g128` owner wide enough for
/// the index split, through the pipelined sparse kernel.
#[test]
fn an_fp8_chunk_under_the_pipe_and_the_index_split_reads_the_assembled_view() {
    const CHILD: &str = "ATLAS_TEST_SHARD_PIPE";
    if std::env::var_os(CHILD).is_none() {
        let name = concat!(
            module_path!(),
            "::an_fp8_chunk_under_the_pipe_and_the_index_split_reads_the_assembled_view"
        );
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", name.split_once("::").unwrap().1, "--nocapture"]);
        for (k, _) in std::env::vars_os() {
            if k.to_string_lossy().starts_with("ATLAS_") {
                cmd.env_remove(k);
            }
        }
        for flag in [
            CHILD,
            "ATLAS_GLM_KV_SHARD",
            "ATLAS_GLM_SPARSE_PREFILL_TC",
            "ATLAS_GLM_SPARSE_PREFILL_KV_REUSE",
            "ATLAS_GLM_SPARSE_PREFILL_PIPE",
            "ATLAS_GLM_INDEX_SPLIT",
        ] {
            cmd.env(flag, "1");
        }
        let out = cmd.output().unwrap();
        assert!(String::from_utf8_lossy(&out.stdout).contains("running 1 test"));
        assert!(
            out.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }
    // 288 rows from token 4096: 274 blocks, 137 of them the peer's. One CTA
    // per row is already the cheapest split, so the rows take the pipe.
    let p = pass(KvCacheDtype::Fp8G128, 4096, 288, 0);
    let (quarter, block) = (72 * 2051 * 4, 16 * 528);
    assert_eq!(
        p.exchanges,
        [quarter, quarter, 8, 137 * block],
        "the split selection's two quarters, the table verdicts, the peer's blocks"
    );
    let view = p.scratch(p.layout.view);
    let [attn] = p.of(PIPE_ATTN)[..] else {
        panic!(
            "one pipelined launch, got {:?}",
            p.launches.iter().map(|l| l.0).collect::<Vec<_>>()
        );
    };
    assert_eq!(attn[1..3], [view.clone(), view], "fp8_g128 read in place");
    assert_eq!(attn[5], ptr(p.shard.identity));
    assert_eq!(attn[6], word(288));
    assert!(p.of(LOCALIZE).is_empty() && p.of(SPARSE_ATTN).is_empty());
}
