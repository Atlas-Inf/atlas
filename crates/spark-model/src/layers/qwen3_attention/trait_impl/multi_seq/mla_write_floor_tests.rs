// SPDX-License-Identifier: AGPL-3.0-only
//! The GLM paged prefill honours the Marconi KV write floor: rows below it
//! replay positions held in shared prefix-cache blocks and must reach neither
//! the latent cache nor the semantic index. Recorded launches establish the
//! written row ranges, not CUDA numerics.
use super::*;

const LATENT_WRITE: u64 = 821;
const TAIL_WRITE: u64 = 801;
const POOL_FINALIZE: u64 = 802;
const KEY_ROW: usize = 128 * 2;
const LATENT_ROW: usize = 512 * 2;
/// BF16, and the production `fp8_g128` latent (K side only: V aliases K).
const DTYPES: [KvCacheDtype; 2] = [KvCacheDtype::Bf16, KvCacheDtype::Fp8G128];

fn ptr(p: DevicePtr) -> Vec<u8> {
    p.0.to_ne_bytes().to_vec()
}

/// One cache write: `(kernel, row-major source pointers, slots, rows)`.
type Write = (u64, Vec<Vec<u8>>, Vec<u8>, u32);
/// The writes of one pass, then its scratch and slot bases.
type Pass = (Vec<Write>, [DevicePtr; 2]);

/// Cache writes of one `prefill_attention_paged` call over `rows` rows from
/// `seq_len_start` on a `dtype` cache, in launch order. `edit` runs on the
/// layer first.
fn try_writes(
    dtype: KvCacheDtype,
    seq_len_start: usize,
    rows: usize,
    floor: usize,
    edit: impl FnOnce(&mut Qwen3AttentionLayer),
) -> Result<Pass> {
    let mut out = None;
    fixture_with(dtype, |gpu, config, layer| {
        edit(layer);
        let layer = &*layer;
        let mut arena = BufferArena::new(config, 64, 32768, 16, 1, gpu).unwrap();
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
            max_blocks_per_seq: 256,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
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
            comm: None,
            graph_capture: false,
            gdn_exact_replay: true,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let mut cache = PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 512,
                num_layers: 1,
                dtype,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            256,
            gpu,
        )
        .unwrap();
        cache
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
            .unwrap();
        let (typed_before, launch_before) = (gpu.1.lock().unwrap().len(), gpu.launch_count());
        let ran = layer.prefill_attention_paged(
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
        if let Err(e) = ran {
            out = Some(Err(e));
            return;
        }
        let is_write = |k: u64| [LATENT_WRITE, TAIL_WRITE, POOL_FINALIZE].contains(&k);
        let grids: Vec<u32> = gpu.launches_snapshot()[launch_before..]
            .iter()
            .filter(|l| is_write(l.func))
            .map(|l| l.grid[0])
            .collect();
        let typed = gpu.1.lock().unwrap();
        let calls: Vec<_> = typed[typed_before..]
            .iter()
            .filter(|(k, _)| is_write(*k))
            .collect();
        assert_eq!(calls.len(), grids.len(), "every cache write is typed");
        // Latent write arguments: BF16 `(k, v, k_pool, v_pool, slots, ..)`,
        // fp8_g128 `(k, k_pool, slots, ..)`.
        let sides = if dtype == KvCacheDtype::Bf16 { 2 } else { 1 };
        let found = calls
            .into_iter()
            .zip(grids)
            .map(|((k, a), rows)| {
                if *k == LATENT_WRITE {
                    assert_eq!(a[sides], ptr(cache.k_pool_ptr(0)), "latent pool");
                    return (*k, a[..sides].to_vec(), a[2 * sides].clone(), rows);
                }
                // The index kernels also bound their rows by `num_tokens`.
                assert_eq!(a[5], rows.to_ne_bytes().to_vec(), "kernel {k} rows");
                // The finalize reads the raw tails, not the scratch keys.
                let sources = if *k == TAIL_WRITE {
                    a[..2].to_vec()
                } else {
                    Vec::new()
                };
                (*k, sources, a[4].clone(), rows)
            })
            .collect();
        out = Some(Ok((found, [arena.ssm_qkvz(), meta.slot])));
    });
    out.unwrap()
}

fn writes(dtype: KvCacheDtype, seq_len_start: usize, rows: usize, floor: usize) -> Pass {
    try_writes(dtype, seq_len_start, rows, floor, |_| ()).unwrap()
}

/// The index writes an owner of `rows` rows at stacked row `row0` issues when
/// its first `skip` rows are floored.
fn owner_index_writes(
    [scratch, slot]: [DevicePtr; 2],
    row0: usize,
    rows: usize,
    skip: usize,
) -> Vec<Write> {
    let slots = ptr(slot.offset((row0 + skip) * 8));
    let keys = scratch.offset(skip * KEY_ROW);
    let gates = scratch.offset((rows + skip) * KEY_ROW);
    let n = (rows - skip) as u32;
    vec![
        (TAIL_WRITE, vec![ptr(keys), ptr(gates)], slots.clone(), n),
        (POOL_FINALIZE, Vec::new(), slots, n),
    ]
}

fn latent_write(
    dtype: KvCacheDtype,
    [scratch, slot]: [DevicePtr; 2],
    total: usize,
    floor: usize,
) -> Write {
    let mut sources = vec![ptr(scratch.offset(floor * LATENT_ROW))];
    if dtype == KvCacheDtype::Bf16 {
        sources.push(ptr(scratch.offset((total + floor) * LATENT_ROW)));
    }
    (
        LATENT_WRITE,
        sources,
        ptr(slot.offset(floor * 8)),
        (total - floor) as u32,
    )
}

#[test]
fn no_floor_writes_every_row_from_the_unshifted_operands() {
    // Floor 0 is every cold prefill: the launches carry the base pointers and
    // the full row count, i.e. exactly what the path issued before the floor.
    for dtype in DTYPES {
        let (found, at) = writes(dtype, 32, 48, 0);
        let mut want = vec![latent_write(dtype, at, 48, 0)];
        want.extend(owner_index_writes(at, 0, 48, 0));
        assert_eq!(found, want, "{dtype:?}");
    }
}

#[test]
fn warm_replay_rows_below_the_floor_are_not_written() {
    // Snapshot at 32, radix match at 48: rows [0, 16) replay shared blocks.
    for dtype in DTYPES {
        let (found, at) = writes(dtype, 32, 48, 16);
        let mut want = vec![latent_write(dtype, at, 48, 16)];
        want.extend(owner_index_writes(at, 0, 48, 16));
        assert_eq!(found, want, "{dtype:?}");
    }
}

#[test]
fn a_chunk_wholly_below_the_floor_writes_nothing() {
    for dtype in DTYPES {
        for floor in [48, 1000] {
            let (found, _) = writes(dtype, 32, 48, floor);
            assert!(found.is_empty(), "{dtype:?} floor {floor}");
        }
    }
}

#[test]
fn the_floor_carries_across_the_dense_and_sparse_pieces() {
    // BF16 only: the fp8_g128 sparse reader is an env-gated kernel, and the
    // writes under test precede and ignore the reader.
    let dtype = KvCacheDtype::Bf16;
    // Rows [0, 32) end at the top-k boundary (dense), rows [32, 64) select.
    // One joint latent write, then each piece's index update.
    let (found, at) = writes(dtype, 2016, 64, 0);
    let mut want = vec![latent_write(dtype, at, 64, 0)];
    want.extend(owner_index_writes(at, 0, 32, 0));
    want.extend(owner_index_writes(at, 32, 32, 0));
    assert_eq!(found, want);

    // Floor inside the first piece.
    let (found, at) = writes(dtype, 2016, 64, 16);
    let mut want = vec![latent_write(dtype, at, 64, 16)];
    want.extend(owner_index_writes(at, 0, 32, 16));
    want.extend(owner_index_writes(at, 32, 32, 0));
    assert_eq!(found, want);

    // Floor past the first piece: it updates no index, the second skips 16.
    let (found, at) = writes(dtype, 2016, 64, 48);
    let mut want = vec![latent_write(dtype, at, 64, 48)];
    want.extend(owner_index_writes(at, 32, 32, 16));
    assert_eq!(found, want);
}

#[test]
fn a_floor_inside_an_index_pool_is_refused() {
    // Pools hold 4 tokens. A floor ending at token 50 would finalize pool
    // [48, 52) from two cached and two new raw tails.
    for dtype in DTYPES {
        let err = try_writes(dtype, 32, 48, 18, |_| ()).unwrap_err();
        assert!(
            format!("{err:#}").contains("splits an index pool"),
            "{err:#}"
        );
    }
    // A snapshot inside a pool is fine: the floor still ends on one.
    let (found, at) = writes(KvCacheDtype::Bf16, 30, 50, 18);
    let mut want = vec![latent_write(KvCacheDtype::Bf16, at, 50, 18)];
    want.extend(owner_index_writes(at, 0, 50, 18));
    assert_eq!(found, want);
}

#[test]
fn an_mla_path_that_writes_every_row_refuses_a_floor() {
    // The generic MLA paged prefill has no floor: it must not drop one.
    let generic = |layer: &mut Qwen3AttentionLayer| {
        layer.mla.as_mut().unwrap().glm_indexer = None;
    };
    let err = try_writes(KvCacheDtype::Bf16, 32, 48, 16, generic).unwrap_err();
    assert!(
        format!("{err:#}").contains("cannot honour a KV write floor"),
        "{err:#}"
    );
}
