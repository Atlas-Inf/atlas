// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

pub(super) fn run_case(
    gpu: &dyn GpuBackend,
    case: Case,
    seed: u64,
    place: Placement,
) -> Result<CaseOut> {
    let c = config();
    let mem = Allocs::new(gpu);
    // Contents and queries from `seed`, tables from the placement alone.
    let mut rng = Rng(seed | 1);
    let mut placing = Rng(place.seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1);
    let (rows, bb) = (case.rows, block_bytes(case.dtype));
    let blocks = case.ctx.div_ceil(BS);
    let half = blocks.max(place.pool_blocks).div_ceil(2) + 3;
    let logical: Vec<Vec<u8>> = (0..blocks)
        .map(|_| latent_block(case.dtype, &mut rng))
        .collect();
    // Unsharded: the whole history, any ids. Sharded: each rank its own
    // table (the ranks' ids differ, their residues do not) and half pool.
    let table_u = table(blocks, half + 4, false, &mut placing);
    let pool_u = mem.alloc(2 * (half + 4) * bb, place.poison)?;
    for (l, &b) in table_u.iter().enumerate() {
        gpu.copy_h2d(&logical[l], pool_u.offset(b as usize * bb))?;
    }
    let mut ranks = vec![];
    for r in 0..2u32 {
        let t = table(blocks, half, true, &mut placing);
        let pool = mem.alloc(half * bb, place.poison)?;
        for (l, &b) in t.iter().enumerate().filter(|(_, b)| **b % 2 == r) {
            gpu.copy_h2d(&logical[l], pool.offset((b / 2) as usize * bb))?;
        }
        let identity: Vec<u32> = (0..half as u32).collect();
        ranks.push((mem.upload(&t)?, pool, mem.upload(&identity)?));
    }
    let table_u = mem.upload(&table_u)?;

    let qn = rows as usize * 32 * 512;
    let mut owners = vec![];
    for o in 0..case.owners {
        let mut q: [Vec<u16>; 2] =
            [0, 1].map(|_| (0..qn).map(|_| bf16(rng.unit() * 0.6)).collect());
        let mut sel = selection(&case, &mut rng);
        if case.pinned_row {
            let mut pin = Rng(0x5eed_0001 + o as u64);
            for q in &mut q {
                q[..32 * 512]
                    .iter_mut()
                    .for_each(|x| *x = bf16(pin.unit() * 0.6));
            }
            if let Some(s) = sel.as_mut() {
                let one = selection(&Case { rows: 1, ..case }, &mut pin).context("one row")?;
                s[..WIDTH as usize].copy_from_slice(&one);
            }
        }
        let dev_sel = sel.as_ref().map(|s| mem.upload(s)).transpose()?;
        owners.push((
            q.clone(),
            [mem.upload(&q[0])?, mem.upload(&q[1])?],
            sel,
            dev_sel,
        ));
    }
    let causal = match case.selection {
        Selection::Causal(start) => start,
        Selection::Sparse { .. } | Selection::Indexer => 0,
    };
    let splits = shard::MERGE_SPLITS;
    let m = MergeLayout::new(rows, splits);
    let (out_bytes, lse_bytes) = (qn * 2, rows as usize * 32 * 4);

    // The shard: both ranks at once, each owner in turn on one scratch.
    let (posts, meet) = (Mutex::new([(0, 0); 2]), Barrier::new(2));
    let shard_out: Vec<Vec<(Vec<u8>, Vec<u8>)>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2usize)
            .map(|r| {
                let (owners, ranks, posts, meet, c, mem) =
                    (&owners, &ranks, &posts, &meet, &c, &mem);
                scope.spawn(move || -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
                    gpu.bind_to_thread()?;
                    let stream = gpu.create_stream()?;
                    let comm = MemPair {
                        gpu,
                        rank: r,
                        posts,
                        meet,
                    };
                    let (table, pool, identity) = ranks[r];
                    let scratch = mem.alloc(m.total, place.poison)?;
                    let merge = ShardMerge {
                        gpu,
                        comm: &comm,
                        config: c,
                        shard: LatentShard {
                            spec: LatentShardSpec {
                                rank: r,
                                world: 2,
                                scratch_bytes: m.total,
                                view_blocks: 0,
                                write_rows: 0,
                                lane: false,
                            },
                            local_blocks: half,
                            scratch,
                            identity,
                            lane: None,
                        },
                        work: 0,
                        work_bytes: m.total,
                        dtype: case.dtype,
                        pool,
                        scale: 0.0625,
                        lane: None,
                    };
                    let output = mem.alloc(out_bytes, place.poison)?;
                    let mut got = vec![];
                    for (_, q, _, sel) in owners {
                        let a = ShardRows {
                            query: q[r],
                            selected: *sel,
                            causal_start: causal,
                            block_table: table,
                            rows,
                            end: None,
                            queries_swapped: false,
                        };
                        merge.run(a, output, stream)?;
                        gpu.synchronize(stream)?;
                        got.push((
                            download(gpu, output, out_bytes)?,
                            download(gpu, scratch.offset(m.out_lse), lse_bytes)?,
                        ));
                    }
                    Ok(got)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect::<Result<Vec<_>>>()
    })?;

    // The canonical form, rank by rank on the main thread.
    let (scratch, canonical_bytes) = match place.scratch {
        Some(region) => region,
        None => {
            let bytes = ops::CanonicalLayout::bytes(rows);
            (mem.alloc(bytes, place.poison)?, bytes)
        }
    };
    let layout = ops::CanonicalLayout::new(rows, splits);
    let output = mem.alloc(out_bytes, place.poison)?;
    let stream = gpu.default_stream();
    let mut differing = 0;
    let mut bits = vec![];
    for (o, (q_host, q, sel, dev_sel)) in owners.iter().enumerate() {
        for r in 0..2u32 {
            let a = ops::GlmSparsePrefillTc {
                config: &c,
                dtype: case.dtype,
                identical_kv_latent: true,
                query: q[r as usize],
                k_cache: pool_u,
                v_cache: pool_u,
                indices: dev_sel.unwrap_or(DevicePtr::NULL),
                output,
                block_table: table_u,
                rows,
                heads: 32,
                head_dim: 512,
                index_width: WIDTH,
                block_size: 16,
                scale: 0.0625,
            };
            let region = (scratch, canonical_bytes);
            ops::glm_sparse_canonical(gpu, &a, *dev_sel, causal, r, region, stream)?;
            gpu.synchronize(stream)?;
            let out = download(gpu, output, out_bytes)?;
            let lse = download(gpu, scratch.offset(layout.out_lse), lse_bytes)?;
            let (s_out, s_lse) = &shard_out[r as usize][o];
            // Every owner in these cases attends at least one key. Only a
            // rank-local partial may have LSE=-inf; these are fully merged.
            for (label, output, lse) in [
                ("canonical attention", &out, &lse),
                ("sharded attention", s_out, s_lse),
            ] {
                crate::layers::glm_gpu_test_backend::finite_bf16(label, output)?;
                crate::layers::glm_gpu_test_backend::finite_f32(label, lse)?;
            }
            let diff = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(a, b)| a != b).count();
            let d = diff(&out, s_out) + diff(&lse, s_lse);
            if d != 0 {
                println!("  rank {r} owner {o}: {d} bytes differ");
            }
            differing += d;
            bits.extend([out.clone(), lse]);
            if case.dtype == KvCacheDtype::Bf16 && rows <= 4 {
                let ids: Vec<Vec<i32>> = (0..rows as usize)
                    .map(|row| match sel {
                        Some(s) => s[row * WIDTH as usize..][..WIDTH as usize]
                            .iter()
                            .copied()
                            .filter(|&t| t >= 0)
                            .collect(),
                        None => (0..=(causal as i32 + row as i32)).collect(),
                    })
                    .collect();
                let want = reference(&case, &logical, &q_host[r as usize], &ids);
                let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
                let max = want.iter().fold(0f64, |m, v| m.max(v.abs()));
                let worst = out
                    .chunks_exact(2)
                    .zip(&want)
                    .map(|(h, w)| (from_bf16(u16::from_le_bytes([h[0], h[1]])) as f64 - w).abs())
                    .fold(0f64, f64::max);
                let tolerance = 0.02 * rms + max / 64.0;
                ensure!(
                    worst <= tolerance,
                    "{case:?} rank {r}: canonical vs CPU attention {worst:e} > {tolerance:e}"
                );
            }
        }
    }
    for (out, lse) in shard_out.into_iter().flatten() {
        bits.extend([out, lse]);
    }
    Ok(CaseOut { differing, bits })
}
