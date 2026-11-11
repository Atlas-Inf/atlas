// SPDX-License-Identifier: AGPL-3.0-only

//! The chain merge's and view's GPU cases.

use super::*;

/// One case on all four ranks; the worst error over the tolerance.
fn run_case(gpu: &dyn GpuBackend, rows: u32, ctx: usize, sel: Selection, seed: u64) -> Result<f64> {
    let c = config();
    let live = Mutex::new(vec![]);
    let mut rng = Rng(seed | 1);
    let blocks = ctx.div_ceil(BS);
    let slots = blocks.div_ceil(WORLD) + 3;
    let logical: Vec<Vec<u8>> = (0..blocks)
        .map(|_| {
            (0..BS * 512)
                .flat_map(|_| bf16(rng.unit()).to_le_bytes())
                .collect()
        })
        .collect();
    let q_host: Vec<u16> = (0..rows as usize * 64 * 512)
        .map(|_| bf16(rng.unit() * 0.6))
        .collect();
    let (sel_flat, ids) = rows_ids(rows as usize, ctx, sel, &mut rng);
    let causal = match sel {
        Selection::Causal(s) => s,
        Selection::Sparse { .. } => 0,
    };
    let sel_dev = sel_flat
        .as_ref()
        .map(|s| {
            upload(
                gpu,
                &live,
                &s.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
            )
        })
        .transpose()?;
    let m = ChainLayout::new(rows, WORLD);
    let local = 16usize;
    let out_bytes = rows as usize * local * 512 * 2;
    let mut per_rank = vec![];
    for r in 0..WORLD {
        let t = table(blocks, slots, &mut rng);
        let pool = alloc(gpu, &live, slots * BB, 0x7f)?;
        for (l, &b) in t
            .iter()
            .enumerate()
            .filter(|(_, b)| **b as usize % WORLD == r)
        {
            gpu.copy_h2d(&logical[l], pool.offset(b as usize / WORLD * BB))?;
        }
        let identity: Vec<u32> = (0..slots as u32).collect();
        let q: Vec<u8> = (0..rows as usize)
            .flat_map(|row| q_host[(row * 64 + r * local) * 512..][..local * 512].to_vec())
            .flat_map(|h| h.to_le_bytes())
            .collect();
        per_rank.push((
            upload(gpu, &live, &u32s(&t))?,
            pool,
            upload(gpu, &live, &u32s(&identity))?,
            upload(gpu, &live, &q)?,
            alloc(gpu, &live, m.total, 0x7f)?,
            alloc(gpu, &live, out_bytes, 0x7f)?,
        ));
    }
    let (posts, meet) = (Mutex::new([(0, 0, 0); WORLD]), Barrier::new(WORLD));
    let outs: Vec<Vec<u8>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..WORLD)
            .map(|r| {
                let (per_rank, posts, meet, c) = (&per_rank, &posts, &meet, &c);
                scope.spawn(move || -> Result<Vec<u8>> {
                    gpu.bind_to_thread()?;
                    let stream = gpu.create_stream()?;
                    let comm = MemChain {
                        gpu,
                        rank: r,
                        posts,
                        meet,
                    };
                    let (table, pool, identity, query, scratch, output) = per_rank[r];
                    let merge = ChainMerge {
                        gpu,
                        comm: &comm,
                        config: c,
                        shard: LatentShard {
                            spec: LatentShardSpec {
                                rank: r,
                                world: WORLD,
                                scratch_bytes: m.total,
                                view_blocks: 0,
                                write_rows: 0,
                                lane: false,
                            },
                            local_blocks: slots,
                            scratch,
                            identity,
                            lane: None,
                        },
                        work: 0,
                        work_bytes: m.total,
                        dtype: KvCacheDtype::Bf16,
                        pool,
                        scale: 0.0625,
                    };
                    let a = ShardRows {
                        query,
                        selected: sel_dev,
                        causal_start: causal,
                        block_table: table,
                        rows,
                        end: None,
                        queries_swapped: false,
                        localized: false,
                    };
                    merge.run(a, output, stream)?;
                    gpu.synchronize(stream)?;
                    let mut out = vec![0u8; out_bytes];
                    gpu.copy_d2h(output, &mut out)?;
                    Ok(out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect::<Result<Vec<_>>>()
    })?;
    let mut worst_ratio = 0f64;
    for (r, out) in outs.iter().enumerate() {
        let want = reference(&logical, &q_host, &ids, r * local, local);
        let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
        let max = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        let worst = out
            .chunks_exact(2)
            .zip(&want)
            .map(|(h, w)| (from_bf16(u16::from_le_bytes([h[0], h[1]])) as f64 - w).abs())
            .fold(
                0f64,
                |a, e| if e.is_nan() { f64::INFINITY } else { a.max(e) },
            );
        worst_ratio = worst_ratio.max(worst / (0.02 * rms + max / 64.0));
    }
    for p in live.into_inner().unwrap() {
        gpu.free(p)?;
    }
    Ok(worst_ratio)
}

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn chain_shard_merge_matches_attention() -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    let sparse = |skip| Selection::Sparse { skip };
    let cases = [
        (1, 65536, sparse(None)),
        (3, 16384, sparse(None)),
        (8, 32768, sparse(Some(3))),
        (5, 8192, sparse(Some(0))),
        (16, 4096, sparse(None)),
        (64, 4352, sparse(None)),
        (8, 64, Selection::Causal(0)),
        (3, 2048, Selection::Causal(1000)),
        (1, 40, Selection::Causal(20)),
    ];
    let mut failed = 0;
    for (i, &(rows, ctx, sel)) in cases.iter().enumerate() {
        let ratio = run_case(gpu, rows, ctx, sel, 0x51ab_c4a1 ^ i as u64)?;
        let ok = ratio <= 1.0;
        println!(
            "{} rows={rows} ctx={ctx} {sel:?}: worst/tolerance {ratio:.3}",
            if ok { "OK" } else { "FAIL" }
        );
        failed += (!ok) as usize;
    }
    ensure!(
        failed == 0,
        "{failed} of {} cases off the reference",
        cases.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires a GB10 and a glm-5.3-flash kernel build"]
fn chain_shard_view_assembles_the_history() -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
        .context("CUDA backend")?;
    let gpu: &dyn GpuBackend = &gpu;
    for (case, &blocks) in [1usize, 5, 4099, 9001].iter().enumerate() {
        let live = Mutex::new(vec![]);
        let mut rng = Rng(0x7e57 + case as u64);
        let slots = blocks.div_ceil(WORLD) + 2;
        let logical: Vec<Vec<u8>> = (0..blocks)
            .map(|_| (0..BB).map(|_| rng.next() as u8).collect())
            .collect();
        let layout = ScratchLayout::for_world(WORLD, blocks, 0, BB);
        let mut per_rank = vec![];
        for r in 0..WORLD {
            let t = table(blocks, slots, &mut rng);
            let pool = alloc(gpu, &live, slots * BB, 0)?;
            let mut mine_slot = vec![];
            let mut mine_dst = vec![];
            for (l, &b) in t
                .iter()
                .enumerate()
                .filter(|(_, b)| **b as usize % WORLD == r)
            {
                gpu.copy_h2d(&logical[l], pool.offset(b as usize / WORLD * BB))?;
                mine_slot.push(b / WORLD as u32);
                mine_dst.push(l as u32);
            }
            let scratch = alloc(gpu, &live, layout.total, 0x5a)?;
            let view = scratch.offset(layout.view);
            let slot_dev = upload(gpu, &live, &u32s(&mine_slot))?;
            let dst_dev = upload(gpu, &live, &u32s(&mine_dst))?;
            ops::glm_kv_shard_copy_blocks(
                gpu,
                pool,
                Some(slot_dev),
                view,
                Some(dst_dev),
                mine_slot.len(),
                BB,
                0,
            )?;
            gpu.synchronize(0)?;
            per_rank.push((pool, scratch, slot_dev));
        }
        let (posts, meet) = (Mutex::new([(0, 0, 0); WORLD]), Barrier::new(WORLD));
        let views: Vec<Vec<u8>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..WORLD)
                .map(|r| {
                    let (per_rank, posts, meet, layout) = (&per_rank, &posts, &meet, &layout);
                    scope.spawn(move || -> Result<Vec<u8>> {
                        gpu.bind_to_thread()?;
                        let stream = gpu.create_stream()?;
                        let comm = MemChain {
                            gpu,
                            rank: r,
                            posts,
                            meet,
                        };
                        let (pool, scratch, mine_slot) = per_rank[r];
                        let s = LatentShard {
                            spec: LatentShardSpec {
                                rank: r,
                                world: WORLD,
                                scratch_bytes: layout.total,
                                view_blocks: blocks,
                                write_rows: 0,
                                lane: false,
                            },
                            local_blocks: slots,
                            scratch,
                            identity: DevicePtr::NULL,
                            lane: None,
                        };
                        let view = scratch.offset(layout.view);
                        let parts = ViewParts {
                            pool,
                            view,
                            mine_slot,
                            blocks,
                            block_bytes: BB,
                        };
                        parts.gather(gpu, &comm, &s, layout, stream)?;
                        gpu.synchronize(stream)?;
                        let mut out = vec![0u8; blocks * BB];
                        gpu.copy_d2h(view, &mut out)?;
                        Ok(out)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread"))
                .collect::<Result<Vec<_>>>()
        })?;
        let want: Vec<u8> = logical.concat();
        for (r, v) in views.iter().enumerate() {
            ensure!(*v == want, "{blocks} blocks: rank {r}'s view differs");
        }
        println!("OK view of {blocks} blocks on all {WORLD} ranks");
        for p in live.into_inner().unwrap() {
            gpu.free(p)?;
        }
    }
    Ok(())
}
