// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention): rows with no routes through
//! `ATLAS_GLM_MOE_ROUTE_FUSE`. See `route_fuse_test_gpu.rs`.

use super::*;

/// Blend `out` (one routing, read back) with expert rows that are a function
/// of (token, expert) alone, so two routings sorting the same routes to
/// other rows blend the same values: `[rows, H]` BF16 words.
fn blend(gpu: &dyn GpuBackend, b: &Bufs, out: &Out, shared: DevicePtr) -> Result<Vec<u8>> {
    let a = &b.args;
    let te = (a.rows * TOPK) as usize;
    let sorted_tokens = words(gpu, a.sorted_token_ids, te)?;
    let mut rows = vec![0u16; te * H as usize];
    for (p, row) in rows.chunks_mut(H as usize).enumerate() {
        let (t, e) = (sorted_tokens[p], out.sorted_experts[p]);
        for (c, v) in row.iter_mut().enumerate() {
            let x = ((t * 131 + e * 17 + c as u32 * 7) % 1013) as f32 / 1013.0 - 0.5;
            *v = bf16(x);
        }
    }
    let expert_output = upload(gpu, &bytes_of(&rows))?;
    let output = upload(gpu, &vec![POISON; (a.rows * H) as usize * 2])?;
    KernelLaunch::new(gpu, gpu.kernel("moe", "moe_unpermute_blend_ep_vec8")?)
        .grid([a.rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_output)
        .arg_ptr(output)
        .arg_ptr(a.token_to_perm)
        .arg_ptr(a.indices)
        .arg_ptr(a.weights)
        .arg_ptr(shared)
        .arg_ptr(shared)
        .arg_ptr(DevicePtr(0))
        .arg_u32(a.rows)
        .arg_u32(TOPK)
        .arg_u32(0)
        .arg_u32(E)
        .launch(gpu.default_stream())?;
    gpu.synchronize(gpu.default_stream())?;
    raw(gpu, output, (a.rows * H) as usize * 2)
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn dead_rows_route_nowhere() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let mut rng = Rng(0xdead_0105);
    let layer = Layer::new(&gpu, &mut rng)?;
    let row_bytes = H as usize * 2;
    // The test chain's router takes up to 32 rows.
    for (owners, width) in [(2u32, 4u32), (4, 4), (8, 4), (4, 8), (5, 6)] {
        let rows = owners * width;
        // Owner o keeps its first `1 + (o * 5) % width` rows (every owner a
        // different prefix; one owner wholly live).
        let live: Vec<u32> = (0..rows)
            .map(|r| u32::from(r % width < 1 + (r / width * 5) % width))
            .collect();
        let live_dev = upload(&gpu, &bytes_of(&live))?;
        let shared: Vec<u16> = (0..rows * H).map(|_| bf16(rng.unit())).collect();
        let shared_dev = upload(&gpu, &bytes_of(&shared))?;
        let x = input(&gpu, &mut rng, rows, None)?;
        for sort in ["moe_sort_by_expert_scan", "moe_sort_by_expert"] {
            let (full, dead) = (
                Bufs::new(&gpu, &layer, x, rows)?,
                Bufs::new(&gpu, &layer, x, rows)?,
            );
            full.chain_with(&gpu, None, sort, stream)?;
            dead.chain_with(&gpu, Some(live_dev), sort, stream)?;
            let (want, got) = (full.read(&gpu, stream)?, dead.read(&gpu, stream)?);
            // The fused pair (and the one-launch kernel at <= 8 rows) masks
            // the same rows to the same words.
            let fuse = RouteFuse::load(&gpu);
            if fuse.loaded() && sort == "moe_sort_by_expert_scan" {
                let mut fused = Bufs::new(&gpu, &layer, x, rows)?;
                fused.args.row_live = live_dev;
                fuse.launch(&gpu, &fused.args, stream)?;
                ensure!(
                    fused.read(&gpu, stream)? == got,
                    "{owners}x{width}: fused routing masks dead rows differently"
                );
            }
            let live_routes = live.iter().filter(|&&l| l != 0).count() * TOPK as usize;
            ensure!(
                got.offsets[E as usize] as usize == live_routes,
                "{owners}x{width} {sort}: {} routes sorted for {live_routes} live",
                got.offsets[E as usize]
            );
            for (i, (&w, &g)) in want.ids.iter().zip(&got.ids).enumerate() {
                let alive = live[i / TOPK as usize] != 0;
                ensure!(
                    if alive { g == w } else { g == E },
                    "{owners}x{width} {sort}: route {i} is {g}, want {}",
                    if alive { w } else { E }
                );
            }
            for e in 0..E as usize {
                let kept: Vec<(u32, u32)> = want.groups[e]
                    .iter()
                    .copied()
                    .filter(|&(t, _)| live[t as usize] != 0)
                    .collect();
                ensure!(
                    got.groups[e] == kept,
                    "{owners}x{width} {sort}: expert {e} holds other rows"
                );
            }
            ensure!(
                got.tiles[0] <= want.tiles[0],
                "{owners}x{width} {sort}: more tiles with dead rows"
            );
            let (a, b) = (
                blend(&gpu, &full, &want, shared_dev)?,
                blend(&gpu, &dead, &got, shared_dev)?,
            );
            let shared_bytes = bytes_of(&shared);
            for r in 0..rows as usize {
                let span = r * row_bytes..(r + 1) * row_bytes;
                if live[r] != 0 {
                    ensure!(
                        a[span.clone()] == b[span],
                        "{owners}x{width} {sort}: live row {r} blends differently"
                    );
                } else {
                    ensure!(
                        b[span.clone()] == shared_bytes[span],
                        "{owners}x{width} {sort}: dead row {r} is not its shared row"
                    );
                }
            }
        }
    }
    println!(
        "dead rows: no route, no tile, routed 0; live rows' routes, groups and blend unchanged"
    );
    Ok(())
}
