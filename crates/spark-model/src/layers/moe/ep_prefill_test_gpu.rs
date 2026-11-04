// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`; a GB10 and a glm-5.3-flash kernel build): the EP
//! unpermute-reduce at a TP4 prefill chunk's shape (8192 rows, top-8 of 288
//! experts, 72 local), timed.
//!
//!   cargo test --release -p spark-model --lib unpermute_ep_gpu -- --ignored --nocapture

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

const H: u32 = 4096;
const E: u32 = 288;
const TOPK: u32 = 8;

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

/// Routes of `n` tokens: distinct experts per token, sorted rows by expert.
pub(super) fn routes(n: u32, seed: u64) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let mut s = seed;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut ids = Vec::with_capacity((n * TOPK) as usize);
    let mut weights = Vec::with_capacity((n * TOPK) as usize);
    for _ in 0..n {
        let mut chosen: Vec<i32> = Vec::new();
        while chosen.len() < TOPK as usize {
            let e = (next() % E as u64) as i32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        ids.extend(&chosen);
        weights.extend((0..TOPK).map(|_| (next() % 1000) as f32 / 400.0));
    }
    // Stable sort of the (token, slot) routes by expert.
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&i| ids[i]);
    let mut to_perm = vec![0i32; ids.len()];
    for (pos, &i) in order.iter().enumerate() {
        to_perm[i] = pos as i32;
    }
    (ids, weights, to_perm)
}

pub(super) fn bf16_values(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .flat_map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 3.0).to_bits();
            (((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn unpermute_ep_gpu_timing() -> Result<()> {
    if std::env::var("UNPERMUTE_BENCH_PDL").as_deref() == Ok("1") {
        // SAFETY: set before any reader, in this test only.
        unsafe { std::env::set_var("ATLAS_PDL", "1") };
        spark_runtime::cuda_backend::configure_pdl("glm-5.3-flash");
    }
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let n = 8192u32;
    let (ids, weights, to_perm) = routes(n, 7);
    let as_bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let ids_d = upload(&gpu, &as_bytes(&ids))?;
    let perm_d = upload(&gpu, &as_bytes(&to_perm))?;
    let w_d = upload(
        &gpu,
        &weights
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let rows = (n * TOPK) as usize;
    let experts = upload(&gpu, &bf16_values(rows * H as usize, 3))?;
    let out = gpu.alloc((n * H * 2) as usize)?;
    let k = gpu.kernel("moe", "moe_unpermute_reduce_indexed_ep_vec8")?;
    let whole = gpu.alloc((n * H * 2) as usize)?;
    for (start, end, ctas) in [
        (72u32, 144u32, n),
        (0, 288, n),
        (0, 288, 384),
        (0, 288, 192),
        (0, 288, 96),
        (0, 288, 64),
    ] {
        let launch = || {
            ops::moe_unpermute_reduce_indexed_ep_grid(
                &gpu, k, experts, out, perm_d, ids_d, w_d, H, n, TOPK, start, end, ctas, None,
                stream,
            )
        };
        launch()?;
        gpu.synchronize(stream)?;
        let t = std::time::Instant::now();
        for _ in 0..20 {
            launch()?;
        }
        gpu.synchronize(stream)?;
        let us = t.elapsed().as_secs_f64() * 1e6 / 20.0;
        let local = ids
            .iter()
            .filter(|&&e| e as u32 >= start && (e as u32) < end)
            .count();
        let bytes = (local as f64 + n as f64) * H as f64 * 2.0;
        println!(
            "unpermute ep vec8 experts {start}..{end}, {ctas} blocks: {us:.0} us, {:.0} GB/s",
            bytes / us / 1e3
        );
        // A short grid leaves the whole grid's bytes.
        let mut got = vec![0u8; (n * H * 2) as usize];
        gpu.copy_d2h(out, &mut got)?;
        if ctas == n {
            gpu.copy_d2d(out, whole, got.len())?;
            // The vec8 kernel sums the scalar kernel's products in its order.
            let scalar = gpu.kernel("moe", "moe_unpermute_reduce_indexed_ep")?;
            ops::moe_unpermute_reduce_indexed_ep_grid(
                &gpu, scalar, experts, out, perm_d, ids_d, w_d, H, n, TOPK, start, end, n, None,
                stream,
            )?;
            let mut want = vec![0u8; got.len()];
            gpu.copy_d2h(out, &mut want)?;
            anyhow::ensure!(
                got == want,
                "experts {start}..{end}: vec8 bytes differ from the scalar kernel"
            );
        } else {
            let mut want = vec![0u8; got.len()];
            gpu.copy_d2h(whole, &mut want)?;
            anyhow::ensure!(
                got == want,
                "{ctas} blocks: bytes differ from the whole grid"
            );
        }
    }
    Ok(())
}

/// The fused reduce-scatter's producer (`moe_unpermute_rs_fused_vec8`) on a
/// stand-in channel in device memory: every piece, in a scrambled order,
/// raw ones to their own staging, holds the vec8 kernel's rows, and is
/// published once (raw to `ready`, kept to the state's own words).
///
///   cargo test --release -p spark-model --lib unpermute_rs_fused_gpu -- --ignored --nocapture
#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn unpermute_rs_fused_gpu_matches_vec8() -> Result<()> {
    use spark_comm::{RS_FUSED_MAX_PIECES, RsFusedPiece, RsFusedPlan};
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let (n, pieces) = (2048u32, 16usize);
    let row = H as usize * 2;
    let piece_bytes = n as usize / pieces * row;
    let (ids, weights, to_perm) = routes(n, 11);
    let as_bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let ids_d = upload(&gpu, &as_bytes(&ids))?;
    let perm_d = upload(&gpu, &as_bytes(&to_perm))?;
    let w_d = upload(
        &gpu,
        &weights
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let experts = upload(&gpu, &bf16_values((n * TOPK) as usize * H as usize, 5))?;
    let (start, end) = (72u32, 144u32);
    let want_d = gpu.alloc(n as usize * row)?;
    let vec8 = gpu.kernel("moe", "moe_unpermute_reduce_indexed_ep_vec8")?;
    ops::moe_unpermute_reduce_indexed_ep_grid(
        &gpu, vec8, experts, want_d, perm_d, ids_d, w_d, H, n, TOPK, start, end, 32, None, stream,
    )?;
    let zeros = |bytes: usize| -> Result<DevicePtr> { upload(&gpu, &vec![0u8; bytes]) };
    let (out, staging) = (zeros(n as usize * row)?, zeros(n as usize * row)?);
    let (state, stage, poison, ready) = (zeros(1024)?, zeros(64)?, zeros(64)?, zeros(8 * 64)?);
    let mut plan = RsFusedPlan {
        ready: ready.0,
        state: state.0,
        stage: stage.0,
        poison: poison.0,
        timeout_ns: 10_000_000_000,
        code: 1,
        npieces: pieces as u32,
        piece_bytes: piece_bytes as u32,
        mode: 1,
        pad: 0,
        piece: [RsFusedPiece::default(); RS_FUSED_MAX_PIECES],
    };
    // A scrambled order; every third piece raw.
    for (i, p) in (0..pieces).map(|i| (i * 7) % pieces).enumerate() {
        let raw = p % 3 == 0;
        plan.piece[i] = RsFusedPiece {
            out: out.offset(p * piece_bytes).0,
            s: if raw {
                staging.offset(p * piece_bytes).0
            } else {
                0
            },
            p: p as u32,
            pad: 0,
        };
    }
    let fused = gpu.kernel("moe_unpermute_fused", "moe_unpermute_rs_fused_vec8")?;
    crate::layers::glm_sp_moe_seam::unpermute_rs_fused(
        &gpu,
        fused,
        experts,
        out,
        (perm_d, ids_d, w_d),
        H,
        TOPK,
        (start, end),
        32,
        None,
        plan.words(),
        stream,
    )?;
    gpu.synchronize(stream)?;
    let read = |p: DevicePtr, bytes: usize| -> Result<Vec<u8>> {
        let mut v = vec![0u8; bytes];
        gpu.copy_d2h(p, &mut v)?;
        Ok(v)
    };
    let (want, got, raw) = (
        read(want_d, n as usize * row)?,
        read(out, n as usize * row)?,
        read(staging, n as usize * row)?,
    );
    let (st, rd) = (read(state, 1024)?, read(ready, 8 * 64)?);
    let word = |v: &[u8], at: usize| u64::from_le_bytes(v[at..at + 8].try_into().unwrap());
    for p in 0..pieces {
        let span = p * piece_bytes..(p + 1) * piece_bytes;
        let (bytes, flag) = if p % 3 == 0 {
            (&raw[span.clone()], word(&rd, p * 8))
        } else {
            (&got[span.clone()], word(&st, 512 + p * 8))
        };
        anyhow::ensure!(
            bytes == &want[span],
            "piece {p}: bytes differ from the vec8 kernel"
        );
        anyhow::ensure!(flag == 1, "piece {p}: published {flag}, not seq 1");
        let produced = u32::from_le_bytes(st[256 + p * 4..260 + p * 4].try_into().unwrap());
        anyhow::ensure!(produced == 0, "piece {p}: writer count left at {produced}");
    }
    Ok(())
}
