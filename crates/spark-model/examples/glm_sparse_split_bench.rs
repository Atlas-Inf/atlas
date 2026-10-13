// SPDX-License-Identifier: AGPL-3.0-only
//! GLM sparse MLA attention at verify widths: the unsplit tensor-core kernel
//! (one CTA per row) against its split-over-selected-IDs variant plus
//! `glm_sparse_decode_split_merge`, on a synthetic BF16 latent cache.
//! Reports time and the largest output difference (the split sums in a
//! different order, so outputs agree to BF16 rounding, not bit for bit).
//!
//! Exit: 0 pass, 1 difference above tolerance.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example glm_sparse_split_bench

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const MODULE: &str = "glm_sparse_prefill_kv_reuse";
const KERNEL: &str = "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad";
const HEADS: u32 = 32;
const DIM: usize = 512;
const WIDTH: u32 = 2051;
const TOKENS: usize = 32768;
const BLOCK: u32 = 16;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn time(g: &dyn GpuBackend, f: &dyn Fn() -> Result<()>) -> Result<f64> {
    f()?;
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() / 20.0)
}

fn to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let unsplit = g.kernel(MODULE, KERNEL)?;
    let split = g.kernel(MODULE, &format!("{KERNEL}_split"))?;
    let merge = g.kernel(
        "glm_sparse_decode_split_merge",
        "glm_sparse_decode_split_merge",
    )?;
    let mut rng = Lcg(0x5A5A);
    let cache: Vec<u8> = (0..TOKENS * DIM).flat_map(|_| bf16(rng.f())).collect();
    let table: Vec<u8> = (0..(TOKENS as u32 / BLOCK))
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let (d_cache, d_table) = (up(g, &cache)?, up(g, &table)?);
    let mut fail = false;
    for rows in [1u32, 6, 8, 24] {
        let q: Vec<u8> = (0..rows as usize * HEADS as usize * DIM)
            .flat_map(|_| bf16(rng.f() * 2.0))
            .collect();
        // Selected IDs: distinct-ish tokens, the last three slots empty.
        let idx: Vec<u8> = (0..rows * WIDTH)
            .flat_map(|i| {
                let t = if i % WIDTH >= WIDTH - 3 {
                    -1
                } else {
                    (rng.next() % TOKENS as u64) as i32
                };
                t.to_le_bytes()
            })
            .collect();
        let (d_q, d_idx) = (up(g, &q)?, up(g, &idx)?);
        let out_bytes = rows as usize * HEADS as usize * DIM * 2;
        let (d_ref, d_new) = (g.alloc(out_bytes)?, g.alloc(out_bytes)?);
        let launch = |k, z: u32, out: DevicePtr, part: Option<(DevicePtr, DevicePtr)>| {
            let mut l = KernelLaunch::new(g, k)
                .grid([1, rows, z])
                .block([256, 1, 1])
                .shared_mem(69376)
                .arg_ptr(d_q)
                .arg_ptr(d_cache)
                .arg_ptr(d_cache)
                .arg_ptr(d_idx)
                .arg_ptr(out)
                .arg_ptr(d_table)
                .arg_u32(rows)
                .arg_u32(HEADS)
                .arg_u32(DIM as u32)
                .arg_u32(WIDTH)
                .arg_u32(BLOCK)
                .arg_f32(1.0 / 16.0);
            if let Some((po, pl)) = part {
                l = l.arg_ptr(po).arg_ptr(pl);
            }
            l.launch(0)
        };
        let t_ref = time(g, &|| launch(unsplit, 1, d_ref, None))?;
        let mut want = vec![0u8; out_bytes];
        g.copy_d2h(d_ref, &mut want)?;
        for splits in [2u32, 4, 6, 8, 12, 16] {
            let rh = rows as usize * HEADS as usize;
            let d_po = g.alloc(splits as usize * rh * DIM * 4)?;
            let (d_pl, d_ol) = (g.alloc(splits as usize * rh * 4)?, g.alloc(rh * 4)?);
            let run = || -> Result<()> {
                launch(split, splits, d_new, Some((d_po, d_pl)))?;
                KernelLaunch::new(g, merge)
                    .grid([rows * HEADS, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_po)
                    .arg_ptr(d_pl)
                    .arg_ptr(d_new)
                    .arg_ptr(d_ol)
                    .arg_u32(rows)
                    .arg_u32(HEADS)
                    .arg_u32(DIM as u32)
                    .arg_u32(splits)
                    .launch(0)
            };
            let t_new = time(g, &run)?;
            let mut got = vec![0u8; out_bytes];
            g.copy_d2h(d_new, &mut got)?;
            let (w, n) = (to_f32(&want), to_f32(&got));
            let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            let diff = w
                .iter()
                .zip(&n)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let ok = diff <= scale * 1.6e-2;
            fail |= !ok;
            println!(
                "rows {rows:2}: unsplit {:7.1}us, {splits:2} splits {:7.1}us, max |diff| {diff:.2e} of {scale:.2e} {}",
                t_ref * 1e6,
                t_new * 1e6,
                if ok { "ok" } else { "TOO LARGE" }
            );
            for p in [d_po, d_pl, d_ol] {
                g.free(p)?;
            }
        }
        for p in [d_q, d_idx, d_ref, d_new] {
            g.free(p)?;
        }
    }
    std::process::exit(if fail { 1 } else { 0 });
}
