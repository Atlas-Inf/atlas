// SPDX-License-Identifier: AGPL-3.0-only
//! GLM semantic-index top-K (`glm_index_topk_expand`) against a CPU
//! reference: every pool above the Kth pooled score, then the lowest-indexed
//! pools equal to it, expanded to four token IDs in ascending order, plus the
//! causal tail. Logits are quantized so ties at the threshold are common.
//! Reports the time of a long-context chunk (4096 rows).
//!
//! Exit: 0 pass, 1 mismatch.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example glm_topk_bench

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const TOPK: u32 = 2048;
const POOL: u32 = 4;
const WIDTH: u32 = TOPK + POOL - 1;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

/// CPU selection for one row of `seq_len` tokens.
fn reference(logits: &[f32], seq_len: u32) -> Vec<i32> {
    let pools = (seq_len / POOL) as usize;
    let select = ((TOPK / POOL) as usize).min(pools);
    let mut out = vec![-1i32; WIDTH as usize];
    let mut order: Vec<usize> = (0..pools).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let mut chosen: Vec<usize> = order[..select].to_vec();
    chosen.sort_unstable();
    for (d, &p) in chosen.iter().enumerate() {
        for i in 0..POOL as usize {
            out[d * POOL as usize + i] = (p * POOL as usize + i) as i32;
        }
    }
    let tail = pools as u32 * POOL;
    for i in 0..seq_len - tail {
        out[(TOPK + i) as usize] = (tail + i) as i32;
    }
    out
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kernel = g.kernel("glm_indexer", "glm_index_topk_expand")?;
    let mut rng = Lcg(0x70BC);
    let mut fail = false;
    for (rows, start) in [(64u32, 1000u32), (64, 30_000), (4096, 112_000)] {
        let stride = (start + rows).div_ceil(POOL);
        // Scores on a 1/64 grid: many exact ties.
        let logits: Vec<f32> = (0..rows as usize * stride as usize)
            .map(|_| (rng.next() % 4096) as f32 / 64.0 - 32.0)
            .collect();
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let d_logits = up(g, &bytes)?;
        let d_out = g.alloc(rows as usize * WIDTH as usize * 4)?;
        let run = || {
            KernelLaunch::new(g, kernel)
                .grid([rows, 1, 1])
                .block([256, 1, 1])
                .shared_mem(16)
                .arg_ptr(d_logits)
                .arg_ptr(d_out)
                .arg_u32(rows)
                .arg_u32(start)
                .arg_u32(stride)
                .arg_u32(TOPK)
                .arg_u32(POOL)
                .arg_u32(WIDTH)
                .launch(0)
        };
        run()?;
        g.synchronize(0)?;
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            run()?;
        }
        g.synchronize(0)?;
        let t = t0.elapsed().as_secs_f64() / 10.0;
        let mut got = vec![0u8; rows as usize * WIDTH as usize * 4];
        g.copy_d2h(d_out, &mut got)?;
        let got: Vec<i32> = got
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let check_rows = rows.min(64);
        let mut bad = 0;
        for r in 0..check_rows as usize {
            let row = &logits[r * stride as usize..(r + 1) * stride as usize];
            let want = reference(row, start + r as u32 + 1);
            bad += (want != got[r * WIDTH as usize..(r + 1) * WIDTH as usize]) as usize;
        }
        fail |= bad != 0;
        println!(
            "rows {rows:4} at context {start:6}: {:8.1}us  {}",
            t * 1e6,
            if bad == 0 {
                format!("exact ({check_rows} rows checked)")
            } else {
                format!("MISMATCH in {bad} rows")
            }
        );
        g.free(d_logits)?;
        g.free(d_out)?;
    }
    std::process::exit(if fail { 1 } else { 0 });
}
