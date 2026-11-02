// SPDX-License-Identifier: AGPL-3.0-only
//! Prefill-shaped benchmark + bitwise gate for the GLM mHC seam kernels
//! (`glm_hc_mix_ss_bf16`, `glm_hc_post_mix_ss_bf16`) at a 4100-row chunk:
//! each variant in `VARIANTS` must reproduce the first one's raw mix, sum of
//! squares and (post) highway byte for byte. Measured 0.85 / 1.74 ms (serving
//! nsys: 1.2-1.4 / 2.3-2.5 ms); 16- and 8-token tiles were no faster.
//!
//! Exit: 0 pass, 1 mismatch.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example glm_hc_seam_bench

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const MODULE: &str = "glm_hc_prefill_vec";
const T: usize = 4100;
const H: usize = 4096;
const K: usize = 4 * H;
/// (tokens per CTA, kernel name suffix); the first is the reference.
const VARIANTS: &[(u32, &str)] = &[(32, "")];

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
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

fn fetch(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
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

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut rng = Lcg(0x4C4C);
    let streams: Vec<u8> = (0..T * K).flat_map(|_| bf16(rng.f() * 4.0)).collect();
    let block: Vec<u8> = (0..T * H).flat_map(|_| bf16(rng.f())).collect();
    let post: Vec<u8> = (0..T * 4)
        .flat_map(|_| (1.0 + rng.f()).to_le_bytes())
        .collect();
    let comb: Vec<u8> = (0..T * 16)
        .flat_map(|_| (0.25 + 0.2 * rng.f()).to_le_bytes())
        .collect();
    let hc_fn: Vec<u8> = (0..24 * K)
        .flat_map(|_| (0.01 * rng.f()).to_le_bytes())
        .collect();
    let (d_pristine, d_block, d_post, d_comb, d_fn) = (
        up(g, &streams)?,
        up(g, &block)?,
        up(g, &post)?,
        up(g, &comb)?,
        up(g, &hc_fn)?,
    );
    let d_streams = g.alloc(streams.len())?;
    let (d_mix, d_ss) = (g.alloc(T * 24 * 4)?, g.alloc(T * 4)?);
    let mut fail = false;
    for post_mix in [false, true] {
        let base = if post_mix {
            "glm_hc_post_mix_ss_bf16"
        } else {
            "glm_hc_mix_ss_bf16"
        };
        let mut want: Option<Vec<u8>> = None;
        for &(tile, suffix) in VARIANTS {
            let name = format!("{base}{suffix}");
            let Ok(kernel) = g.kernel(MODULE, &name) else {
                println!("{name}: absent");
                continue;
            };
            let run = || {
                let mut l = KernelLaunch::new(g, kernel)
                    .grid([(T as u32).div_ceil(tile), 1, 1])
                    .block([256, 1, 1]);
                if post_mix {
                    l = l.arg_ptr(d_block);
                }
                l = l.arg_ptr(d_streams);
                if post_mix {
                    l = l.arg_ptr(d_post).arg_ptr(d_comb);
                }
                l.arg_ptr(d_fn)
                    .arg_ptr(d_mix)
                    .arg_ptr(d_ss)
                    .arg_u32(T as u32)
                    .launch(0)
            };
            // Timing mutates the highway in place (post); the check runs once
            // from the pristine copy afterwards.
            g.copy_d2d(d_pristine, d_streams, streams.len())?;
            let t = time(g, &run)?;
            g.copy_d2d(d_pristine, d_streams, streams.len())?;
            run()?;
            g.synchronize(0)?;
            let mut got = fetch(g, d_mix, T * 24 * 4)?;
            got.extend(fetch(g, d_ss, T * 4)?);
            if post_mix {
                got.extend(fetch(g, d_streams, streams.len())?);
            }
            let verdict = match &want {
                None => {
                    want = Some(got);
                    "reference".to_string()
                }
                Some(w) => {
                    let diff = w.iter().zip(&got).filter(|(a, b)| a != b).count();
                    fail |= diff != 0;
                    if diff == 0 {
                        "bitwise".to_string()
                    } else {
                        format!("MISMATCH {diff} bytes")
                    }
                }
            };
            println!("{name:32} {tile:3} tokens/CTA {:7.1}us  {verdict}", t * 1e6);
        }
    }
    std::process::exit(if fail { 1 } else { 0 });
}
