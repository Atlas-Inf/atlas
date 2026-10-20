// SPDX-License-Identifier: AGPL-3.0-only
//! End-to-end check of the qwen4_exp mHC PREFILL collapse through the real
//! Rust dispatch (`ops::hc_post_site` / `ops::hc_pre_site` /
//! `ops::hc_head_site`, and the seam `ops::qwen4exp_prefill_hc::
//! hc_post_pre_seam`), at the real site shape (hidden 2560, 4 streams, rank
//! 320) over several prompt lengths, slab tails included.
//!
//! Each length runs one GDN-layer shape: post(attn) + pre(ffn) as one seam,
//! then post(ffn) + pre(next attn) through the cross-layer deferral
//! (`ops::qwen4exp_prefill_seam`), then the head. Prints an FNV-1a hash of every
//! output (highway, `y`, `inj`) and the time per length; two runs, one with
//! `ATLAS_QWEN4EXP_PREFILL_HC=1` and one without, must print the SAME hashes.
//! Add `ATLAS_QWEN4EXP_PREFILL_HC_CHECK=100000` to the switched run and every
//! slab is also compared byte for byte in-process against the default arm.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo build -p spark-model --release \
//!     --features cuda,gpu-examples --example qwen4exp_hc_prefill_check
//!   ./qwen4exp_hc_prefill_check > off.txt
//!   ATLAS_QWEN4EXP_PREFILL_HC=1 ./qwen4exp_hc_prefill_check > on.txt
//!   diff <(grep hash off.txt) <(grep hash on.txt)

use anyhow::Result;
use spark_model::layers::ops;
use spark_model::layers::qwen3_attention::{HcLowRank, HcSiteWeights, HcWeights};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

const H: usize = 2560;
const HC: usize = 4;
const RANK: usize = 320;
const EPS: f32 = 1e-6;

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

fn bf16s(rng: &mut Lcg, n: usize, scale: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| (((rng.f() * scale).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn f32s(rng: &mut Lcg, n: usize, f: impl Fn(f32) -> f32) -> Vec<u8> {
    (0..n).flat_map(|_| f(rng.f()).to_le_bytes()).collect()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(256))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn fnv(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<u64> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    }))
}

fn site(g: &dyn GpuBackend, rng: &mut Lcg, inject: bool) -> Result<HcSiteWeights> {
    let wide = HC * H;
    Ok(HcSiteWeights {
        hc_fn: DevicePtr::NULL,
        hc_fn_bf16: DevicePtr::NULL,
        hc_base: DevicePtr::NULL,
        hc_scale: DevicePtr::NULL,
        lowrank: Some(HcLowRank {
            norm_w: up(g, &bf16s(rng, wide, 0.1))?,
            down_w: up(g, &bf16s(rng, RANK * wide, 0.02))?,
            up_w: up(g, &bf16s(rng, RANK * wide, 0.05))?,
            inject_w: if inject {
                up(g, &bf16s(rng, HC * wide, 0.01))?
            } else {
                DevicePtr::NULL
            },
            rank: RANK,
        }),
    })
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let mut rng = Lcg(0x5EA7);
    let (attn, ffn) = (site(g, &mut rng, true)?, site(g, &mut rng, true)?);
    let head = site(g, &mut rng, false)?;
    let hc = HcWeights {
        attn,
        ffn,
        head: None,
        hc_mult: HC,
        sinkhorn_iters: 0,
        hc_eps: 0.0,
        is_first_model_layer: false,
        is_last_model_layer: false,
    };
    let k_pre = g.kernel("hyper_connection", "hc_pre")?;
    let k_post = g.kernel("hyper_connection", "hc_post")?;
    let k_head = g.kernel("hyper_connection", "hc_head")?;
    let max_t = 16046usize;
    let lay = max_t.min(2048);
    let scratch_bytes = spark_runtime::buffers::hc_pre_scratch_layout(lay, HC * H, RANK, HC).total;
    let scratch = g.alloc(scratch_bytes)?;
    let comb = g.alloc(max_t * HC * HC * 4)?;
    println!(
        "ATLAS_QWEN4EXP_PREFILL_HC={}",
        std::env::var("ATLAS_QWEN4EXP_PREFILL_HC").unwrap_or_default()
    );
    for &t in &[9usize, 100, 256, 512, 1024, 2048, 2049, 5000, max_t] {
        let mut rng = Lcg(t as u64);
        let streams0 = f32s(&mut rng, t * HC * H, |x| x * 3.0);
        let inj0 = f32s(&mut rng, t * HC, |x| 1.0 + x);
        let (block_a, block_f) = (bf16s(&mut rng, t * H, 1.0), bf16s(&mut rng, t * H, 1.0));
        let d_s0 = up(g, &streams0)?;
        let d_streams = g.alloc(streams0.len())?;
        let d_inj = up(g, &inj0)?;
        let (d_ba, d_bf) = (up(g, &block_a)?, up(g, &block_f)?);
        let (d_y, d_y2) = (g.alloc(t * H * 2)?, g.alloc(t * H * 2)?);
        let (n, h) = (t as u32, H as u32);
        let run = || -> Result<()> {
            // post(attn out) + pre(ffn): the seam when it serves, else the pair.
            if !ops::qwen4exp_prefill_hc::hc_post_pre_seam(
                g, &hc, &hc.ffn, d_ba, d_streams, d_y, d_inj, scratch, n, h, EPS, 0,
            )? {
                ops::hc_post_site(
                    g, k_post, &hc, d_ba, d_streams, d_inj, comb, d_streams, n, h, 0,
                )?;
                ops::hc_pre_site(
                    g, k_pre, d_streams, &hc.ffn, &hc, d_y, d_inj, comb, scratch, n, h, EPS, 0,
                )?;
            }
            // post(ffn out) + pre(next attn): the cross-layer seam, deferred
            // and fused when the switch serves it, else as two calls.
            if !ops::qwen4exp_prefill_seam::defer_post(g, &hc, d_bf, n, h) {
                ops::hc_post_site(
                    g, k_post, &hc, d_bf, d_streams, d_inj, comb, d_streams, n, h, 0,
                )?;
            }
            if !ops::qwen4exp_prefill_seam::pre_with_pending(
                g, k_post, &hc, &hc.attn, d_streams, d_y, d_inj, scratch, n, h, EPS, 0,
            )? {
                ops::hc_pre_site(
                    g, k_pre, d_streams, &hc.attn, &hc, d_y, d_inj, comb, scratch, n, h, EPS, 0,
                )?;
            }
            let head_w = spark_model::layers::qwen3_attention::HcHeadWeights {
                hc_fn: DevicePtr::NULL,
                hc_base: DevicePtr::NULL,
                hc_scale: DevicePtr::NULL,
                lowrank: head.lowrank,
            };
            ops::hc_head_site(
                g, k_head, d_streams, &head_w, &hc, d_y2, scratch, n, h, EPS, 0,
            )
        };
        g.copy_d2d(d_s0, d_streams, streams0.len())?;
        g.copy_h2d(&inj0, d_inj)?;
        run()?;
        g.synchronize(0)?;
        println!(
            "T={t:5} hash highway {:016x} y {:016x} inj {:016x} head {:016x}",
            fnv(g, d_streams, streams0.len())?,
            fnv(g, d_y, t * H * 2)?,
            fnv(g, d_inj, t * HC * 4)?,
            fnv(g, d_y2, t * H * 2)?
        );
        let reps = if t >= 2048 { 10 } else { 50 };
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            run()?;
        }
        g.synchronize(0)?;
        println!(
            "T={t:5} time {:.3} ms per layer-shaped round (2 posts, 3 collapses)",
            t0.elapsed().as_secs_f64() * 1e3 / reps as f64
        );
        for p in [d_s0, d_streams, d_inj, d_ba, d_bf, d_y, d_y2] {
            g.free(p)?;
        }
    }
    Ok(())
}
