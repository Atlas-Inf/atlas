// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-path oracle for `gated_delta_rule::gated_delta_rule_decode` —
//! single-step state+output comparison vs the CPU recurrent SSOT, at the
//! Qwen3.8 GDN dims. The Windows durability bisection exonerated the BF16
//! decode GEMV; this covers the GDN decode recurrence.
//!
//! The model-override kernel (qwen3.6-27b/nvfp4, reused by 3.8 via
//! kernel_source) takes FP32 q/k/v — the f32 conv output feeds it directly
//! in production (trait_decode_batched_conv_gdn.rs).

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const HR: usize = NV / NK;

struct Lcg(u64);
impl Lcg {
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + (hi - lo) * (((self.0 >> 11) as f64) / ((1u64 << 53) as f64))
    }
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kern = g.kernel("gated_delta_rule", "gated_delta_rule_decode")?;
    println!("gdn_decode oracle — single step, NK={NK} KD={KD} NV={NV} VD={VD}");

    let mut rng = Lcg(0x51A7_C0DE);

    // Single-step inputs.
    let q: Vec<f32> = (0..NK * KD).map(|_| rng.r(-0.05, 0.05) as f32).collect();
    let kw: Vec<f32> = (0..NK * KD).map(|_| rng.r(-0.05, 0.05) as f32).collect();
    let vw: Vec<f32> = (0..NV * VD).map(|_| rng.r(-0.05, 0.05) as f32).collect();
    let gate: Vec<f32> = (0..NV).map(|_| rng.r(0.80, 0.99) as f32).collect();
    let beta: Vec<f32> = (0..NV).map(|_| rng.r(0.0, 1.0) as f32).collect();
    let h0: Vec<f32> = (0..NV * KD * VD).map(|_| rng.r(-0.01, 0.01) as f32).collect();

    // CPU SSOT: one step.
    let mut s: Vec<f64> = h0.iter().map(|&x| x as f64).collect();
    let mut o_ref = vec![0.0f32; NV * VD];
    let scale = (KD as f64).powf(-0.5);
    for vh in 0..NV {
        let kh = vh / HR;
        let gg = gate[vh] as f64;
        let bt = beta[vh] as f64;
        for v in 0..VD {
            let mut hk = 0.0f64;
            for k in 0..KD {
                hk += s[(vh * KD + k) * VD + v] * kw[kh * KD + k] as f64;
            }
            let v_new = (vw[vh * VD + v] as f64 - gg * hk) * bt;
            let mut qd = 0.0f64;
            for k in 0..KD {
                let idx = (vh * KD + k) * VD + v;
                let hn = gg * s[idx] + kw[kh * KD + k] as f64 * v_new;
                s[idx] = hn;
                qd += hn * q[kh * KD + k] as f64;
            }
            o_ref[vh * VD + v] = (qd * scale) as f32;
        }
    }

    // GPU: one step.
    let qp = up_f32(g, &q)?;
    let kp = up_f32(g, &kw)?;
    let vp = up_f32(g, &vw)?;
    let gp = up_f32(g, &gate)?;
    let bp = up_f32(g, &beta)?;
    let hp = up_f32(g, &h0)?;
    let op = g.alloc(NV * VD * 2)?;
    KernelLaunch::new(g, kern)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(hp)
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(gp)
        .arg_ptr(bp)
        .arg_ptr(op)
        .arg_u32(1)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .launch(0)?;
    g.synchronize(0)?;

    let gpu_h = dn_f32(g, hp, NV * KD * VD)?;

    // Output is BF16.
    let mut o_raw = vec![0u8; NV * VD * 2];
    g.copy_d2h(op, &mut o_raw)?;
    let gpu_o: Vec<f32> = o_raw
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let mut max_o = 0.0f64;
    for (a, b) in gpu_o.iter().zip(o_ref.iter()) {
        max_o = max_o.max((*a as f64 - *b as f64).abs());
    }
    let mut max_h = 0.0f64;
    let mut worst = (0usize, 0usize, 0usize);
    for vh in 0..NV {
        for k in 0..KD {
            for v in 0..VD {
                let d = (gpu_h[(vh * KD + k) * VD + v] as f64 - s[(vh * KD + k) * VD + v]).abs();
                if d > max_h {
                    max_h = d;
                    worst = (vh, k, v);
                }
            }
        }
    }
    println!("single step: output max|Δ|={max_o:.6}  state max|Δ|={max_h:.6} at vh/k/v={worst:?}");
    let pass = max_h <= 1e-4 && max_o <= 0.01;
    println!("{}", if pass { "RESULT: PASS" } else { "RESULT: FAIL" });
    if !pass {
        bail!("gdn_decode oracle FAILED");
    }
    Ok(())
}

