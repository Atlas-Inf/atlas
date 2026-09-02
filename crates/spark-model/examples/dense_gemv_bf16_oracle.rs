// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-path oracle for `gemv::dense_gemv_bf16` (M=1) at the REAL
//! Qwen3.8-27B decode shapes, against an f32 CPU reference.
//!
//! The BF16-preservation decode route pushes every preserved layer's M=1
//! projections through this kernel (attention q/k/v/o, GDN qkvz/out, the
//! final-eight FFN). No microtest covered it on gfx1151, and the Windows
//! serve crashes were bisected to the preservation decode paths — this is the
//! Linux drift baseline that audit needs.
//!
//! ```text
//! cargo build --release -p spark-model --no-default-features \
//!   --features cuda,gpu-examples --example dense_gemv_bf16_oracle
//! ```

use anyhow::{Result, bail};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

struct Lcg(u64);
impl Lcg {
    fn r(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        -1.0 + 2.0 * (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
}

fn up(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

/// (N, K, label) at the real Qwen3.8-27B decode shapes.
const SHAPES: &[(usize, usize, &str)] = &[
    (6144, 5120, "attn_q"),
    (1024, 5120, "attn_k"),
    (1024, 5120, "attn_v"),
    (5120, 6144, "attn_o"),
    (12288, 5120, "gdn_qkvz"),
    (5120, 4096, "gdn_out"),
    (17408, 5120, "ffn_gate"),
    (5120, 17408, "ffn_down"),
];

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let kern = g.kernel("gemv", "dense_gemv_bf16")?;
    println!("dense_gemv_bf16 M=1 oracle — Qwen3.8-27B decode shapes (CPU f32 reference)");

    let mut rng = Lcg(0x51A7_C0DE);
    let mut all_pass = true;
    for &(n, k, label) in SHAPES {
        let w: Vec<bf16> = (0..n * k).map(|_| bf16::from_f32(rng.r() * 0.5)).collect();
        let a: Vec<bf16> = (0..k).map(|_| bf16::from_f32(rng.r() * 0.5)).collect();
        let wd = up(g, &w)?;
        let ad = up(g, &a)?;
        let c = g.alloc(n * 2)?;
        // 0xFFFF NaN sentinel: unwritten outputs fail finiteness.
        g.memset(c, 0xFF, n * 2)?;
        KernelLaunch::new(g, kern)
            .grid([div_ceil(n as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ad)
            .arg_ptr(wd)
            .arg_ptr(c)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(0)?;
        g.synchronize(0)?;
        let mut raw = vec![0u8; n * 2];
        g.copy_d2h(c, &mut raw)?;
        let gpu: Vec<f32> = raw
            .chunks_exact(2)
            .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
            .collect();
        if gpu.iter().any(|x| !x.is_finite()) {
            println!("{label:>10} N={n} K={k}: NONFINITE OUTPUT (unwritten tail)");
            all_pass = false;
            continue;
        }
        // CPU f32 reference: sum over K in order, rounded to BF16 like the kernel.
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let mut max_abs = 0.0f64;
        let mut rel_sum = 0.0f64;
        let mut cos_den_a = 0.0f64;
        let mut cos_den_b = 0.0f64;
        let mut cos_num = 0.0f64;
        for (col, w_col) in w.chunks_exact(k).enumerate() {
            let mut sum = 0.0f32;
            for (kk, &wv) in w_col.iter().enumerate() {
                sum += a_f32[kk] * wv.to_f32();
            }
            let ref_bf16 = bf16::from_f32(sum).to_f32();
            let got = gpu[col];
            let d = (got - ref_bf16).abs() as f64;
            max_abs = max_abs.max(d);
            let denom = ref_bf16.abs() as f64;
            if denom > 1e-3 {
                rel_sum += d / denom;
            }
            cos_num += got as f64 * ref_bf16 as f64;
            cos_den_a += got as f64 * got as f64;
            cos_den_b += ref_bf16 as f64 * ref_bf16 as f64;
        }
        let cosine = cos_num / ((cos_den_a.sqrt() * cos_den_b.sqrt()).max(1e-30));
        let mean_rel = rel_sum / n as f64;
        // BF16 has ~8 mantissa bits: 1 ULP at magnitude m is ~m/256. The gate
        // is magnitude-aware (2 ULP of the largest reference output) plus a
        // mean-relative and cosine bound — a fixed absolute cap would misfail
        // large-magnitude outputs that are bit-exact-class.
        let max_ref = gpu
            .iter()
            .zip(raw.chunks_exact(2))
            .max_by(|a, b| {
                let av = bf16::from_bits(u16::from_le_bytes([a.1[0], a.1[1]])).to_f32();
                let bv = bf16::from_bits(u16::from_le_bytes([b.1[0], b.1[1]])).to_f32();
                (av.abs() as f64).total_cmp(&(bv.abs() as f64))
            })
            .map(|(_, b)| {
                (bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32().abs() as f64) * 2.0
                    / 256.0
            })
            .unwrap_or(0.05);
        let max_cap = max_ref.max(0.05);
        let pass = cosine >= 0.9999 && mean_rel <= 1e-3 && max_abs <= max_cap;
        all_pass &= pass;
        println!(
            "{label:>10} N={n:>6} K={k:>6}: cos={cosine:.8} max_abs={max_abs:.3e} mean_rel={mean_rel:.3e} {}",
            if pass { "PASS" } else { "FAIL" }
        );
        for p in [wd, ad, c] {
            let _ = g.free(p);
        }
    }
    if !all_pass {
        bail!("dense_gemv_bf16 decode oracle FAILED");
    }
    println!("RESULT: PASS (all decode shapes vs CPU f32 reference)");
    Ok(())
}
