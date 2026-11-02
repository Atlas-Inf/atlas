// SPDX-License-Identifier: AGPL-3.0-only
//! Native numerical/layout gate for token-interleaved BF16 head GEMMs.
#![cfg(feature = "cuda")]
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::{
    cuda_backend::AtlasCudaBackend,
    gpu::{DevicePtr, GpuBackend},
};
const CANARY: u16 = 0x7fc1;
fn upload(gpu: &dyn GpuBackend, values: &[u16]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(values.len() * 2)?;
    gpu.copy_h2d(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        ptr,
    )?;
    Ok(ptr)
}
fn av(t: usize, h: usize, k: usize) -> f32 {
    bf16::from_f32(((t * 3 + h * 5 + k * 7) % 17) as f32 / 13.0 - 0.5).to_f32()
}
fn bv(h: usize, n: usize, k: usize) -> f32 {
    bf16::from_f32(((h * 11 + n * 13 + k * 3) % 19) as f32 / 23.0 - 0.4).to_f32()
}
fn check(gpu: &dyn GpuBackend, m: usize, g: usize, n: usize, k: usize, pad: usize) -> Result<()> {
    let stream = gpu.default_stream();
    let astride = g * k + pad;
    let cstride = g * n + pad;
    let mut a = vec![CANARY; m * astride + 64];
    let mut b = vec![CANARY; g * n * k];
    for t in 0..m {
        for h in 0..g {
            for x in 0..k {
                a[t * astride + h * k + x] = bf16::from_f32(av(t, h, x)).to_bits();
            }
        }
    }
    for h in 0..g {
        for y in 0..n {
            for x in 0..k {
                b[(h * n + y) * k + x] = bf16::from_f32(bv(h, y, x)).to_bits();
            }
        }
    }
    let act = upload(gpu, &a)?;
    let weight = upload(gpu, &b)?;
    let out = upload(gpu, &vec![CANARY; m * cstride + 128])?;
    let output = out.offset(128);
    // Fill both head and row padding with NaN so wrong batch/leading strides
    // cannot accidentally satisfy a zero-filled-output oracle.
    spark_runtime::cublaslt::bf16_grouped_gemm_act_weight_t(
        act.0,
        weight.0,
        output.0,
        m as u32,
        g as u32,
        n as u32,
        k as u32,
        astride as u32,
        cstride as u32,
        stream,
    )?;
    let mut bytes = vec![0; (m * cstride + 128) * 2];
    gpu.copy_d2h_on_stream(out, &mut bytes, stream)?;
    let got: Vec<_> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    // Periodic inputs permit an independent full-output FP32 CPU reference
    // without billions of repeated dot products in the larger prefill shapes.
    let mut reference = vec![0f32; 17 * g * 19];
    for t in 0..17 {
        for h in 0..g {
            for y in 0..19 {
                let mut sum = 0f32;
                for x in 0..k {
                    sum += av(t, h, x) * bv(h, y, x);
                }
                reference[(t * g + h) * 19 + y] = sum;
            }
        }
    }
    let mut max_error = 0f32;
    for t in 0..m {
        for h in 0..g {
            for y in 0..n {
                let expected = bf16::from_f32(reference[((t % 17) * g + h) * 19 + y % 19]).to_f32();
                let actual = bf16::from_bits(got[64 + t * cstride + h * n + y]).to_f32();
                let ulp = if expected == 0.0 {
                    f32::MIN_POSITIVE
                } else {
                    2f32.powi(expected.abs().log2().floor() as i32 - 7)
                };
                let error = (actual - expected).abs();
                max_error = max_error.max(error);
                ensure!(
                    actual.is_finite() && error <= 2.0 * ulp + 1e-4,
                    "CPU mismatch M={m} G={g} N={n} K={k} t={t} h={h} y={y}: {actual} vs{expected}"
                );
            }
        }
    }
    for t in 0..m {
        ensure!(
            got[64 + t * cstride + g * n..64 + (t + 1) * cstride]
                .iter()
                .all(|&b| b == CANARY),
            "output row padding overwritten"
        );
    }
    ensure!(
        got[..64]
            .iter()
            .chain(got[64 + m * cstride..].iter())
            .all(|&b| b == CANARY),
        "output redzone overwritten"
    );
    if m <= 17 {
        let oracle = upload(gpu, &vec![CANARY; m * cstride])?;
        ops::grouped_gemm_mla(
            gpu,
            gpu.kernel("grouped_gemm_mla", "grouped_gemm_mla")?,
            act,
            weight,
            oracle,
            m as u32,
            g as u32,
            k as u32,
            n as u32,
            astride as u32,
            cstride as u32,
            stream,
        )?;
        let mut raw = vec![0; m * cstride * 2];
        gpu.copy_d2h_on_stream(oracle, &mut raw, stream)?;
        for t in 0..m {
            for i in 0..g * n {
                let old = bf16::from_bits(u16::from_le_bytes(
                    raw[(t * cstride + i) * 2..(t * cstride + i) * 2 + 2]
                        .try_into()
                        .unwrap(),
                ))
                .to_f32();
                let new = bf16::from_bits(got[64 + t * cstride + i]).to_f32();
                let ulp = if old == 0.0 {
                    f32::MIN_POSITIVE
                } else {
                    2f32.powi(old.abs().log2().floor() as i32 - 7)
                };
                ensure!(
                    (old - new).abs() <= 2.0 * ulp + 1e-4,
                    "scalar MLA mismatch t={t} i={i}: {new} vs{old}"
                );
            }
        }
        gpu.free(oracle)?;
    }
    println!("PASS M={m} G={g} N={n} K={k} pad={pad} max_error={max_error}");
    for ptr in [act, weight, out] {
        gpu.free(ptr)?;
    }
    Ok(())
}
#[test]
#[ignore = "requires available GB10 GPU, cuBLASLt, and compiled MLA kernel; no checkpoint"]
fn bf16_head_batches_match_cpu_and_scalar_mla() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    for (m, g, n, k, pad) in [
        (3, 2, 16, 16, 16),
        (17, 32, 512, 256, 32),
        (17, 32, 256, 512, 32),
        (1024, 32, 512, 256, 0),
        (2048, 32, 256, 512, 0),
    ] {
        check(&gpu, m, g, n, k, pad)?;
    }
    Ok(())
}
