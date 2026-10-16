// SPDX-License-Identifier: AGPL-3.0-only

//! GPU bit-exactness test: `w8a16_gemv` (serial decode, M=1) vs the batched
//! verify GEMVs `w8a16_gemv_batch4` and `w8a16_gemv_batch16` — the pair the
//! multi-seq/verify path dispatches on GB10 (batch8/vl2/n2 are Strix-only).
//!
//! Native-FP8 attention (ATLAS_NEMOTRON_NATIVE_FP8_ATTN) feeds this family
//! from `QuantWeight::Fp8`, so spec-off == verify needs the batched rows to
//! be byte-identical to the serial pick the scheduler compares drafts
//! against. Shapes match the omni attention projections: q/o [4096, 4096],
//! k/v [256, 4096].

use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

fn gpu() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    spark_runtime::cuda_backend::AtlasCudaBackend::new(
        0,
        &atlas_kernels::ptx_for_exact_target("nemotron-3.5-super-vl-120b-a12b", "nvfp4")
            .expect("omni target PTX — build with ATLAS_TARGET_MODEL='*'")
            .modules,
    )
    .expect("CUDA backend")
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len().max(1)).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v).unwrap();
    v
}

/// Deterministic pseudo-random byte stream (SplitMix-ish LCG).
fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u8
        })
        .collect()
}

/// Deterministic bf16 activations in [-1, 1).
fn bf16_seq(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let f = ((x >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;
        out.extend_from_slice(&bf16::from_f64(f).to_le_bytes());
    }
    out
}

// Launchers delegate to the production ops — `ops::w8a16_gemv` takes
// (input, weight, scale, output, n, k) — no m arg; the batch twins take
// (input, weight, block_scale, output, m, n, k[, div]). The previous
// hand-rolled launch passed a stray m first arg into the serial call and
// read garbage n/k, which is exactly the class of error this test exists
// to catch.
use crate::layers::ops::{w8a16_gemv, w8a16_gemv_batch4_div};

fn parity_one_shape(n: u32, k: u32, m: u32) {
    let g = gpu();
    let stream = g.default_stream();
    let single = g.kernel("w8a16_gemv", "w8a16_gemv").unwrap();
    let batch4 = g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4").unwrap();
    let batch16 = g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16").unwrap();

    // FP8 E4M3 weights — nonzero, finite encodings only (mask NaN 0x7F/0xFF).
    let mut w = lcg_bytes((n * k) as usize, 0x9e37 + n as u64);
    for b in w.iter_mut() {
        if *b == 0x7f || *b == 0xff {
            *b = 0x3f;
        }
    }
    // FP32 block scales [N/128, K/128], small positive deterministic values.
    let n_blocks = (n as usize).div_ceil(128);
    let k_blocks = (k as usize).div_ceil(128);
    let mut scale = Vec::with_capacity(n_blocks * k_blocks * 4);
    for i in 0..(n_blocks * k_blocks) {
        let v = 0.25f32 + (i % 7) as f32 * 0.125;
        scale.extend_from_slice(&v.to_le_bytes());
    }
    let x = bf16_seq((m as usize) * (k as usize), 0x5eed + n as u64);

    let input = upload(&g, &x);
    let weight = upload(&g, &w);
    let scale_p = upload(&g, &scale);
    let out_single = g.alloc((n * 2) as usize * m as usize).unwrap();
    let out_batch = g.alloc((n * 2) as usize * m as usize).unwrap();

    // Serial reference: one w8a16_gemv per row (production M=1 launcher).
    for r in 0..m as usize {
        w8a16_gemv(
            &g,
            single,
            input.offset(r * (k as usize) * 2),
            weight,
            scale_p,
            out_single.offset(r * (n as usize) * 2),
            n,
            k,
            stream,
        )
        .unwrap();
    }
    g.synchronize(stream).unwrap();

    // Batched: batch4 (verify M<=4) and batch16 (verify M<=16 on GB10).
    // batch4 covers M<=4; batch16 covers M<=16 on GB10 (Strix batch8 arms
    // are cfg(atlas_hip)-gated). Test each only inside its own row capacity.
    let mut cases: Vec<(spark_runtime::gpu::KernelHandle, u32, &str)> = Vec::new();
    if m <= 4 {
        cases.push((batch4, 4, "batch4"));
    }
    cases.push((batch16, 4, "batch16"));
    let cases = cases.as_slice();
    for (kh, div, name) in cases {
        w8a16_gemv_batch4_div(
            &g, *kh, input, weight, scale_p, out_batch, m, n, k, *div, stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let a = download(&g, out_single, (n * 2) as usize * m as usize);
        let b = download(&g, out_batch, (n * 2) as usize * m as usize);
        assert_eq!(
            a, b,
            "{name} row-outputs differ from serial w8a16_gemv at N={n} K={k} M={m}"
        );
    }
    g.free(input).unwrap();
    g.free(weight).unwrap();
    g.free(scale_p).unwrap();
    g.free(out_single).unwrap();
    g.free(out_batch).unwrap();
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn w8a16_gemv_batch_equals_serial_on_attention_shapes() {
    // q/o: N=4096 K=4096; k/v: N=256 K=4096 — M=9 covers the batch16 range.
    for m in [4u32, 9] {
        parity_one_shape(4096, 4096, m);
        parity_one_shape(256, 4096, m);
    }
}
