// SPDX-License-Identifier: AGPL-3.0-only

//! GPU parity + timing for the native batch-1 EXL3 GEMV (`exl3_dense`).
//!
//! Per tensor (fixtures + ATLAS_EXL3_TEST_DATA):
//! - core: `exl3_gemv_raw_k<K>` on the GPU-transformed fp16 `xh` against
//!   `xh · decode_inner(trellis)` accumulated in f64 (same fp16 operands) —
//!   pins the trellis decode + MMA + reduction.
//! - input transform: the bf16 kernel's in-block `had_r128(x * suh)` must give
//!   BIT-identical fp32 `y_inner` to the raw kernel fed `exl3_had_r128_pre`.
//! - end to end: native bf16 out vs the BF16-materialized path
//!   (`exl3_dense_bf16_nk` + `dense_gemv_bf16`, what decode runs today) and
//!   both vs the f64 CPU reference `x · reconstruct(..)`.
//!
//! Timing (`exl3_gemv_timing`): the decode shapes of CYBER-FROST, native
//! (random K8 trellis) vs `dense_gemv_bf16` (random BF16).

use super::*;
use crate::layers::ops::{Exl3GemvKernels, exl3_gemv_bf16, exl3_gemv_inner};

fn download_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// (max |a - b|, ||a - b|| / ||b||)
fn diff(a: &[f64], b: &[f64]) -> (f64, f64) {
    let (mut mx, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        mx = mx.max((x - y).abs());
        num += (x - y) * (x - y);
        den += y * y;
    }
    (mx, (num / den.max(1e-300)).sqrt())
}

fn check_gemv(
    g: &dyn GpuBackend,
    k: &Exl3Kernels,
    kg: &Exl3GemvKernels,
    t: &Tensor,
) -> (f64, f64, f64) {
    let stream = g.default_stream();
    let (i, o) = (t.shape.in_features, t.shape.out_features);
    let w = Exl3Weight {
        trellis: upload(g, &u16_bytes(&t.trellis)),
        suh: upload(g, &f16_bytes(&t.suh)),
        svh: upload(g, &f16_bytes(&t.svh)),
        shape: t.shape,
    };
    let mut seed = 0x9E37_0001_ABCD_0000u64 ^ ((i * 7919 + o) as u64);
    let x_bits: Vec<u16> = (0..i)
        .map(|_| {
            let u = (mix64(&mut seed) >> 16) as f64 / (1u64 << 48) as f64;
            ((((u * 2.0 - 1.0) as f32).to_bits()) >> 16) as u16
        })
        .collect();
    let x_dev = upload(g, &u16_bytes(&x_bits));
    let x: Vec<f64> = x_bits.iter().copied().map(bf16_f64).collect();

    // xh on the GPU with the existing linear-path kernels.
    let xh = upload(g, &vec![0u8; i * 2]);
    exl3_convert(g, k.bf16_to_f16, x_dev, xh, i as u32, stream).unwrap();
    exl3_had_r128(g, k.had_pre, xh, xh, w.suh, 1, i as u32, stream).unwrap();
    let xh_h: Vec<f64> = download_u16(g, xh, i)
        .into_iter()
        .map(|b| f16::from_bits(b).to_f64())
        .collect();

    // Core: raw kernel vs f64 over the same fp16 operands.
    let y_raw = upload(g, &vec![0u8; o * 4]);
    exl3_gemv_inner(g, kg, true, xh, &w, y_raw, stream).unwrap();
    let y_raw_h = download_f32(g, y_raw, o);
    let inner = exl3::decode_inner(&t.trellis, &t.shape).unwrap();
    let mut y_core = vec![0.0f64; o];
    for (kk, xv) in xh_h.iter().enumerate() {
        let row = &inner[kk * o..(kk + 1) * o];
        for (c, wv) in row.iter().enumerate() {
            y_core[c] += xv * wv.to_f64();
        }
    }
    let y_raw64: Vec<f64> = y_raw_h.iter().map(|&v| v as f64).collect();
    let (core_max, core_rel) = diff(&y_raw64, &y_core);

    // Input transform inside the bf16 kernel: bit-identical y_inner.
    let y_bf = upload(g, &vec![0u8; o * 4]);
    exl3_gemv_inner(g, kg, false, x_dev, &w, y_bf, stream).unwrap();
    let y_bf_h = download_f32(g, y_bf, o);
    let bits_diff = y_bf_h
        .iter()
        .zip(&y_raw_h)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();

    // End to end, native.
    let out_nat = upload(g, &vec![0u8; o * 2]);
    exl3_gemv_bf16(g, kg, x_dev, &w, y_bf, out_nat, o, stream).unwrap();
    let nat: Vec<f64> = download_u16(g, out_nat, o)
        .into_iter()
        .map(bf16_f64)
        .collect();

    // End to end, BF16 materialized + dense_gemv_bf16 (today's decode).
    let wt = upload(g, &vec![0u8; i * o * 2]);
    exl3_dense_bf16_nk(g, k, &w, wt, stream).unwrap();
    let out_bf = upload(g, &vec![0u8; o * 2]);
    let gemv_k = g.kernel("gemv", "dense_gemv_bf16").unwrap();
    crate::layers::ops::dense_gemv(
        g,
        gemv_k,
        x_dev,
        &crate::weight_map::DenseWeight { weight: wt },
        out_bf,
        o as u32,
        i as u32,
        stream,
    )
    .unwrap();
    let bf: Vec<f64> = download_u16(g, out_bf, o)
        .into_iter()
        .map(bf16_f64)
        .collect();

    // f64 CPU reference on the exact W.
    let w_ref = exl3::reconstruct_ref(&t.trellis, &t.suh, &t.svh, &t.shape).unwrap();
    let mut y_ref = vec![0.0f64; o];
    for (kk, xv) in x.iter().enumerate() {
        let row = &w_ref[kk * o..(kk + 1) * o];
        for (c, wv) in row.iter().enumerate() {
            y_ref[c] += xv * *wv as f64;
        }
    }
    let (nr_max, nr_rel) = diff(&nat, &y_ref);
    let (br_max, br_rel) = diff(&bf, &y_ref);
    let (nb_max, nb_rel) = diff(&nat, &bf);
    let ymax = y_ref.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    eprintln!(
        "GEMV {} K{} {}x{}: core max|d| {core_max:.3e} rel {core_rel:.3e} | in-xform bitdiff {bits_diff}/{o} \
         | native-vs-f64 max {nr_max:.3e} rel {nr_rel:.3e} | bf16path-vs-f64 max {br_max:.3e} rel {br_rel:.3e} \
         | native-vs-bf16path max {nb_max:.3e} rel {nb_rel:.3e} | max|y| {ymax:.3e}",
        t.name, t.shape.bits, i, o
    );
    assert!(
        core_rel < 1e-5,
        "{}: GEMV core rel {core_rel:.3e} >= 1e-5",
        t.name
    );
    assert!(nr_rel < 5e-3, "{}: native rel {nr_rel:.3e} >= 5e-3", t.name);
    for p in [
        w.trellis, w.suh, w.svh, x_dev, xh, y_raw, y_bf, out_nat, wt, out_bf,
    ] {
        g.free(p).unwrap();
    }
    (nr_rel, br_rel, nb_rel)
}

#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn exl3_gemv_matches_cpu_and_bf16_path() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let k = Exl3Kernels::resolve(g).unwrap();
    let kg = Exl3GemvKernels::resolve(g).unwrap();
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    let mut n = 0usize;
    for t in fixture_tensors().into_iter().chain(real_tensors()) {
        if !(t.shape.in_features.is_multiple_of(128)
            && t.shape.out_features.is_multiple_of(128)
            && (2..=8).contains(&t.shape.bits))
        {
            eprintln!("GEMV {}: shape/bits outside the kernel, skipped", t.name);
            continue;
        }
        let r = check_gemv(g, &k, &kg, &t);
        worst = (worst.0.max(r.0), worst.1.max(r.1), worst.2.max(r.2));
        n += 1;
    }
    eprintln!(
        "GEMV summary over {n} tensors: worst rel native-vs-f64 {:.3e}, bf16path-vs-f64 {:.3e}, native-vs-bf16path {:.3e}",
        worst.0, worst.1, worst.2
    );
}

/// Decode-shape timing: native K8 GEMV vs dense_gemv_bf16 (random data;
/// timing only). Host wall time over `iters` launches on one stream.
#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn exl3_gemv_timing() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let kg = Exl3GemvKernels::resolve(g).unwrap();
    let gemv_k = g.kernel("gemv", "dense_gemv_bf16").unwrap();
    let stream = g.default_stream();
    let bits = 8u32;
    // (name, in, out, count per token)
    let shapes: [(&str, usize, usize, usize); 7] = [
        ("gdn.in_proj_qkv", 2560, 10240, 36),
        ("gdn.in_proj_z", 2560, 6144, 36),
        ("gdn.out_proj", 6144, 2560, 36),
        ("attn.q_proj", 2560, 12288, 12),
        ("attn.k/v_proj", 2560, 512, 24),
        ("attn.o_proj", 6144, 2560, 12),
        ("lm_head", 2560, 248320, 1),
    ];
    let (mut tot_nat, mut tot_bf) = (0.0f64, 0.0f64);
    for (name, i, o, per_tok) in shapes {
        let mut seed = 0xABCDu64 ^ (i * o) as u64;
        let tw = i / 16 * (o / 16) * 16 * bits as usize;
        let mut rnd = |n: usize| -> Vec<u8> {
            let mut v = Vec::with_capacity(n);
            while v.len() < n {
                v.extend_from_slice(&mix64(&mut seed).to_le_bytes());
            }
            v.truncate(n);
            v
        };
        let trellis = upload(g, &rnd(tw * 2));
        let ones = |n: usize| upload(g, &f16_bytes(&vec![f16::from_f32(1.0); n]));
        let w = Exl3Weight {
            trellis,
            suh: ones(i),
            svh: ones(o),
            shape: Exl3Shape {
                in_features: i,
                out_features: o,
                bits,
            },
        };
        let xb: Vec<u16> = (0..i).map(|_| bf16_uniform(&mut seed) & 0x3fff).collect();
        let x = upload(g, &u16_bytes(&xb));
        let y = upload(g, &vec![0u8; o * 4]);
        let out = upload(g, &vec![0u8; o * 2]);
        // BF16 weight: small random bf16 values.
        let wb: Vec<u16> = (0..i * o)
            .map(|j| 0x3c00u16 ^ ((j as u16).wrapping_mul(40503) & 0x00ff))
            .collect();
        let wbf = upload(g, &u16_bytes(&wb));
        let dw = crate::weight_map::DenseWeight { weight: wbf };
        let iters = if o > 100_000 { 50 } else { 400 };
        let time = |f: &dyn Fn()| -> f64 {
            for _ in 0..10 {
                f();
            }
            g.synchronize(stream).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            g.synchronize(stream).unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / iters as f64
        };
        let nat = time(&|| exl3_gemv_bf16(g, &kg, x, &w, y, out, o, stream).unwrap());
        let bfu = time(&|| {
            crate::layers::ops::dense_gemv(g, gemv_k, x, &dw, out, o as u32, i as u32, stream)
                .unwrap()
        });
        let nat_gbs = (i * o) as f64 * bits as f64 / 8.0 / nat / 1e3;
        let bf_gbs = (i * o) as f64 * 2.0 / bfu / 1e3;
        eprintln!(
            "TIMING {name:16} {i}x{o}: native K8 {nat:8.1} us ({nat_gbs:5.1} GB/s) | bf16 {bfu:8.1} us ({bf_gbs:5.1} GB/s) | x{per_tok}/token"
        );
        tot_nat += nat * per_tok as f64;
        tot_bf += bfu * per_tok as f64;
        for p in [w.trellis, w.suh, w.svh, x, y, out, wbf] {
            g.free(p).unwrap();
        }
    }
    eprintln!(
        "TIMING per-token dense sum: native {:.2} ms vs bf16 {:.2} ms",
        tot_nat / 1e3,
        tot_bf / 1e3
    );
}
