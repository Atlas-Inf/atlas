// SPDX-License-Identifier: AGPL-3.0-only
// Portions of this file are derived from ExLlamaV3 (MIT, see NOTICE.md).

//! GPU parity for the EXL3 linear path (`exl3_linear_bf16`, `exl3_hgemm`):
//! pinned to a relative norm against an f64 CPU reference. Helpers come from
//! the parent `gpu_tests` module.

use super::*;

/// `y = x · diag(suh) H W_inner H diag(svh)` for bf16 `x`: the CPU reference
/// accumulates `x · reconstruct(..)` in f64, the GPU runs the kernel path.
fn check_linear(g: &dyn GpuBackend, k: &Exl3Kernels, t: &Tensor) {
    let stream = g.default_stream();
    let (i, o) = (t.shape.in_features, t.shape.out_features);
    let w = Exl3Weight {
        trellis: upload(g, &u16_bytes(&t.trellis)),
        suh: upload(g, &f16_bytes(&t.suh)),
        svh: upload(g, &f16_bytes(&t.svh)),
        shape: t.shape,
    };
    let inner = upload(g, &vec![0u8; i * o * 2]);
    exl3_reconstruct(g, k, &w, inner, stream).unwrap();

    // Deterministic bf16 activations, uploaded and kept as f64 for the
    // reference (exact for bf16).
    let mut seed = 0x1234_ABCD_0000_0001u64 ^ ((i * 1_000_003 + o) as u64);
    let x_bits: Vec<u16> = (0..M * i).map(|_| bf16_uniform(&mut seed)).collect();
    let x_bf16 = upload(g, &u16_bytes(&x_bits));
    let x: Vec<f64> = x_bits.iter().copied().map(bf16_f64).collect();

    // CPU: W in f32 (the validated reference), the product in f64.
    let w_ref = exl3::reconstruct_ref(&t.trellis, &t.suh, &t.svh, &t.shape).unwrap();
    let mut y_ref = vec![0.0f64; M * o];
    for r in 0..M {
        for c in 0..o {
            let mut acc = 0.0f64;
            for kk in 0..i {
                acc += x[r * i + kk] * w_ref[kk * o + c] as f64;
            }
            y_ref[r * o + c] = acc;
        }
    }

    let xh = upload(g, &vec![0u8; M * i * 2]);
    let y = upload(g, &vec![0u8; M * o * 2]);
    let out = upload(g, &vec![0u8; M * o * 2]);
    exl3_linear_bf16(g, k, x_bf16, M as u32, &w, inner, xh, y, out, stream).unwrap();
    let got = download_u16(g, out, M * o);

    let (mut ngg, mut ngw) = (0.0f64, 0.0f64);
    let mut nw = 0.0f64;
    let mut worst = (0.0f64, 0usize);
    for (r, &got_bits) in got.iter().enumerate() {
        let v = bf16_f64(got_bits);
        ngg += v * v;
        ngw += v * y_ref[r];
        nw += y_ref[r] * y_ref[r];
        let d = (v - y_ref[r]).abs();
        if d > worst.0 {
            worst = (d, r);
        }
    }
    // ||y_gpu - y_ref|| / ||y_ref||, from the inner products above.
    let ratio = (ngg - 2.0 * ngw + nw).sqrt() / nw.sqrt();
    eprintln!(
        "{}: linear rel_err = {ratio:.3e} (worst |diff| {:.3e} at {})",
        t.name, worst.0, worst.1
    );
    assert!(
        ratio < 1e-2,
        "{}: relative error {ratio:.3e} >= 1e-2",
        t.name
    );
    for p in [w.trellis, w.suh, w.svh, inner, x_bf16, xh, y, out] {
        g.free(p).unwrap();
    }
}

#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn linear_matches_cpu_reference() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let k = Exl3Kernels::resolve(g).unwrap();
    for t in fixture_tensors().into_iter().chain(real_tensors()) {
        check_linear(g, &k, &t);
    }
}

/// The GEMM's own edges: m, n and k all non-multiples of the 16-wide tile, so
/// every guarded load and store in `exl3_hgemm_f16` is exercised.
#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn hgemm_matches_cpu_on_odd_shapes() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let k = Exl3Kernels::resolve(g).unwrap();
    let stream = g.default_stream();
    let (m, n, kd) = (5usize, 37usize, 19usize);

    let mut seed = 0xDEAD_BEEF_CAFE_F00Du64;
    let fp16_uniform = |seed: &mut u64| {
        let u = (mix64(seed) >> 16) as f32 / (1u64 << 48) as f32;
        f16::from_f32(u * 2.0 - 1.0)
    };
    let a: Vec<f16> = (0..m * kd).map(|_| fp16_uniform(&mut seed)).collect();
    let b: Vec<f16> = (0..kd * n).map(|_| fp16_uniform(&mut seed)).collect();
    let a_dev = upload(g, &f16_bytes(&a));
    let b_dev = upload(g, &f16_bytes(&b));
    let c_dev = upload(g, &vec![0u8; m * n * 2]);
    exl3_hgemm(
        g, &k, a_dev, b_dev, c_dev, m as u32, n as u32, kd as u32, stream,
    )
    .unwrap();
    let got = download_u16(g, c_dev, m * n);

    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..kd {
                acc += f64::from(a[r * kd + kk]) * f64::from(b[kk * n + c]);
            }
            max_ref = max_ref.max(acc.abs());
            max_abs = max_abs.max((f64::from(f16::from_bits(got[r * n + c])) - acc).abs());
        }
    }
    eprintln!("hgemm odd: max|diff| = {max_abs:.3e}, max|ref| = {max_ref:.3e}");
    assert!(
        max_abs <= 1e-3 * max_ref + 1e-3,
        "hgemm: max|diff| {max_abs:.3e} > 1e-3 * {max_ref:.3e} + 1e-3"
    );
    for p in [a_dev, b_dev, c_dev] {
        g.free(p).unwrap();
    }
}

/// The two conversions, pinned on raw bits so a wrong rounding mode cannot hide
/// inside a tolerance: bf16 -> fp16 rounds the widened f32 once, fp16 -> bf16
/// rounds the f32 value once (RNE, as `__float2half_rn` / `__float2bfloat16_rn`).
#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn conversions_match_cpu_bit_exact() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let k = Exl3Kernels::resolve(g).unwrap();
    let stream = g.default_stream();

    // Finite f32s of O(1) magnitude, as bf16 bits.
    let mut seed = 0x5EED_1234_9ABC_DEF0u64;
    let bf: Vec<u16> = (0..4096)
        .map(|_| {
            let v = ((mix64(&mut seed) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            (v.to_bits() >> 16) as u16
        })
        .collect();
    let bf_dev = upload(g, &u16_bytes(&bf));
    let h_dev = upload(g, &vec![0u8; bf.len() * 2]);
    exl3_convert(g, k.bf16_to_f16, bf_dev, h_dev, bf.len() as u32, stream).unwrap();
    let got = download_u16(g, h_dev, bf.len());
    let want: Vec<u16> = bf
        .iter()
        .map(|&b| f16::from_f32(bf16_f64(b) as f32).to_bits())
        .collect();
    let bad: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
    assert!(
        bad.is_empty(),
        "bf16_to_f16: {}/{} bits differ, first 5 (idx, gpu, cpu): {:?}",
        bad.len(),
        got.len(),
        bad.iter()
            .take(5)
            .map(|&i| (i, got[i], want[i]))
            .collect::<Vec<_>>()
    );

    // fp16 -> bf16 of the same values: one rounding via f32.
    let back = upload(g, &vec![0u8; got.len() * 2]);
    exl3_convert(g, k.f16_to_bf16, h_dev, back, got.len() as u32, stream).unwrap();
    let got2 = download_u16(g, back, got.len());
    let want2: Vec<u16> = got
        .iter()
        .map(|&h| (f16::from_bits(h).to_f32().to_bits() >> 16) as u16)
        .collect();
    let bad2: Vec<usize> = (0..got2.len()).filter(|&i| got2[i] != want2[i]).collect();
    assert!(
        bad2.is_empty(),
        "f16_to_bf16: {}/{} bits differ, first 5 (idx, gpu, cpu): {:?}",
        bad2.len(),
        got2.len(),
        bad2.iter()
            .take(5)
            .map(|&i| (i, got2[i], want2[i]))
            .collect::<Vec<_>>()
    );
    for p in [bf_dev, h_dev, back] {
        g.free(p).unwrap();
    }
}

/// One plain Hadamard block against the CPU reference's butterfly, so a wrong
/// butterfly order or scale cannot hide inside the linear's tolerance. The
/// device folds the 1/sqrt(128) into seven halvings, so the fp16 results agree
/// only up to a few ulps — the count of differing bits is what is pinned (a
/// mis-ordered butterfly differs everywhere).
#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn had_plain_matches_cpu_128_block() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let k = Exl3Kernels::resolve(g).unwrap();
    let stream = g.default_stream();

    let mut seed = 0x0DADEAD1_BEEF0001u64;
    let x: Vec<f16> = (0..128)
        .map(|_| f16::from_f32(((mix64(&mut seed) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0))
        .collect();
    let dev = upload(g, &f16_bytes(&x));
    exl3_had_r128(g, k.had_plain, dev, dev, DevicePtr::NULL, 1, 128, stream).unwrap();
    let got = download_u16(g, dev, 128);

    let mut v: Vec<f32> = x.iter().map(|&h| h.to_f32()).collect();
    let mut len = 1usize;
    while len < 128 {
        for base in (0..128).step_by(len * 2) {
            for i in 0..len {
                let (a, b) = (v[base + i], v[base + len + i]);
                v[base + i] = a + b;
                v[base + len + i] = a - b;
            }
        }
        len *= 2;
    }
    for e in v.iter_mut() {
        *e *= exl3::HAD_SCALE;
    }
    let want: Vec<u16> = v.iter().map(|&e| f16::from_f32(e).to_bits()).collect();
    let bad: Vec<usize> = (0..128).filter(|&i| got[i] != want[i]).collect();
    let max_ulp = bad
        .iter()
        .map(|&i| {
            let (a, b) = (
                f16::from_bits(got[i]).to_f32(),
                f16::from_bits(want[i]).to_f32(),
            );
            ((a - b).abs() / b.abs().max(1e-30)) as f64
        })
        .fold(0.0f64, f64::max);
    eprintln!(
        "had_plain: {}/128 bits differ, max rel dev {max_ulp:.3e}",
        bad.len()
    );
    assert!(
        bad.len() <= 16 && max_ulp < 1e-2,
        "had_plain: {} of 128 bits differ (rel dev {max_ulp:.3e}) — a mis-ordered \
         butterfly or a wrong scale differs everywhere",
        bad.len()
    );
    g.free(dev).unwrap();
}
