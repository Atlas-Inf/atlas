// SPDX-License-Identifier: AGPL-3.0-only

//! E2 `dense_gemv_bf16` pass-pair fusion (strix-hip shadow) — GPU tests.
//! f64 ref at the real decode shapes, batchm rows bitwise == M=1,
//! flushed microbench (rotating weight copies).
//! Run: `cargo test --release -p spark-model dense_tiled -- --test-threads 1`

#![cfg(all(test, atlas_hip))]

use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

fn gpu() -> AtlasCudaBackend {
    AtlasCudaBackend::new(
        0,
        &atlas_kernels::ptx_for_exact_target("nemotron-3.5-super-vl-120b-a12b", "nvfp4")
            .expect("n35vl target PTX")
            .modules,
    )
    .expect("CUDA backend")
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len().max(1)).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn download_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u16> {
    let mut v = vec![0u8; n * 2];
    g.copy_d2h(p, &mut v).unwrap();
    v.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn bf16_seq(n: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            half::bf16::from_f32(u * scale).to_bits()
        })
        .collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn ref_row(a: &[u16], b: &[u16], m: usize, n: usize, k: usize) -> f64 {
    let mut acc = 0.0f64;
    for kk in 0..k {
        acc += half::bf16::from_bits(a[m * k + kk]).to_f32() as f64
            * half::bf16::from_bits(b[n * k + kk]).to_f32() as f64;
    }
    acc
}

#[allow(non_snake_case)]
fn run_case(n: usize, k: usize, m: usize, tag: &str) {
    let g = gpu();
    let kk = g
        .kernel("gemv", "dense_gemv_bf16")
        .expect("dense_gemv_bf16 resolves");
    let ko = g
        .kernel("dense_gemv_bf16_legacy", "dense_gemv_bf16_legacy")
        .expect("legacy resolves");
    let kb = g
        .kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
        .expect("batchm resolves");
    let a = bf16_seq(m * k, 0xBEEF, 0.5);
    let b = bf16_seq(n * k, 0xF00D, 0.5);
    let a_p = upload(&g, &u16_bytes(&a));
    let b_p = upload(&g, &u16_bytes(&b));

    // M=1 vs f64
    let c1 = g.alloc(n * 2).unwrap();
    ops::dense_gemv(
        &g,
        kk,
        a_p,
        &DenseWeight { weight: b_p },
        c1,
        n as u32,
        k as u32,
        0,
    )
    .unwrap();
    g.synchronize(0).unwrap();
    let out = download_u16(&g, c1, n);
    let mut max_rel = 0.0f64;
    for nn in 0..n {
        let got = half::bf16::from_bits(out[nn]).to_f32() as f64;
        let want = ref_row(&a, &b, 0, nn, k);
        let rel = if want.abs() < 1e-6 {
            got.abs()
        } else {
            (got - want).abs() / want.abs()
        };
        if rel > max_rel {
            max_rel = rel
        }
    }
    eprintln!("{tag}: M=1 vs f64 max_rel={max_rel:.6}");
    assert!(max_rel < 0.02, "{tag} max_rel {max_rel}");

    // batchm rows bitwise == M=1
    let cb = g.alloc(m * n * 2).unwrap();
    ops::dense_gemv_batchm(
        &g,
        kb,
        a_p,
        &DenseWeight { weight: b_p },
        cb,
        m as u32,
        n as u32,
        k as u32,
        n as u32,
        0,
    )
    .unwrap();
    g.synchronize(0).unwrap();
    let bout = download_u16(&g, cb, m * n);
    for t in 0..m {
        let c_row = g.alloc(n * 2).unwrap();
        let ar = upload(&g, &u16_bytes(&a[t * k..(t + 1) * k]));
        ops::dense_gemv(
            &g,
            kk,
            ar,
            &DenseWeight { weight: b_p },
            c_row,
            n as u32,
            k as u32,
            0,
        )
        .unwrap();
        g.synchronize(0).unwrap();
        let m1 = download_u16(&g, c_row, n);
        assert_eq!(
            &m1[..],
            &bout[t * n..(t + 1) * n],
            "{tag} row {t} not bitwise"
        );
    }
    eprintln!("{tag}: {m} batchm rows bitwise == M=1");

    // flushed microbench: rotate >=8 B copies
    let n_copies = (512 * 1024 * 1024 / (n * k * 2)).max(8);
    let copies: Vec<DevicePtr> = (0..n_copies).map(|_| upload(&g, &u16_bytes(&b))).collect();
    let _ = a_p;
    for (kern, lab) in [(ko, "old"), (kk, "new")] {
        for _ in 0..3 {
            ops::dense_gemv(
                &g,
                kern,
                a_p,
                &DenseWeight { weight: copies[0] },
                c1,
                n as u32,
                k as u32,
                0,
            )
            .unwrap();
        }
        g.synchronize(0).unwrap();
        let t0 = std::time::Instant::now();
        let iters = 60;
        for i in 0..iters {
            ops::dense_gemv(
                &g,
                kern,
                a_p,
                &DenseWeight {
                    weight: copies[i % copies.len()],
                },
                c1,
                n as u32,
                k as u32,
                0,
            )
            .unwrap();
        }
        g.synchronize(0).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        let bytes = (n * k * 2 + k * 2 + n * 2) as f64;
        eprintln!(
            "{tag} {lab}: {:.1} GB/s ({:.0} us/launch)",
            bytes * iters as f64 / dt / 1e9,
            dt * 1e6 / iters as f64
        );
    }
}

#[test]
fn dense_gemv_bf16_e2_shapes() {
    run_case(512, 4096, 8, "router");
    run_case(1024, 4096, 8, "fc1");
    run_case(4096, 1024, 8, "fc2");
    run_case(4096, 4096, 8, "attn_proj");
}
