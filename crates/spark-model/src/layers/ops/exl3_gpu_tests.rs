// SPDX-License-Identifier: AGPL-3.0-only
// Portions of this file are derived from ExLlamaV3 (MIT, see NOTICE.md).

//! GPU parity for the `exl3` kernels, against the CPU reference in
//! `weight_map/exl3/cpu_ref.rs` — the same fixtures the CPU tests hash.
//!
//! `reconstruct` is pinned BIT-exact (every fp16 element's raw bits), because
//! both sides are the same pure index -> half expansion; anything else (a
//! mis-set rounding mode, a wrong fragment layout) shows up as differing bits,
//! not as noise. The linear path is pinned to a relative norm: its GEMM runs
//! fp32-accumulate / fp16-store against an f64 reference.
//!
//! GPU tests (`#[ignore]` per repo convention):
//!   cargo test -p spark-model --release exl3 -- --ignored --nocapture
//! Real checkpoint tensors (optional): point ATLAS_EXL3_TEST_DATA at a
//! safetensors file holding `t0..t3.{trellis,suh,svh}` (see bench/exl3/st_io.py).

use half::f16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::weight_map::exl3;
use crate::weight_map::exl3::fixtures::{fixtures, halves, lanes};
use crate::weight_map::exl3::{Exl3Shape, Exl3Weight};

/// The activation rows of the linear test.
const M: usize = 5;

/// fp16 values as raw little-endian bytes (what the device holds).
fn f16_bytes(v: &[f16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// u16 lanes as raw little-endian bytes.
fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn gpu() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    spark_runtime::cuda_backend::AtlasCudaBackend::new(
        0,
        &atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
            .expect("build with ATLAS_TARGET_MODEL='*'")
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
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// SplitMix64 — deterministic pseudo-randomness with no dependency on the
/// platform's RNG.
fn mix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform [-1, 1) rounded to bf16, as raw bits.
fn bf16_uniform(seed: &mut u64) -> u16 {
    let u = (mix64(seed) >> 16) as f64 / (1u64 << 48) as f64;
    f16::to_bits(f16::from_f32(u as f32 * 2.0 - 1.0))
}

/// The bf16 bits as f64 (exact: bf16 is a truncation of f32).
fn bf16_f64(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

/// One test tensor: trellis and scales as raw lanes, plus a name. The GPU and
/// the CPU reference read the same bytes.
struct Tensor {
    name: String,
    trellis: Vec<u16>,
    suh: Vec<f16>,
    svh: Vec<f16>,
    shape: Exl3Shape,
}

fn fixture_tensors() -> Vec<Tensor> {
    fixtures()
        .blocks
        .iter()
        .map(|b| Tensor {
            name: b.name.clone(),
            trellis: lanes(&b.trellis_i16_b64),
            suh: halves(&b.suh_f16_b64),
            svh: halves(&b.svh_f16_b64),
            shape: Exl3Shape {
                in_features: b.in_features,
                out_features: b.out_features,
                bits: b.bits,
            },
        })
        .collect()
}

/// Real checkpoint tensors if ATLAS_EXL3_TEST_DATA names a safetensors file,
/// else empty (the tests then print a skip note). Parsed by hand — the repo
/// carries no safetensors dependency; the layout is 8-byte LE header length,
/// the JSON header, then the data blob.
fn real_tensors() -> Vec<Tensor> {
    let Ok(path) = std::env::var("ATLAS_EXL3_TEST_DATA") else {
        eprintln!("ATLAS_EXL3_TEST_DATA unset: skipping the real checkpoint tensors");
        return Vec::new();
    };
    let blob = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let hl = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
    let head: serde_json::Value =
        serde_json::from_slice(&blob[8..8 + hl]).expect("safetensors header");
    let data = &blob[8 + hl..];
    let tensor = |key: &str| -> Option<(Vec<u8>, Vec<usize>, String)> {
        let t = head.get(key)?;
        let (a, b) = (
            t["data_offsets"][0].as_u64()?,
            t["data_offsets"][1].as_u64()?,
        );
        Some((
            data[a as usize..b as usize].to_vec(),
            t["shape"]
                .as_array()?
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect(),
            t["dtype"].as_str()?.to_string(),
        ))
    };
    let u16s = |v: &[u8]| -> Vec<u16> {
        v.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let mut out = Vec::new();
    for i in 0..4 {
        let Some((tr, tr_dims, tr_dt)) = tensor(&format!("t{i}.trellis")) else {
            continue;
        };
        assert_eq!((tr_dims.len(), tr_dt.as_str()), (3, "I16"), "t{i}.trellis");
        let (su, su_dims, su_dt) = tensor(&format!("t{i}.suh")).expect("t{i}.suh");
        let (sv, sv_dims, sv_dt) = tensor(&format!("t{i}.svh")).expect("t{i}.svh");
        assert_eq!((su_dims.len(), su_dt.as_str()), (1, "F16"), "t{i}.suh");
        assert_eq!((sv_dims.len(), sv_dt.as_str()), (1, "F16"), "t{i}.svh");
        out.push(Tensor {
            name: format!("t{i}"),
            trellis: u16s(&tr),
            suh: u16s(&su).iter().copied().map(f16::from_bits).collect(),
            svh: u16s(&sv).iter().copied().map(f16::from_bits).collect(),
            shape: Exl3Shape::from_trellis_dims(&tr_dims).unwrap(),
        });
    }
    out
}

/// Every element's fp16 bits against `decode_inner`, bit-exact.
fn check_inner_bit_exact(g: &dyn GpuBackend, k: &Exl3Kernels, t: &Tensor) {
    let stream = g.default_stream();
    let w = Exl3Weight {
        trellis: upload(g, &u16_bytes(&t.trellis)),
        suh: upload(g, &f16_bytes(&t.suh)),
        svh: upload(g, &f16_bytes(&t.svh)),
        shape: t.shape,
    };
    let n = t.shape.in_features * t.shape.out_features;
    let inner = upload(g, &vec![0u8; n * 2]);
    exl3_reconstruct(g, k, &w, inner, stream).unwrap();
    let got = download_u16(g, inner, n);

    let want = exl3::decode_inner(&t.trellis, &t.shape).unwrap();
    let mism: Vec<usize> = got
        .iter()
        .zip(want.iter())
        .enumerate()
        .filter(|(_, (g, w))| **g != w.to_bits())
        .map(|(i, _)| i)
        .collect();
    assert!(
        mism.is_empty(),
        "{}: {}/{} fp16 bits differ from decode_inner; first 5 (idx, gpu, cpu): {:?}",
        t.name,
        mism.len(),
        n,
        mism.iter()
            .take(5)
            .map(|&i| (i, got[i], want[i].to_bits()))
            .collect::<Vec<_>>()
    );
    for p in [w.trellis, w.suh, w.svh, inner] {
        g.free(p).unwrap();
    }
}

#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn reconstruct_matches_cpu_on_fixtures() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let k = Exl3Kernels::resolve(g).unwrap();
    for t in fixture_tensors() {
        check_inner_bit_exact(g, &k, &t);
    }
}

#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn reconstruct_matches_cpu_on_real_tensors() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let k = Exl3Kernels::resolve(g).unwrap();
    for t in real_tensors() {
        check_inner_bit_exact(g, &k, &t);
    }
}

/// `exl3_transpose_f16` against a CPU transpose, on a ragged shape so every
/// guarded load and store runs: bit-exact, the buffers hold no rounding.
#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn transpose_f16_matches_cpu() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let k = Exl3Kernels::resolve(g).unwrap();
    let stream = g.default_stream();

    let (rows, cols) = (37usize, 70usize);
    let mut seed = 0xF00D_1234_9ABC_DEF0u64;
    let x: Vec<f16> = (0..rows * cols)
        .map(|_| {
            let u = (mix64(&mut seed) >> 16) as f32 / (1u64 << 48) as f32;
            f16::from_f32(u * 2.0 - 1.0)
        })
        .collect();
    let src = upload(g, &f16_bytes(&x));
    let dst = upload(g, &vec![0u8; rows * cols * 2]);
    exl3_transpose_f16(g, &k, src, dst, rows as u32, cols as u32, stream).unwrap();
    let got = download_u16(g, dst, rows * cols);

    let mut bad = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let want = x[r * cols + c].to_bits();
            if got[c * rows + r] != want {
                bad.push((r, c));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "transpose: {}/{} entries differ, first 5 (row, col): {:?}",
        bad.len(),
        rows * cols,
        bad.iter().take(5).collect::<Vec<_>>()
    );
    for p in [src, dst] {
        g.free(p).unwrap();
    }
}

/// One EXL3 linear dequantized to BF16 `[out, in]`, against the CPU reference:
/// `reconstruct(..)` gives W `[in, out]` in f32, the GPU walks the transpose
/// through two fp16 Hadamard passes and one rounding to bf16.
fn check_dense_bf16(g: &dyn GpuBackend, k: &Exl3Kernels, t: &Tensor) {
    let stream = g.default_stream();
    let (i, o) = (t.shape.in_features, t.shape.out_features);
    let w = Exl3Weight {
        trellis: upload(g, &u16_bytes(&t.trellis)),
        suh: upload(g, &f16_bytes(&t.suh)),
        svh: upload(g, &f16_bytes(&t.svh)),
        shape: t.shape,
    };
    let out = upload(g, &vec![0u8; i * o * 2]);
    exl3_dense_bf16_nk(g, k, &w, out, stream).unwrap();
    let got = download_u16(g, out, i * o);

    let w_ref = exl3::reconstruct_ref(&t.trellis, &t.suh, &t.svh, &t.shape).unwrap();
    let (mut ngg, mut ngw, mut nw) = (0.0f64, 0.0f64, 0.0f64);
    for r in 0..i {
        for c in 0..o {
            let v = bf16_f64(got[c * i + r]);
            let x = w_ref[r * o + c] as f64;
            ngg += v * v;
            ngw += v * x;
            nw += x * x;
        }
    }
    // ||gpu - cpu|| / ||cpu||, from the inner products above.
    let ratio = (ngg - 2.0 * ngw + nw).sqrt() / nw.sqrt();
    eprintln!("{}: dense_bf16 rel_err = {ratio:.3e}", t.name);
    assert!(
        ratio < 5e-3,
        "{}: relative error {ratio:.3e} >= 5e-3",
        t.name
    );
    for p in [w.trellis, w.suh, w.svh, out] {
        g.free(p).unwrap();
    }
}

#[test]
#[ignore = "needs a GB10 GPU and a real kernel build"]
fn dense_bf16_nk_matches_cpu_reconstruct() {
    let gpu = gpu();
    let g: &dyn GpuBackend = &gpu;
    let k = Exl3Kernels::resolve(g).unwrap();
    for t in fixture_tensors() {
        check_dense_bf16(g, &k, &t);
    }
    let ts = real_tensors();
    if ts.is_empty() && std::env::var_os("ATLAS_EXL3_TEST_DATA").is_none() {
        eprintln!("ATLAS_EXL3_TEST_DATA unset: real tensors skipped");
    }
    for t in ts {
        check_dense_bf16(g, &k, &t);
    }
}

// The linear-path tests (fp32-accumulate GEMM, relative-norm pinned) live in a
// sibling file to keep each file under the 500-line cap.
#[path = "exl3_gpu_linear_tests.rs"]
mod linear;
