// SPDX-License-Identifier: AGPL-3.0-only

//! GPU checks of the `qwen4exp_prefill` launchers against the default kernels
//! they replace, through the production Rust launch code (`#[ignore]`: run on
//! a GB10 with `--ignored`).

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::weight_map::DenseWeight;

fn lcg_bytes(seed: &mut u64, n: usize, f: impl Fn(f32) -> Vec<u8>) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f(((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        })
        .collect()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

/// `qwen4exp_ba_gates_prefill_rows` must equal `dense_gemm_ba_gates_prefill`
/// byte for byte at the TP2 rank shape (48 BA outputs, 24 v-heads, hidden
/// 2560): odd and even row counts, one row, a prefill chunk.
#[test]
#[ignore]
fn ba_gates_rows_matches_default_bitwise() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k_def = g
        .kernel("ssm_preprocess", "dense_gemm_ba_gates_prefill")
        .unwrap();
    let (nv, k) = (24usize, 2560usize);
    let (n, vpg, gs) = (2 * nv, 2usize, 2 * nv);
    let mut seed = 0x0ba5_u64;
    let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes().to_vec();
    let w = DenseWeight {
        weight: up(g, &lcg_bytes(&mut seed, n * k, |x| bf(0.05 * x))),
    };
    let a_log = up(g, &lcg_bytes(&mut seed, nv, |x| x.to_le_bytes().to_vec()));
    let dt_bias = up(g, &lcg_bytes(&mut seed, nv, |x| x.to_le_bytes().to_vec()));
    for m in [1usize, 7, 30, 4097, 16016] {
        let a = up(g, &lcg_bytes(&mut seed, m * k, bf));
        let (o1, o2) = (g.alloc(m * gs * 4).unwrap(), g.alloc(m * gs * 4).unwrap());
        super::super::dense_gemm_ba_gates_prefill(
            g, k_def, a, &w, a_log, dt_bias, o1, m as u32, n as u32, k as u32, k as u32, gs as u32,
            nv as u32, vpg as u32, stream,
        )
        .unwrap();
        let dims = [m, n, k, k, gs, nv, vpg].map(|v| v as u32);
        assert!(
            super::launch_ba_gates_rows(g, [a, w.weight, a_log, dt_bias, o2], dims, stream)
                .unwrap(),
            "the rows kernel must serve the TP2 shape"
        );
        g.synchronize(stream).unwrap();
        let (mut v1, mut v2) = (vec![0u8; m * gs * 4], vec![0u8; m * gs * 4]);
        g.copy_d2h(o1, &mut v1).unwrap();
        g.copy_d2h(o2, &mut v2).unwrap();
        let diff = v1.iter().zip(&v2).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "m={m}: {diff} gate bytes differ");
        println!("ba gates rows == default at m={m}");
        for p in [a, o1, o2] {
            g.free(p).unwrap();
        }
    }
}

/// `qwen4exp_fp8_gemm_w2` must equal `fp8_fp8_gemm_t_m128` byte for byte at
/// the attention prefill shapes (TP2 q+gate, k/v), full and partial tiles.
#[test]
#[ignore]
fn fp8_gemm_w2_matches_default_bitwise() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k_def = g.kernel("w4a16", "fp8_fp8_gemm_t_m128").unwrap();
    let mut seed = 0xf8_u64;
    // Finite E4M3 bytes: drop the two NaN encodings.
    let e4m3 = |x: f32| {
        let b = ((x + 1.0) * 127.5) as u8;
        vec![if b & 0x7F == 0x7F { b & 0xF0 } else { b }]
    };
    for (m, n, k) in [
        (16016usize, 6144usize, 2560usize),
        (16016, 256, 2560),
        (1001, 6144, 2560),
        (129, 300, 2560),
    ] {
        let a = up(g, &lcg_bytes(&mut seed, m * k, e4m3));
        let b = up(g, &lcg_bytes(&mut seed, n * k, e4m3));
        let (o1, o2) = (g.alloc(m * n * 2).unwrap(), g.alloc(m * n * 2).unwrap());
        super::super::fp8_fp8_gemm_n128_m128(
            g, k_def, a, b, o1, m as u32, n as u32, k as u32, stream,
        )
        .unwrap();
        assert!(
            super::launch_fp8_gemm_w2(g, [a, b, o2], [m, n, k].map(|v| v as u32), stream).unwrap()
        );
        g.synchronize(stream).unwrap();
        let (mut v1, mut v2) = (vec![0u8; m * n * 2], vec![0u8; m * n * 2]);
        g.copy_d2h(o1, &mut v1).unwrap();
        g.copy_d2h(o2, &mut v2).unwrap();
        let diff = v1.iter().zip(&v2).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 0, "{m}x{n}x{k}: {diff} bytes differ");
        println!("fp8 gemm w2 == default at {m}x{n}x{k}");
        for p in [a, b, o1, o2] {
            g.free(p).unwrap();
        }
    }
}

/// `fp8_gemm_t_m128` (the qwen4_exp attention o_proj, BF16 x FP8) is row
/// independent: any row ranges, any sizes, give the whole GEMM's bytes --
/// what `ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE`'s piecewise o_proj relies on.
#[test]
#[ignore]
fn fp8_gemm_m128_rows_are_independent() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let kern = g.kernel("w4a16", "fp8_gemm_t_m128").unwrap();
    let mut seed = 0x0f_u64;
    let bf = |x: f32| ((x.to_bits() >> 16) as u16).to_le_bytes().to_vec();
    let e4m3 = |x: f32| {
        let b = ((x + 1.0) * 127.5) as u8;
        vec![if b & 0x7F == 0x7F { b & 0xF0 } else { b }]
    };
    let (m, n, k) = (16016usize, 2560usize, 2048usize);
    let a = up(g, &lcg_bytes(&mut seed, m * k, bf));
    let b = up(g, &lcg_bytes(&mut seed, n * k, e4m3));
    let (o1, o2) = (g.alloc(m * n * 2).unwrap(), g.alloc(m * n * 2).unwrap());
    super::super::fp8_gemm_n128_m128(g, kern, a, b, o1, m as u32, n as u32, k as u32, stream)
        .unwrap();
    for (r0, rows) in [(0usize, 77usize), (77, 2048), (2125, 1), (2126, m - 2126)] {
        super::super::fp8_gemm_n128_m128(
            g,
            kern,
            a.offset(r0 * k * 2),
            b,
            o2.offset(r0 * n * 2),
            rows as u32,
            n as u32,
            k as u32,
            stream,
        )
        .unwrap();
    }
    g.synchronize(stream).unwrap();
    let (mut v1, mut v2) = (vec![0u8; m * n * 2], vec![0u8; m * n * 2]);
    g.copy_d2h(o1, &mut v1).unwrap();
    g.copy_d2h(o2, &mut v2).unwrap();
    let diff = v1.iter().zip(&v2).filter(|(x, y)| x != y).count();
    assert_eq!(
        diff, 0,
        "{diff} bytes differ between the row ranges and the whole"
    );
}
