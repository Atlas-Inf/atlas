// SPDX-License-Identifier: AGPL-3.0-only

//! GPU bit-exactness test: `mamba2_ssm_verify_exact_persistent` (the K-row
//! batched verify kernel) vs K sequential `mamba2_ssm_decode` launches.
//!
//! nemotron_h_omni's generic-DFlash path runs this kernel as its verify
//! default (`mamba_verify_exact` lever); the pairing is only valid if the
//! kernel's outputs are byte-identical to the per-row decode loop it
//! replaces — y rows, final H, and the per-row H intermediates the verify
//! commit path replays from.
//!
//! Runs only where a CUDA backend and the target's PTX exist (spark2-style
//! builds); it is a no-op failure elsewhere, matching the
//! `exl3_gpu_tests` precedent of failing loudly without CUDA.

use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::ops::mamba2_ssm_verify;

// Nemotron-3-Super/3.5-VL dims.
const HEADS: u32 = 128;
const HEAD_DIM: u32 = 64;
const STATE: u32 = 128;
const GROUPS: u32 = 8;
const D_INNER: usize = (HEADS * HEAD_DIM) as usize; // 8192
const GS: usize = (GROUPS * STATE) as usize; // 1024
const H_ELEMS: usize = (HEADS * HEAD_DIM * STATE) as usize; // f32

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

/// Deterministic pseudo-random bf16 in [-1, 1] — SplitMix-ish LCG, no
/// `rand` dependency. bf16 bits are the top 16 of f32 bits.
fn bf16_seq(n: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((x >> 33) & 0xFFFF) as f32 / 65535.0;
            bf16::from_f32((u * 2.0 - 1.0) * scale).to_bits()
        })
        .collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn f32_seq(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((x >> 33) & 0xFFFF) as f32 / 65535.0;
            lo + u * (hi - lo)
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Sequential reference: K `mamba2_ssm_decode` launches + post-row H
/// snapshot copies, exactly what `decode_batched_verify`'s fallback loop
/// does (minus conv, which is row-wise identical either way).
fn run_sequential(
    g: &dyn GpuBackend,
    k: usize,
    x: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    dt: DevicePtr,
    h_state: DevicePtr,
    y: DevicePtr,
    h_inter: DevicePtr,
    params: (DevicePtr, DevicePtr, DevicePtr),
    stream: u64,
) {
    let decode_k = g.kernel("mamba2_ssm", "mamba2_ssm_decode").unwrap();
    for t in 0..k {
        KernelLaunch::new(g, decode_k)
            .grid([HEADS, 1, 1])
            .block([STATE, 1, 1])
            .arg_ptr(h_state)
            .arg_ptr(x.offset(t * D_INNER * 2))
            .arg_ptr(b.offset(t * GS * 2))
            .arg_ptr(c.offset(t * GS * 2))
            .arg_ptr(dt.offset(t * HEADS as usize * 2))
            .arg_ptr(params.0)
            .arg_ptr(params.1)
            .arg_ptr(params.2)
            .arg_ptr(y.offset(t * D_INNER * 2))
            .arg_u32(1)
            .arg_u32(HEADS)
            .arg_u32(HEAD_DIM)
            .arg_u32(STATE)
            .arg_u32(GROUPS)
            .arg_f32(1e-9)
            .arg_f32(1e9)
            .launch(stream)
            .unwrap();
        if t + 1 < k {
            g.copy_d2d_async(
                h_state,
                h_inter.offset(t * H_ELEMS * 4),
                H_ELEMS * 4,
                stream,
            )
            .unwrap();
        }
    }
    g.synchronize(stream).unwrap();
}

fn run_case(k: usize) {
    let g = gpu();
    let verify_k = g
        .kernel("mamba2_ssm", "mamba2_ssm_verify_exact_persistent")
        .expect("exact persistent verify kernel must resolve on this target");
    let stream = 0u64;

    let x = upload(&g, &u16_bytes(&bf16_seq(k * D_INNER, 1, 0.5)));
    let b = upload(&g, &u16_bytes(&bf16_seq(k * GS, 2, 0.5)));
    let c = upload(&g, &u16_bytes(&bf16_seq(k * GS, 3, 0.5)));
    // dt raw: moderate range so softplus stays well-conditioned.
    let dt = upload(&g, &u16_bytes(&bf16_seq(k * HEADS as usize, 4, 1.0)));
    let a_log = upload(&g, &f32_bytes(&f32_seq(HEADS as usize, 5, -4.0, -1.0)));
    let d_param = upload(&g, &f32_bytes(&f32_seq(HEADS as usize, 6, -1.0, 1.0)));
    let dt_bias = upload(&g, &f32_bytes(&f32_seq(HEADS as usize, 7, -2.0, 0.0)));
    let h0_bytes = f32_bytes(&f32_seq(H_ELEMS, 8, -0.5, 0.5));

    // Sequential reference.
    let h_seq = upload(&g, &h0_bytes);
    let y_seq = g.alloc(k * D_INNER * 2).unwrap();
    let h_inter_seq = g.alloc(k.max(1) * H_ELEMS * 4).unwrap();
    run_sequential(
        &g,
        k,
        x,
        b,
        c,
        dt,
        h_seq,
        y_seq,
        h_inter_seq,
        (a_log, d_param, dt_bias),
        stream,
    );

    // Batched exact verify.
    let h_ver = upload(&g, &h0_bytes);
    let y_ver = g.alloc(k * D_INNER * 2).unwrap();
    let h_inter_ver = g.alloc(k.max(1) * H_ELEMS * 4).unwrap();
    mamba2_ssm_verify(
        &g,
        verify_k,
        h_ver,
        x,
        b,
        c,
        dt,
        a_log,
        d_param,
        dt_bias,
        y_ver,
        h_inter_ver,
        (k - 1) as u32,
        H_ELEMS as u32,
        1,
        k as u32,
        HEADS,
        HEAD_DIM,
        STATE,
        GROUPS,
        1e-9,
        1e9,
        D_INNER as u32,
        GS as u32,
        HEADS,
        D_INNER as u32,
        true,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    assert_eq!(
        download(&g, y_seq, k * D_INNER * 2),
        download(&g, y_ver, k * D_INNER * 2),
        "y rows diverge at K={k}"
    );
    assert_eq!(
        download(&g, h_seq, H_ELEMS * 4),
        download(&g, h_ver, H_ELEMS * 4),
        "final H diverges at K={k}"
    );
    if k > 1 {
        assert_eq!(
            download(&g, h_inter_seq, (k - 1) * H_ELEMS * 4),
            download(&g, h_inter_ver, (k - 1) * H_ELEMS * 4),
            "H intermediates diverge at K={k}"
        );
    }
}

// ── Batch-M twin bit-identity ───────────────────────────────────────────
//
// `mamba_verify_exact`/`moe_verify_exact` batch the verify projections
// through the M-row twins instead of calling the serial GEMV K times.
// Their contract is per-row bit-identity with the M=1 kernel — same
// K-iteration order, accumulator precision and reduction tree — so each
// twin is checked directly against its serial form at real omni dims,
// including large-magnitude inputs that stress the ordering claim.

/// w8a16 block-scale tensor: [n/128, k/128] f32 scales ~1.0.
fn block_scales(n: usize, k: usize) -> Vec<u8> {
    f32_bytes(&f32_seq((n / 128) * (k / 128), 42, 0.7, 1.3))
}

fn run_w8a16_twin_case(m: usize, n: usize, k: usize) {
    let g = gpu();
    let serial = g.kernel("w8a16_gemv", "w8a16_gemv").unwrap();
    let twin = g
        .kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")
        .expect("w8a16_gemv_batch16 must resolve");
    let stream = 0u64;

    // Activations with a wide range — bf16 inputs at ±0.5 and ±64.
    let mut a = bf16_seq(m * k, 11, 0.5);
    a.extend(bf16_seq(m * k, 12, 64.0));
    let a = upload(&g, &u16_bytes(&a[..m * k]));
    let w = upload(
        &g,
        &bf16_seq(n * k, 13, 0.5)
            .iter()
            .map(|&b| (b & 0x7f) as u8)
            .collect::<Vec<u8>>(),
    );
    let scales = upload(&g, &block_scales(n, k));

    let c_seq = g.alloc(m * n * 2).unwrap();
    for t in 0..m {
        crate::layers::ops::w8a16_gemv(
            &g,
            serial,
            a.offset(t * k * 2),
            w,
            scales,
            c_seq.offset(t * n * 2),
            n as u32,
            k as u32,
            stream,
        )
        .unwrap();
    }
    g.synchronize(stream).unwrap();

    let c_twin = g.alloc(m * n * 2).unwrap();
    crate::layers::ops::w8a16_gemv_batch4_div(
        &g, twin, a, w, scales, c_twin, m as u32, n as u32, k as u32, 4, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    assert_eq!(
        download(&g, c_seq, m * n * 2),
        download(&g, c_twin, m * n * 2),
        "w8a16_gemv_batch16 diverges from serial w8a16_gemv at m={m} n={n} k={k}"
    );
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn w8a16_batch16_identical_to_serial_m1() {
    run_w8a16_twin_case(1, 18560, 4096); // omni in_proj shape
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn w8a16_batch16_identical_to_serial_m2() {
    run_w8a16_twin_case(2, 4096, 8192); // omni out_proj shape
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn w8a16_batch16_identical_to_serial_m8() {
    run_w8a16_twin_case(8, 18560, 4096);
    run_w8a16_twin_case(8, 4096, 8192);
    run_w8a16_twin_case(8, 5376, 4096); // shared-expert up
}

fn run_dense_twin_case(m: usize, n: usize, k: usize) {
    use crate::weight_map::DenseWeight;
    let g = gpu();
    let serial = g.kernel("gemv", "dense_gemv_bf16").unwrap();
    let twin = g
        .kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
        .expect("dense_gemv_bf16_batchm must resolve");
    let stream = 0u64;

    let a = upload(&g, &u16_bytes(&bf16_seq(m * k, 21, 4.0)));
    let w = upload(&g, &u16_bytes(&bf16_seq(n * k, 22, 0.5)));
    let weight = DenseWeight { weight: w };

    let c_seq = g.alloc(m * n * 2).unwrap();
    for t in 0..m {
        crate::layers::ops::dense_gemv(
            &g,
            serial,
            a.offset(t * k * 2),
            &weight,
            c_seq.offset(t * n * 2),
            n as u32,
            k as u32,
            stream,
        )
        .unwrap();
    }
    g.synchronize(stream).unwrap();

    let c_twin = g.alloc(m * n * 2).unwrap();
    crate::layers::ops::dense_gemv_batchm(
        &g, twin, a, &weight, c_twin, m as u32, n as u32, k as u32, n as u32, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    assert_eq!(
        download(&g, c_seq, m * n * 2),
        download(&g, c_twin, m * n * 2),
        "dense_gemv_bf16_batchm diverges from serial dense_gemv at m={m} n={n} k={k}"
    );
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn dense_batchm_identical_to_serial() {
    run_dense_twin_case(1, 512, 4096); // router gate
    run_dense_twin_case(2, 1024, 4096); // fc1_latent
    run_dense_twin_case(8, 512, 4096);
    run_dense_twin_case(8, 4096, 1024); // fc2_latent
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn verify_exact_persistent_matches_sequential_decode_k1() {
    run_case(1);
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn verify_exact_persistent_matches_sequential_decode_k2() {
    run_case(2);
}

#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn verify_exact_persistent_matches_sequential_decode_k8() {
    run_case(8);
}
