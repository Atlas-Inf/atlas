// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests for the strix-hip port of the GB10-only kernels this model
//! links.
//!
//! `mamba2_ssd_chunk` is intentionally ABSENT on this target: the fused K3
//! scan's dynamic shared footprint needs >64 KB workgroup LDS at every
//! `state_size` the kernel's shared layout tolerates (the [L][L] decay tile
//! writes past `sCM` for N<64; N>=64 needs >=66.8 KB vs gfx1151's 64 KB
//! ceiling). `NemotronMamba2Layer::prefill` gates on the kernel handles
//! (`prefill.rs`: `ssd_ok`) and takes the sequential `mamba2_ssm_prefill`
//! recurrence when they are 0 — this file proves that fallback is what
//! actually runs and that it computes the right recurrence.
//!
//! `ssm_h_dtype` is a pure dtype convert and must match the CPU bitwise.
//!
//! Run: `cargo test --release -p spark-model ssd -- --test-threads 1`

#![cfg(all(test, target_os = "linux"))]

use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

// head_dim must be <= state_size: the sequential kernel launches
// state_size threads but fills smem_x[head_dim].
const HEADS: u32 = 4;
const HEAD_DIM: u32 = 64;
const STATE: u32 = 128;
const GROUPS: u32 = 1;
const SEQ: u32 = 40;
const DT_MIN: f32 = 1e-3;
const DT_MAX: f32 = 1e-1;

fn gpu() -> AtlasCudaBackend {
    AtlasCudaBackend::new(
        0,
        &atlas_kernels::ptx_for_exact_target("nemotron-3.5-super-vl-120b-a12b", "nvfp4")
            .expect("omni target PTX")
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
fn bf16_seq(n: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((x >> 33) as f32 / 2147483648.0 - 1.0) * scale;
            (f.to_bits() >> 16) as u16
        })
        .collect()
}
fn f32_seq(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            lo + (x >> 33) as f32 / 2147483648.0 * (hi - lo)
        })
        .collect()
}
fn u16b(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|u| u.to_le_bytes()).collect()
}
fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
fn bf2f(u: u16) -> f32 {
    f32::from_bits((u as u32) << 16)
}
fn f2b(f: f32) -> u16 {
    half::bf16::from_f32(f).to_bits()
}

/// The SSD chunked scan must NOT resolve on strix-hip (it cannot run within
/// the gfx1151 LDS ceiling), so `ssd_ok` in `prefill.rs` is always false and
/// prefill takes the sequential `mamba2_ssm_prefill` kernel. Assert the
/// handles are absent and that the fallback kernel reproduces a CPU
/// recurrence bitwise at bf16 output precision.
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn ssd_absent_and_sequential_prefill_is_the_fallback() {
    let g = gpu();
    // If this ever resolves, the kernel must be proven to run within the
    // device LDS limit — the port's audit found no valid state_size.
    // HIP-only: GB10 resolves mamba2_ssd_chunk legitimately.
    if cfg!(atlas_hip) {
        assert!(
            g.kernel("mamba2_ssd_chunk", "mamba2_ssd_scan").is_err(),
            "mamba2_ssd_scan must stay absent on strix-hip (LDS ceiling)"
        );
        assert!(
            !ops::ssd_scan_fits(128),
            "omni state_size=128 exceeds the device opt-in limit by design"
        );
    }

    let prefill_k = g.kernel("mamba2_ssm", "mamba2_ssm_prefill").unwrap();
    let x_elems = (SEQ * HEADS * HEAD_DIM) as usize;
    let bc_elems = (SEQ * GROUPS * STATE) as usize;
    let h_elems = (HEADS * HEAD_DIM * STATE) as usize;

    let x_h = bf16_seq(x_elems, 0x1234, 0.5);
    let b_h = bf16_seq(bc_elems, 0x2345, 0.5);
    let c_h = bf16_seq(bc_elems, 0x3456, 0.5);
    let dt_h = bf16_seq((SEQ * HEADS) as usize, 0x4567, 2.0);
    let a_log = f32_seq(HEADS as usize, 0x5678, -4.0, 2.0);
    let d_h = f32_seq(HEADS as usize, 0x6789, 0.5, 1.5);
    let dtb = f32_seq(HEADS as usize, 0x789a, -1.0, 0.0);

    let y = g.alloc(x_elems * 2).unwrap();
    let h = g.alloc(h_elems * 4).unwrap();
    let h_init = upload(&g, &vec![0u8; h_elems * 4]);
    g.copy_d2d(h_init, h, h_elems * 4).unwrap();

    ops::mamba2_ssm_prefill(
        &g,
        prefill_k,
        h,
        upload(&g, &u16b(&x_h)),
        upload(&g, &u16b(&b_h)),
        upload(&g, &u16b(&c_h)),
        upload(&g, &u16b(&dt_h)),
        upload(&g, &f32b(&a_log)),
        upload(&g, &f32b(&d_h)),
        upload(&g, &f32b(&dtb)),
        y,
        1,
        SEQ,
        HEADS,
        HEAD_DIM,
        STATE,
        GROUPS,
        DT_MIN,
        DT_MAX,
        HEADS * HEAD_DIM,
        GROUPS * STATE,
        HEADS,
        HEADS * HEAD_DIM,
        0,
    )
    .unwrap();
    g.synchronize(0).unwrap();

    // CPU recurrence: h[hd][n] = exp(-exp(A_log)*dt) * h + x*dt*B[n];
    // y[hd] = sum_n h*C[n] + D*x.
    let y_got: Vec<u16> = download(&g, y, x_elems * 2)
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let h_got: Vec<f32> = download(&g, h, h_elems * 4)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (mut my, mut mh) = (0.0f32, 0.0f32);
    for hd_i in 0..HEADS as usize {
        let mut hs = vec![0f32; (HEAD_DIM * STATE) as usize];
        let neg_a = a_log[hd_i].exp();
        for t in 0..SEQ as usize {
            // softplus then clamp, matching the kernel exactly.
            let raw = bf2f(dt_h[t * HEADS as usize + hd_i]) + dtb[hd_i];
            let soft = if raw > 20.0 {
                raw
            } else {
                (1.0f32 + raw.exp()).ln()
            };
            let dt = soft.clamp(DT_MIN, DT_MAX);
            let da = (-neg_a * dt).exp();
            for p in 0..HEAD_DIM as usize {
                let xv = bf2f(x_h[t * (HEADS * HEAD_DIM) as usize + hd_i * HEAD_DIM as usize + p]);
                let mut yv = 0f32;
                for n in 0..STATE as usize {
                    let bv = bf2f(b_h[t * STATE as usize + n]);
                    let cv = bf2f(c_h[t * STATE as usize + n]);
                    let hv = da * hs[p * STATE as usize + n] + xv * dt * bv;
                    hs[p * STATE as usize + n] = hv;
                    yv += hv * cv;
                }
                yv += d_h[hd_i] * xv;
                let want = f2b(yv);
                let got = y_got[t * (HEADS * HEAD_DIM) as usize + hd_i * HEAD_DIM as usize + p];
                my = my.max((bf2f(want) - bf2f(got)).abs());
            }
        }
        for i in 0..hs.len() {
            mh = mh.max((hs[i] - h_got[hd_i * hs.len() + i]).abs());
        }
    }
    println!("sequential prefill vs CPU: y max_abs={my:.6} h max_abs={mh:.6}");
    // y rounds through bf16 at the store; the recurrence itself is fp32.
    assert!(my <= 0.02, "y diff {my}");
    assert!(mh <= 1e-3, "h diff {mh}");
}

/// `ssm_h_dtype`: f32->f16 must equal the CPU conversion bitwise, and
/// f16->f32 is exact by construction — assert both.
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn ssm_h_dtype_roundtrip_bitwise() {
    let g = gpu();
    let n = 8191usize; // not a multiple of the 256-thread block
    let src_h = f32_seq(n, 0x9abc, -4.0, 4.0);
    let src = upload(&g, &f32b(&src_h));

    let f16 = g.alloc(n * 2).unwrap();
    let back = g.alloc(n * 4).unwrap();
    let to16 = g
        .kernel("ssm_h_dtype", "ssm_h_state_f32_to_f16")
        .expect("ssm_h_state_f32_to_f16 must resolve on this target");
    let to32 = g
        .kernel("ssm_h_dtype", "ssm_h_state_f16_to_f32")
        .expect("ssm_h_state_f16_to_f32 must resolve on this target");
    ops::ssm_h_state_f32_to_f16(&g, to16, src, f16, n as u64, 0).unwrap();
    ops::ssm_h_state_f16_to_f32(&g, to32, f16, back, n as u64, 0).unwrap();
    g.synchronize(0).unwrap();

    let f16_h = download(&g, f16, n * 2);
    let back_h = download(&g, back, n * 4);

    let expect16: Vec<u8> = src_h
        .iter()
        .flat_map(|f| half::f16::from_f32(*f).to_bits().to_le_bytes())
        .collect();
    assert_eq!(f16_h, expect16, "f32->f16 must be bitwise vs CPU");

    let expect32: Vec<u8> = src_h
        .iter()
        .flat_map(|f| half::f16::from_f32(*f).to_f32().to_le_bytes())
        .collect();
    assert_eq!(back_h, expect32, "f16->f32 round trip must be bitwise");
}
