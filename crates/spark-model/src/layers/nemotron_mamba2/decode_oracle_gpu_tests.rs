// SPDX-License-Identifier: AGPL-3.0-only

//! CPU f64-reference oracles for the nemotron Mamba-2 L0 decode-path kernels,
//! at this model's real dims (heads=128, head_dim=64, state=128, n_groups=8,
//! d_inner=8192, d_xbc=10240, in_proj=18560). Twin tests only compare HIP vs
//! HIP — these check the numbers are right.
//!
//! Runs where a CUDA backend + the target's PTX exist.

use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const HEADS: usize = 128;
const HEAD_DIM: usize = 64;
const STATE: usize = 128;
const GROUPS: usize = 8;
const D_INNER: usize = HEADS * HEAD_DIM; // 8192
const D_XBC: usize = D_INNER + 2 * GROUPS * STATE; // 10240
const IN_PROJ: usize = D_XBC + HEADS; // 18560 (z|xBC|dt layout)
const D_CONV: usize = 4;

fn gpu() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    spark_runtime::cuda_backend::AtlasCudaBackend::new(
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

fn download_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut v = vec![0u8; n * 4];
    g.copy_d2h(p, &mut v).unwrap();
    v.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn download_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut v = vec![0u8; n * 2];
    g.copy_d2h(p, &mut v).unwrap();
    v.chunks_exact(2)
        .map(|c| f32::from_bits(((u16::from_le_bytes([c[0], c[1]])) as u32) << 16))
        .collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| {
            let b = (bf16::from_f32(*x)).to_bits();
            b.to_le_bytes()
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn seq(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
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

fn metrics(want: &[f64], got: &[f32]) -> (f64, f64, f64) {
    let mut ma = 0.0f64;
    let mut mr = 0.0f64;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (w, g) in want.iter().zip(got) {
        let g = *g as f64;
        ma = ma.max((g - w).abs());
        if w.abs() > 1e-6 {
            mr = mr.max(((g - w) / w).abs());
        }
        dot += g * w;
        na += g * g;
        nb += w * w;
    }
    (ma, mr, dot / (na.sqrt() * nb.sqrt()))
}

fn report(name: &str, want: &[f64], got: &[f32]) {
    let (ma, mr, cos) = metrics(want, got);
    println!("{name}: max_abs={ma:.6e} max_rel={mr:.6e} cos={cos:.8}");
    assert!(cos >= 0.9999, "{name}: cosine {cos} < 0.9999");
}

// ─── causal_conv1d_update (decode, biased) ───────────────────────
// Kernel args: conv_state[f32,b,d,d_conv](in/out), input[bf16,b,d],
// weight[bf16,d,d_conv], bias[f32,d], out[bf16,b,d], batch, dim, d_conv.
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn conv1d_update_decode_oracle() {
    let g = gpu();
    let dim = D_XBC;
    let state = seq(dim * D_CONV, 0xC0FFEE, -1.0, 1.0);
    let input = seq(dim, 0x11, -1.0, 1.0);
    let weight = seq(dim * D_CONV, 0x22, -1.0, 1.0);
    let bias = seq(dim, 0x33, -0.5, 0.5);

    let state_dev = upload(&g, &f32_bytes(&state));
    let input_dev = upload(&g, &bf16_bytes(&input));
    let weight_dev = upload(&g, &bf16_bytes(&weight));
    let bias_dev = upload(&g, &f32_bytes(&bias));
    let out_dev = upload(&g, &vec![0u8; dim * 2]);

    let k = g.kernel("causal_conv1d", "causal_conv1d_update").unwrap();
    KernelLaunch::new(&g, k)
        .grid([div_ceil(dim as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(state_dev)
        .arg_ptr(input_dev)
        .arg_ptr(weight_dev)
        .arg_ptr(bias_dev)
        .arg_ptr(out_dev)
        .arg_u32(1)
        .arg_u32(dim as u32)
        .arg_u32(D_CONV as u32)
        .launch(0)
        .unwrap();
    g.synchronize(0).unwrap();

    // CPU f64 reference: shift state, insert, conv, SiLU.
    // bf16 round the input/weight exactly as the device sees them.
    let in_bf: Vec<f64> = input
        .iter()
        .map(|x| bf16::from_f32(*x).to_f32() as f64)
        .collect();
    let w_bf: Vec<f64> = weight
        .iter()
        .map(|x| bf16::from_f32(*x).to_f32() as f64)
        .collect();
    let mut want_out = vec![0f64; dim];
    let mut want_state = vec![0f64; dim * D_CONV];
    for ch in 0..dim {
        let s: [f64; D_CONV] = core::array::from_fn(|i| {
            if i + 1 < D_CONV {
                state[ch * D_CONV + i + 1] as f64
            } else {
                in_bf[ch]
            }
        });
        for i in 0..D_CONV {
            want_state[ch * D_CONV + i] = s[i];
        }
        let mut acc = bias[ch] as f64;
        for i in 0..D_CONV {
            acc += s[i] * w_bf[ch * D_CONV + i];
        }
        want_out[ch] = acc / (1.0 + (-acc).exp());
    }
    let got_out = download_bf16(&g, out_dev, dim);
    let got_state = download_f32(&g, state_dev, dim * D_CONV);
    report("conv1d_update out", &want_out, &got_out);
    report("conv1d_update state", &want_state, &got_state);
    let _ = input;
}

// ─── mamba2_ssm_decode ─────────────────────────────────────────
// Args: h_state[f32,b,h,hd,s], x[bf16,b,h*hd], B[bf16,b,g*s], C[bf16,b,g*s],
// dt_raw[bf16,b,h], A_log[f32,h], D[f32,h], dt_bias[f32,h], out[bf16,b,h*hd],
// batch, heads, head_dim, state, groups, dt_min, dt_max.
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn mamba2_ssm_decode_oracle() {
    let g = gpu();
    let n_h = HEADS * HEAD_DIM * STATE;
    let h = seq(n_h, 0xA11CE, -1.0, 1.0);
    let x = seq(D_INNER, 0x44, -1.0, 1.0);
    let b_p = seq(GROUPS * STATE, 0x55, -1.0, 1.0);
    let c_p = seq(GROUPS * STATE, 0x66, -1.0, 1.0);
    let dt_raw = seq(HEADS, 0x77, -1.0, 1.0);
    let a_log = seq(HEADS, 0x88, -4.0, 1.0);
    let d_par = seq(HEADS, 0x99, 0.5, 2.0);
    let dt_bias = seq(HEADS, 0xAA, -4.0, 0.5);

    let h_dev = upload(&g, &f32_bytes(&h));
    let x_dev = upload(&g, &bf16_bytes(&x));
    let b_dev = upload(&g, &bf16_bytes(&b_p));
    let c_dev = upload(&g, &bf16_bytes(&c_p));
    let dt_dev = upload(&g, &bf16_bytes(&dt_raw));
    let a_dev = upload(&g, &f32_bytes(&a_log));
    let dp_dev = upload(&g, &f32_bytes(&d_par));
    let db_dev = upload(&g, &f32_bytes(&dt_bias));
    let out_dev = upload(&g, &vec![0u8; D_INNER * 2]);

    let k = g.kernel("mamba2_ssm", "mamba2_ssm_decode").unwrap();
    KernelLaunch::new(&g, k)
        .grid([HEADS as u32, 1, 1])
        .block([STATE as u32, 1, 1])
        .arg_ptr(h_dev)
        .arg_ptr(x_dev)
        .arg_ptr(b_dev)
        .arg_ptr(c_dev)
        .arg_ptr(dt_dev)
        .arg_ptr(a_dev)
        .arg_ptr(dp_dev)
        .arg_ptr(db_dev)
        .arg_ptr(out_dev)
        .arg_u32(1)
        .arg_u32(HEADS as u32)
        .arg_u32(HEAD_DIM as u32)
        .arg_u32(STATE as u32)
        .arg_u32(GROUPS as u32)
        .arg_f32(1e-9)
        .arg_f32(1e9)
        .launch(0)
        .unwrap();
    g.synchronize(0).unwrap();

    // CPU f64 reference, same formula as the kernel's comments.
    let x_bf: Vec<f64> = x
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let b_bf: Vec<f64> = b_p
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let c_bf: Vec<f64> = c_p
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let dt_bf: Vec<f64> = dt_raw
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let hpg = HEADS / GROUPS;
    let mut want_y = vec![0f64; D_INNER];
    let mut want_h = vec![0f64; n_h];
    for hd_i in 0..HEADS {
        let mut dt = dt_bf[hd_i] + dt_bias[hd_i] as f64;
        if dt <= 20.0 {
            dt = (1.0 + dt.exp()).ln();
        }
        dt = dt.clamp(1e-9, 1e9);
        let da = (-a_log[hd_i].exp() as f64 * dt).exp();
        let d_v = d_par[hd_i] as f64;
        let grp = hd_i / hpg;
        for s in 0..STATE {
            let b = b_bf[grp * STATE + s];
            let c = c_bf[grp * STATE + s];
            for d in 0..HEAD_DIM {
                let i = (hd_i * HEAD_DIM + d) * STATE + s;
                let xhd = x_bf[hd_i * HEAD_DIM + d];
                let mut hv = da * h[i] as f64 + dt * xhd * b;
                hv = hv.clamp(-200.0, 200.0);
                want_h[i] = hv;
                want_y[hd_i * HEAD_DIM + d] += hv * c;
            }
        }
        for d in 0..HEAD_DIM {
            want_y[hd_i * HEAD_DIM + d] += d_v * x_bf[hd_i * HEAD_DIM + d];
        }
    }
    let got_y = download_bf16(&g, out_dev, D_INNER);
    let got_h = download_f32(&g, h_dev, n_h);
    report("mamba2_ssm_decode y", &want_y, &got_y);
    report("mamba2_ssm_decode H", &want_h, &got_h);
    let _ = h;
}

// ─── gated_rms_norm ─────────────────────────────────────────────
// Args: input[bf16], gate[bf16,gate_stride], weight[bf16], out[bf16],
// hidden_size, eps, gate_stride, group_size(unused — full-hidden RMS).
// out[i] = x[i]*rms(x over hidden)*w[i]*silu(g[i]).
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn gated_rms_norm_oracle() {
    let g = gpu();
    let hidden = D_INNER;
    let eps = 1e-5f32;
    // gate is laid at stride IN_PROJ starting at proj offset 0 (z region):
    // emulate by giving a flat gate buffer of IN_PROJ elems.
    let x = seq(hidden, 0xBB, -1.0, 1.0);
    let mut gate_full = vec![0f32; IN_PROJ];
    gate_full[..hidden].copy_from_slice(&seq(hidden, 0xCC, -1.0, 1.0));
    let w = seq(hidden, 0xDD, 0.5, 1.5);

    let x_dev = upload(&g, &bf16_bytes(&x));
    let g_dev = upload(&g, &bf16_bytes(&gate_full));
    let w_dev = upload(&g, &bf16_bytes(&w));
    let out_dev = upload(&g, &vec![0u8; hidden * 2]);

    let k = g.kernel("norm", "gated_rms_norm").unwrap();
    KernelLaunch::new(&g, k)
        .grid([1, 1, 1])
        .block([hidden.min(1024) as u32, 1, 1])
        .arg_ptr(x_dev)
        .arg_ptr(g_dev)
        .arg_ptr(w_dev)
        .arg_ptr(out_dev)
        .arg_u32(hidden as u32)
        .arg_f32(eps)
        .arg_u32(IN_PROJ as u32)
        .arg_u32(1024)
        .launch(0)
        .unwrap();
    g.synchronize(0).unwrap();

    // HF `MambaRMSNormGated` semantics (what the kernel implements):
    //   t = x * silu(z);  per-group RMS over t (group_size = d_inner/n_groups
    //   = 1024);  y = t * rsqrt(mean(t^2) + eps) * w.
    // (The previous oracle normed x alone over the whole row then multiplied
    // by the gate — the Qwen-style common semantics, not this model's.)
    let x_bf: Vec<f64> = x
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let g_bf: Vec<f64> = gate_full[..hidden]
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let w_bf: Vec<f64> = w
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let t: Vec<f64> = (0..hidden)
        .map(|i| x_bf[i] * (g_bf[i] / (1.0 + (-g_bf[i]).exp())))
        .collect();
    const GROUP: usize = 1024;
    let mut want = vec![0f64; hidden];
    for g0 in (0..hidden).step_by(GROUP) {
        let g1 = (g0 + GROUP).min(hidden);
        let n = (g1 - g0) as f64;
        let ss: f64 = t[g0..g1].iter().map(|v| v * v).sum();
        let rms = 1.0 / (ss / n + eps as f64).sqrt();
        for i in g0..g1 {
            want[i] = t[i] * rms * w_bf[i];
        }
    }
    let got = download_bf16(&g, out_dev, hidden);
    report("gated_rms_norm", &want, &got);
}

// mamba2_ssm_prefill (sequential path): per-token recurrence
// H = dA*H + dt*x*B^T ; y = H.C + D*x, at omni dims, seq_len=19.
#[test]
#[ignore = "needs a GPU and a real kernel build"]
fn mamba2_ssm_prefill_oracle() {
    let g = gpu();
    let n = 19usize;
    let h_elems = HEADS * HEAD_DIM * STATE;
    let h_init = seq(h_elems, 0x5EED, -0.5, 0.5);
    let x = seq(n * D_INNER, 0x12, -1.0, 1.0);
    let b_p = seq(n * GROUPS * STATE, 0x13, -1.0, 1.0);
    let c_p = seq(n * GROUPS * STATE, 0x14, -1.0, 1.0);
    let dt_raw = seq(n * HEADS, 0x15, -1.0, 1.0);
    let a_log = seq(HEADS, 0x16, -4.0, 1.0);
    let d_par = seq(HEADS, 0x17, 0.5, 2.0);
    let dt_bias = seq(HEADS, 0x18, -4.0, 0.5);

    let h_dev = upload(&g, &f32_bytes(&h_init));
    let x_dev = upload(&g, &bf16_bytes(&x));
    let b_dev = upload(&g, &bf16_bytes(&b_p));
    let c_dev = upload(&g, &bf16_bytes(&c_p));
    let dt_dev = upload(&g, &bf16_bytes(&dt_raw));
    let a_dev = upload(&g, &f32_bytes(&a_log));
    let dp_dev = upload(&g, &f32_bytes(&d_par));
    let db_dev = upload(&g, &f32_bytes(&dt_bias));
    let out_dev = upload(&g, &vec![0u8; n * D_INNER * 2]);

    // Strides: x/bc contiguous per token (d_xbc layout), dt contiguous,
    // y contiguous [n, d_inner]. Kernel: mamba2_ssm_prefill.
    let k = g.kernel("mamba2_ssm", "mamba2_ssm_prefill").unwrap();
    KernelLaunch::new(&g, k)
        .grid([HEADS as u32, 1, 1])
        .block([STATE as u32, 1, 1])
        .arg_ptr(h_dev)
        .arg_ptr(x_dev)
        .arg_ptr(b_dev)
        .arg_ptr(c_dev)
        .arg_ptr(dt_dev)
        .arg_ptr(a_dev)
        .arg_ptr(dp_dev)
        .arg_ptr(db_dev)
        .arg_ptr(out_dev)
        .arg_u32(1)
        .arg_u32(n as u32)
        .arg_u32(HEADS as u32)
        .arg_u32(HEAD_DIM as u32)
        .arg_u32(STATE as u32)
        .arg_u32(GROUPS as u32)
        .arg_f32(1e-9)
        .arg_f32(1e9)
        .arg_u32(D_INNER as u32) // x_stride
        .arg_u32((GROUPS * STATE) as u32) // bc_stride
        .arg_u32(HEADS as u32) // dt_stride
        .arg_u32(D_INNER as u32) // y_stride
        .launch(0)
        .unwrap();
    g.synchronize(0).unwrap();

    // CPU f64 reference (sequential recurrence, h_state carried).
    let xb: Vec<f64> = x
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let bb: Vec<f64> = b_p
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let cb: Vec<f64> = c_p
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let dbv: Vec<f64> = dt_raw
        .iter()
        .map(|v| bf16::from_f32(*v).to_f32() as f64)
        .collect();
    let hpg = HEADS / GROUPS;
    // Layouts: x[t*10240 + head*64 + d], B/C[t*10240 + grp*128 + s],
    // dt[t*128 + head] — but x is uploaded as [n, d_inner] flat here, so
    // index x[t*d_inner + head*hd + d] to mirror x_stride=D_INNER.
    let mut hcur: Vec<f64> = h_init.iter().map(|v| *v as f64).collect();
    let mut wy = vec![0f64; n * D_INNER];
    for t in 0..n {
        for hd_i in 0..HEADS {
            let mut dt = dbv[t * HEADS + hd_i] + dt_bias[hd_i] as f64;
            if dt <= 20.0 {
                dt = (1.0 + dt.exp()).ln();
            }
            dt = dt.clamp(1e-9, 1e9);
            let da = (-a_log[hd_i].exp() as f64 * dt).exp();
            let grp = hd_i / hpg;
            for s_i in 0..STATE {
                let bv = bb[t * (GROUPS * STATE) + grp * STATE + s_i];
                let cv = cb[t * (GROUPS * STATE) + grp * STATE + s_i];
                for d_i in 0..HEAD_DIM {
                    let xhd = xb[t * D_INNER + hd_i * HEAD_DIM + d_i];
                    let i = (hd_i * HEAD_DIM + d_i) * STATE + s_i;
                    let hv = da * hcur[i] + dt * xhd * bv;
                    hcur[i] = hv;
                    wy[t * D_INNER + hd_i * HEAD_DIM + d_i] += hv * cv;
                }
            }
            for d_i in 0..HEAD_DIM {
                let xhd = xb[t * D_INNER + hd_i * HEAD_DIM + d_i];
                wy[t * D_INNER + hd_i * HEAD_DIM + d_i] += d_par[hd_i] as f64 * xhd;
            }
        }
    }
    let got_y = download_bf16(&g, out_dev, n * D_INNER);
    let got_h = download_f32(&g, h_dev, h_elems);
    report("mamba2_ssm_prefill y", &wy, &got_y);
    report("mamba2_ssm_prefill H", &hcur, &got_h);
}
