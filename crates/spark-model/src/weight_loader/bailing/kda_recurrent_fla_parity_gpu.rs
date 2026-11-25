// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-level parity of `ops::kda_recurrent` (`kda_recurrent_bf16`, the
//! `atlas_kda_recurrent_body` path) against the authoritative fla
//! semantics, on REAL layer-0 and layer-3 weights. `#[ignore]`d:
//!
//! ```sh
//! ATLAS_LING_CKPT=/home/spark/ling/Ling-3.2-air-NVFP4 \
//! cargo test --release -p spark-model --lib \
//!     bailing::layers::kda_recurrent_fla_parity_gpu -- --ignored --nocapture
//! ```
//!
//! The decay contract under test (fla-org/flash-linear-attention,
//! `fla/ops/kda/fused_recurrent.py`, `USE_LOWER_BOUND` branch — identical
//! to the chunk path's gate, whose docstring reads literally
//! `g = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`):
//!
//! ```text
//! a          = exp(A_log[head])                    // NOTE the exp
//! log_decay  = lower_bound * sigmoid(a * (g + dt_bias))
//! H'         = H * exp(log_decay) + beta * (v - (H * exp(log_decay)) k) k^T
//! out        = H' q
//! ```
//!
//! with q/k L2-normalised in kernel (1e-6 eps, 1/sqrt(dim) on q only) and
//! beta = sigmoid(b_proj(x)). The kernel's `lower_bound / (1+expf(-a*...))`
//! form is the same sigmoid written as a quotient.
//!
//! This pins a footgun the e2e oracle hit: a reference that uses `A_log`
//! RAW (missing the `exp`) is exact at token 0 — at H=0 every decay error
//! cancels (H'*decay = 0, delta = v*beta) — and first diverges at token 1
//! with a per-head error proportional to A_log (exp(A_log) in [0.88, 1.74]
//! vs A_log in [-0.12, 0.55] for layer 0), which is why the wrong formula
//! produced "row 1 kda_out up to 1-cos 0.25 per head with clean inputs and
//! clean state". The harness prints per-head 1-cos under BOTH formulas so
//! the discrimination is visible on every run.
//!
//! Inputs are staged host-side from the REAL checkpoint through the same
//! chain a decode row takes (rms-norm -> q/k/v/f/b projections -> 4-tap
//! causal conv1d + SiLU -> packed [q|k|v]), then fed to the kernel in five
//! serial rows=1 launches (the decode path) AND one rows=5 launch (the
//! prefill path's internal loop), which must agree bit-exactly.

use std::sync::Arc;

use anyhow::{Result, ensure};
use atlas_core::config::parse_config;
use atlas_core::numeric::f32_to_bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::layers_moe_parity_gpu::{Ckpt, gpu_backend, widen, x_rows};
use crate::layers::ops;
use crate::model::bailing_c1::{LingKernels, LingSpec};

const HIDDEN: usize = 2560;
const HEADS: usize = 32;
const DIM: usize = 128;
const T: usize = HEADS * DIM;
const H_STATE_ELEMS: usize = HEADS * DIM * DIM;
const CONV: usize = 4;
const ROWS: usize = 5;
/// fla-faithful reference agrees to ~1e-4 per head on real dumps; the
/// raw-A_log (oracle) formula shows up to 0.25. 1e-3 sits 10x above the
/// noise and 250x below the wrong-semantics signal.
const COS_TOL: f32 = 1e-3;

#[test]
#[ignore = "needs a GB10, a ling-3.2-air kernel build and ATLAS_LING_CKPT"]
fn ling_kda_recurrent_fla_parity_gpu() -> Result<()> {
    let gpu = gpu_backend()?;
    let dir = std::env::var("ATLAS_LING_CKPT")
        .map(Into::into)
        .unwrap_or_else(|_| "/home/spark/ling/Ling-3.2-air-NVFP4".to_owned());
    let config = parse_config(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let spec = LingSpec::from_config(&config)?;
    eprintln!("lower_bound = {}", spec.lower_bound);
    let kernels = Arc::new(LingKernels::resolve(gpu)?);
    let mut ckpt = Ckpt::open()?;

    for layer in [0usize, 3usize] {
        let w = LayerWeights::load(&mut ckpt, layer)?;
        run_layer(gpu, &kernels, &w, layer, spec.lower_bound)?;
    }
    Ok(())
}

// ── Real-weight loading ────────────────────────────────────────────────────

struct LayerWeights {
    norm_w: Vec<f32>,
    /// [plane][out=4096][in=2560] q|k|v projections, bf16 widened.
    qkv_proj: [Vec<f32>; 3],
    f_proj: Vec<f32>,
    b_proj: Vec<f32>,
    /// [plane][T][CONV] conv taps.
    conv: [Vec<f32>; 3],
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
}

impl LayerWeights {
    fn load(ckpt: &mut Ckpt, layer: usize) -> Result<Self> {
        let lp = format!("model.layers.{layer}");
        let norm_w = checkpoint_name(ckpt, &format!("{lp}.input_layernorm.weight"))?;
        let f_proj = checkpoint_name(ckpt, &format!("{lp}.attention.f_proj.weight"))?;
        let b_proj = checkpoint_name(ckpt, &format!("{lp}.attention.b_proj.weight"))?;
        let a_log = name_f32(ckpt, &format!("{lp}.attention.A_log"))?;
        let dt_bias = name_f32(ckpt, &format!("{lp}.attention.dt_bias"))?;
        let mut qkv_proj = [Vec::new(), Vec::new(), Vec::new()];
        let mut conv = [Vec::new(), Vec::new(), Vec::new()];
        for (plane, stem) in ["q", "k", "v"].iter().enumerate() {
            qkv_proj[plane] = checkpoint_name(ckpt, &format!("{lp}.attention.{stem}_proj.weight"))?;
            conv[plane] = checkpoint_name(ckpt, &format!("{lp}.attention.{stem}_conv1d.weight"))?;
        }
        Ok(Self {
            norm_w,
            qkv_proj,
            f_proj,
            b_proj,
            conv,
            a_log,
            dt_bias,
        })
    }
}

/// A bf16 checkpoint tensor, flattened (projection matrices are 2-D
/// [out, in]; the conv1d taps may be stored 3-D [T, 1, conv] — either way
/// the bytes are row-major so the flat slice is the layout we index).
fn checkpoint_name(ckpt: &mut Ckpt, name: &str) -> Result<Vec<f32>> {
    let (_, shape, data) = ckpt.tensor(name)?;
    let elems: usize = shape.iter().product();
    ensure!(
        elems * 2 == data.len(),
        "{name}: shape/bytes mismatch {shape:?} vs {}",
        data.len()
    );
    Ok(data
        .chunks_exact(2)
        .map(|c| widen(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn name_f32(ckpt: &mut Ckpt, name: &str) -> Result<Vec<f32>> {
    let (_, shape, data) = ckpt.tensor(name)?;
    ensure!(shape.len() == 1, "{name}: expected 1-D, got {shape:?}");
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// ── Host staging: the decode chain a row takes, bf16-rounded ────────────────

fn stage_rows(w: &LayerWeights, layer: usize) -> Result<RowInputs> {
    let x = x_rows(ROWS, layer);
    let x_bits: Vec<u16> = x.iter().map(|v| f32_to_bf16(*v)).collect();
    let mut conv_state = [[0f64; CONV]; 3 * T];
    let mut out = RowInputs {
        qkv: vec![0u16; ROWS * 3 * T],
        gate: vec![0u16; ROWS * T],
        beta: vec![0u16; ROWS * HEADS],
    };
    for t in 0..ROWS {
        let xb = &x_bits[t * HIDDEN..(t + 1) * HIDDEN];
        let xw: Vec<f64> = xb.iter().map(|b| widen(*b) as f64).collect();
        let sq = xw.iter().map(|v| v * v).sum::<f64>() / HIDDEN as f64;
        let inv = (sq + 1e-6f64).recip().sqrt();
        let n: Vec<f64> = (0..HIDDEN)
            .map(|i| xw[i] * inv * w.norm_w[i] as f64)
            .collect();
        // Projections: raw planes + the recurrence gate and beta.
        let mut raw = [[0u16; T]; 3];
        for (plane, proj) in w.qkv_proj.iter().enumerate() {
            for i in 0..T {
                raw[plane][i] = dot_bf16(proj, i * HIDDEN, &n);
            }
        }
        let mut gate = [0u16; T];
        for i in 0..T {
            gate[i] = dot_bf16(&w.f_proj, i * HIDDEN, &n);
        }
        for h in 0..HEADS {
            out.beta[t * HEADS + h] = dot_bf16(&w.b_proj, h * HIDDEN, &n);
        }
        // 4-tap causal conv1d + SiLU, one persistent window per channel.
        for plane in 0..3 {
            let conv = &w.conv[plane];
            for ch in 0..T {
                let state = &mut conv_state[plane * T + ch];
                state.copy_within(1..CONV, 0);
                state[CONV - 1] = widen(raw[plane][ch]) as f64;
                let taps = &conv[ch * CONV..(ch + 1) * CONV];
                let acc = (0..CONV).map(|k| state[k] * taps[k] as f64).sum::<f64>();
                let silu = acc * (1.0 / (1.0 + (-acc).exp()));
                out.qkv[t * 3 * T + plane * T + ch] = f32_to_bf16(silu as f32);
            }
        }
        out.gate[t * T..(t + 1) * T].copy_from_slice(&gate);
    }
    Ok(out)
}

fn dot_bf16(w: &[f32], off: usize, x: &[f64]) -> u16 {
    let acc: f64 = (0..HIDDEN).map(|i| w[off + i] as f64 * x[i]).sum();
    f32_to_bf16(acc as f32)
}

struct RowInputs {
    /// Packed [row][q|k|v] post-conv bf16 bits.
    qkv: Vec<u16>,
    gate: Vec<u16>,
    beta: Vec<u16>,
}

// ── Host reference: the fla recurrence in f64 ──────────────────────────────

/// One head's step. `a_raw_mode` uses A_log directly (the e2e oracle's
/// bug) instead of exp(A_log); kept for the printed discrimination only.
fn ref_step(
    h_state: &mut [f64],
    qkv: &[u16],
    gate: &[u16],
    beta_raw: &[u16],
    a_log: &[f32],
    dt_bias: &[f32],
    lower_bound: f32,
    a_raw_mode: bool,
) -> Vec<f64> {
    let mut out = vec![0f64; T];
    for h in 0..HEADS {
        let a = if a_raw_mode {
            a_log[h] as f64
        } else {
            (a_log[h] as f64).exp()
        };
        let (mut qn, mut kn, mut decay) = ([0f64; DIM], [0f64; DIM], [0f64; DIM]);
        let (mut sq, mut sk) = (0f64, 0f64);
        for d in 0..DIM {
            let qi = widen(qkv[h * DIM + d]) as f64;
            let ki = widen(qkv[T + h * DIM + d]) as f64;
            sq += qi * qi;
            sk += ki * ki;
            let gi = widen(gate[h * DIM + d]) as f64;
            decay[d] = (lower_bound as f64
                / (1.0 + (-a * (gi + dt_bias[h * DIM + d] as f64)).exp()))
            .exp();
            qn[d] = qi;
            kn[d] = ki;
        }
        let inv_q = (sq + 1e-6).recip().sqrt() * (DIM as f64).sqrt().recip();
        let inv_k = (sk + 1e-6).recip().sqrt();
        for d in 0..DIM {
            qn[d] *= inv_q;
            kn[d] *= inv_k;
        }
        let beta = 1.0 / (1.0 + (-(widen(beta_raw[h]) as f64)).exp());
        let hs = &mut h_state[h * DIM * DIM..];
        for vv in 0..DIM {
            let mut dot = 0f64;
            for kk in 0..DIM {
                dot += hs[kk * DIM + vv] * decay[kk] * kn[kk];
            }
            let delta = (widen(qkv[2 * T + h * DIM + vv]) as f64 - dot) * beta;
            for kk in 0..DIM {
                hs[kk * DIM + vv] = hs[kk * DIM + vv] * decay[kk] + delta * kn[kk];
            }
            let mut o = 0f64;
            for kk in 0..DIM {
                o += hs[kk * DIM + vv] * qn[kk];
            }
            out[h * DIM + vv] = o;
        }
    }
    out
}

// ── The GPU legs ───────────────────────────────────────────────────────────

fn alloc_bf16(gpu: &dyn GpuBackend, bits: &[u16]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

#[allow(clippy::too_many_lines)]
fn run_layer(
    gpu: &dyn GpuBackend,
    kernels: &Arc<LingKernels>,
    w: &LayerWeights,
    layer: usize,
    lower_bound: f32,
) -> Result<()> {
    let stream = gpu.default_stream();
    let inp = stage_rows(w, layer)?;

    let qkv_p = alloc_bf16(gpu, &inp.qkv)?;
    let gate_p = alloc_bf16(gpu, &inp.gate)?;
    let beta_p = alloc_bf16(gpu, &inp.beta)?;
    let a_log_p = {
        let b: Vec<u8> = w.a_log.iter().flat_map(|f| f.to_le_bytes()).collect();
        let p = gpu.alloc(b.len())?;
        gpu.copy_h2d(&b, p)?;
        p
    };
    let dt_bias_p = {
        let b: Vec<u8> = w.dt_bias.iter().flat_map(|f| f.to_le_bytes()).collect();
        let p = gpu.alloc(b.len())?;
        gpu.copy_h2d(&b, p)?;
        p
    };
    let out_p = gpu.alloc(ROWS * T * 2)?;

    // Leg A: five serial rows=1 launches (the decode path), advancing one
    // f64 fla-reference state in lockstep.
    let fresh_state = |gpu: &dyn GpuBackend| -> Result<DevicePtr> {
        let p = gpu.alloc(H_STATE_ELEMS * 4)?;
        gpu.memset(p, 0, H_STATE_ELEMS * 4)?;
        Ok(p)
    };
    let mut ref_fla = vec![0f64; H_STATE_ELEMS];
    let mut ref_raw = vec![0f64; H_STATE_ELEMS];
    let mut serial_out = vec![0u16; ROWS * T];
    // ONE persistent state buffer, five launches — how decode advances it.
    let state_a = fresh_state(gpu)?;
    for t in 0..ROWS {
        ops::kda_recurrent(
            gpu,
            kernels.kda_recurrent_k,
            qkv_p.offset(t * 3 * T * 2),
            gate_p.offset(t * T * 2),
            beta_p.offset(t * HEADS * 2),
            a_log_p,
            dt_bias_p,
            state_a,
            out_p.offset(t * T * 2),
            1,
            HEADS as u32,
            DIM as u32,
            lower_bound,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut bits = vec![0u8; T * 2];
        gpu.copy_d2h(out_p.offset(t * T * 2), &mut bits)?;
        let row_bits = bytemuck_bits(&bits);
        serial_out[t * T..(t + 1) * T].copy_from_slice(&row_bits);

        let want_fla = ref_step(
            &mut ref_fla,
            &inp.qkv[t * 3 * T..(t + 1) * 3 * T],
            &inp.gate[t * T..(t + 1) * T],
            &inp.beta[t * HEADS..(t + 1) * HEADS],
            &w.a_log,
            &w.dt_bias,
            lower_bound,
            false,
        );
        let got: Vec<f32> = serial_out[t * T..(t + 1) * T]
            .iter()
            .map(|b| widen(*b))
            .collect();
        let mut worst_fla = 0f32;
        for h in 0..HEADS {
            let c = head_cos(
                &got[h * DIM..(h + 1) * DIM],
                &want_fla[h * DIM..(h + 1) * DIM],
            );
            worst_fla = worst_fla.max(c);
        }
        ensure!(
            worst_fla <= COS_TOL,
            "layer {layer} row {t}: fla-semantics per-head 1-cos {worst_fla:.2e} > {COS_TOL}"
        );
        // The raw-A_log (oracle-bug) formula, printed only: it must NOT be
        // under tolerance from row 1 on, or this harness lost its teeth.
        let want_raw = ref_step(
            &mut ref_raw,
            &inp.qkv[t * 3 * T..(t + 1) * 3 * T],
            &inp.gate[t * T..(t + 1) * T],
            &inp.beta[t * HEADS..(t + 1) * HEADS],
            &w.a_log,
            &w.dt_bias,
            lower_bound,
            true,
        );
        let mut worst_raw = 0f32;
        for h in 0..HEADS {
            let c = head_cos(
                &got[h * DIM..(h + 1) * DIM],
                &want_raw[h * DIM..(h + 1) * DIM],
            );
            worst_raw = worst_raw.max(c);
        }
        eprintln!(
            "layer {layer} row {t}: fla max per-head 1-cos {worst_fla:.2e}   \
             raw-A_log (oracle bug) {worst_raw:.2e}"
        );
    }

    // Leg B: ONE rows=5 launch (the prefill loop) must reproduce leg A
    // bit-exactly — same kernel, same inputs, same fp32 order.
    let state_b = fresh_state(gpu)?;
    let out_b = gpu.alloc(ROWS * T * 2)?;
    ops::kda_recurrent(
        gpu,
        kernels.kda_recurrent_k,
        qkv_p,
        gate_p,
        beta_p,
        a_log_p,
        dt_bias_p,
        state_b,
        out_b,
        ROWS as u32,
        HEADS as u32,
        DIM as u32,
        lower_bound,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let mut bits = vec![0u8; ROWS * T * 2];
    gpu.copy_d2h(out_b, &mut bits)?;
    let batch_out = bytemuck_bits(&bits);
    ensure!(
        batch_out == serial_out,
        "layer {layer}: rows=5 batch launch != five serial rows=1 launches"
    );
    eprintln!("layer {layer}: rows=5 batch == five serial launches, bit-exact");
    Ok(())
}

fn bytemuck_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn head_cos(got: &[f32], want: &[f64]) -> f32 {
    let (mut dot, mut ng, mut nw) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        dot += *g as f64 * w;
        ng += (*g as f64) * (*g as f64);
        nw += w * w;
    }
    (1.0 - dot / (ng.sqrt() * nw.sqrt() + 1e-30)) as f32
}
