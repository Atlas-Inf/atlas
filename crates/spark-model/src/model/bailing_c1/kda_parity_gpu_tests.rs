// SPDX-License-Identifier: AGPL-3.0-only

//! On-GPU KDA parity test for Ling-3.2-air: one REAL KDA layer (layer 3)
//! from the checkpoint, driven token by token exactly like
//! `bailing_c1::kda::LingKdaLayer::attention_half` at decode time, against a
//! naive-f64 Rust reference of `BailingMoeV3KimiDeltaAttention`
//! (context/modeling_bailing_moe_v3_vl.py :1237, forward :1301; fla
//! `fused_recurrent_kda` decode semantics: safe-gate decay
//! `exp(lower_bound * sigmoid(exp(A_log) * (f + dt_bias)))` with
//! lower_bound -5.0, q/k L2-norm in recurrence, per-head sigmoid beta, conv1d
//! kernel 4 with fused SiLU, then FusedRMSNormGated o_norm with a sigmoid
//! gate).
//!
//! Per-stage isolation: the reference computes each kernel's output from
//! the GPU's own intermediate bits (downloaded), so a difference measures
//! ONLY that kernel's arithmetic, never upstream drift. BF16 outputs are
//! compared in BF16 ulps (both sides round to the grid; float noise across
//! an accumulation flips at most a rounding boundary). The FP32 recurrent
//! state is compared with a per-element error floored by the tensor's RMS.
//!
//! ```sh
//! ATLAS_LING_CKPT=/home/spark/ling/Ling-3.2-air-NVFP4 \
//! cargo test --release -p spark-model --lib \
//!     bailing_c1::kda_parity_gpu_tests -- --ignored --nocapture --test-threads=1
//! ```

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::{AtlasCudaBackend, configure_pdl};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::kernels::LingKernels;
use super::spec::{KDA_HEAD_DIM, KDA_HEADS};
use crate::layers::ops;
use crate::weight_map::DenseWeight;
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;

const HIDDEN: usize = 2560;
const T: usize = KDA_HEADS * KDA_HEAD_DIM;
const CONV: usize = 4;
/// `rms_norm_eps` of the text config (ground truth over inference).
const EPS: f32 = 1e-6;
/// `kda_lower_bound` of the text config.
const LOWER_BOUND: f32 = -5.0;
/// Layer 3: a KDA layer (the gated-MLA layers are 5, 11, 17, 23, 29, 35, 41).
const LAYER: usize = 3;
const TOKENS: usize = 16;

fn ckpt() -> Result<std::path::PathBuf> {
    Ok(std::env::var("ATLAS_LING_CKPT")
        .map(Into::into)
        .unwrap_or_else(|_| "/home/spark/ling/Ling-3.2-air-NVFP4".into()))
}

fn ling_gpu() -> Result<&'static AtlasCudaBackend> {
    static GPU: OnceLock<Result<AtlasCudaBackend, String>> = OnceLock::new();
    let gpu = GPU.get_or_init(|| {
        configure_pdl("ling-3.2-air");
        let set = atlas_kernels::ptx_for_exact_target("ling-3.2-air", "nvfp4")
            .ok_or_else(|| "ling-3.2-air/nvfp4 absent from this build".to_owned())?;
        AtlasCudaBackend::new(0, &set.modules).map_err(|e| format!("CUDA backend: {e:#}"))
    });
    let gpu = gpu.as_ref().map_err(|e| anyhow::anyhow!(e.clone()))?;
    gpu.bind_to_thread()?;
    Ok(gpu)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Read tensor `name`, check its element count and dtype, keep the host
/// bytes (to upload) and f32 view (for the reference).
fn tensor_f32(dir: &Path, name: &str, elems: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    let (dtype, shape, bytes) = read_checkpoint_tensor(dir, name)?;
    ensure!(
        bytes.len() == elems * 2 && dtype == "BF16" && shape.iter().product::<usize>() == elems,
        "{name}: expected BF16 [{elems}], got {dtype} {shape:?} ({} bytes)",
        bytes.len()
    );
    let vals = bytes
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Ok((bytes, vals))
}

fn tensor_pure_f32(dir: &Path, name: &str, elems: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    let (dtype, shape, bytes) = read_checkpoint_tensor(dir, name)?;
    ensure!(
        bytes.len() == elems * 4 && dtype == "F32" && shape.iter().product::<usize>() == elems,
        "{name}: expected F32 [{elems}], got {dtype} {shape:?} ({} bytes)",
        bytes.len()
    );
    let vals = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((bytes, vals))
}

/// Download `elems` BF16 words from `ptr`.
fn bf16_bits(gpu: &dyn GpuBackend, ptr: DevicePtr, elems: usize) -> Result<Vec<u16>> {
    let mut bytes = vec![0u8; elems * 2];
    gpu.copy_d2h(ptr, &mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// BF16-word vectors' per-element error in BF16 ulps: both sides sit on the
/// grid, so an accumulation-order flip of the pre-rounding value moves a
/// word by whole ulps of the local magnitude. Elements below the vector's
/// noise floor (`vec_max * 1e-5`, the FP32-cancellation scale of a 128-term
/// dot) are measured against the floor, not their own magnitude.
fn bf16_ulp_error(what: &str, got: &[u16], want: &[u16], ulps_allowed: f64) -> Result<f64> {
    ensure!(got.len() == want.len(), "{what}: length mismatch");
    let vec_max = got
        .iter()
        .chain(want)
        .map(|&b| bf16_to_f32(b) as f64)
        .fold(0f64, |m, v| m.max(v.abs()))
        .max(1e-30);
    let floor = vec_max * 1e-5;
    let mut worst = 0f64;
    let mut flipped = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g == w {
            continue;
        }
        flipped += 1;
        let (gf, wf) = (bf16_to_f32(g) as f64, bf16_to_f32(w) as f64);
        let mag = gf.abs().max(wf.abs()).max(floor);
        let ulps = (gf - wf).abs() / (mag * (1f64 / 256.0));
        if ulps > worst {
            worst = ulps;
            if ulps > ulps_allowed {
                ensure!(
                    false,
                    "{what}[{i}]: got {gf:e} want {wf:e} = {ulps:.2} BF16 ulps \
                     (limit {ulps_allowed}); {flipped} words differ"
                );
            }
        }
    }
    eprintln!(
        "{what}: {flipped}/{} words differ, worst {worst:.2} ulps",
        got.len()
    );
    Ok(worst)
}

/// FP32 per-element error floored by the tensor's RMS (recurrence state).
fn f32_rel_error(what: &str, got: &[f32], want: &[f64], rel_allowed: f64) -> Result<f64> {
    ensure!(got.len() == want.len(), "{what}: length mismatch");
    let rms = want.iter().map(|v| v * v).sum::<f64>().sqrt() / want.len() as f64;
    let floor = (rms * 1e-3).max(f64::MIN_POSITIVE);
    let mut worst = 0f64;
    let mut worst_at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let rel = ((g as f64) - w).abs() / w.abs().max(floor);
        if rel > worst {
            worst = rel;
            worst_at = i;
            ensure!(
                rel <= rel_allowed,
                "{what}[{i}]: got {g:e} want {w:e} (rel {rel:.2e}, \
                 limit {rel_allowed:.0e}, rms {rms:.2e})"
            );
        }
    }
    eprintln!("{what}: worst rel {worst:.2e} at {worst_at} (rms {rms:.2e})");
    Ok(worst)
}

// ── naive f64 reference stages (the modeling-file semantics, one per
// GPU kernel, always from the GPU's own input bits). ──────────────────

fn ref_rms_norm(x: &[u16], w: &[f32]) -> Vec<u16> {
    let xf: Vec<f64> = x.iter().map(|&b| bf16_to_f32(b) as f64).collect();
    let ms = xf.iter().map(|v| v * v).sum::<f64>() / HIDDEN as f64;
    let inv = 1.0 / (ms + EPS as f64).sqrt();
    xf.iter()
        .zip(w)
        .map(|(&v, &wv)| atlas_core::numeric::f32_to_bf16((v * wv as f64 * inv) as f32))
        .collect()
}

fn ref_gemv(w: &[f32], x: &[u16], n: usize, k: usize) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let acc: f64 = w[i * k..(i + 1) * k]
                .iter()
                .zip(x.iter().map(|&b| bf16_to_f32(b) as f64))
                .map(|(&wv, xv)| wv as f64 * xv)
                .sum();
            atlas_core::numeric::f32_to_bf16(acc as f32)
        })
        .collect()
}

/// One decode token's conv+SiLU over the raw q|k|v planes, FP32 state.
fn ref_conv_silu(state: &mut Vec<f32>, raw: &[u16], conv: &[f32]) -> Vec<u16> {
    let mut out = vec![0u16; 3 * T];
    for ch in 0..3 * T {
        let (lo, hi) = (ch * CONV, (ch + 1) * CONV);
        state.copy_within(lo + 1..hi, lo);
        state[hi - 1] = bf16_to_f32(raw[ch]);
        let acc: f64 = (0..CONV)
            .map(|j| state[lo + j] as f64 * conv[ch * CONV + j] as f64)
            .sum();
        out[ch] = atlas_core::numeric::f32_to_bf16((acc / (1.0 + (-acc).exp())) as f32);
    }
    out
}

/// The KDA recurrence (fla fused_recurrent_kda semantics) over one token,
/// advancing `h_state` (heads, K, V; V contiguous in each K row).
fn ref_recurrent(
    h_state: &mut [f64],
    qkv: &[u16],
    gate: &[u16],
    beta_raw: &[u16],
    a_log: &[f32],
    dt_bias: &[f32],
) -> Vec<u16> {
    let mut out = vec![0u16; T];
    for h in 0..KDA_HEADS {
        let a = (a_log[h] as f64).exp();
        let (mut qn, mut kn, mut decay) = (
            [0f64; KDA_HEAD_DIM],
            [0f64; KDA_HEAD_DIM],
            [0f64; KDA_HEAD_DIM],
        );
        let (mut sq, mut sk) = (0f64, 0f64);
        for d in 0..KDA_HEAD_DIM {
            let qi = bf16_to_f32(qkv[h * KDA_HEAD_DIM + d]) as f64;
            let ki = bf16_to_f32(qkv[T + h * KDA_HEAD_DIM + d]) as f64;
            sq += qi * qi;
            sk += ki * ki;
            let gi = bf16_to_f32(gate[h * KDA_HEAD_DIM + d]) as f64;
            decay[d] = (LOWER_BOUND as f64
                / (1.0 + (-a * (gi + dt_bias[h * KDA_HEAD_DIM + d] as f64)).exp()))
            .exp();
            qn[d] = qi;
            kn[d] = ki;
        }
        let inv_q = 1.0 / (sq + 1e-6).sqrt() * (KDA_HEAD_DIM as f64).sqrt().recip();
        let inv_k = 1.0 / (sk + 1e-6).sqrt();
        for d in 0..KDA_HEAD_DIM {
            qn[d] *= inv_q;
            kn[d] *= inv_k;
        }
        let beta = 1.0 / (1.0 + (-(bf16_to_f32(beta_raw[h]) as f64)).exp());
        let hs = &mut h_state[h * KDA_HEAD_DIM * KDA_HEAD_DIM..];
        for vv in 0..KDA_HEAD_DIM {
            let mut dot = 0f64;
            for kk in 0..KDA_HEAD_DIM {
                dot += hs[kk * KDA_HEAD_DIM + vv] * decay[kk] * kn[kk];
            }
            let delta = (bf16_to_f32(qkv[2 * T + h * KDA_HEAD_DIM + vv]) as f64 - dot) * beta;
            let mut o = 0f64;
            for kk in 0..KDA_HEAD_DIM {
                let h_new = hs[kk * KDA_HEAD_DIM + vv] * decay[kk] + delta * kn[kk];
                hs[kk * KDA_HEAD_DIM + vv] = h_new;
                o += h_new * qn[kk];
            }
            out[h * KDA_HEAD_DIM + vv] = atlas_core::numeric::f32_to_bf16(o as f32);
        }
    }
    out
}

/// FusedRMSNormGated(128, eps, activation='sigmoid') rows, gated by o_g.
fn ref_gated_norm(rec: &[u16], o_g: &[u16], w: &[f32]) -> Vec<u16> {
    let mut out = vec![0u16; T];
    for h in 0..KDA_HEADS {
        let row: Vec<f64> = (0..KDA_HEAD_DIM)
            .map(|d| bf16_to_f32(rec[h * KDA_HEAD_DIM + d]) as f64)
            .collect();
        let ms = row.iter().map(|v| v * v).sum::<f64>() / KDA_HEAD_DIM as f64;
        let inv = 1.0 / (ms + EPS as f64).sqrt();
        for d in 0..KDA_HEAD_DIM {
            let y = row[d] * inv * w[d] as f64
                / (1.0 + (-(bf16_to_f32(o_g[h * KDA_HEAD_DIM + d]) as f64)).exp());
            out[h * KDA_HEAD_DIM + d] = atlas_core::numeric::f32_to_bf16(y as f32);
        }
    }
    out
}

/// Deterministic BF16 pseudo-activation stream (xorshift, GLM GPU-test
/// house pattern).
fn hidden_stream(tokens: usize) -> Vec<Vec<u16>> {
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15 ^ (TOKENS as u64);
    (0..tokens)
        .map(|_| {
            (0..HIDDEN)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    atlas_core::numeric::f32_to_bf16(
                        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0,
                    )
                })
                .collect()
        })
        .collect()
}

struct Worst(f64);

impl Worst {
    fn keep(&mut self, v: f64) {
        self.0 = self.0.max(v);
    }
}

#[test]
#[ignore = "needs a GB10, a ling-3.2-air kernel build and ATLAS_LING_CKPT"]
fn kda_layer_3_decode_matches_the_naive_reference() -> Result<()> {
    let gpu = ling_gpu()?;
    let stream = gpu.default_stream();
    let k = LingKernels::resolve(gpu)?;
    let dir = ckpt()?;
    let attn = |stem: &str| format!("model.layers.{LAYER}.attention.{stem}");

    // Weights: checkpoint host bytes (asserted shape) uploaded verbatim.
    let proj = |stem: &str, n: usize, h: usize| -> Result<(DevicePtr, Vec<f32>)> {
        let (bytes, vals) = tensor_f32(&dir, &format!("{}.weight", attn(stem)), n * h)?;
        let ptr = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, ptr)?;
        Ok((ptr, vals))
    };
    let (q_w, q) = proj("q_proj", T, HIDDEN)?;
    let (k_w, kk) = proj("k_proj", T, HIDDEN)?;
    let (v_w, vv) = proj("v_proj", T, HIDDEN)?;
    let (b_w, bb) = proj("b_proj", KDA_HEADS, HIDDEN)?;
    let (f_w, ff) = proj("f_proj", T, HIDDEN)?;
    let (g_w, gg) = proj("g_proj", T, HIDDEN)?;
    let (o_w, oo) = proj("o_proj", HIDDEN, T)?;
    let (mut conv_bytes, mut conv) = (Vec::new(), Vec::new());
    for stem in ["q_conv1d", "k_conv1d", "v_conv1d"] {
        let (bytes, vals) = tensor_f32(&dir, &format!("{}.weight", attn(stem)), T * CONV)?;
        conv_bytes.extend_from_slice(&bytes);
        conv.extend_from_slice(&vals);
    }
    ensure!(conv.len() == 3 * T * CONV, "conv blob size");
    let conv_w = gpu.alloc(conv_bytes.len())?;
    gpu.copy_h2d(&conv_bytes, conv_w)?;
    let (input_norm_bytes, input_norm) = tensor_f32(
        &dir,
        &format!("model.layers.{LAYER}.input_layernorm.weight"),
        HIDDEN,
    )?;
    let input_norm_w = gpu.alloc(input_norm_bytes.len())?;
    gpu.copy_h2d(&input_norm_bytes, input_norm_w)?;
    let (o_norm_bytes, o_norm) = tensor_f32(&dir, &attn("o_norm.weight"), KDA_HEAD_DIM)?;
    let o_norm_w = gpu.alloc(o_norm_bytes.len())?;
    gpu.copy_h2d(&o_norm_bytes, o_norm_w)?;
    let (a_log_bytes, a_log) = tensor_pure_f32(&dir, &attn("A_log"), KDA_HEADS)?;
    let a_log_p = gpu.alloc(a_log_bytes.len())?;
    gpu.copy_h2d(&a_log_bytes, a_log_p)?;
    let (dt_bytes, dt_bias) = tensor_pure_f32(&dir, &attn("dt_bias"), T)?;
    let dt_bias_p = gpu.alloc(dt_bytes.len())?;
    gpu.copy_h2d(&dt_bytes, dt_bias_p)?;

    // Scratch, exactly LingScratch's KDA lanes at one decode row.
    let lane = |elems: usize| -> Result<DevicePtr> {
        let p = gpu.alloc(elems * 2)?;
        gpu.memset(p, 0, elems * 2)?;
        Ok(p)
    };
    let normed = lane(HIDDEN)?;
    let kda_qkv = lane(3 * T)?;
    let kda_gate = lane(T)?;
    let kda_o_g = lane(T)?;
    let kda_beta = lane(KDA_HEADS)?;
    let kda_out = lane(T)?;
    let kda_o = lane(T)?;
    let attn_out = lane(HIDDEN)?;
    let hidden = lane(HIDDEN)?;
    let conv_state = gpu.alloc(3 * T * CONV * 4)?;
    gpu.memset(conv_state, 0, 3 * T * CONV * 4)?;
    let h_state = gpu.alloc(KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4)?;
    gpu.memset(h_state, 0, KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4)?;

    let mut ref_conv_state = vec![0f32; 3 * T * CONV];
    let mut ref_h_state = vec![0f64; KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM];

    // Per-token GPU bits kept for the multi-token leg below: the recurrence
    // and gated norm re-run over two rows and must reproduce them exactly.
    let mut t_qkv: Vec<Vec<u16>> = Vec::new();
    let mut t_gate = Vec::new();
    let mut t_beta = Vec::new();
    let mut t_o_g = Vec::new();
    let mut t_out = Vec::new();
    let mut t_norm = Vec::new();
    let mut h_states: Vec<Vec<u8>> = Vec::new();

    let stream_tokens = hidden_stream(TOKENS);
    let mut worst = Worst(0.0);
    for (t, xbits) in stream_tokens.iter().enumerate() {
        eprintln!("── token {t}");
        let xbytes: Vec<u8> = xbits.iter().flat_map(|b| b.to_le_bytes()).collect();
        gpu.copy_h2d(&xbytes, hidden)?;

        // --- the decode-time chain of bailing_c1/kda.rs, verbatim ---
        ops::rms_norm(
            gpu,
            k.rms_norm_k,
            hidden,
            &DenseWeight {
                weight: input_norm_w,
            },
            normed,
            1,
            HIDDEN as u32,
            EPS,
            stream,
        )?;
        for (plane, weight) in [q_w, k_w, v_w].iter().enumerate() {
            ops::dense_gemv(
                gpu,
                k.gemv_k,
                normed,
                &DenseWeight { weight: *weight },
                kda_qkv.offset(plane * T * 2),
                T as u32,
                HIDDEN as u32,
                stream,
            )?;
        }
        ops::dense_gemv(
            gpu,
            k.gemv_k,
            normed,
            &DenseWeight { weight: b_w },
            kda_beta,
            KDA_HEADS as u32,
            HIDDEN as u32,
            stream,
        )?;
        ops::dense_gemv(
            gpu,
            k.gemv_k,
            normed,
            &DenseWeight { weight: f_w },
            kda_gate,
            T as u32,
            HIDDEN as u32,
            stream,
        )?;
        ops::dense_gemv(
            gpu,
            k.gemv_k,
            normed,
            &DenseWeight { weight: g_w },
            kda_o_g,
            T as u32,
            HIDDEN as u32,
            stream,
        )?;

        // Isolate the projections: reference from the GPU's normed row.
        gpu.synchronize(stream)?;
        let normed_bits = bf16_bits(gpu, normed, HIDDEN)?;
        let e = bf16_ulp_error(
            "rms_norm",
            &normed_bits,
            &ref_rms_norm(xbits, &input_norm),
            2.0,
        )?;
        worst.keep(e);
        let raw_qkv = bf16_bits(gpu, kda_qkv, 3 * T)?;
        let mut want_raw = Vec::with_capacity(3 * T);
        for w in [&q, &kk, &vv] {
            want_raw.extend(ref_gemv(w, &normed_bits, T, HIDDEN));
        }
        let e = bf16_ulp_error("q/k/v gemv", &raw_qkv, &want_raw, 2.0)?;
        worst.keep(e);

        // conv1d_update (fused SiLU), in place on the packed planes — the
        // raws are downloaded, so the reference conv sees the same bits.
        ops::conv1d_update(
            gpu,
            k.conv_update_k,
            conv_state,
            kda_qkv,
            &DenseWeight { weight: conv_w },
            kda_qkv,
            (3 * T) as u32,
            CONV as u32,
            1,
            stream,
        )?;
        ops::dense_gemv(
            gpu,
            k.gemv_k,
            normed,
            &DenseWeight { weight: b_w },
            kda_beta,
            KDA_HEADS as u32,
            HIDDEN as u32,
            stream,
        )?;
        ops::dense_gemv(
            gpu,
            k.gemv_k,
            normed,
            &DenseWeight { weight: f_w },
            kda_gate,
            T as u32,
            HIDDEN as u32,
            stream,
        )?;
        ops::dense_gemv(
            gpu,
            k.gemv_k,
            normed,
            &DenseWeight { weight: g_w },
            kda_o_g,
            T as u32,
            HIDDEN as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let qkv_bits = bf16_bits(gpu, kda_qkv, 3 * T)?;
        let want_qkv = ref_conv_silu(&mut ref_conv_state, &raw_qkv, &conv);
        worst.keep(bf16_ulp_error("conv+silu", &qkv_bits, &want_qkv, 2.5)?);
        let beta_bits = bf16_bits(gpu, kda_beta, KDA_HEADS)?;
        let gate_bits = bf16_bits(gpu, kda_gate, T)?;
        let o_g_bits = bf16_bits(gpu, kda_o_g, T)?;
        worst.keep(bf16_ulp_error(
            "b gemv",
            &beta_bits,
            &ref_gemv(&bb, &normed_bits, KDA_HEADS, HIDDEN),
            2.0,
        )?);
        worst.keep(bf16_ulp_error(
            "f gemv",
            &gate_bits,
            &ref_gemv(&ff, &normed_bits, T, HIDDEN),
            2.0,
        )?);
        worst.keep(bf16_ulp_error(
            "g gemv",
            &o_g_bits,
            &ref_gemv(&gg, &normed_bits, T, HIDDEN),
            2.0,
        )?);

        // kda_recurrent_bf16: one token against the reference recurrence.
        ops::kda_recurrent(
            gpu,
            k.kda_recurrent_k,
            kda_qkv,
            kda_gate,
            kda_beta,
            a_log_p,
            dt_bias_p,
            h_state,
            kda_out,
            1,
            KDA_HEADS as u32,
            KDA_HEAD_DIM as u32,
            LOWER_BOUND,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let out_bits = bf16_bits(gpu, kda_out, T)?;
        let want_out = ref_recurrent(
            &mut ref_h_state,
            &qkv_bits,
            &gate_bits,
            &beta_bits,
            &a_log,
            &dt_bias,
        );
        worst.keep(bf16_ulp_error("kda_recurrent", &out_bits, &want_out, 2.5)?);

        // kda_sigmoid_gated_rms_norm.
        ops::kda_sigmoid_gated_norm(
            gpu,
            k.kda_gated_norm_k,
            kda_out,
            kda_o_g,
            o_norm_w,
            kda_o,
            1,
            KDA_HEADS as u32,
            KDA_HEAD_DIM as u32,
            EPS,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let norm_bits = bf16_bits(gpu, kda_o, T)?;
        let want_onorm = ref_gated_norm(&out_bits, &o_g_bits, &o_norm);
        worst.keep(bf16_ulp_error(
            "o_norm (gated rms)",
            &norm_bits,
            &want_onorm,
            2.5,
        )?);
        t_qkv.push(qkv_bits);
        t_gate.push(gate_bits);
        t_beta.push(beta_bits);
        t_o_g.push(o_g_bits);
        t_out.push(out_bits);
        t_norm.push(norm_bits.clone());
        let mut h_bytes = vec![0u8; KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4];
        gpu.copy_d2h(h_state, &mut h_bytes)?;
        h_states.push(h_bytes);

        // o_proj GEMV out of the o_norm rows.
        ops::dense_gemv(
            gpu,
            k.gemv_k,
            kda_o,
            &DenseWeight { weight: o_w },
            attn_out,
            HIDDEN as u32,
            T as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let attn_bits = bf16_bits(gpu, attn_out, HIDDEN)?;
        let want_attn = ref_gemv(&oo, &norm_bits, HIDDEN, T);
        worst.keep(bf16_ulp_error("o_proj gemv", &attn_bits, &want_attn, 2.0)?);
        ensure!(
            attn_bits.iter().all(|b| bf16_to_f32(*b).is_finite()),
            "token {t}: non-finite GPU output"
        );
    }

    // The FP32 recurrence state, elementwise against the f64 reference
    // advanced from the same GPU inputs every step.
    let mut bytes = vec![0u8; KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4];
    gpu.copy_d2h(h_state, &mut bytes)?;
    let got_h: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    worst.keep(f32_rel_error("h_state", &got_h, &ref_h_state, 3.0e-2)?);

    // Multi-token leg: the driver's prefill launches `kda_recurrent_bf16`
    // and the gated norm with rows > 1. Re-run tokens 12 and 13 through a
    // two-row launch starting from the state token 11 left (both bits kept
    // above); the serial loop must reproduce the per-token outputs EXACTLY.
    {
        let (a, b) = (12usize, 13);
        gpu.copy_h2d(&h_states[a - 1], h_state)?;
        let qkv2 = gpu.alloc(2 * 3 * T * 2)?;
        let gate2 = gpu.alloc(2 * T * 2)?;
        let beta2 = gpu.alloc(2 * KDA_HEADS * 2)?;
        let out2 = gpu.alloc(2 * T * 2)?;
        let o_g2 = gpu.alloc(2 * T * 2)?;
        let norm2 = gpu.alloc(2 * T * 2)?;
        for (row, t) in [a, b].iter().enumerate() {
            let to = |src: &[u16], dst: DevicePtr| -> Result<()> {
                let bytes: Vec<u8> = src.iter().flat_map(|w| w.to_le_bytes()).collect();
                gpu.copy_h2d(&bytes, dst.offset(row * src.len() * 2))?;
                Ok(())
            };
            to(&t_qkv[*t], qkv2)?;
            to(&t_gate[*t], gate2)?;
            to(&t_beta[*t], beta2)?;
            to(&t_o_g[*t], o_g2)?;
        }
        ops::kda_recurrent(
            gpu,
            k.kda_recurrent_k,
            qkv2,
            gate2,
            beta2,
            a_log_p,
            dt_bias_p,
            h_state,
            out2,
            2,
            KDA_HEADS as u32,
            KDA_HEAD_DIM as u32,
            LOWER_BOUND,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let pair_out = bf16_bits(gpu, out2, 2 * T)?;
        let mut want_pair = t_out[a].clone();
        want_pair.extend_from_slice(&t_out[b]);
        worst.keep(bf16_ulp_error(
            "kda_recurrent (2 rows)",
            &pair_out,
            &want_pair,
            0.5,
        )?);
        // The two-row recurrence advanced the state past token 13; bring it
        // back so the gated-norm leg pairs row-for-row against the same rows.
        gpu.copy_h2d(&h_states[a - 1], h_state)?;
        ops::kda_sigmoid_gated_norm(
            gpu,
            k.kda_gated_norm_k,
            out2,
            o_g2,
            o_norm_w,
            norm2,
            2,
            KDA_HEADS as u32,
            KDA_HEAD_DIM as u32,
            EPS,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let pair_norm = bf16_bits(gpu, norm2, 2 * T)?;
        let mut want_pair = t_norm[a].clone();
        want_pair.extend_from_slice(&t_norm[b]);
        worst.keep(bf16_ulp_error(
            "o_norm (2 rows)",
            &pair_norm,
            &want_pair,
            0.5,
        )?);
    }

    eprintln!("16-token worst: {} ulps / rel", worst.0);
    ensure!(
        worst.0 < 10.0,
        "layer 3's KDA decode chain diverges from the naive reference"
    );
    Ok(())
}
