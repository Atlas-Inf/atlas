// SPDX-License-Identifier: AGPL-3.0-only

//! On-GPU parity of the KDA prefill branch `LingKdaLayer::attention_half`
//! with rows > 1 — the leg the per-kernel harnesses bypass (they drive
//! `kda_recurrent`/norms directly with hand-uploaded features): the GEMM
//! staging into `kda_preconv` (plane-major), the three strided
//! `causal_conv1d_update_prefill` launches, the side GEMMs' [rows, N]
//! layouts, the recurrence over rows, the gated norm, and o_proj + the
//! residual. The reference is the driver itself at rows == 1 — the decode
//! branch is GPU-proven clean (the sibling harness's per-token leg) and is
//! what a single-token prompt takes. `#[ignore]`d:
//!
//! ```sh
//! ATLAS_LING_CKPT=/home/spark/ling/Ling-3.2-air-NVFP4 \
//! cargo test --release -p spark-model --lib \
//!     bailing::layers::kda_prefill_parity_gpu -- --ignored --nocapture
//! ```
//!
//! Every stage of the prefill run (rows in {2, 3}) is compared token by
//! token against the row-by-row decode run: the normed rows, the
//! post-conv q|k|v rows, beta/gate/o_g, the recurrent out, the gated-norm
//! out, o_proj + the residual, and the final h_/conv_ states — plus one
//! more decode token ON TOP of the prefilled state, which fails loudly if
//! prefill leaves the state mispositioned for the next decode. Layers 0
//! (dense-FFN prefix) and 3 (plain MoE layer), REAL weights, REAL driver
//! load path.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, ensure};
use atlas_core::config::parse_config;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::layers_moe_parity_gpu::{Ckpt, gpu_backend, widen, x_rows};
use crate::layer::SsmLayerState;
use crate::model::bailing_c1::{FfnKind, LingKdaLayer, LingKernels, LingScratch, LingSpec};

const HIDDEN: usize = 2560;
const T: usize = 32 * 128;
const H_STATE_ELEMS: usize = 32 * 128 * 128;
const CONV_STATE_ELEMS: usize = 3 * T * 4;

/// One run's per-stage values (bf16 widened to f32; states FP32).
struct Run {
    normed: Vec<f32>,
    qkv: Vec<f32>,
    beta: Vec<f32>,
    gate: Vec<f32>,
    o_g: Vec<f32>,
    rec_out: Vec<f32>,
    o_normed: Vec<f32>,
    attn: Vec<f32>,
    hidden: Vec<f32>,
    h_state: Vec<f32>,
    conv_state: Vec<f32>,
}

/// The compared stages, in driver order: the first strongly divergent one
/// is the bug's stage.
const STAGES: [(&str, fn(&Run) -> &[f32]); 11] = [
    ("normed", |r| &r.normed),
    ("qkv (post-conv)", |r| &r.qkv),
    ("beta", |r| &r.beta),
    ("gate (f)", |r| &r.gate),
    ("o_g (g)", |r| &r.o_g),
    ("kda_out (recurrent)", |r| &r.rec_out),
    ("kda_o (gated norm)", |r| &r.o_normed),
    ("attn_out (o_proj)", |r| &r.attn),
    ("hidden (residual)", |r| &r.hidden),
    ("h_state", |r| &r.h_state),
    ("conv_state", |r| &r.conv_state),
];

fn fresh_state(gpu: &dyn GpuBackend) -> Result<SsmLayerState> {
    let h = gpu.alloc(H_STATE_ELEMS * 4)?;
    gpu.memset(h, 0, H_STATE_ELEMS * 4)?;
    let conv = gpu.alloc(CONV_STATE_ELEMS * 4)?;
    gpu.memset(conv, 0, CONV_STATE_ELEMS * 4)?;
    Ok(SsmLayerState {
        h_state: h,
        conv_state: conv,
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        gdn_commit_qkv: DevicePtr::NULL,
        gdn_commit_gb: DevicePtr::NULL,
        gdn_commit_pending: false,
        gdn_fuse_n: DevicePtr::NULL,
        kda_records: DevicePtr::NULL,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    })
}

#[test]
#[ignore = "needs a GB10, a ling-3.2-air kernel build and ATLAS_LING_CKPT"]
fn ling_kda_prefill_parity_gpu() -> Result<()> {
    let gpu = gpu_backend()?;
    let dir = std::env::var("ATLAS_LING_CKPT")
        .map(Into::into)
        .unwrap_or_else(|_| "/home/spark/ling/Ling-3.2-air-NVFP4".to_owned());
    let config = parse_config(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let spec = LingSpec::from_config(&config)?;
    ensure!(spec.hidden == HIDDEN, "unexpected hidden {}", spec.hidden);
    let kernels = Arc::new(LingKernels::resolve(gpu)?);
    let scratch = LingScratch::arc(gpu, 8, HIDDEN, spec.experts, spec.top_k, 6144, 16)?;
    let mut ckpt = Ckpt::open()?;
    for layer in [0usize, 3usize] {
        let dfn = build_layer(gpu, &mut ckpt, &spec, &kernels, &scratch, layer)?;
        for m in [2usize, 3usize] {
            run_one(gpu, &dfn, &scratch, layer, m)?;
        }
    }
    Ok(())
}

/// Real weights, real loader: `load_kda_weights` + `load_ffn` on a fresh
/// store built from the checkpoint (the MoE layer uploads its 512 experts
/// through the same names the harness of PR #120 uses).
fn build_layer(
    gpu: &dyn GpuBackend,
    ckpt: &mut Ckpt,
    spec: &LingSpec,
    kernels: &Arc<LingKernels>,
    scratch: &Arc<LingScratch>,
    layer: usize,
) -> Result<LingKdaLayer> {
    let lp = format!("model.layers.{layer}");
    let mut names: Vec<String> = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        .iter()
        .map(|s| format!("model.layers.{layer}.{s}"))
        .collect();
    for stem in [
        "q_proj", "k_proj", "v_proj", "b_proj", "f_proj", "g_proj", "o_proj", "q_conv1d",
        "k_conv1d", "v_conv1d", "o_norm",
    ] {
        names.push(format!("{lp}.attention.{stem}.weight"));
    }
    names.push(format!("{lp}.attention.A_log"));
    names.push(format!("{lp}.attention.dt_bias"));
    match spec.layers[layer].ffn {
        FfnKind::Dense => {
            for p in ["gate_proj", "up_proj", "down_proj"] {
                names.push(format!("{lp}.mlp.{p}.weight"));
            }
        }
        FfnKind::Moe => {
            names.push(format!("{lp}.mlp.gate.weight"));
            names.push(format!("{lp}.mlp.gate.expert_bias"));
            for p in ["gate_proj", "up_proj", "down_proj"] {
                names.push(format!("{lp}.mlp.shared_experts.{p}.weight"));
                for e in 0..spec.experts {
                    for part in [
                        "weight_packed",
                        "weight_scale",
                        "weight_global_scale",
                        "input_global_scale",
                    ] {
                        names.push(format!("{lp}.mlp.experts.{e}.{p}.{part}"));
                    }
                }
            }
        }
    }
    let store = upload(gpu, ckpt, &names)?;
    let weights = super::load_kda_weights(&store, &lp, gpu)?;
    let ffn = super::load_ffn(&store, &lp, spec.layers[layer], spec, gpu, kernels, scratch)?;
    let input_norm = crate::weight_map::dense(
        &store,
        &format!("model.layers.{layer}.input_layernorm.weight"),
    )?;
    let post_norm = crate::weight_map::dense(
        &store,
        &format!("model.layers.{layer}.post_attention_layernorm.weight"),
    )?;
    LingKdaLayer::new(
        input_norm,
        post_norm,
        weights,
        ffn,
        Arc::clone(kernels),
        Arc::clone(scratch),
        spec.lower_bound,
        spec.eps,
    )
}

fn upload(gpu: &dyn GpuBackend, ckpt: &mut Ckpt, names: &[String]) -> Result<WeightStore> {
    let mut map = HashMap::new();
    for name in names {
        let (dtype, shape, data) = ckpt.tensor(name)?;
        let ptr = gpu.alloc(data.len().max(1))?;
        gpu.copy_h2d(&data, ptr)?;
        map.insert(
            name.clone(),
            spark_runtime::weights::WeightTensor { ptr, dtype, shape },
        );
    }
    Ok(WeightStore::from_map(map))
}

#[allow(clippy::too_many_lines)]
fn run_one(
    gpu: &dyn GpuBackend,
    layer: &LingKdaLayer,
    scratch: &LingScratch,
    layer_idx: usize,
    m: usize,
) -> Result<()> {
    let stream = gpu.default_stream();
    let x = x_rows(m + 1, layer_idx); // +1 token for the post-prefill leg
    let x_bits: Vec<u16> = x
        .iter()
        .map(|v| atlas_core::numeric::f32_to_bf16(*v))
        .collect();
    let to_bytes =
        |bits: &[u16]| -> Vec<u8> { bits.iter().flat_map(|b| b.to_le_bytes()).collect() };

    // ── Reference: the rows==1 decode branch, token by token.
    let mut ref_state = fresh_state(gpu)?;
    let row_buf = gpu.alloc(HIDDEN * 2)?;
    let mut reference = Vec::new();
    for t in 0..m {
        gpu.copy_h2d(&to_bytes(&x_bits[t * HIDDEN..(t + 1) * HIDDEN]), row_buf)?;
        layer.attention_half(gpu, row_buf, &mut ref_state, 1, stream)?;
        gpu.synchronize(stream)?;
        reference.push(snapshot(gpu, scratch, row_buf, &ref_state, 1)?);
    }

    // ── Prefill branch: one rows=m call on zeroed state.
    let mut pre_state = fresh_state(gpu)?;
    let pre_buf = gpu.alloc(m * HIDDEN * 2)?;
    gpu.copy_h2d(&to_bytes(&x_bits[0..m * HIDDEN]), pre_buf)?;
    layer.attention_half(gpu, pre_buf, &mut pre_state, m as u32, stream)?;
    gpu.synchronize(stream)?;
    let prefill = snapshot(gpu, scratch, pre_buf, &pre_state, m)?;

    // ── Stage-by-stage compare: the first divergent stage is the bug.
    let flat_reference = concat_runs(&reference);
    let mut first_bad: Option<&'static str> = None;
    for (name, get) in STAGES {
        let (got, want) = (get(&prefill), get(&flat_reference));
        let (bit_eq, bad) = cmp_stage(name, got, want);
        eprintln!(
            "layer {layer_idx} rows={m} {name}: {bad} values off-tolerance ({bit_eq}/{} bit-equal)",
            got.len()
        );
        if bad > 0 && first_bad.is_none() {
            first_bad = Some(name);
        }
    }
    ensure!(
        first_bad.is_none(),
        "layer {layer_idx} rows={m}: prefill diverges first at {:?}",
        first_bad.unwrap()
    );

    // ── One more decode token on each ending state.
    let tail = {
        let bytes = to_bytes(&x_bits[m * HIDDEN..(m + 1) * HIDDEN]);
        gpu.copy_h2d(&bytes, row_buf)?;
        layer.attention_half(gpu, row_buf, &mut pre_state, 1, stream)?;
        gpu.synchronize(stream)?;
        let a = snapshot(gpu, scratch, row_buf, &pre_state, 1)?;
        gpu.copy_h2d(&bytes, row_buf)?;
        layer.attention_half(gpu, row_buf, &mut ref_state, 1, stream)?;
        gpu.synchronize(stream)?;
        let b = snapshot(gpu, scratch, row_buf, &ref_state, 1)?;
        (a, b)
    };
    let mut tail_bad: Option<&'static str> = None;
    for (name, get) in STAGES {
        let (a, b) = (get(&tail.0), get(&tail.1));
        let (bit_eq, bad) = cmp_stage(name, a, b);
        eprintln!(
            "layer {layer_idx} rows={m} tail {name}: {bad} values off-tolerance ({bit_eq}/{} bit-equal)",
            a.len()
        );
        if bad > 0 && tail_bad.is_none() {
            tail_bad = Some(name);
        }
    }
    ensure!(
        tail_bad.is_none(),
        "layer {layer_idx} rows={m}: the first decode after prefill diverges at {tail_bad:?}"
    );
    eprintln!("layer {layer_idx} rows={m}: prefill + decode-after-prefill CLEAN");
    Ok(())
}

fn snapshot(
    gpu: &dyn GpuBackend,
    s: &LingScratch,
    hidden: DevicePtr,
    state: &SsmLayerState,
    m: usize,
) -> Result<Run> {
    let bf16 = |p: DevicePtr, n: usize| -> Result<Vec<f32>> {
        let mut b = vec![0u8; n * 2];
        gpu.copy_d2h(p, &mut b)?;
        Ok(b.chunks_exact(2)
            .map(|c| widen(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    };
    let f32s = |p: DevicePtr, n: usize| -> Result<Vec<f32>> {
        let mut b = vec![0u8; n * 4];
        gpu.copy_d2h(p, &mut b)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    Ok(Run {
        normed: bf16(s.normed, m * HIDDEN)?,
        qkv: bf16(s.kda_qkv, m * 3 * T)?,
        beta: bf16(s.kda_beta, m * 32)?,
        gate: bf16(s.kda_gate, m * T)?,
        o_g: bf16(s.kda_o_g, m * T)?,
        rec_out: bf16(s.kda_out, m * T)?,
        o_normed: bf16(s.kda_o, m * T)?,
        attn: bf16(s.attn_out, m * HIDDEN)?,
        hidden: bf16(hidden, m * HIDDEN)?,
        h_state: f32s(state.h_state, H_STATE_ELEMS)?,
        conv_state: f32s(state.conv_state, CONV_STATE_ELEMS)?,
    })
}

/// Token-ordered concatenation of the per-token decode snapshots (the
/// FP32 states are cumulative: the reference run carries the LAST token's
/// state).
fn concat_runs(runs: &[Run]) -> Run {
    let cat =
        |get: fn(&Run) -> &[f32]| -> Vec<f32> { runs.iter().flat_map(get).copied().collect() };
    Run {
        normed: cat(|r| &r.normed),
        qkv: cat(|r| &r.qkv),
        beta: cat(|r| &r.beta),
        gate: cat(|r| &r.gate),
        o_g: cat(|r| &r.o_g),
        rec_out: cat(|r| &r.rec_out),
        o_normed: cat(|r| &r.o_normed),
        attn: cat(|r| &r.attn),
        hidden: cat(|r| &r.hidden),
        // States are cumulative: the concatenated reference carries the
        // LAST token's state.
        h_state: runs.last().map(|r| r.h_state.clone()).unwrap_or_default(),
        conv_state: runs
            .last()
            .map(|r| r.conv_state.clone())
            .unwrap_or_default(),
    }
}

/// `(bit-equal count, off-tolerance count)`. BF16 stages are widened f32s
/// of bf16 words: tolerance ~2.5 bf16 ulps (gemm vs gemv accumulation
/// order); the FP32 states get a small relative band (microscopic cells
/// are noisy — the sibling harness measured ~4e-2 rel on ~4e-9 cells).
///
/// Near-zero cells need an ABSOLUTE band too: when the reference itself
/// lands at ~1e-5 by cancellation (o_proj sums K=4096 bf16 products, and the
/// prefill/decode recurrences accumulate in different orders), the
/// per-element relative band can flag a 32% "error" on a 1e-5 cell whose
/// absolute drift is 5e-6 — one band-width kda_o drift (2.5% of a ~2e-3
/// cell) through one o_proj tap. The floor is 0.5% of the reference
/// stage's RMS, two orders below bug-scale shifts: the conv plane-stride
/// bug moved whole k/v planes by O(stage RMS) — 200x the floor — and every
/// real layout bug shifts whole rows, so the floor cannot mask one.
fn cmp_stage(name: &str, got: &[f32], want: &[f32]) -> (usize, usize) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let state = name == "h_state" || name == "conv_state";
    let rms = (want.iter().map(|w| w * w).sum::<f32>() / want.len() as f32).sqrt();
    let mut bit_eq = 0;
    let mut bad = 0;
    let mut worst: Vec<(f32, usize, f32)> = Vec::new();
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g == w {
            bit_eq += 1;
            continue;
        }
        let scale = w.abs().max(if state { 1e-6 } else { 1e-5 });
        let tol = if state { 0.03 } else { 0.025 };
        // FP32 recurrence states drift further on microscopic cells (the
        // sibling harness measured ~4e-2 rel on ~4e-9 cells) than one
        // band-width-through-a-tap, so their floor is 3% of stage RMS.
        let floor = if state { 3e-2 * rms } else { 5e-3 * rms };
        if (g - w).abs() > (tol * scale).max(floor) {
            bad += 1;
        }
        worst.push(((g - w).abs() / scale, i, *w));
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (rel, i, w) in worst.iter().take(8) {
        let g = got[*i];
        eprintln!(
            "    worst[{name}] #{i}: ref={w:.3e} got={g:.3e} rel={rel:.2e} abs={:.3e} rms={rms:.2e}",
            (g - w).abs()
        );
    }
    (bit_eq, bad)
}
