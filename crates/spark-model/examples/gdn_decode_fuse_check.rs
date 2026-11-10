// SPDX-License-Identifier: AGPL-3.0-only
//! Byte-for-byte gate for `ATLAS_QWEN38_DECODE_FUSE` at Qwen3.8-27B GDN shapes
//! (nk 16, nv 48, head dim 128, d_conv 4, BA input 5120).
//!
//! CHAIN is serial decode's single-token GDN step as `ssm_forward` issues it:
//! `dense_gemv_ba_gates`, `causal_conv1d_update_l2norm_f32`,
//! `gated_delta_rule_decode_f32`, `gated_rms_norm_f32_input` (the default arm:
//! FP32 conv and h-state, no `--gdn-fused-norm`). FUSED is
//! `ops::qwen38_decode_fuse::gdn_decode` (`qwen35_gdn_decode_fused`), the
//! same host dispatch the layer runs. Both start from the same state and are
//! fed the same token rows; every step the gates, betas, conv window and
//! normed output, and every 8 steps (and at the end) the recurrence state,
//! must be BYTE-IDENTICAL. Then both are captured as CUDA graphs of
//! `STEPS` steps, replayed, compared again (so a PDL hazard under
//! `ATLAS_QWEN38_PDL_DECODE=1` shows up), and timed.
//!
//! GPU REQUIRED. On the GB10, with the qwen3.8-27b kernel set:
//!   ATLAS_TARGET_MODEL=qwen3.8-27b cargo run --release -p spark-model \
//!       --features cuda,gpu-examples --example gdn_decode_fuse_check
//! and again with `ATLAS_QWEN38_PDL_DECODE=1` in front for the PDL launches.
//! Exit codes: 0 = pass, 1 = FAIL, 2 = skipped (the fused kernel is not in
//! this build's kernel set — the gate did NOT run).
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::layers::ops::qwen4exp_decode_fuse::GdnDecode;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::{AtlasCudaBackend, configure_pdl, pdl_enabled};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const D_CONV: usize = 4;
const BA_K: usize = 5120;
const KEY_DIM: usize = NK * KD;
const VALUE_DIM: usize = NV * VD;
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;
const QKVZ: usize = CONV_DIM + VALUE_DIM;
const H_ELEMS: usize = NV * KD * VD;
const CONV_ELEMS: usize = CONV_DIM * D_CONV;
const L2_EPS: f32 = 1e-6;
/// Qwen3.8-27B `rms_norm_eps`.
const EPS: f32 = 1e-6;
const SEEDS: u64 = 8;
const STEPS: usize = 32;
const REPLAYS: usize = 20;

struct Lcg(u64);
impl Lcg {
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)
    }
    fn bf16(&mut self, n: usize, lo: f64, hi: f64) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f64(self.r(lo, hi)).to_bits().to_le_bytes())
            .collect()
    }
    fn f32(&mut self, n: usize, lo: f64, hi: f64) -> Vec<u8> {
        (0..n)
            .flat_map(|_| (self.r(lo, hi) as f32).to_le_bytes())
            .collect()
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len())?;
    g.copy_h2d(b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// The layer's weights and the token rows, shared by both twins.
struct Shared {
    qkvz: DevicePtr,  // [STEPS, QKVZ] BF16: Q | K | V | Z
    ba_in: DevicePtr, // [STEPS, BA_K] BF16 (the mixer's normed input)
    ba_w: DevicePtr,  // [2 * NV, BA_K] BF16
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    conv_w: DevicePtr,  // [CONV_DIM, D_CONV] BF16
    norm_w: DevicePtr,  // [VD] BF16
    scratch: DevicePtr, // chain only: FP32 conv out | FP32 GDN out
    h0: Vec<u8>,
    conv0: Vec<u8>,
}

impl Shared {
    fn new(g: &dyn GpuBackend, seed: u64) -> Result<Self> {
        let mut r = Lcg(seed);
        let mut qkvz = Vec::with_capacity(STEPS * QKVZ * 2);
        for _ in 0..STEPS {
            qkvz.extend(r.bf16(CONV_DIM, -1.0, 1.0));
            qkvz.extend(r.bf16(VALUE_DIM, -3.0, 3.0));
        }
        Ok(Self {
            qkvz: up(g, &qkvz)?,
            ba_in: up(g, &r.bf16(STEPS * BA_K, -2.0, 2.0))?,
            ba_w: up(g, &r.bf16(2 * NV * BA_K, -0.03, 0.03))?,
            a_log: up(g, &r.f32(NV, -1.0, 1.5))?,
            dt_bias: up(g, &r.f32(NV, -1.0, 1.0))?,
            conv_w: up(g, &r.bf16(CONV_ELEMS, -0.5, 0.5))?,
            norm_w: up(g, &r.bf16(VD, 0.5, 1.5))?,
            scratch: g.alloc((CONV_DIM + VALUE_DIM) * 4)?,
            h0: r.f32(H_ELEMS, -0.1, 0.1),
            conv0: r.f32(CONV_ELEMS, -0.5, 0.5),
        })
    }
    fn qkvz_row(&self, t: usize) -> DevicePtr {
        self.qkvz.offset(t * QKVZ * 2)
    }
    fn ba_row(&self, t: usize) -> DevicePtr {
        self.ba_in.offset(t * BA_K * 2)
    }
}

/// One twin's state and outputs.
struct Twin {
    h: DevicePtr,
    conv: DevicePtr,
    gates: DevicePtr, // [gate NV | beta NV] FP32
    out: DevicePtr,   // [VALUE_DIM] BF16
}

impl Twin {
    fn new(g: &dyn GpuBackend, sh: &Shared) -> Result<Self> {
        Ok(Self {
            h: up(g, &sh.h0)?,
            conv: up(g, &sh.conv0)?,
            gates: g.alloc(2 * NV * 4)?,
            out: g.alloc(VALUE_DIM * 2)?,
        })
    }
    fn reset(&self, g: &dyn GpuBackend, sh: &Shared) -> Result<()> {
        g.copy_h2d(&sh.h0, self.h)?;
        g.copy_h2d(&sh.conv0, self.conv)
    }
    /// gates+betas, conv window, normed output; with the h-state when `h`.
    fn bytes(&self, g: &dyn GpuBackend, h: bool) -> Result<[Vec<u8>; 4]> {
        Ok([
            dn(g, self.gates, 2 * NV * 4)?,
            dn(g, self.conv, CONV_ELEMS * 4)?,
            dn(g, self.out, VALUE_DIM * 2)?,
            if h {
                dn(g, self.h, H_ELEMS * 4)?
            } else {
                Vec::new()
            },
        ])
    }
}

struct Chain {
    ba: KernelHandle,
    conv: KernelHandle,
    gdn: KernelHandle,
    norm: KernelHandle,
}

impl Chain {
    /// `ssm_forward` steps 3-7 on the four-kernel FP32 arm.
    fn step(&self, g: &dyn GpuBackend, sh: &Shared, tw: &Twin, t: usize, s: u64) -> Result<()> {
        let w = |weight| DenseWeight { weight };
        let beta = tw.gates.offset(NV * 4);
        let (n, k, vpg) = ((2 * NV) as u32, BA_K as u32, (NV / NK) as u32);
        let ba_w = w(sh.ba_w);
        ops::dense_gemv_ba_gates(
            g,
            self.ba,
            sh.ba_row(t),
            &ba_w,
            sh.a_log,
            sh.dt_bias,
            tw.gates,
            beta,
            n,
            k,
            vpg,
            s,
        )?;
        let conv_out = sh.scratch;
        ops::conv1d_update_l2norm(
            g,
            self.conv,
            tw.conv,
            sh.qkvz_row(t),
            &w(sh.conv_w),
            conv_out,
            CONV_DIM as u32,
            D_CONV as u32,
            1,
            (2 * KEY_DIM) as u32,
            KD as u32,
            L2_EPS,
            s,
        )?;
        let gdn_out = sh.scratch.offset(CONV_DIM * 4);
        ops::gdn_decode(
            g,
            self.gdn,
            tw.h,
            conv_out,
            conv_out.offset(KEY_DIM * 4),
            conv_out.offset(2 * KEY_DIM * 4),
            tw.gates,
            beta,
            gdn_out,
            1,
            NK as u32,
            NV as u32,
            KD as u32,
            VD as u32,
            s,
        )?;
        ops::gated_rms_norm(
            g,
            self.norm,
            gdn_out,
            sh.qkvz_row(t).offset(CONV_DIM * 2),
            &w(sh.norm_w),
            tw.out,
            NV as u32,
            VD as u32,
            VD as u32,
            EPS,
            VD as u32,
            s,
        )
    }
}

/// The fused step through the layer's own dispatch; errors if it declines.
fn fused_step(g: &dyn GpuBackend, sh: &Shared, tw: &Twin, t: usize, s: u64) -> Result<()> {
    let b = GdnDecode {
        h_state: tw.h,
        conv_state: tw.conv,
        qkvz: sh.qkvz_row(t),
        conv_w: sh.conv_w,
        ba_in: sh.ba_row(t),
        ba_w: sh.ba_w,
        a_log: sh.a_log,
        dt_bias: sh.dt_bias,
        gate_out: tw.gates,
        beta_out: tw.gates.offset(NV * 4),
        norm_w: sh.norm_w,
        out: tw.out,
    };
    let (nk, nv, kd, vd) = (NK as u32, NV as u32, KD as u32, VD as u32);
    let ran = ops::qwen38_decode_fuse::gdn_decode(
        g,
        &b,
        nk,
        nv,
        kd,
        vd,
        D_CONV as u32,
        BA_K as u32,
        L2_EPS,
        EPS,
        s,
    )?;
    ensure!(ran, "qwen38_decode_fuse::gdn_decode declined the 27B shape");
    Ok(())
}

const NAMES: [&str; 4] = ["gates/betas", "conv window", "normed out", "h-state"];

/// Count of differing bytes per output, first mismatch reported.
fn compare(what: &str, a: &[Vec<u8>; 4], b: &[Vec<u8>; 4]) -> bool {
    let mut same = true;
    for (i, name) in NAMES.iter().enumerate() {
        if let Some(at) = a[i].iter().zip(&b[i]).position(|(x, y)| x != y) {
            let diff = a[i].iter().zip(&b[i]).filter(|(x, y)| x != y).count();
            eprintln!("  MISMATCH {what}: {name}: {diff} bytes differ, first at byte {at}");
            same = false;
        }
    }
    same
}

/// Graph of `STEPS` steps of `step` on `stream`, replayed `REPLAYS` times
/// from the twin's initial state; returns (bytes, us per step).
fn graphed(
    g: &dyn GpuBackend,
    sh: &Shared,
    tw: &Twin,
    stream: u64,
    step: &dyn Fn(usize, u64) -> Result<()>,
) -> Result<([Vec<u8>; 4], f64)> {
    tw.reset(g, sh)?;
    g.begin_capture(stream)?;
    let captured = (0..STEPS).try_for_each(|t| step(t, stream));
    if let Err(e) = captured {
        g.abort_capture_if_active(stream);
        return Err(e);
    }
    let graph = g.end_capture(stream)?;
    g.launch_graph(graph, stream)?; // warm-up
    g.synchronize(stream)?;
    tw.reset(g, sh)?;
    let start = std::time::Instant::now();
    for _ in 0..REPLAYS {
        g.launch_graph(graph, stream)?;
    }
    g.synchronize(stream)?;
    let us = start.elapsed().as_secs_f64() * 1e6 / (REPLAYS * STEPS) as f64;
    g.destroy_graph(graph)?;
    Ok((tw.bytes(g, true)?, us))
}

fn main() -> Result<()> {
    // Before anything reads the environment: the fused step's lever.
    // SAFETY: single-threaded here; nothing else reads or writes the env yet.
    unsafe { std::env::set_var("ATLAS_QWEN38_DECODE_FUSE", "1") };
    let set = atlas_kernels::ptx_for_model("qwen3.8-27b")
        .or_else(|| atlas_kernels::ptx_for_model("qwen3.6-27b"))
        .expect("no qwen3.8-27b or qwen3.6-27b kernel set in this build");
    // PDL marks kernels at lookup, so the target is recorded first.
    configure_pdl(set.target.model);
    eprintln!(
        "kernel set: {}  PDL launches: {}",
        set.target.model,
        pdl_enabled()
    );
    let gpu = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &gpu;
    if g.kernel("qwen35_decode_fuse", "qwen35_gdn_decode_fused")
        .is_err()
    {
        eprintln!("SKIPPED: qwen35_gdn_decode_fused is not in this kernel set");
        std::process::exit(2);
    }
    let chain = Chain {
        ba: g.kernel("ssm_preprocess", "dense_gemv_ba_gates")?,
        conv: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32")?,
        gdn: g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32")?,
        norm: g.kernel("norm", "gated_rms_norm_f32_input")?,
    };
    let stream = g.create_stream()?;
    let mut ok = true;
    let (mut chain_us, mut fused_us) = (Vec::new(), Vec::new());
    for seed in 0..SEEDS {
        let sh = Shared::new(g, 0x27B0_0000 + seed)?;
        let (a, b) = (Twin::new(g, &sh)?, Twin::new(g, &sh)?);
        // Eager, step by step.
        for t in 0..STEPS {
            chain.step(g, &sh, &a, t, 0)?;
            fused_step(g, &sh, &b, t, 0)?;
            g.synchronize(0)?;
            let h = t % 8 == 7 || t + 1 == STEPS;
            if !compare(
                &format!("seed {seed} step {t}"),
                &a.bytes(g, h)?,
                &b.bytes(g, h)?,
            ) {
                ok = false;
                break;
            }
        }
        // Sanity: the step did something (a wiring bug can match trivially).
        let [_, _, out, h] = b.bytes(g, true)?;
        let nonzero = out.chunks_exact(2).filter(|c| c != &[0, 0]).count();
        if nonzero * 10 < VALUE_DIM * 9 || h == sh.h0 {
            eprintln!(
                "  seed {seed}: degenerate step ({nonzero} nonzero outputs, h moved: {})",
                h != sh.h0
            );
            ok = false;
        }
        // Graphs (PDL edges when the decode switch is on).
        let (ga, ta) = graphed(g, &sh, &a, stream, &|t, s| chain.step(g, &sh, &a, t, s))?;
        let (gb, tb) = graphed(g, &sh, &b, stream, &|t, s| fused_step(g, &sh, &b, t, s))?;
        ok &= compare(&format!("seed {seed} graph replays"), &ga, &gb);
        chain_us.push(ta);
        fused_us.push(tb);
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|x, y| x.total_cmp(y));
        v[v.len() / 2]
    };
    let (c, f) = (median(&mut chain_us), median(&mut fused_us));
    println!(
        "graph us per GDN step (median over {SEEDS} seeds): chain {c:.2}, fused {f:.2}; \
         x48 GDN layers = {:.0} us saved per decode step",
        (c - f) * 48.0
    );
    if ok {
        println!(
            "PASS: qwen35_gdn_decode_fused is byte-identical to the four-kernel chain \
             ({SEEDS} seeds x {STEPS} steps, eager and graphed)"
        );
        Ok(())
    } else {
        println!("FAIL: the fused GDN decode step differs from the four-kernel chain");
        std::process::exit(1);
    }
}
