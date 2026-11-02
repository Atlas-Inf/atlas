// SPDX-License-Identifier: AGPL-3.0-only
//! Native KDA integration gate. Run ONLY on an idle GB10 built with native KDA
//! and GLM kernel modules. No FlashKDA Python package or bridge is required.
//! The fast algorithm rounds state internally; these are declared numerical
//! bounds, not bit-exact equivalence to the FP32 recurrence. Full-model/long-
//! context quality remains a separate gate.
#![cfg(feature = "cuda")]
#[path = "arm2_common/support.rs"]
mod support;

use anyhow::{Result, ensure};
use spark_model::layers::ops;
use spark_runtime::{
    glm_kda,
    gpu::{DevicePtr, GpuBackend},
};
use support::*;
const H: usize = 32;
const D: usize = 128;
const P: usize = H * D;
const KEYS: [usize; 2] = [3, 91];
const VALUES: [usize; 2] = [7, 119];

// Two separated active dimensions cross CUDA tiles/warps while keeping an
// independent FP64 oracle. The fixture exercises every
// head, all interleaved Q/K/V strides, beta row/head order and asymmetric state.
struct Fixture {
    qkv: Vec<u16>,
    gate: Vec<u16>,
    beta: Vec<u16>,
    state: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    expected_output: Vec<f64>,
    expected_state: Vec<f64>,
}
impl Fixture {
    fn new(rows: usize) -> Self {
        let mut f = Self {
            qkv: vec![0; rows * 3 * P],
            gate: vec![0; rows * P],
            beta: vec![0; rows * H],
            state: vec![0.0; H * D * D],
            a_log: (0..H).map(|h| (h % 7) as f32 * 0.125 - 0.375).collect(),
            dt_bias: (0..P).map(|i| (i % 13) as f32 * 0.0625 - 0.375).collect(),
            expected_output: vec![0.0; rows * P],
            expected_state: vec![0.0; H * D * D],
        };
        let quant = |x| f32_to_bf16_bits(x);
        let value = |x| bf16_bits_to_f32(x) as f64;
        for h in 0..H {
            let mut s = [[0.0125f64, -0.0375], [0.075, -0.025]];
            for k in 0..2 {
                for v in 0..2 {
                    f.state[h * D * D + KEYS[k] * D + VALUES[v]] = s[k][v] as f32;
                    s[k][v] = f.state[h * D * D + KEYS[k] * D + VALUES[v]] as f64;
                }
            }
            for t in 0..rows {
                let at = t * 3 * P + h * D;
                let tiny = if t % 97 == 0 { 1.0e-8 } else { 1.0 };
                let q = [
                    quant((0.25 + ((t + h) % 7) as f32 * 0.0625) * tiny),
                    quant(-0.375 * tiny),
                ];
                let k = [
                    quant(-0.125 * tiny),
                    quant((0.5 + (h % 3) as f32 * 0.125) * tiny),
                ];
                let v = [quant(((t + 3 * h) % 11) as f32 * 0.125 - 0.5), quant(-0.25)];
                let b = quant(((t + h) % 9) as f32 - 4.0);
                f.beta[t * H + h] = b;
                let b = 1.0 / (1.0 + (-value(b)).exp());
                let qn = (value(q[0]).powi(2) + value(q[1]).powi(2) + 1.0e-6).sqrt();
                let kn = (value(k[0]).powi(2) + value(k[1]).powi(2) + 1.0e-6).sqrt();
                let q = [value(q[0]) / qn, value(q[1]) / qn];
                let k = [value(k[0]) / kn, value(k[1]) / kn];
                for i in 0..2 {
                    // Store original raw operands, never their normalized values.
                    let qraw = if i == 0 {
                        (0.25 + ((t + h) % 7) as f32 * 0.0625) * tiny
                    } else {
                        -0.375 * tiny
                    };
                    let kraw = if i == 0 {
                        -0.125 * tiny
                    } else {
                        (0.5 + (h % 3) as f32 * 0.125) * tiny
                    };
                    f.qkv[at + KEYS[i]] = quant(qraw);
                    f.qkv[at + P + KEYS[i]] = quant(kraw);
                    f.qkv[at + 2 * P + VALUES[i]] = v[i];
                    let g = quant(((h + t + i) % 5) as f32 - 2.0);
                    f.gate[t * P + h * D + KEYS[i]] = g;
                    let logit =
                        (f.a_log[h] as f64).exp() * (value(g) + f.dt_bias[h * D + KEYS[i]] as f64);
                    let decay = (-5.0 / (1.0 + (-logit).exp())).exp();
                    for item in &mut s[i] {
                        *item *= decay;
                    }
                }
                for vi in 0..2 {
                    let delta = b * (value(v[vi]) - k[0] * s[0][vi] - k[1] * s[1][vi]);
                    for ki in 0..2 {
                        s[ki][vi] += k[ki] * delta;
                    }
                    f.expected_output[t * P + h * D + VALUES[vi]] =
                        (q[0] * s[0][vi] + q[1] * s[1][vi]) / (D as f64).sqrt();
                }
            }
            for k in 0..2 {
                for v in 0..2 {
                    f.expected_state[h * D * D + KEYS[k] * D + VALUES[v]] = s[k][v];
                }
            }
        }
        f
    }
}

fn compare(name: &str, actual: &[f64], expected: &[f64], absolute: f64, relative: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: result shape mismatch"
    );
    let (mut e2, mut r2, mut max) = (0.0, 0.0, 0.0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite(), "{name}: nonfinite result");
        e2 += (a - b).powi(2);
        r2 += b * b;
        max = max.max((a - b).abs());
    }
    let rel = (e2 / r2.max(1.0e-30)).sqrt();
    eprintln!("{name}: max_abs={max:e}, relative_rms={rel:e}");
    assert!(
        max <= absolute && rel <= relative,
        "{name}: numerical bound exceeded"
    );
}
fn read_state(gpu: &dyn GpuBackend, ptr: DevicePtr) -> Result<Vec<f64>> {
    Ok(rd_u32(gpu, ptr, H * D * D)?
        .into_iter()
        .map(|b| f32::from_bits(b) as f64)
        .collect())
}

fn run(rows: usize, split: bool) -> Result<()> {
    assert!(
        glm_kda::available(),
        "build Atlas native KDA before running this GPU gate"
    );
    let target =
        atlas_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4").ok_or_else(|| {
            anyhow::anyhow!("build the GLM CUDA target; ATLAS_SKIP_BUILD must be unset")
        })?;
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &target.modules)?;
    ensure!(
        gpu.free_memory()? > 2 * 1024 * 1024 * 1024,
        "native KDA fixture requires 2GiB free on an idle GPU"
    );
    let stream = gpu.create_stream()?;
    let f = Fixture::new(rows);
    let q = up_u16(&gpu, &f.qkv)?;
    let g = up_u16(&gpu, &f.gate)?;
    let b = up_u16(&gpu, &f.beta)?;
    let a = up_f32(&gpu, &f.a_log)?;
    let bias = up_f32(&gpu, &f.dt_bias)?;
    let state = up_f32(&gpu, &f.state)?;
    let reference_state = up_f32(&gpu, &f.state)?;
    let out = gpu.alloc(rows * P * 2)?;
    let reference_out = gpu.alloc(rows * P * 2)?;
    let bytes = glm_kda::ScratchLayout::new(rows).unwrap().bytes as usize;
    let scratch = gpu.alloc(bytes)?;
    let params = glm_kda::Params::new(
        q.0, g.0, b.0, a.0, bias.0, state.0, out.0, scratch.0, bytes, rows, stream,
    )?;
    // SAFETY: all checked operands were allocated on this backend and remain
    // owned until stream completion and all comparisons below.
    unsafe {
        glm_kda::launch(&params)?;
    }
    let reference = gpu.kernel("kda", "kda_recurrent_bf16")?;
    ops::kda_recurrent(
        &gpu,
        reference,
        q,
        g,
        b,
        a,
        bias,
        reference_state,
        reference_out,
        rows as u32,
        H as u32,
        D as u32,
        -5.0,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let fast_bits = rd_u16(&gpu, out, rows * P)?;
    let fast: Vec<f64> = fast_bits
        .iter()
        .map(|&x| bf16_bits_to_f32(x) as f64)
        .collect();
    let native: Vec<f64> = rd_u16(&gpu, reference_out, rows * P)?
        .into_iter()
        .map(|x| bf16_bits_to_f32(x) as f64)
        .collect();
    compare("fast output/FP64", &fast, &f.expected_output, 0.01, 0.05);
    compare("FP32 output/FP64", &native, &f.expected_output, 0.001, 0.01);
    compare("fast output/FP32", &fast, &native, 0.01, 0.05);
    compare(
        "fast state/FP64",
        &read_state(&gpu, state)?,
        &f.expected_state,
        0.03,
        0.05,
    );
    compare(
        "FP32 state/FP64",
        &read_state(&gpu, reference_state)?,
        &f.expected_state,
        0.001,
        0.01,
    );
    if split {
        let split_state = up_f32(&gpu, &f.state)?;
        let split_out = gpu.alloc(rows * P * 2)?;
        for start in [0, 2048] {
            let p = glm_kda::Params::new(
                q.offset(start * 3 * P * 2).0,
                g.offset(start * P * 2).0,
                b.offset(start * H * 2).0,
                a.0,
                bias.0,
                split_state.0,
                split_out.offset(start * P * 2).0,
                scratch.0,
                bytes,
                2048,
                stream,
            )?;
            // SAFETY: disjoint owned slices on the same stream; state carries across pieces.
            unsafe {
                glm_kda::launch(&p)?;
            }
        }
        gpu.synchronize(stream)?;
        assert!(
            fast_bits == rd_u16(&gpu, split_out, rows * P)?,
            "chunk-aligned split output drift"
        );
        assert!(
            rd_u32(&gpu, state, H * D * D)? == rd_u32(&gpu, split_state, H * D * D)?,
            "split state drift"
        );
        gpu.free(split_state)?;
        gpu.free(split_out)?;
    }
    // Continuing both states through the unchanged FP32 one-token decode
    // catches an incorrect persistent-state orientation even after prefill.
    // Use row1, whose Q/K are ordinary scale (row0 is the tiny-norm case).
    let dq = q.offset(3 * P * 2);
    let dg = g.offset(P * 2);
    let db = b.offset(H * 2);
    ops::kda_recurrent(
        &gpu, reference, dq, dg, db, a, bias, state, out, 1, H as u32, D as u32, -5.0, stream,
    )?;
    ops::kda_recurrent(
        &gpu,
        reference,
        dq,
        dg,
        db,
        a,
        bias,
        reference_state,
        reference_out,
        1,
        H as u32,
        D as u32,
        -5.0,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let decode = |p| -> Result<Vec<f64>> {
        Ok(rd_u16(&gpu, p, P)?
            .into_iter()
            .map(|x| bf16_bits_to_f32(x) as f64)
            .collect())
    };
    compare(
        "continued FP32 decode",
        &decode(out)?,
        &decode(reference_out)?,
        0.01,
        0.05,
    );
    for p in [
        q,
        g,
        b,
        a,
        bias,
        state,
        reference_state,
        out,
        reference_out,
        scratch,
    ] {
        gpu.free(p)?;
    }
    gpu.destroy_stream(stream)?;
    Ok(())
}

#[test]
#[ignore = "requires idle GB10, native KDA and GLM kernel build; not a full quality gate"]
fn native_kda_matches_fp64_and_fp32_with_tail_and_continuation() -> Result<()> {
    for rows in [2048, 2051, 4100] {
        run(rows, false)?;
    }
    Ok(())
}
#[test]
#[ignore = "requires idle GB10, native KDA and GLM kernel build"]
fn native_kda_chunk_aligned_split_preserves_state_and_outputs() -> Result<()> {
    run(4096, true)
}
