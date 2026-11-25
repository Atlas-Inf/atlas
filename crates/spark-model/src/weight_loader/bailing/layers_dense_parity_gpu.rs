// SPDX-License-Identifier: AGPL-3.0-only

//! On-GPU parity of the dense-MLP FFN of Ling-3.2-air layers 0-1 — the only
//! FFN path no parity harness has covered: plain BF16
//! `model.layers.{0,1}.mlp.{gate,up,down}_proj.weight`, intermediate 6144,
//! no routed experts. Per the family driver the dense FFN reuses the
//! `ling_moe_shared_*` kernels on the 32-row bucket split with no swiglu
//! limit (`bailing_c1/ffn.rs` `forward_dense`), loaded through the real
//! `load_ffn`. `#[ignore]`d — needs a GB10, a `ling-3.2-air` kernel build
//! and the real checkpoint:
//!
//! ```sh
//! ATLAS_LING_CKPT=/home/spark/ling/Ling-3.2-air-NVFP4 \
//! cargo test --release -p spark-model --lib \
//!     bailing::layers::layers_dense_parity_gpu -- --ignored --nocapture
//! ```
//!
//! Stages: H = clamped-SwiGLU-less gate/up + silu (GPU vs the host
//! reference of `BailingMoeV3MLP.forward` with `swiglu_limit=None`,
//! modeling_bailing_moe_v3_vl.py:421-428), then Y = down. Batches M=1
//! (decode), M=8 (verify-shaped) and M=33 (crosses the 32-row sub-batch
//! seam: one MR32 bucket then one MR1 bucket).

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use atlas_core::config::parse_config;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightStore, WeightTensor};

use super::layers_moe_parity_gpu::{Ckpt, bf16, gpu_backend, silu_clamped, widen, x_rows};
use crate::model::bailing_c1::{FfnKind, LingKernels, LingScratch, LingSpec};

const HIDDEN: usize = 2560;
const DENSE_INTER: usize = 6144;

#[test]
#[ignore = "needs a GB10, a ling-3.2-air kernel build and ATLAS_LING_CKPT"]
fn ling_dense_ffn_parity_gpu() -> Result<()> {
    let gpu = gpu_backend()?;
    let dir: PathBuf = std::env::var("ATLAS_LING_CKPT")
        .map(Into::into)
        .unwrap_or_else(|_| "/home/spark/ling/Ling-3.2-air-NVFP4".into());
    let config = parse_config(&std::fs::read_to_string(dir.join("config.json"))?)?;
    ensure!(
        config.hidden_size == HIDDEN && config.intermediate_size == DENSE_INTER,
        "unexpected dense-MLP shapes: {} x {}",
        config.hidden_size,
        config.intermediate_size
    );
    let spec = LingSpec::from_config(&config)?;
    let kernels = std::sync::Arc::new(LingKernels::resolve(gpu)?);
    // inter = the dense intermediate: shared_h is [rows, inter] and the
    // dense path is what needs the WIDE bucket (rows >= 33 for the seam).
    let scratch = LingScratch::arc(gpu, 40, HIDDEN, spec.experts, spec.top_k, DENSE_INTER, 16)?;
    let mut ckpt = Ckpt::open()?;
    for layer in [0usize, 1usize] {
        let layer_spec = spec.layers[layer];
        ensure!(
            layer_spec.ffn == FfnKind::Dense,
            "layer {layer} is not a dense-MLP layer in the plan"
        );
        parity_one_layer(
            gpu,
            &mut ckpt,
            &spec,
            &kernels,
            &scratch,
            &format!("model.layers.{layer}"),
            layer,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn parity_one_layer(
    gpu: &dyn GpuBackend,
    ckpt: &mut Ckpt,
    spec: &LingSpec,
    kernels: &std::sync::Arc<LingKernels>,
    scratch: &std::sync::Arc<LingScratch>,
    lp: &str,
    layer: usize,
) -> Result<()> {
    // One read per tensor: raw bytes to the store the production loader
    // reads, widened f32s for the host reference.
    let mut map = HashMap::new();
    let (mut gate, mut up, mut down) = (Vec::new(), Vec::new(), Vec::new());
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let name = format!("{lp}.mlp.{proj}.weight");
        let (dt, sh, data) = ckpt.tensor(&name)?;
        let expected: Vec<usize> = if proj == "down_proj" {
            vec![HIDDEN, DENSE_INTER]
        } else {
            vec![DENSE_INTER, HIDDEN]
        };
        ensure!(
            dt == spark_runtime::weights::WeightDtype::BF16 && sh == expected,
            "{name}: {dt:?} {sh:?}"
        );
        let host: Vec<f32> = data
            .chunks_exact(2)
            .map(|c| widen(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let ptr = gpu.alloc(data.len())?;
        gpu.copy_h2d(&data, ptr)?;
        map.insert(
            name,
            WeightTensor {
                ptr,
                dtype: dt,
                shape: sh,
            },
        );
        match proj {
            "gate_proj" => gate = host,
            "up_proj" => up = host,
            _ => down = host,
        }
    }
    let store = WeightStore::from_map(map);

    let ffn = super::load_ffn(&store, lp, spec.layers[layer], spec, gpu, kernels, scratch)?;

    let stream = gpu.default_stream();
    for m in [1usize, 8, 33] {
        let x = x_rows(m, layer);
        let x_bits: Vec<u16> = x
            .iter()
            .map(|v| atlas_core::numeric::f32_to_bf16(*v))
            .collect();
        let x_bytes: Vec<u8> = x_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let x_dev = gpu.alloc(x_bytes.len())?;
        gpu.copy_h2d(&x_bytes, x_dev)?;
        let out = gpu.alloc(m * HIDDEN * 2)?;
        gpu.memset(out, 0, m * HIDDEN * 2)?;
        ffn.forward(gpu, x_dev, out, m, stream)?;
        gpu.synchronize(stream)?;
        compare_stages(gpu, scratch, layer, &gate, &up, &down, &x, out, m)?;
        gpu.free(x_dev)?;
        gpu.free(out)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compare_stages(
    gpu: &dyn GpuBackend,
    scratch: &LingScratch,
    layer: usize,
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    x: &[f32],
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    let d2h = |gpu: &dyn GpuBackend, p: DevicePtr, n: usize| -> Result<Vec<u8>> {
        let mut b = vec![0u8; n];
        gpu.copy_d2h(p, &mut b)?;
        Ok(b)
    };
    let h_b = d2h(gpu, scratch.shared_h, m * DENSE_INTER * 2)?;
    let out_b = d2h(gpu, out, m * HIDDEN * 2)?;

    // Reference H: bf16(silu(bf16(g_dot)) * bf16(u_dot)) — the modeling
    // chain with swiglu_limit None (ling_gate_up_mul's bf16 product, the
    // limit-0 case of silu_clamped).
    let ref_h: Vec<Vec<f32>> = (0..m)
        .map(|r| {
            let xs = &x[r * HIDDEN..(r + 1) * HIDDEN];
            (0..DENSE_INTER)
                .map(|c| {
                    let mut g_acc = 0f64;
                    let mut u_acc = 0f64;
                    for t in 0..HIDDEN {
                        g_acc += (xs[t] * gate[c * HIDDEN + t]) as f64;
                        u_acc += (xs[t] * up[c * HIDDEN + t]) as f64;
                    }
                    bf16(silu_clamped(bf16(g_acc as f32), 0.0) * bf16(u_acc as f32))
                })
                .collect()
        })
        .collect();

    let mut h_err2 = 0f64;
    let mut h_ref2 = 0f64;
    let mut h_bit_eq = 0usize;
    for r in 0..m {
        for c in 0..DENSE_INTER {
            let i = (r * DENSE_INTER + c) * 2;
            let hv = widen(u16::from_le_bytes([h_b[i], h_b[i + 1]]));
            ensure!(hv.is_finite(), "dense H non-finite at row {r} col {c}");
            h_err2 += ((hv - ref_h[r][c]) as f64).powi(2);
            h_ref2 += (ref_h[r][c] as f64).powi(2);
            h_bit_eq += usize::from(hv == ref_h[r][c]);
        }
    }

    // Reference Y: one bf16 round of the down dot of the reference H.
    let mut y_err2 = 0f64;
    let mut y_ref2 = 0f64;
    let mut y_bit_eq = 0usize;
    for r in 0..m {
        for c in 0..HIDDEN {
            let mut acc = 0f64;
            for t in 0..DENSE_INTER {
                acc += (ref_h[r][t] * down[c * DENSE_INTER + t]) as f64;
            }
            let y_ref = bf16(acc as f32);
            let i = (r * HIDDEN + c) * 2;
            let yv = widen(u16::from_le_bytes([out_b[i], out_b[i + 1]]));
            ensure!(yv.is_finite(), "dense Y non-finite at row {r} col {c}");
            y_err2 += ((yv - y_ref) as f64).powi(2);
            y_ref2 += (y_ref as f64).powi(2);
            y_bit_eq += usize::from(yv == y_ref);
        }
    }
    let h_rms = (h_err2 / h_ref2.max(f64::MIN_POSITIVE)).sqrt();
    let y_rms = (y_err2 / y_ref2.max(f64::MIN_POSITIVE)).sqrt();
    let h_n = m * DENSE_INTER;
    let y_n = m * HIDDEN;
    eprintln!(
        "layer {layer} M={m} dense FFN: H rel RMS {h_rms:.5} ({h_bit_eq}/{h_n} bit-equal), Y rel RMS {y_rms:.5} ({y_bit_eq}/{y_n} bit-equal)"
    );
    ensure!(h_rms < 0.02, "dense H diverges (rel RMS {h_rms})");
    ensure!(y_rms < 0.02, "dense Y diverges (rel RMS {y_rms})");
    Ok(())
}
