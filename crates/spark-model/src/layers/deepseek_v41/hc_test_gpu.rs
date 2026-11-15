// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): both Hyper-Connection sites of layers 0 to 2,
//! replayed from DeepSeek's recorded streams on the release's weights.
//!
//!   ATLAS_DSV41_REC=<recording with --ops> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::hc -- --ignored --nocapture --test-threads=1

use std::path::Path;

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::layers::deepseek_v41::test_support::{
    Recording, backend, bf16s, closeness, download, picks, rows_of, upload,
};
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;

const SHAPE: HcShape = HcShape {
    hidden: 5120,
    copies: 4,
    sinkhorn_iters: 20,
    norm_eps: 1e-20,
    hc_eps: 1e-6,
};

fn site(gpu: &dyn GpuBackend, ckpt: &str, layer: usize, name: &str) -> Result<HcSite> {
    let tensor = |part: &str| -> Result<DevicePtr> {
        let (dtype, _, bytes) =
            read_checkpoint_tensor(Path::new(ckpt), &format!("layers.{layer}.hc_{name}_{part}"))?;
        anyhow::ensure!(dtype == "F32", "hc_{name}_{part}: {dtype}");
        upload(gpu, &bytes)
    };
    Ok(HcSite {
        fn_: tensor("fn")?,
        scale: tensor("scale")?,
        base: tensor("base")?,
    })
}

/// The largest relative difference between FP32 values at `p` and recorded ones.
fn f32_gap(gpu: &dyn GpuBackend, p: DevicePtr, want: &[u8]) -> Result<f32> {
    gpu.synchronize(gpu.default_stream())?;
    let mut got = vec![0u8; want.len()];
    gpu.copy_d2h(p, &mut got)?;
    let values = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    Ok(values(&got)
        .iter()
        .zip(values(want))
        .map(|(a, b)| (a - b).abs() / b.abs().max(1e-6))
        .fold(0.0, f32::max))
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn hyper_connections_replay_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let ckpt = std::env::var("ATLAS_DSV41_CKPT").context("ATLAS_DSV41_CKPT=<checkpoint dir>")?;
    let gpu = backend()?;
    let (k, stream) = (HcKernels::load(gpu)?, gpu.default_stream());
    let (hidden, copies) = (SHAPE.hidden, SHAPE.copies);
    let picks = picks(&rec, 40);
    let tokens = picks.len();
    // FP32 rows are twice as wide in bytes as the BF16 rows `rows_of` counts.
    let recorded = |name: &str, width: usize| -> Result<Vec<u8>> {
        Ok(rows_of(&rec.prefill(name)?.1, width, &picks))
    };
    let (pre, post, comb) = (
        gpu.alloc(tokens * copies * 4)?,
        gpu.alloc(tokens * copies * 4)?,
        gpu.alloc(tokens * copies * copies * 4)?,
    );
    let collapsed = gpu.alloc(tokens * hidden * 2)?;

    for layer in 0..3 {
        let l = format!("L{layer}.");
        let streams = upload(gpu, &recorded(&format!("{l}block_in"), copies * hidden)?)?;
        let carried = recorded(&format!("{l}pre_mix_in"), copies * 2)?;
        if layer == 0 {
            let identity: Vec<u8> = IDENTITY_PRE.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert!(carried.chunks_exact(16).all(|row| row == identity));
        }
        let mut pre_in = upload(gpu, &carried)?;
        let mut sublayer_out = format!("{l}attn_out");
        for name in ["attn", "ffn"] {
            let weights = site(gpu, &ckpt, layer, name)?;
            let op = format!("{l}op.hc.{name}.");
            mix(
                gpu, k.mix, &SHAPE, &weights, streams, pre, post, comb, tokens, stream,
            )?;
            let gaps = [
                f32_gap(gpu, pre, &recorded(&format!("{op}pre"), copies * 2)?)?,
                f32_gap(gpu, post, &recorded(&format!("{op}post"), copies * 2)?)?,
                f32_gap(
                    gpu,
                    comb,
                    &recorded(&format!("{op}comb"), copies * copies * 2)?,
                )?,
            ];

            collapse(
                gpu, k.collapse, &SHAPE, streams, pre_in, collapsed, tokens, stream,
            )?;
            let got = download(gpu, collapsed, tokens * hidden)?;
            let (c_same, _) =
                closeness(&got, &bf16s(&recorded(&format!("{op}collapsed"), hidden)?));

            let x = upload(gpu, &recorded(&sublayer_out, hidden)?)?;
            expand(
                gpu, k.expand, &SHAPE, x, streams, post, comb, streams, tokens, stream,
            )?;
            let got = download(gpu, streams, tokens * copies * hidden)?;
            let (e_same, _) = closeness(
                &got,
                &bf16s(&recorded(&format!("{op}expanded"), copies * hidden)?),
            );
            println!(
                "layer {layer} {name}: pre/post/comb within {:.1e}/{:.1e}/{:.1e}; collapsed {:.4}% and expanded {:.4}% of values equal",
                gaps[0],
                gaps[1],
                gaps[2],
                100.0 * c_same,
                100.0 * e_same
            );
            assert!(
                gaps.iter().all(|&g| g < 1e-4),
                "layer {layer} {name}: {gaps:?}"
            );
            assert!(
                c_same > 0.999 && e_same > 0.999,
                "layer {layer} {name}: {c_same} {e_same}"
            );

            // The next collapse uses this site's `pre`. Each step starts from
            // the reference's own stream, so one step's rounding is not read
            // as the next step's error.
            gpu.free(pre_in)?;
            pre_in = upload(gpu, &recorded(&format!("{op}pre"), copies * 2)?)?;
            gpu.copy_h2d(
                &recorded(&format!("{op}expanded"), copies * hidden)?,
                streams,
            )?;
            sublayer_out = format!("{l}op.ffn.out");
            for p in [x, weights.fn_, weights.scale, weights.base] {
                gpu.free(p)?;
            }
        }
        if layer == 2 {
            // Before the head, the stream collapses with the last FFN site's `pre`.
            collapse(
                gpu, k.collapse, &SHAPE, streams, pre_in, collapsed, tokens, stream,
            )?;
            let got = download(gpu, collapsed, tokens * hidden)?;
            let want = bf16s(&recorded("L2.op.hc.final.collapsed", hidden)?);
            let (same, _) = closeness(&got, &want);
            assert!(same > 0.999, "final collapse: {same}");
        }
        gpu.free(pre_in)?;
        gpu.free(streams)?;
    }
    Ok(())
}
