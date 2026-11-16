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
    let partials = gpu.alloc(SHAPE.partials(tokens))?;

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
                gpu, &k, &SHAPE, &weights, streams, partials, pre, post, comb, tokens, stream,
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
        gpu.free(pre_in)?;
        gpu.free(streams)?;
    }
    // Before the head, the last block's stream collapses with its FFN
    // site's `pre`.
    let (last, want) = rec.final_collapse()?;
    let l = format!("L{last}.");
    let streams = upload(gpu, &recorded(&format!("{l}block_out"), copies * hidden)?)?;
    let pre_in = upload(gpu, &recorded(&format!("{l}pre_mix_out"), copies * 2)?)?;
    collapse(
        gpu, k.collapse, &SHAPE, streams, pre_in, collapsed, tokens, stream,
    )?;
    let got = download(gpu, collapsed, tokens * hidden)?;
    let (same, _) = closeness(&got, &bf16s(&rows_of(&want, hidden, &picks)));
    println!(
        "final collapse after layer {last}: {:.4}% of values equal",
        100.0 * same
    );
    assert!(same > 0.999, "final collapse: {same}");
    Ok(())
}

/// A site's buffers, for one token.
#[derive(Clone, Copy)]
struct SiteBuffers {
    streams: DevicePtr,
    pre: DevicePtr,
    post: DevicePtr,
    comb: DevicePtr,
    out: SiteOut,
}

impl SiteBuffers {
    fn alloc(gpu: &dyn GpuBackend) -> Result<Self> {
        let (hidden, copies) = (SHAPE.hidden, SHAPE.copies);
        Ok(Self {
            streams: gpu.alloc(copies * hidden * 2)?,
            pre: gpu.alloc(copies * 4)?,
            post: gpu.alloc(copies * 4)?,
            comb: gpu.alloc(copies * copies * 4)?,
            out: SiteOut {
                collapsed: gpu.alloc(hidden * 2)?,
                normed: gpu.alloc(hidden * 2)?,
                quantized: gpu.alloc(hidden * 2)?,
                partials: gpu.alloc(SHAPE.partials(1))?,
            },
        })
    }

    /// Everything a site leaves, as bytes.
    fn bytes(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let (hidden, copies) = (SHAPE.hidden, SHAPE.copies);
        gpu.synchronize(gpu.default_stream())?;
        let mut all = Vec::new();
        for (p, bytes) in [
            (self.streams, copies * hidden * 2),
            (self.pre, copies * 4),
            (self.post, copies * 4),
            (self.comb, copies * copies * 4),
            (self.out.collapsed, hidden * 2),
            (self.out.normed, hidden * 2),
            (self.out.quantized, hidden * 2),
        ] {
            let mut got = vec![0u8; bytes];
            gpu.copy_d2h(p, &mut got)?;
            all.extend(got);
        }
        Ok(all)
    }
}

/// `site` is two launches for the seven a site was, and `expand_parts` one
/// for two: each leaves the bytes of the plain kernels, with and without a
/// sublayer to expand first, on DeepSeek's recorded streams.
#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn a_site_in_two_launches_leaves_the_plain_kernels_bytes() -> Result<()> {
    use crate::layers::deepseek_v41::ops::{Kernels, fp8_quant_dequant};
    use crate::layers::deepseek_v41::parts::{self, PARTS, PartKernels};
    use crate::layers::ops::rms_norm;
    use crate::weight_map::DenseWeight;

    const LAYER: usize = 1;
    const SLOTS: usize = 3;
    let rec = Recording::open()?;
    let ckpt = std::env::var("ATLAS_DSV41_CKPT").context("ATLAS_DSV41_CKPT=<checkpoint dir>")?;
    let gpu = backend()?;
    let (k, stream) = (HcKernels::load(gpu)?, gpu.default_stream());
    let (part_kernels, cache) = (PartKernels::load(gpu)?, Kernels::load(gpu)?);
    let norm_kernel = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
    let (hidden, copies) = (SHAPE.hidden, SHAPE.copies);
    let picks = picks(&rec, 24);
    let l = format!("L{LAYER}.");
    let recorded = |name: &str, width: usize| -> Result<Vec<u8>> {
        Ok(rows_of(
            &rec.prefill(&format!("{l}{name}"))?.1,
            width,
            &picks,
        ))
    };
    let streams = recorded("block_in", copies * hidden)?;
    let carried = recorded("pre_mix_in", copies * 2)?;
    let outputs = bf16s(&recorded("attn_out", hidden)?);
    let (attn, ffn) = (
        site(gpu, &ckpt, LAYER, "attn")?,
        site(gpu, &ckpt, LAYER, "ffn")?,
    );
    let (_, _, norm) =
        read_checkpoint_tensor(Path::new(&ckpt), &format!("layers.{LAYER}.ffn_norm.weight"))?;
    let norm = DenseWeight {
        weight: upload(gpu, &norm)?,
    };
    let (plain, fused) = (SiteBuffers::alloc(gpu)?, SiteBuffers::alloc(gpu)?);
    let (pre_in, x) = (gpu.alloc(copies * 4)?, gpu.alloc(hidden * 2)?);
    let parts_dev = gpu.alloc(PARTS * SLOTS * hidden * 4)?;

    for token in 0..picks.len() {
        let stream_in = &streams[token * copies * hidden * 2..(token + 1) * copies * hidden * 2];
        // A sublayer's output in four unequal parts, a slot or three.
        let split = [0.5f32, 0.25, 0.125, 0.12501];
        let mut host = vec![0f32; PARTS * SLOTS * hidden];
        for (i, value) in host.iter_mut().enumerate() {
            let (part, slot, d) = (i / (SLOTS * hidden), i / hidden % SLOTS, i % hidden);
            let bits = outputs[token * hidden + d];
            *value = f32::from_bits(u32::from(bits) << 16) * split[part] * (1 + slot) as f32;
        }
        let host: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        for slots in [0, 1, SLOTS] {
            gpu.synchronize(stream)?;
            gpu.copy_h2d(
                &carried[token * copies * 4..(token + 1) * copies * 4],
                pre_in,
            )?;
            // `slots` slots of the parts, each part's together.
            for part in 0..PARTS {
                let at = part * SLOTS * hidden * 4;
                gpu.copy_h2d(
                    &host[at..at + slots * hidden * 4],
                    parts_dev.offset(part * slots * hidden * 4),
                )?;
            }
            // Both start at the recorded stream, with the attention site's
            // post and comb to expand under.
            for set in [&plain, &fused] {
                gpu.copy_h2d(stream_in, set.streams)?;
                mix(
                    gpu,
                    &k,
                    &SHAPE,
                    &attn,
                    set.streams,
                    set.out.partials,
                    set.pre,
                    set.post,
                    set.comb,
                    1,
                    stream,
                )?;
            }
            let before = (slots > 0).then_some(SublayerParts {
                parts: parts_dev,
                slots,
                rows: 1,
            });
            // The plain kernels, a launch each.
            if before.is_some() {
                parts::combine(
                    gpu,
                    part_kernels.combine,
                    parts_dev,
                    x,
                    slots * hidden,
                    slots,
                    hidden,
                    stream,
                )?;
                expand(
                    gpu,
                    k.expand,
                    &SHAPE,
                    x,
                    plain.streams,
                    plain.post,
                    plain.comb,
                    plain.streams,
                    1,
                    stream,
                )?;
            }
            mix(
                gpu,
                &k,
                &SHAPE,
                &ffn,
                plain.streams,
                plain.out.partials,
                plain.pre,
                plain.post,
                plain.comb,
                1,
                stream,
            )?;
            collapse(
                gpu,
                k.collapse,
                &SHAPE,
                plain.streams,
                pre_in,
                plain.out.collapsed,
                1,
                stream,
            )?;
            rms_norm(
                gpu,
                norm_kernel,
                plain.out.collapsed,
                &norm,
                plain.out.normed,
                1,
                hidden as u32,
                SHAPE.norm_eps,
                stream,
            )?;
            fp8_quant_dequant(
                gpu,
                cache.fp8_quant_dequant,
                plain.out.normed,
                plain.out.quantized,
                hidden,
                stream,
            )?;
            // The site in two.
            super::site(
                gpu,
                &k,
                &SHAPE,
                &ffn,
                before,
                fused.streams,
                pre_in,
                fused.pre,
                fused.post,
                fused.comb,
                norm.weight,
                fused.out,
                1,
                stream,
            )?;
            assert!(
                plain.bytes(gpu)? == fused.bytes(gpu)?,
                "token {token}, {slots} slots: the site in two launches differs"
            );

            // And the expansion alone, from the same parts, under the FFN
            // site's post and comb, which both sets now hold.
            let Some(before) = before else { continue };
            for set in [&plain, &fused] {
                gpu.copy_h2d(stream_in, set.streams)?;
            }
            expand(
                gpu,
                k.expand,
                &SHAPE,
                x,
                plain.streams,
                plain.post,
                plain.comb,
                plain.streams,
                1,
                stream,
            )?;
            expand_parts(
                gpu,
                k.expand_parts,
                &SHAPE,
                before,
                fused.streams,
                fused.post,
                fused.comb,
                1,
                stream,
            )?;
            assert!(
                plain.bytes(gpu)? == fused.bytes(gpu)?,
                "token {token}, {slots} slots: the expansion from parts differs"
            );
        }
    }
    println!(
        "layer {LAYER}: a site in two launches, and an expansion from parts, leave the plain kernels' bytes on {} tokens, with no sublayer before, the attention's and the MoE's",
        picks.len()
    );
    Ok(())
}
