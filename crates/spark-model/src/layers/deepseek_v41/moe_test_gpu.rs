// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`): DeepSeek's recorded MoE inputs through the plain
//! MoE kernels, on the release's weights.
//!
//!   ATLAS_DSV41_REC=<recording with --ops> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::moe -- --ignored --nocapture --test-threads=1

use std::path::Path;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::layers::deepseek_v41::ops::{self, Kernels};
use crate::layers::deepseek_v41::parts::{PartKernels, project};
use crate::layers::deepseek_v41::test_support::{
    Recording, Rng, activations, backend, bf16s, checkpoint, closeness, download, le_bytes,
    load_fp8, picks, rows_of, upload,
};
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;

const HIDDEN: usize = 5120;
const INTER: usize = 2304;
const EXPERTS: usize = 384;
const TOP_K: usize = 6;
const ROUTE_SCALE: f32 = 1.5;
const SWIGLU_LIMIT: f32 = 10.0;

fn tensor(gpu: &dyn GpuBackend, name: &str) -> Result<DevicePtr> {
    let (_, _, bytes) = read_checkpoint_tensor(Path::new(&checkpoint()?), name)?;
    upload(gpu, &bytes)
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn routing_replays_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let k = MoeKernels::load(gpu)?;
    let picks = picks(&rec, 200);
    for layer in 0..3 {
        let l = format!("L{layer}.op.ffn.");
        let x = upload(
            gpu,
            &rows_of(&rec.prefill(&format!("{l}in"))?.1, HIDDEN, &picks),
        )?;
        let gate = tensor(gpu, &format!("layers.{layer}.ffn.gate.weight"))?;
        let bias = tensor(gpu, &format!("layers.{layer}.ffn.gate.bias"))?;
        let (chosen, weights) = (
            gpu.alloc(picks.len() * TOP_K * 4)?,
            gpu.alloc(picks.len() * TOP_K * 4)?,
        );
        // The plain routing first, then the one the block runs: they must
        // choose alike, and it is the block's that is held to the reference.
        route(
            gpu,
            k.route,
            x,
            gate,
            bias,
            chosen,
            weights,
            picks.len(),
            HIDDEN,
            EXPERTS,
            TOP_K,
            ROUTE_SCALE,
            gpu.default_stream(),
        )?;
        gpu.synchronize(gpu.default_stream())?;
        let mut plain = vec![0u8; picks.len() * TOP_K * 4];
        gpu.copy_d2h(chosen, &mut plain)?;
        let scores = gpu.alloc(picks.len() * EXPERTS * 4)?;
        route_fast(
            gpu,
            &k,
            x,
            gate,
            bias,
            scores,
            chosen,
            weights,
            picks.len(),
            HIDDEN,
            EXPERTS,
            TOP_K,
            ROUTE_SCALE,
            false,
            false,
            gpu.default_stream(),
        )?;
        gpu.synchronize(gpu.default_stream())?;
        let mut fast = vec![0u8; picks.len() * TOP_K * 4];
        gpu.copy_d2h(chosen, &mut fast)?;
        assert!(
            plain == fast,
            "layer {layer}: the two routings choose differently"
        );
        gpu.free(scores)?;
        gpu.synchronize(gpu.default_stream())?;
        let mut got = vec![0u8; picks.len() * TOP_K * 4];
        gpu.copy_d2h(chosen, &mut got)?;
        let got_chosen: Vec<u32> = got
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        gpu.copy_d2h(weights, &mut got)?;
        let got_weights: Vec<f32> = got
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Recorded: indices i64 [tokens, 6] in order of selection, weights f32.
        let want_chosen: Vec<u32> = rows_of(
            &rec.prefill(&format!("{l}gate.indices"))?.1,
            TOP_K * 4,
            &picks,
        )
        .chunks_exact(8)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
        let want_weights: Vec<f32> = rows_of(
            &rec.prefill(&format!("{l}gate.weights"))?.1,
            TOP_K * 2,
            &picks,
        )
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
        let same_sets = got_chosen
            .chunks_exact(TOP_K)
            .zip(want_chosen.chunks_exact(TOP_K))
            .filter(|(a, b)| {
                let (mut a, mut b) = (a.to_vec(), b.to_vec());
                a.sort_unstable();
                b.sort_unstable();
                a == b
            })
            .count();
        let same_order = got_chosen == want_chosen;
        let gap = got_weights
            .iter()
            .zip(&want_weights)
            .map(|(a, b)| (a - b).abs() / b.abs())
            .fold(0.0f32, f32::max);
        println!(
            "layer {layer} routing: {same_sets} of {} tokens choose the same six experts (same order: {same_order}); weights within {gap:.1e}",
            picks.len()
        );
        assert!(
            same_sets == picks.len(),
            "layer {layer}: {same_sets} tokens route alike"
        );
        assert!(
            !same_order || gap < 1e-5,
            "layer {layer}: weights {gap:e} apart"
        );
        for p in [x, gate, bias, chosen, weights] {
            gpu.free(p)?;
        }
    }
    Ok(())
}

/// A routed expert's three projections: `(w4, scales)` each.
struct Expert {
    w1: (DevicePtr, DevicePtr),
    w3: (DevicePtr, DevicePtr),
    w2: (DevicePtr, DevicePtr),
}

impl Expert {
    fn load(gpu: &dyn GpuBackend, layer: usize, expert: u32) -> Result<Self> {
        let part = |name: &str| -> Result<(DevicePtr, DevicePtr)> {
            let prefix = format!("layers.{layer}.ffn.experts.{expert}.{name}");
            Ok((
                tensor(gpu, &format!("{prefix}.weight"))?,
                tensor(gpu, &format!("{prefix}.scale"))?,
            ))
        };
        Ok(Self {
            w1: part("w1")?,
            w3: part("w3")?,
            w2: part("w2")?,
        })
    }

    fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for (w, s) in [self.w1, self.w3, self.w2] {
            gpu.free(w)?;
            gpu.free(s)?;
        }
        Ok(())
    }
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn the_moe_replays_the_reference() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let (k, cache, fp8, stream) = (
        MoeKernels::load(gpu)?,
        Kernels::load(gpu)?,
        PartKernels::load(gpu)?.fp8,
        gpu.default_stream(),
    );
    let picks = picks(&rec, 8);
    let (xq, gate, up, h, down) = (
        gpu.alloc(HIDDEN * 2)?,
        gpu.alloc(INTER * 2)?,
        gpu.alloc(INTER * 2)?,
        gpu.alloc(INTER * 2)?,
        gpu.alloc(HIDDEN * 2)?,
    );
    let (acc, y) = (gpu.alloc(HIDDEN * 4)?, gpu.alloc(HIDDEN * 2)?);
    for layer in [0, 2] {
        let l = format!("L{layer}.op.ffn.");
        let prefix = format!("layers.{layer}.ffn.shared_experts");
        let (s1, _, _) = load_fp8(gpu, &format!("{prefix}.w1"))?;
        let (s3, _, _) = load_fp8(gpu, &format!("{prefix}.w3"))?;
        let (s2, _, _) = load_fp8(gpu, &format!("{prefix}.w2"))?;
        let inputs = rows_of(&rec.prefill(&format!("{l}in"))?.1, HIDDEN, &picks);
        let outputs = bf16s(&rows_of(
            &rec.prefill(&format!("{l}out"))?.1,
            HIDDEN,
            &picks,
        ));
        let chosen = rows_of(
            &rec.prefill(&format!("{l}gate.indices"))?.1,
            TOP_K * 4,
            &picks,
        );
        let weights = rows_of(
            &rec.prefill(&format!("{l}gate.weights"))?.1,
            TOP_K * 2,
            &picks,
        );
        let mut got = Vec::new();
        let (mut fast_same, mut fast_seen) = (0usize, 0usize);
        for token in 0..picks.len() {
            let x = upload(gpu, &inputs[token * HIDDEN * 2..(token + 1) * HIDDEN * 2])?;
            // Every projection of the token reads the same FP8 activation.
            ops::fp8_quant_dequant(gpu, cache.fp8_quant_dequant, x, xq, HIDDEN, stream)?;
            gpu.memset(acc, 0, HIDDEN * 4)?;
            // Routed experts in expert order, each with its routing weight.
            let mut routed: Vec<(u32, f32)> = (0..TOP_K)
                .map(|slot| {
                    let i = &chosen[(token * TOP_K + slot) * 8..][..4];
                    let w = &weights[(token * TOP_K + slot) * 4..][..4];
                    (
                        u32::from_le_bytes([i[0], i[1], i[2], i[3]]),
                        f32::from_le_bytes([w[0], w[1], w[2], w[3]]),
                    )
                })
                .collect();
            routed.sort_unstable_by_key(|&(expert, _)| expert);
            for (expert, weight) in routed {
                let e = Expert::load(gpu, layer, expert)?;
                // The tensor-core projection against the plain one.
                fp4_gemv(
                    gpu, k.fp4_gemv, xq, e.w1.0, e.w1.1, gate, INTER, HIDDEN, stream,
                )?;
                fp4_gemv_tc(
                    gpu,
                    k.fp4_gemv_tc,
                    xq,
                    e.w1.0,
                    e.w1.1,
                    up,
                    INTER,
                    HIDDEN,
                    stream,
                )?;
                let (plain, fast) = (download(gpu, gate, INTER)?, download(gpu, up, INTER)?);
                fast_same += plain.iter().zip(&fast).filter(|(a, b)| a == b).count();
                fast_seen += INTER;
                fp4_project(gpu, &k, xq, e.w1.0, e.w1.1, gate, INTER, HIDDEN, stream)?;
                fp4_project(gpu, &k, xq, e.w3.0, e.w3.1, up, INTER, HIDDEN, stream)?;
                swiglu(
                    gpu,
                    k.swiglu,
                    gate,
                    up,
                    h,
                    INTER,
                    weight,
                    SWIGLU_LIMIT,
                    stream,
                )?;
                ops::fp8_quant_dequant(gpu, cache.fp8_quant_dequant, h, h, INTER, stream)?;
                fp4_project(gpu, &k, h, e.w2.0, e.w2.1, down, HIDDEN, INTER, stream)?;
                accumulate(gpu, k.accumulate, acc, down, HIDDEN, stream)?;
                gpu.synchronize(stream)?;
                e.free(gpu)?;
            }
            // The shared expert: FP8 weights, no routing weight.
            project(
                gpu, fp8, xq, s1.data, s1.scales, gate, INTER, HIDDEN, stream,
            )?;
            project(gpu, fp8, xq, s3.data, s3.scales, up, INTER, HIDDEN, stream)?;
            swiglu(gpu, k.swiglu, gate, up, h, INTER, 1.0, SWIGLU_LIMIT, stream)?;
            ops::fp8_quant_dequant(gpu, cache.fp8_quant_dequant, h, h, INTER, stream)?;
            project(gpu, fp8, h, s2.data, s2.scales, down, HIDDEN, INTER, stream)?;
            accumulate(gpu, k.accumulate, acc, down, HIDDEN, stream)?;
            round_bf16(gpu, k.round_bf16, acc, y, HIDDEN, stream)?;
            got.extend(download(gpu, y, HIDDEN)?);
            gpu.free(x)?;
        }
        let (same, distance) = closeness(&got, &outputs);
        println!(
            "layer {layer} MoE ({}): {:.3}% of values equal, {distance:.2e} away; the tensor-core projection gives {:.4}% of the plain one's values",
            if k.plain_fp4 {
                "plain FP4"
            } else {
                "tensor-core FP4"
            },
            100.0 * same,
            100.0 * fast_same as f64 / fast_seen as f64
        );
        assert!(
            fast_same * 1000 >= fast_seen * 995,
            "tensor-core FP4 differs from plain"
        );
        assert!(
            same > 0.995 && distance < 5e-4,
            "layer {layer} MoE: {same} equal, {distance:e} away"
        );
        for w in [s1, s3, s2] {
            gpu.free(w.data)?;
            gpu.free(w.scales)?;
        }
    }
    Ok(())
}

/// The router's scores of `tokens` rows of `x` in one launch, as FP32 bits.
fn scores_of(
    gpu: &dyn GpuBackend,
    k: &MoeKernels,
    x: DevicePtr,
    gate: DevicePtr,
    bias: DevicePtr,
    tokens: usize,
    experts: usize,
) -> Result<Vec<u32>> {
    let scores = gpu.alloc(tokens * experts * 4)?;
    let (chosen, weights) = (
        gpu.alloc(tokens * TOP_K * 4)?,
        gpu.alloc(tokens * TOP_K * 4)?,
    );
    let stream = gpu.default_stream();
    route_fast(
        gpu,
        k,
        x,
        gate,
        bias,
        scores,
        chosen,
        weights,
        tokens,
        HIDDEN,
        experts,
        TOP_K,
        ROUTE_SCALE,
        true,
        false,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let mut bytes = vec![0u8; tokens * experts * 4];
    gpu.copy_d2h(scores, &mut bytes)?;
    for p in [scores, chosen, weights] {
        gpu.free(p)?;
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn rows_scored_together_score_as_each_alone() -> Result<()> {
    let gpu = backend()?;
    let k = MoeKernels::load(gpu)?;
    let mut rng = Rng(0x5c0_4e5);
    // A count of experts the blocks do not divide, and more rows than a
    // block takes at once.
    let (experts, most) = (386usize, 13usize);
    let x = activations(&mut rng, most * HIDDEN);
    let gate = upload(
        gpu,
        &le_bytes(&activations(&mut rng, experts * HIDDEN), u16::to_le_bytes),
    )?;
    let bias = upload(gpu, &vec![0u8; experts * 4])?;
    let rows = |first: usize, count: usize| -> Result<Vec<u32>> {
        let at = &x[first * HIDDEN..(first + count) * HIDDEN];
        let x = upload(gpu, &le_bytes(at, u16::to_le_bytes))?;
        let scores = scores_of(gpu, &k, x, gate, bias, count, experts);
        gpu.free(x)?;
        scores
    };
    let alone: Vec<u32> = (0..most)
        .map(|row| rows(row, 1))
        .collect::<Result<Vec<_>>>()?
        .concat();
    for tokens in [2, 6, 8, 9, most] {
        assert_eq!(
            rows(0, tokens)?,
            alone[..tokens * experts],
            "{tokens} rows together"
        );
    }
    // The sums' order is part of what a build's tokens are: the scores of
    // one row, as every build since the routing went to two launches left
    // them.
    let digest = alone.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| {
        (h ^ u64::from(v)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    assert_eq!(
        digest, 0x498c_8fb8_9846_aeca,
        "the router's scores of a row"
    );
    for p in [gate, bias] {
        gpu.free(p)?;
    }
    Ok(())
}
