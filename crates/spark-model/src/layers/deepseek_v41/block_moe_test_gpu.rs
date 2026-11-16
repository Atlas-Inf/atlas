// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): the experts of a token in three launches
//! (`moe::experts`) against the launches they stand for, one projection and
//! one activation at a time, on the release's weights and DeepSeek's
//! recorded MoE inputs.
//!
//!   ATLAS_DSV41_REC=<recording with --ops> ATLAS_DSV41_CKPT=<checkpoint> \
//!     cargo test --release -p spark-model --lib deepseek_v41::block_moe -- --ignored --nocapture --test-threads=1

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use spark_runtime::gpu::GpuBackend;

use crate::layers::deepseek_v41::block::{BlockKernels, Handoff, Scratch, Shared};
use crate::layers::deepseek_v41::moe;
use crate::layers::deepseek_v41::ops;
use crate::layers::deepseek_v41::parts::{self, Held, PARTS};
use crate::layers::deepseek_v41::test_support::{
    Recording, SHAPE, backend, block, free_store, picks, rows_of,
};
use crate::layers::deepseek_v41::weights::row_part;

const LAYER: usize = 1;

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash target, ATLAS_DSV41_REC (--ops) and ATLAS_DSV41_CKPT"]
fn the_experts_in_three_launches_leave_the_bytes_of_one_at_a_time() -> Result<()> {
    let rec = Recording::open()?;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let shared = Shared {
        shape: SHAPE,
        held: Held::ALL,
        kernels: BlockKernels::load(gpu)?,
        scratch: Arc::new(Scratch::alloc(gpu, &SHAPE, 16)?),
        handoff: Arc::new(Mutex::new(Handoff::default())),
    };
    let (block, store) = block(gpu, LAYER, true, shared)?;
    let (s, k, sc, w) = (
        &block.shared.shape,
        &block.shared.kernels,
        &block.shared.scratch,
        &block.weights.moe,
    );
    let (slots, rows) = (s.top_k + 1, s.inter / PARTS);
    let picks = picks(&rec, 12);
    let inputs = rows_of(
        &rec.prefill(&format!("L{LAYER}.op.ffn.in"))?.1,
        s.hidden,
        &picks,
    );
    let (gate, up, inner) = (
        gpu.alloc(rows * 2)?,
        gpu.alloc(rows * 2)?,
        gpu.alloc(rows * 2)?,
    );
    let one = gpu.alloc(s.hidden * 4)?;
    let read = |p, bytes: usize| -> Result<Vec<u8>> {
        gpu.synchronize(stream)?;
        let mut out = vec![0u8; bytes];
        gpu.copy_d2h(p, &mut out)?;
        Ok(out)
    };
    let mut compared = 0usize;
    for token in 0..picks.len() {
        gpu.synchronize(stream)?;
        gpu.copy_h2d(
            &inputs[token * s.hidden * 2..(token + 1) * s.hidden * 2],
            sc.normed,
        )?;
        ops::fp8_quant_dequant(
            gpu,
            k.cache.fp8_quant_dequant,
            sc.normed,
            sc.quantized,
            s.hidden,
            stream,
        )?;
        block.moe(gpu, None, stream)?;
        let fused = read(sc.moe_parts, PARTS * slots * s.hidden * 4)?;
        let chosen: Vec<usize> = read(sc.chosen, s.top_k * 4)?
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        let weights: Vec<f32> = read(sc.weights, s.top_k * 4)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(
            chosen.windows(2).all(|pair| pair[0] < pair[1]),
            "token {token}: {chosen:?} is not in expert order"
        );
        for part in 0..PARTS {
            let slab = w.experts[part].context("every part is held")?;
            let shared_w2 = w.shared_w2[part].context("every part is held")?;
            for slot in 0..slots {
                // The projections and the weight of this slot's expert, the
                // last slot being the shared one: FP8, and unweighted.
                let routed = slot < s.top_k;
                let (w1, w3, w2, weight) = match routed {
                    true => {
                        let e = chosen[slot];
                        (slab.w1(e), slab.w3(e), slab.w2(e), weights[slot])
                    }
                    false => (
                        row_part(&w.shared_w1, part),
                        row_part(&w.shared_w3, part),
                        shared_w2,
                        1.0,
                    ),
                };
                let (into, out) = match routed {
                    true => (k.parts.fp4, k.parts.fp4_f32),
                    false => (k.parts.fp8, k.parts.fp8_f32),
                };
                let x = sc.quantized;
                parts::project(
                    gpu, into, x, w1.data, w1.scales, gate, rows, s.hidden, stream,
                )?;
                parts::project(gpu, into, x, w3.data, w3.scales, up, rows, s.hidden, stream)?;
                moe::swiglu(
                    gpu,
                    k.moe.swiglu,
                    gate,
                    up,
                    inner,
                    rows,
                    weight,
                    s.swiglu_limit,
                    stream,
                )?;
                ops::fp8_quant_dequant(gpu, k.cache.fp8_quant_dequant, inner, inner, rows, stream)?;
                parts::project(
                    gpu, out, inner, w2.data, w2.scales, one, s.hidden, rows, stream,
                )?;
                let at = (part * slots + slot) * s.hidden * 4;
                assert!(
                    read(one, s.hidden * 4)? == fused[at..at + s.hidden * 4],
                    "token {token}, part {part}, slot {slot}: the fused launch differs"
                );
                compared += s.hidden;
            }
        }
    }
    println!(
        "layer {LAYER}: {compared} partial sums of {} tokens equal, three launches against {} a part",
        picks.len(),
        slots * 5
    );
    for p in [gate, up, inner, one] {
        gpu.free(p)?;
    }
    free_store(gpu, &store)?;
    block.weights.free_parts(gpu)?;
    Ok(())
}
