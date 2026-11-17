// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): the MoE of a token as the block runs it (every
//! expert's projections a row of one grid, from staged weights, the hidden
//! activations read in the parts they were computed in) against the launches
//! it stands for, one projection and one activation at a time, on the
//! release's weights and DeepSeek's recorded MoE inputs.
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
fn the_moe_leaves_the_bytes_of_one_launch_at_a_time() -> Result<()> {
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
    let (slots, rows, n) = (s.top_k + 1, s.inter / PARTS, s.hidden / PARTS);
    let picks = picks(&rec, 12);
    let inputs = rows_of(
        &rec.prefill(&format!("L{LAYER}.op.ffn.in"))?.1,
        s.hidden,
        &picks,
    );
    // A part's gate and up rows; every slot's whole hidden activation; and
    // one part of the slots' outputs and of their sum.
    let (gate, up) = (gpu.alloc(rows * 2)?, gpu.alloc(rows * 2)?);
    let inner = gpu.alloc(slots * s.inter * 2)?;
    let (slot_out, one) = (gpu.alloc(slots * n * 2)?, gpu.alloc(n * 2)?);
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
        block.moe(gpu, 1, None, None, stream)?;
        let fused = read(sc.moe_out, PARTS * n * 2)?;
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
        // The projections and the weight of a slot's expert, the last slot
        // being the shared one: FP8, and unweighted.
        let expert = |part: usize, slot: usize| -> Result<_> {
            let slab = w.experts[part].context("every part is held")?;
            Ok(match slot < s.top_k {
                true => {
                    let e = chosen[slot];
                    (
                        k.parts.fp4,
                        [slab.w1(e), slab.w3(e), slab.w2(e)],
                        weights[slot],
                    )
                }
                false => (
                    k.parts.fp8,
                    [&w.shared_w1, &w.shared_w3, &w.shared_w2].map(|whole| row_part(whole, part)),
                    1.0,
                ),
            })
        };
        let x = sc.quantized;
        for part in 0..PARTS {
            for slot in 0..slots {
                let (kernel, [w1, w3, _], weight) = expert(part, slot)?;
                parts::project(
                    gpu, kernel, x, w1.data, w1.scales, gate, rows, s.hidden, stream,
                )?;
                parts::project(
                    gpu, kernel, x, w3.data, w3.scales, up, rows, s.hidden, stream,
                )?;
                let at = inner.offset((slot * s.inter + part * rows) * 2);
                moe::swiglu(
                    gpu,
                    k.moe.swiglu,
                    gate,
                    up,
                    at,
                    rows,
                    weight,
                    s.swiglu_limit,
                    stream,
                )?;
                ops::fp8_quant_dequant(gpu, k.cache.fp8_quant_dequant, at, at, rows, stream)?;
            }
        }
        for part in 0..PARTS {
            for slot in 0..slots {
                let (kernel, [_, _, w2], _) = expert(part, slot)?;
                parts::project(
                    gpu,
                    kernel,
                    inner.offset(slot * s.inter * 2),
                    w2.data,
                    w2.scales,
                    slot_out.offset(slot * n * 2),
                    n,
                    s.inter,
                    stream,
                )?;
            }
            parts::slots_sum(
                gpu,
                k.parts.slots_sum,
                slot_out,
                spark_runtime::gpu::DevicePtr::NULL,
                one,
                1,
                slots,
                n,
                stream,
            )?;
            let at = part * n * 2;
            assert!(
                read(one, n * 2)? == fused[at..at + n * 2],
                "token {token}, part {part}: the block's MoE differs"
            );
            compared += n;
        }
    }
    println!(
        "layer {LAYER}: {compared} values of the MoE's output on {} tokens equal, the block against a launch a projection",
        picks.len()
    );
    for p in [gate, up, inner, slot_out, one] {
        gpu.free(p)?;
    }
    free_store(gpu, &store)?;
    block.weights.free_parts(gpu)?;
    Ok(())
}
