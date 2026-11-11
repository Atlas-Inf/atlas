// SPDX-License-Identifier: AGPL-3.0-only

//! The retained per-CTA batch kernel and RC4 grid-wide selector must agree
//! for finite inputs. Bans and confidence use the serial fallback in serving;
//! the batch ABI has neither. Nonfinite-input policies are not asserted equal.
use super::*;

const ANCHORS: [u32; 3] = [0, 1, 8];

fn logits(shape: Shape) -> Vec<u16> {
    (0..ANCHORS.len() * shape.gamma * shape.vocab)
        .map(|i| {
            let id = i % shape.vocab;
            let row = (i / shape.vocab) % shape.gamma;
            // Row 1 changes winner with predecessor parity: the edge can
            // overcome 3/32 only with anchor 1. Other rows include exact
            // unary/score ties and ties across the top-k boundary.
            let value = match (row, id) {
                (1, 1) => 0.09375,
                (1, 2) => 0.0,
                (1, _) => -4.0,
                (_, 3 | 6) => 0.5,
                _ => (id % 3) as f32 * 0.0625 - 1.0,
            };
            f32_to_bf16_bits(value)
        })
        .collect()
}

fn run_batch(shape: Shape) -> Result<()> {
    let (gpu, stream) = setup()?;
    let (single, _) = kernels(&gpu)?;
    let batch = gpu.kernel(
        "dflash2_candidate_selector",
        "dflash2_candidate_selector_batched",
    )?;
    let logits = logits(shape);
    let mut pred = vec![0u16; shape.vocab * shape.rank];
    let mut succ = pred.clone();
    let mut projected = vec![0u16; ANCHORS.len() * shape.gamma * shape.rank];
    for id in 0..shape.vocab {
        pred[(id + 1) * shape.rank - 1] = f32_to_bf16_bits((1 + id % 2) as f32);
        succ[(id + 1) * shape.rank - 1] = f32_to_bf16_bits((id % 3) as f32 * 0.125);
    }
    for row in 0..ANCHORS.len() * shape.gamma {
        projected[(row + 1) * shape.rank - 1] = f32_to_bf16_bits(0.5);
    }
    let d_logits = up_u16(&gpu, &logits)?;
    let d_pred = up_u16(&gpu, &pred)?;
    let d_succ = up_u16(&gpu, &succ)?;
    let d_proj = up_u16(&gpu, &projected)?;
    let d_anchors = up_u32(&gpu, &ANCHORS)?;
    let d_ban = up_u32(&gpu, &[0])?;
    let scratch_bytes = ops::dflash2_selector_scratch_bytes(shape.gamma);
    let scratch = gpu.alloc(scratch_bytes)?;
    gpu.memset(scratch, 0, scratch_bytes)?;
    let count = ANCHORS.len() * shape.gamma;
    let serial_out = gpu.alloc((count + 4) * 4)?;
    let batch_out = gpu.alloc((count + 4) * 4)?;
    for _ in 0..2 {
        gpu.memset(serial_out, 0xff, (count + 4) * 4)?;
        gpu.memset(batch_out, 0xff, (count + 4) * 4)?;
        for sequence in 0..ANCHORS.len() {
            ops::dflash2_candidate_selector(
                &gpu,
                single,
                d_logits.offset(sequence * shape.gamma * shape.vocab * 2),
                d_proj.offset(sequence * shape.gamma * shape.rank * 2),
                d_pred,
                d_succ,
                serial_out.offset(sequence * shape.gamma * 4),
                d_anchors.offset(sequence * 4),
                d_ban,
                [u32::MAX; 4],
                shape.gamma as u32,
                shape.vocab as u32,
                shape.rank as u32,
                shape.top_k as u32,
                scratch,
                stream,
            )?;
        }
        ops::dflash2_candidate_selector_batched(
            &gpu,
            batch,
            d_logits,
            d_proj,
            d_pred,
            d_succ,
            batch_out,
            d_anchors,
            ANCHORS.len() as u32,
            shape.gamma as u32,
            shape.vocab as u32,
            shape.rank as u32,
            shape.top_k as u32,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let serial = rd_u32(&gpu, serial_out, count + 4)?;
        let batched = rd_u32(&gpu, batch_out, count + 4)?;
        ensure!(serial == batched, "batch/serial mismatch at {shape:?}");
        ensure!(serial[count..] == [u32::MAX; 4], "output guard overwritten");
        for (sequence, anchor) in ANCHORS.into_iter().enumerate() {
            let start = sequence * shape.gamma;
            let expected = reference(
                shape,
                &logits[start * shape.vocab..][..shape.gamma * shape.vocab],
                anchor,
                0,
                [u32::MAX; 4],
            );
            compare(&batched[start..start + shape.gamma], &expected, false)?;
        }
    }
    for ptr in [
        d_logits, d_pred, d_succ, d_proj, d_anchors, d_ban, scratch, serial_out, batch_out,
    ] {
        gpu.free(ptr)?;
    }
    gpu.destroy_stream(stream)?;
    Ok(())
}

#[test]
#[ignore = "requires GB10 and the non-stub glm-5.3-flash/nvfp4 kernel target"]
fn batched_selector_matches_single_and_scalar_reference() -> Result<()> {
    for (gamma, vocab, rank, top_k) in [(5, 73, 13, 3), (19, 1003, 64, 7)] {
        run_batch(Shape {
            gamma,
            vocab,
            rank,
            top_k,
        })?;
    }
    Ok(())
}

#[test]
fn batch_fixture_distinguishes_anchors_and_resolves_finite_ties() {
    let shape = Shape {
        gamma: 5,
        vocab: 73,
        rank: 13,
        top_k: 3,
    };
    let logits = logits(shape);
    let results: Vec<_> = ANCHORS
        .into_iter()
        .map(|anchor| reference(shape, &logits, anchor, 0, [u32::MAX; 4]))
        .collect();
    assert_eq!(
        results.iter().map(|r| r[1].0).collect::<Vec<_>>(),
        [1, 2, 1]
    );
    for result in results {
        assert_eq!(result[0].0, 3);
        assert_eq!(result[2].0, 3);
    }
}
