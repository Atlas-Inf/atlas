// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`): the attention's front in one launch, and the
//! attention whose heads share the keys they read and which turns its own
//! query, against the launches they stand for.
//!
//!   cargo test --release -p spark-model --lib deepseek_v41::ops::front -- --ignored --nocapture --test-threads=1

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::layers::deepseek_v41::cache_recipe::rope_freqs;
use crate::layers::deepseek_v41::test_support::{
    Rng, SHAPE, activations, backend, le_bytes, upload,
};
use crate::layers::ops::rms_norm;
use crate::weight_map::DenseWeight;

const ROPE: usize = 64;
const POSITIONS: [u32; 5] = [0, 1, 127, 777, 6001];

fn bytes(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(gpu.default_stream())?;
    let mut out = vec![0u8; n];
    gpu.copy_d2h(p, &mut out)?;
    Ok(out)
}

fn values(gpu: &dyn GpuBackend, rng: &mut Rng, n: usize) -> Result<DevicePtr> {
    upload(gpu, &le_bytes(&activations(rng, n), u16::to_le_bytes))
}

fn ladder(gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let freqs = rope_freqs(ROPE, 65536, 160_000.0, 16.0, 32.0, 1.0);
    upload(gpu, &le_bytes(&freqs, f32::to_le_bytes))
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn the_front_in_one_launch_leaves_the_bytes_of_five() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let k = Kernels::load(gpu)?;
    let norm = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
    let (q_rank, eps) = (SHAPE.q_rank, SHAPE.norm_eps);
    let mut rng = Rng(0xf407_0001);
    let freqs = ladder(gpu)?;
    let at = upload(gpu, &le_bytes(&POSITIONS, u32::to_le_bytes))?;
    let (plain_q, fused_q) = (gpu.alloc(q_rank * 2)?, gpu.alloc(q_rank * 2)?);
    let (turned, plain_slot, fused_slot) = (
        gpu.alloc(HEAD_DIM * 2)?,
        gpu.alloc(HEAD_DIM * 2)?,
        gpu.alloc(HEAD_DIM * 2)?,
    );
    for trial in 0..POSITIONS.len() {
        let (q_low, key) = (
            values(gpu, &mut rng, q_rank)?,
            values(gpu, &mut rng, HEAD_DIM)?,
        );
        let (q_weight, key_weight) = (
            DenseWeight {
                weight: values(gpu, &mut rng, q_rank)?,
            },
            DenseWeight {
                weight: values(gpu, &mut rng, HEAD_DIM)?,
            },
        );
        let turn = Turn {
            positions: at.offset(trial * 4),
            freqs,
            rope: ROPE,
        };
        // A launch each.
        rms_norm(
            gpu,
            norm,
            q_low,
            &q_weight,
            plain_q,
            1,
            q_rank as u32,
            eps,
            stream,
        )?;
        rms_norm(
            gpu,
            norm,
            key,
            &key_weight,
            turned,
            1,
            HEAD_DIM as u32,
            eps,
            stream,
        )?;
        rotate(
            gpu,
            k.rotate,
            turned,
            turn.positions,
            freqs,
            1,
            1,
            HEAD_DIM,
            ROPE,
            false,
            stream,
        )?;
        fp8_quant_dequant(
            gpu,
            k.fp8_quant_dequant,
            turned,
            plain_slot,
            HEAD_DIM,
            stream,
        )?;
        // One for all.
        front(
            gpu,
            k.front,
            q_low,
            q_weight.weight,
            fused_q,
            key,
            key_weight.weight,
            fused_slot,
            turn,
            q_rank,
            1,
            eps,
            stream,
        )?;
        assert!(
            bytes(gpu, plain_q, q_rank * 2)? == bytes(gpu, fused_q, q_rank * 2)?,
            "trial {trial}: the normed query differs"
        );
        assert!(
            bytes(gpu, plain_slot, HEAD_DIM * 2)? == bytes(gpu, fused_slot, HEAD_DIM * 2)?,
            "trial {trial}: the window key differs"
        );
        for p in [q_low, key, q_weight.weight, key_weight.weight] {
            gpu.free(p)?;
        }
    }
    println!(
        "the attention's front in one launch leaves the bytes of five, at positions {POSITIONS:?}"
    );
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn attention_with_the_heads_together_leaves_the_plain_forms_bytes() -> Result<()> {
    const HEADS: usize = 16;
    const ROWS: usize = 700;
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let k = Kernels::load(gpu)?;
    let mut rng = Rng(0xa77e_0002);
    let freqs = ladder(gpu)?;
    let at = upload(gpu, &le_bytes(&POSITIONS, u32::to_le_bytes))?;
    let keys = values(gpu, &mut rng, ROWS * HEAD_DIM)?;
    let sink: Vec<f32> = (0..HEADS)
        .map(|_| (rng.next() % 2000) as f32 / 500.0 - 2.0)
        .collect();
    let sink = upload(gpu, &le_bytes(&sink, f32::to_le_bytes))?;
    let (plain, fused, turned) = (
        gpu.alloc(HEADS * HEAD_DIM * 2)?,
        gpu.alloc(HEADS * HEAD_DIM * 2)?,
        gpu.alloc(HEADS * HEAD_DIM * 2)?,
    );
    let records = gpu.alloc(attn_records(1, HEADS))?;
    // Key counts on and off the 64-key steps, with keys absent among them.
    for (trial, count) in [1, 64, 129, 500, MAX_KEYS].into_iter().enumerate() {
        let idx: Vec<i32> = (0..count)
            .map(|_| match rng.next() % 9 {
                0 => -1,
                _ => (rng.next() % ROWS as u64) as i32,
            })
            .collect();
        let idx = upload(gpu, &le_bytes(&idx, i32::to_le_bytes))?;
        let q = values(gpu, &mut rng, HEADS * HEAD_DIM)?;
        let split = count.min(WINDOW_KEYS);
        let turn = Turn {
            positions: at.offset(trial * 4),
            freqs,
            rope: ROPE,
        };
        // Turn, attend, turn back: a launch each.
        gpu.synchronize(stream)?;
        gpu.copy_d2d(q, turned, HEADS * HEAD_DIM * 2)?;
        let turn_all = |x: DevicePtr, inverse: bool| {
            rotate(
                gpu,
                k.rotate,
                x,
                turn.positions,
                freqs,
                1,
                HEADS,
                HEAD_DIM,
                ROPE,
                inverse,
                stream,
            )
        };
        turn_all(turned, false)?;
        sparse_attn_plain(
            gpu,
            k.sparse_attn,
            turned,
            keys,
            keys,
            idx,
            sink,
            plain,
            1,
            HEADS,
            count,
            split,
            stream,
        )?;
        turn_all(plain, true)?;
        // Three launches for all the heads together, turning inside.
        sparse_attn(
            gpu,
            &k,
            q,
            keys,
            keys,
            idx,
            sink,
            fused,
            records,
            1,
            HEADS,
            count,
            split,
            Some(turn),
            false,
            stream,
        )?;
        let want = bytes(gpu, plain, HEADS * HEAD_DIM * 2)?;
        assert!(
            want == bytes(gpu, fused, HEADS * HEAD_DIM * 2)?,
            "{count} keys: the attention with the heads together differs"
        );
        // And on tensor cores, as a batch runs it: the same products, summed
        // in another order, so a value in a few hundred may round the other
        // way.
        sparse_attn(
            gpu,
            &k,
            q,
            keys,
            keys,
            idx,
            sink,
            fused,
            records,
            1,
            HEADS,
            count,
            split,
            Some(turn),
            true,
            stream,
        )?;
        let got = bytes(gpu, fused, HEADS * HEAD_DIM * 2)?;
        let same = want
            .chunks_exact(2)
            .zip(got.chunks_exact(2))
            .filter(|(a, b)| a == b)
            .count();
        println!(
            "{count} keys: {same} of {} values equal on tensor cores",
            HEADS * HEAD_DIM
        );
        assert!(
            same * 100 >= HEADS * HEAD_DIM * 97,
            "{count} keys: the attention on tensor cores is not the same attention"
        );
        gpu.free(idx)?;
        gpu.free(q)?;
    }
    println!(
        "attention with the heads sharing their keys, turning its own query, leaves the bytes of turn, a block a head, and turn back"
    );
    Ok(())
}

/// Keys of a window, which the index row lists first.
const WINDOW_KEYS: usize = 128;
