// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`; a GB10 and the deepseek-v4.1-flash kernel target)
//! for the V4.1 attention kernels.
//!
//!   cargo test --release -p spark-model --lib deepseek_v41::ops -- --ignored --nocapture --test-threads=1
//!
//! The replay test also needs the reference recording
//! (`sparkseek/results/2026-11-15-reference-layers`, `ATLAS_DSV41_REC`) and
//! the released checkpoint (`ATLAS_DSV41_CKPT`).

use std::path::Path;

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::layers::deepseek_v41::cache_recipe as recipe;
use crate::layers::deepseek_v41::test_support::{
    Recording, Rng, activations, backend, bf16s, download, le_bytes as bytes_of, upload,
};
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash kernel target"]
fn cache_kernels_write_what_the_host_recipe_writes() -> Result<()> {
    let gpu = backend()?;
    let k = Kernels::load(gpu)?;
    let stream = gpu.default_stream();
    let mut rng = Rng(0x0d5e_e41c_ac4e_0001);

    // Quantizers, over 300 keys of 512.
    let host = activations(&mut rng, 300 * HEAD_DIM);
    for (name, launch, rule) in [
        ("fp8", 0, recipe::fp8_quant_dequant as fn(&mut [u16])),
        ("fp4 e8m0", 1, recipe::fp4_quant_dequant_e8m0),
        ("fp4 e4m3", 2, recipe::fp4_quant_dequant_e4m3),
    ] {
        let dev = upload(gpu, &bytes_of(&host, u16::to_le_bytes))?;
        match launch {
            0 => fp8_quant_dequant(gpu, k.fp8_quant_dequant, dev, dev, host.len(), stream)?,
            1 => fp4_quant_dequant(
                gpu,
                k.fp4_quant_dequant,
                dev,
                host.len(),
                Fp4Scale::PowerOfTwoPer32,
                stream,
            )?,
            _ => fp4_quant_dequant(
                gpu,
                k.fp4_quant_dequant,
                dev,
                host.len(),
                Fp4Scale::E4m3Per16,
                stream,
            )?,
        }
        let got = download(gpu, dev, host.len())?;
        let mut want = host.clone();
        rule(&mut want);
        assert!(got == want, "{name}: the kernel and the host rule differ");
        gpu.free(dev)?;
    }

    // Rotation of the last 64 values of 3 vectors a row, at scattered
    // positions, with both of the model's frequency ladders; then back.
    let (rows, vectors, rope) = (40usize, 3usize, 64usize);
    let positions: Vec<u32> = (0..rows as u32).map(|r| r * r * 613 % 1_000_003).collect();
    let dev_positions = upload(gpu, &bytes_of(&positions, u32::to_le_bytes))?;
    for freqs in [
        recipe::rope_freqs(rope, 0, 10_000.0, 16.0, 32.0, 1.0),
        recipe::rope_freqs(rope, 65536, 160_000.0, 16.0, 32.0, 1.0),
    ] {
        let host = activations(&mut rng, rows * vectors * HEAD_DIM);
        let dev = upload(gpu, &bytes_of(&host, u16::to_le_bytes))?;
        let dev_freqs = upload(gpu, &bytes_of(&freqs, f32::to_le_bytes))?;
        rotate(
            gpu,
            k.rotate,
            dev,
            dev_positions,
            dev_freqs,
            rows,
            vectors,
            HEAD_DIM,
            rope,
            false,
            stream,
        )?;
        let got = download(gpu, dev, host.len())?;
        let mut want = host.clone();
        for (v, vector) in want.chunks_exact_mut(HEAD_DIM).enumerate() {
            recipe::rotate_pairs(
                &mut vector[HEAD_DIM - rope..],
                positions[v / vectors] as usize,
                &freqs,
            );
        }
        let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
        println!("rotate: {same} of {} values equal the host's", got.len());
        // sin and cos of a large angle may differ in the last bit between
        // the host's libm and CUDA's; a BF16 result then moves by one step.
        let worst = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
            .max()
            .unwrap_or(0);
        assert!(
            worst <= 1 && same * 1000 >= got.len() * 999,
            "worst step {worst}"
        );
        // The values not rotated are untouched.
        for (a, b) in got.chunks_exact(HEAD_DIM).zip(host.chunks_exact(HEAD_DIM)) {
            assert!(a[..HEAD_DIM - rope] == b[..HEAD_DIM - rope]);
        }
        gpu.free(dev)?;
        gpu.free(dev_freqs)?;
    }
    Ok(())
}

/// One launch's inputs on the host: `m` queries of `h` heads over `k` index
/// columns into `keys`.
struct Case {
    q: Vec<u16>,
    keys: Vec<u16>,
    idx: Vec<i32>,
    reference: Vec<u16>,
    m: usize,
    k: usize,
}

const HEADS: usize = 64;

fn run(gpu: &dyn GpuBackend, k: &Kernels, case: &Case, sink: DevicePtr) -> Result<Vec<u16>> {
    let q = upload(gpu, &bytes_of(&case.q, u16::to_le_bytes))?;
    let keys = upload(gpu, &bytes_of(&case.keys, u16::to_le_bytes))?;
    let idx = upload(gpu, &bytes_of(&case.idx, i32::to_le_bytes))?;
    let out = gpu.alloc(case.q.len() * 2)?;
    sparse_attn(
        gpu,
        k.sparse_attn,
        q,
        keys,
        keys,
        idx,
        sink,
        out,
        case.m,
        HEADS,
        case.k,
        case.k,
        gpu.default_stream(),
    )?;
    let got = download(gpu, out, case.q.len())?;
    for p in [q, keys, idx, out] {
        gpu.free(p)?;
    }
    Ok(got)
}

/// Plain softmax attention of one query head, in f64.
fn exact(q: &[u16], keys: &[u16], idx: &[i32], sink: f32) -> Vec<f64> {
    let value = |bits: u16| f64::from(recipe::bf16_to_f32(bits));
    let rows: Vec<&[u16]> = idx
        .iter()
        .filter(|&&i| i >= 0)
        .map(|&i| &keys[i as usize * HEAD_DIM..][..HEAD_DIM])
        .collect();
    let logits: Vec<f64> = rows
        .iter()
        .map(|row| {
            q.iter()
                .zip(*row)
                .map(|(&a, &b)| value(a) * value(b))
                .sum::<f64>()
                * (HEAD_DIM as f64).powf(-0.5)
        })
        .collect();
    let peak = logits.iter().copied().fold(f64::from(sink), f64::max);
    let weights: Vec<f64> = logits.iter().map(|l| (l - peak).exp()).collect();
    let total = weights.iter().sum::<f64>() + (f64::from(sink) - peak).exp();
    (0..HEAD_DIM)
        .map(|d| {
            rows.iter()
                .zip(&weights)
                .map(|(row, w)| w * value(row[d]))
                .sum::<f64>()
                / total
        })
        .collect()
}

fn rel_l2(got: &[u16], want: &[u16]) -> f64 {
    let value = |bits: u16| f64::from(recipe::bf16_to_f32(bits));
    let (mut diff, mut norm) = (0.0, 0.0);
    for (&a, &b) in got.iter().zip(want) {
        diff += (value(a) - value(b)).powi(2);
        norm += value(b).powi(2);
    }
    (diff / norm).sqrt()
}

#[test]
#[ignore = "needs a GB10 GPU, the deepseek-v4.1-flash kernel target, ATLAS_DSV41_REC and ATLAS_DSV41_CKPT"]
fn attention_replays_the_reference_recording() -> Result<()> {
    let rec = Recording::open()?;
    let ckpt = std::env::var("ATLAS_DSV41_CKPT").context("ATLAS_DSV41_CKPT=<checkpoint dir>")?;
    let gpu = backend()?;
    let k = Kernels::load(gpu)?;
    let (prefill, decode) = (rec.prefill, rec.decode);

    // 20, 21 and 35 are where a whole block's attention is furthest from the
    // reference.
    for layer in rec.sample(&[0, 1, 2], &[20, 21, 35]) {
        let l = format!("L{layer}.");
        let (_, _, sink) =
            read_checkpoint_tensor(Path::new(&ckpt), &format!("layers.{layer}.attn.attn_sink"))?;
        let sink_host: Vec<f32> = sink
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        anyhow::ensure!(
            sink_host.len() == HEADS,
            "attn_sink has {} heads",
            sink_host.len()
        );
        let sink_dev = upload(gpu, &sink)?;

        // Prefill: queries spread over the prompt, and its last positions.
        let (_, q) = rec.prefill(&format!("{l}sa_q"))?;
        let (_, o) = rec.prefill(&format!("{l}sa_o"))?;
        let (_, keys) = rec.prefill(&format!("{l}sa_kv"))?;
        let (idx_shape, idx) = rec.prefill(&format!("{l}sa_idxs"))?;
        let width = idx_shape[2];
        let picks: Vec<usize> = (0..24)
            .map(|i| i * (prefill - 1) / 23)
            .chain(prefill - 24..prefill)
            .collect();
        let per_query = HEADS * HEAD_DIM * 2;
        let gather = |all: &[u8], each: usize| -> Vec<u8> {
            picks
                .iter()
                .flat_map(|&p| all[p * each..(p + 1) * each].iter().copied())
                .collect()
        };
        let case = Case {
            q: bf16s(&gather(&q, per_query)),
            keys: bf16s(&keys),
            idx: gather(&idx, width * 4)
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            reference: bf16s(&gather(&o, per_query)),
            m: picks.len(),
            k: width,
        };
        let got = run(gpu, &k, &case, sink_dev)?;
        let against_reference = rel_l2(&got, &case.reference);
        let equal = got
            .iter()
            .zip(&case.reference)
            .filter(|(a, b)| a == b)
            .count();

        // Plain softmax in f64, on a few heads of a few queries: the
        // reference's schedule is not it, by the BF16 rounding of the weights.
        let (mut diff, mut norm) = (0.0f64, 0.0f64);
        for query in (0..case.m).step_by(6) {
            for head in (0..HEADS).step_by(13) {
                let at = (query * HEADS + head) * HEAD_DIM;
                let want = exact(
                    &case.q[at..at + HEAD_DIM],
                    &case.keys,
                    &case.idx[query * width..(query + 1) * width],
                    sink_host[head],
                );
                for (d, w) in want.iter().enumerate() {
                    diff += (f64::from(recipe::bf16_to_f32(got[at + d])) - w).powi(2);
                    norm += w * w;
                }
            }
        }
        let from_exact = (diff / norm).sqrt();
        println!(
            "layer {layer} prefill: {} queries x {width} keys, {:.3}% of values equal the reference's ({against_reference:.2e} away); {from_exact:.2e} from plain softmax",
            case.m,
            100.0 * equal as f64 / got.len() as f64
        );
        assert!(
            equal * 1000 >= got.len() * 995 && against_reference < 2e-4,
            "layer {layer} prefill: {equal} of {} values equal, {against_reference:.3e} away",
            got.len()
        );
        assert!(
            from_exact < 3e-3,
            "layer {layer} prefill: {from_exact:.3e} from plain softmax"
        );

        // A query alone gives the same bytes as it did among the others.
        for query in [0, case.m / 2, case.m - 1] {
            let alone = Case {
                q: case.q[query * HEADS * HEAD_DIM..(query + 1) * HEADS * HEAD_DIM].to_vec(),
                keys: case.keys.clone(),
                idx: case.idx[query * width..(query + 1) * width].to_vec(),
                reference: Vec::new(),
                m: 1,
                k: width,
            };
            let bytes = run(gpu, &k, &alone, sink_dev)?;
            assert!(
                bytes[..] == got[query * HEADS * HEAD_DIM..(query + 1) * HEADS * HEAD_DIM],
                "layer {layer}: query {query} alone differs"
            );
        }

        // Decode: every recorded step.
        let mut worst = 0.0f64;
        for step in 0..decode {
            let (_, q) = rec.step(&format!("{l}sa_q"), step)?;
            let (_, o) = rec.step(&format!("{l}sa_o"), step)?;
            let (_, keys) = rec.step(&format!("{l}sa_kv"), step)?;
            let (idx_shape, idx) = rec.step(&format!("{l}sa_idxs"), step)?;
            let width = *idx_shape.last().context("idx shape")?;
            let case = Case {
                q: bf16s(&q),
                keys: bf16s(&keys),
                idx: idx
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                reference: bf16s(&o),
                m: 1,
                k: width,
            };
            let got = run(gpu, &k, &case, sink_dev)?;
            worst = worst.max(rel_l2(&got, &case.reference));
        }
        println!("layer {layer} decode: {decode} steps, worst {worst:.2e} from the reference");
        assert!(
            worst < 4e-4,
            "layer {layer} decode: {worst:.3e} from the reference"
        );
        gpu.free(sink_dev)?;
    }
    Ok(())
}
