// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): the draft head's own kernels against the host's
//! arithmetic, and a sequence's ring of stream means across its wrap.
//!
//!   cargo test --release -p spark-model --lib deepseek_v41::draft_ops -- --ignored --nocapture --test-threads=1

use anyhow::Result;

use super::*;
use atlas_core::numeric::f32_to_bf16;

use crate::layers::deepseek_v41::cache_recipe::{WINDOW, bf16_to_f32};
use crate::layers::deepseek_v41::draft_seq::DraftSeq;
use crate::layers::deepseek_v41::test_support::{
    Rng, activations, backend, download, le_bytes, upload,
};

/// BF16 bits of a value BF16 holds exactly.
fn exact(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn the_markov_chain_over_the_parts_is_the_hosts_over_the_vocabulary() -> Result<()> {
    let gpu = backend()?;
    let (stream, k) = (gpu.default_stream(), DraftKernels::load(gpu)?);
    // A share that does not fill its last block, and values whose products
    // and sums are exact, so that the order of a sum cannot matter.
    let (rows, share, rank, token) = (5usize, 300usize, 16usize, 7u32);
    let vocab = PARTS * share;
    let mut rng = Rng(0x3a2c_0de5);
    let mut eighths = |n: usize, span: u64| -> Vec<f32> {
        (0..n)
            .map(|_| (rng.next() % span) as f32 / 8.0 - span as f32 / 16.0)
            .collect()
    };
    let (embed, head) = (eighths(vocab * rank, 33), eighths(vocab * rank, 33));
    // Few distinct logits, so that tokens tie and the tie rule shows.
    let logits = eighths(rows * vocab, 9);
    let bf16 = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| exact(x)).collect() };
    let markov = Markov {
        embed: upload(gpu, &le_bytes(&bf16(&embed), u16::to_le_bytes))?,
        head: upload(gpu, &le_bytes(&bf16(&head), u16::to_le_bytes))?,
        vocab,
        rank,
    };
    // A part's logits of a row, as a rank holds them: [rows, share].
    let part_logits: Vec<DevicePtr> = (0..PARTS)
        .map(|part| {
            let mine: Vec<f32> = (0..rows)
                .flat_map(|row| {
                    let from = row * vocab + part * share;
                    logits[from..from + share].to_vec()
                })
                .collect();
            upload(gpu, &le_bytes(&mine, f32::to_le_bytes))
        })
        .collect::<Result<_>>()?;
    let bytes = markov.bytes();
    let found = gpu.alloc(rows * PARTS * bytes)?;
    let drafts = gpu.alloc(rows * 4)?;
    let none = DevicePtr::NULL;
    for row in 0..rows {
        let (before, chosen) = match row {
            0 => (none, none),
            _ => (
                found.offset((row - 1) * PARTS * bytes),
                drafts.offset((row - 1) * 4),
            ),
        };
        for (part, of_part) in part_logits.iter().enumerate() {
            let mine = found.offset((row * PARTS + part) * bytes);
            let of_row = of_part.offset(row * share * 4);
            markov_round(
                gpu, &k, &markov, part, of_row, before, mine, chosen, token, stream,
            )?;
        }
    }
    let last = found.offset((rows - 1) * PARTS * bytes);
    markov_close(
        gpu,
        &k,
        &markov,
        last,
        drafts.offset((rows - 1) * 4),
        stream,
    )?;
    gpu.synchronize(stream)?;
    let mut got = vec![0u8; rows * 4];
    gpu.copy_d2h(drafts, &mut got)?;
    let got: Vec<u32> = got
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // The host's chain: a row's largest logit plus gain, the lower token
    // among equals.
    let mut want = Vec::new();
    let mut before = token as usize;
    for row in 0..rows {
        let from = &embed[before * rank..(before + 1) * rank];
        let value = |v: usize| -> f32 {
            let gain: f32 = (0..rank).map(|i| from[i] * head[v * rank + i]).sum();
            logits[row * vocab + v] + gain
        };
        let best = (0..vocab).fold(0, |best, v| if value(v) > value(best) { v } else { best });
        want.push(best as u32);
        before = best;
    }
    assert_eq!(got, want);
    for p in [markov.embed, markov.head, found, drafts]
        .into_iter()
        .chain(part_logits)
    {
        gpu.free(p)?;
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn gathered_parts_come_back_a_row_at_a_time() -> Result<()> {
    let gpu = backend()?;
    let (stream, k) = (gpu.default_stream(), DraftKernels::load(gpu)?);
    let (rows, part_n) = (7usize, 96usize);
    let mut rng = Rng(0x9a27_5e11);
    let parts = activations(&mut rng, PARTS * rows * part_n);
    let on_device = upload(gpu, &le_bytes(&parts, u16::to_le_bytes))?;
    let out = gpu.alloc(rows * PARTS * part_n * 2)?;
    rows_of_parts(gpu, &k, on_device, out, rows, part_n, stream)?;
    let got = download(gpu, out, rows * PARTS * part_n)?;
    for row in 0..rows {
        for part in 0..PARTS {
            let from = (part * rows + row) * part_n;
            let to = (row * PARTS + part) * part_n;
            assert_eq!(
                got[to..to + part_n],
                parts[from..from + part_n],
                "row {row}, part {part}"
            );
        }
    }
    gpu.free(on_device)?;
    gpu.free(out)
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn a_layers_means_land_in_the_ring_across_its_wrap() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let mean = gpu.kernel("dsv41_draft", "dsv41_stream_mean")?;
    let (hidden, copies, targets) = (64usize, 4usize, 3usize);
    let mut seq = DraftSeq::alloc(gpu, hidden, targets, 1)?;
    let mut rng = Rng(0x3ea2_0f57);
    // More rows than a window, ending past a multiple of it: the last
    // window of them is kept, and the ring wraps inside it.
    let (position, rows) = (1000usize, 200usize);
    let streams: Vec<Vec<u16>> = (0..targets)
        .map(|_| activations(&mut rng, rows * copies * hidden))
        .collect();
    for (target, values) in streams.iter().enumerate() {
        let on_device = upload(gpu, &le_bytes(values, u16::to_le_bytes))?;
        seq.take(
            gpu, mean, target, on_device, rows, position, hidden, copies, stream,
        )?;
        gpu.synchronize(stream)?;
        gpu.free(on_device)?;
    }
    let ring = download(gpu, seq.means, WINDOW * targets * hidden)?;
    for at in position + rows - WINDOW..position + rows {
        for (target, values) in streams.iter().enumerate() {
            let row = at - position;
            let want: Vec<u16> = (0..hidden)
                .map(|d| {
                    let sum: f32 = (0..copies)
                        .map(|j| bf16_to_f32(values[(row * copies + j) * hidden + d]))
                        .sum();
                    f32_to_bf16(sum / copies as f32)
                })
                .collect();
            let from = ((at % WINDOW) * targets + target) * hidden;
            assert_eq!(
                ring[from..from + hidden],
                want,
                "position {at}, layer {target}"
            );
        }
    }
    // The head may draft at the end of those rows, from a window of them.
    let plan = seq.plan(position + rows).expect("a plan");
    assert_eq!(plan.ingest, position + rows - WINDOW..position + rows);
    assert_eq!(seq.plan(position + rows + 1), None);
    seq.free(gpu)
}
