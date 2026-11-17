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

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn the_largest_logits_of_a_row_are_the_hosts() -> Result<()> {
    let gpu = backend()?;
    let (stream, k) = (gpu.default_stream(), DraftKernels::load(gpu)?);
    let (rows, n, first) = (5usize, 32_320usize, 96_960usize);
    let mut rng = Rng(0x0d5a_7e57);
    // Few distinct values, so that equals are common and the tie rule shows.
    let logits: Vec<f32> = (0..rows * n)
        .map(|_| (rng.next() % 4096) as f32 / 64.0 - 20.0)
        .collect();
    let on_device = upload(gpu, &le_bytes(&logits, f32::to_le_bytes))?;
    let best = gpu.alloc(rows * KEEP * 8)?;
    largest(gpu, &k, on_device, best, rows, n, first, stream)?;
    gpu.synchronize(stream)?;
    let mut bytes = vec![0u8; rows * KEEP * 8];
    gpu.copy_d2h(best, &mut bytes)?;
    for row in 0..rows {
        let mut order: Vec<usize> = (0..n).collect();
        let of = |c: usize| logits[row * n + c];
        order.sort_by(|&a, &b| of(b).total_cmp(&of(a)).then(a.cmp(&b)));
        for (i, &column) in order[..KEEP].iter().enumerate() {
            let at = (row * KEEP + i) * 8;
            let word = |at: usize| u32::from_le_bytes([0, 1, 2, 3].map(|b| bytes[at + b]));
            assert_eq!(
                (f32::from_bits(word(at)), word(at + 4) as usize),
                (of(column), first + column),
                "row {row}, place {i}"
            );
        }
    }
    gpu.free(on_device)?;
    gpu.free(best)
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
