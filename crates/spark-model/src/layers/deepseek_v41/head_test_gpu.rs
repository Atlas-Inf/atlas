// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): the head's pick against `argmax_bf16` over the
//! whole head's row, on a head of four parts held by one rank.
//!
//!   cargo test --release -p spark-model --lib deepseek_v41::head::test_gpu -- --ignored --nocapture --test-threads=1

use anyhow::Result;

use super::*;
use crate::layers::deepseek_v41::cache_recipe::bf16_to_f32;
use crate::layers::deepseek_v41::test_support::{Rng, backend, le_bytes, upload};
use crate::layers::ops::{argmax_bf16, dense_gemv, mxfp8_quantize};
use crate::weight_map::DenseWeight;

const N: usize = 640;
const VOCAB: usize = PARTS * N;
const HIDDEN: usize = 5120;

fn bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // To the nearest, ties to even.
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// About normal, as a head's values are: four uniform draws.
fn bell(rng: &mut Rng) -> f32 {
    let sum: f32 = (0..4).map(|_| (rng.next() % 4096) as f32 / 4096.0).sum();
    sum - 2.0
}

struct Bench {
    gpu: &'static dyn GpuBackend,
    stream: u64,
    pick: HeadPick,
    head: Vec<u16>,
    gemv: KernelHandle,
    argmax: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
}

impl Bench {
    fn new(head: Vec<u16>) -> Result<Self> {
        let gpu: &'static dyn GpuBackend = backend()?;
        let stream = gpu.default_stream();
        let on_device = upload(gpu, &le_bytes(&head, u16::to_le_bytes))?;
        let copy = Mxfp8Weight::alloc(gpu, VOCAB, HIDDEN)?;
        let quantize = gpu.kernel("mxfp8_gemv", "mxfp8_quantize_bf16")?;
        mxfp8_quantize(
            gpu,
            quantize,
            on_device,
            copy.data,
            copy.scales,
            VOCAB,
            HIDDEN,
            stream,
        )?;
        Ok(Self {
            gpu,
            stream,
            pick: HeadPick::load(gpu, on_device, copy, Held::ALL, N, HIDDEN)?,
            head,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            argmax: gpu.kernel("argmax", "argmax_bf16")?,
            logits: gpu.alloc(VOCAB * 2)?,
            out: gpu.alloc(4)?,
        })
    }

    fn words(&self, p: DevicePtr, n: usize) -> Result<Vec<u32>> {
        self.gpu.synchronize(self.stream)?;
        let mut bytes = vec![0u8; n * 4];
        self.gpu.copy_d2h(p, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// The whole head's logits of one hidden row, and their argmax.
    fn whole(&self, x: DevicePtr) -> Result<(Vec<u16>, u32)> {
        let weight = DenseWeight {
            weight: self.pick.head,
        };
        dense_gemv(
            self.gpu,
            self.gemv,
            x,
            &weight,
            self.logits,
            VOCAB as u32,
            HIDDEN as u32,
            self.stream,
        )?;
        argmax_bf16(
            self.gpu,
            self.argmax,
            self.logits,
            self.out,
            VOCAB as u32,
            self.stream,
        )?;
        let token = self.words(self.out, 1)?[0];
        let mut bytes = vec![0u8; VOCAB * 2];
        self.gpu.copy_d2h(self.logits, &mut bytes)?;
        let logits = bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        Ok((logits, token))
    }

    /// The picks of `rows` (hidden rows), each checked against the whole
    /// head, and how many tokens each row kept.
    fn check(&self, what: &str, rows: &[Vec<u16>]) -> Result<Vec<u32>> {
        let flat: Vec<u16> = rows.concat();
        let x = upload(self.gpu, &le_bytes(&flat, u16::to_le_bytes))?;
        // Marked as a step's logits would be, in rows of their own.
        let marks = self.gpu.alloc(rows.len() * VOCAB * 2)?;
        let mark = Mark {
            logits: marks,
            vocab: VOCAB,
            step: 0x0102_0304 + rows.len() as u32,
        };
        let picked = self
            .pick
            .pick(self.gpu, None, x, rows.len(), Some(mark), self.stream)?;
        let picked = self.words(picked, rows.len())?;
        for (r, &token) in picked.iter().enumerate() {
            let mut bytes = [0u8; Mark::BYTES];
            self.gpu.copy_d2h(marks.offset(r * VOCAB * 2), &mut bytes)?;
            assert_eq!(
                Mark::read(&bytes, mark.step),
                Some(token),
                "{what}: row {r}'s mark"
            );
            assert_eq!(
                Mark::read(&bytes, mark.step + 1),
                None,
                "{what}: another step's mark"
            );
        }
        self.gpu.free(marks)?;
        let counts = self.words(self.pick.count, PARTS * ROWS)?;
        let mut kept = Vec::new();
        for (r, &token) in picked.iter().enumerate() {
            let (_, want) = self.whole(x.offset(r * HIDDEN * 2))?;
            kept.push((0..PARTS).map(|p| counts[p * ROWS + r]).sum::<u32>());
            assert_eq!(
                token,
                want,
                "{what}: row {r} of {} picked {token}, the whole head {want} ({} kept)",
                rows.len(),
                kept[r]
            );
        }
        self.gpu.free(x)?;
        println!("{what}: {} rows, tokens kept {kept:?}", rows.len());
        Ok(kept)
    }

    /// Head row `token`, times `gain`: a hidden row that points at it.
    fn toward(&self, token: usize, gain: f32) -> Vec<u16> {
        self.head[token * HIDDEN..(token + 1) * HIDDEN]
            .iter()
            .map(|&w| bf16(bf16_to_f32(w) * gain))
            .collect()
    }
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn the_pick_is_the_whole_heads_argmax() -> Result<()> {
    let mut rng = Rng(0x4ead_91c4);
    let mut head: Vec<u16> = (0..VOCAB * HIDDEN)
        .map(|_| bf16(bell(&mut rng) * 0.03))
        .collect();
    // Rows that are the same, in one part and across parts: their logits
    // tie, and the lowest token of them is the pick.
    let same = [17, 23, N + 5, 3 * N + 9, 3 * N + 600];
    for &token in &same[1..] {
        head.copy_within(same[0] * HIDDEN..(same[0] + 1) * HIDDEN, token * HIDDEN);
    }
    let twin = (2 * N + 77, 2 * N + 31);
    head.copy_within(twin.0 * HIDDEN..(twin.0 + 1) * HIDDEN, twin.1 * HIDDEN);
    let bench = Bench::new(head)?;

    // A hidden row that points at a token: the copy rules nearly all others
    // out, in every part.
    for (i, token) in [0, 1, N - 1, N, 2 * N + 300, VOCAB - 1]
        .into_iter()
        .enumerate()
    {
        let kept = bench.check("a clear pick", &[bench.toward(token, 3.0 + i as f32)])?;
        assert!(
            (kept[0] as usize) < VOCAB / 8,
            "a clear pick kept {} of {VOCAB} tokens",
            kept[0]
        );
    }
    // Ties, where the lowest token wins: among parts and within one.
    bench.check("a tie across parts", &[bench.toward(3 * N + 9, 4.0)])?;
    bench.check("a tie in a part", &[bench.toward(twin.0, 4.0)])?;
    // Rows the copy can rule nothing out of: all logits equal, and logits
    // that lie close together.
    bench.check("a row of zeros", &[vec![0u16; HIDDEN]])?;
    let mut flat = || -> Vec<u16> { (0..HIDDEN).map(|_| bf16(bell(&mut rng) * 0.5)).collect() };
    bench.check("flat rows", &[flat(), flat()])?;
    // Every row count, clear and flat rows mixed.
    for rows in 1..=ROWS {
        let mixed: Vec<Vec<u16>> = (0..rows)
            .map(|r| match r % 3 {
                0 => bench.toward((r * 397 + 11) % VOCAB, 2.5),
                1 => flat(),
                _ => bench.toward(same[r % same.len()], 5.0),
            })
            .collect();
        bench.check("mixed rows", &mixed)?;
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn a_kept_tokens_logit_is_the_heads_bytes() -> Result<()> {
    let mut rng = Rng(0x0b17_e5a1);
    let head: Vec<u16> = (0..VOCAB * HIDDEN)
        .map(|_| bf16(bell(&mut rng) * 0.03))
        .collect();
    let bench = Bench::new(head)?;
    let (gpu, pick, stream) = (bench.gpu, &bench.pick, bench.stream);
    let x: Vec<u16> = (0..HIDDEN).map(|_| bf16(bell(&mut rng) * 2.0)).collect();
    let x = upload(gpu, &le_bytes(&x, u16::to_le_bytes))?;
    let (logits, _) = bench.whole(x)?;
    // A block's four tokens at a time are one token four times, so a block
    // reports that token's logit: a token a block, over every part.
    for part in 0..PARTS {
        let tokens: Vec<u32> = (0..BLOCKS)
            .map(|b| ((b * 131 + part * 7) % N) as u32)
            .collect();
        let list: Vec<u32> = tokens.iter().flat_map(|&t| [t; 4]).collect();
        let count = (list.len() as u32).to_le_bytes();
        gpu.copy_h2d(
            &le_bytes(&list, u32::to_le_bytes),
            pick.list.offset(part * ROWS * N * 4),
        )?;
        gpu.copy_h2d(&count, pick.count.offset(part * ROWS * 4))?;
        KernelLaunch::new(gpu, pick.kernels.exact)
            .grid([BLOCKS as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(pick.head.offset(part * N * HIDDEN * 2))
            .arg_ptr(pick.list.offset(part * ROWS * N * 4))
            .arg_ptr(pick.count.offset(part * ROWS * 4))
            .arg_ptr(pick.found.offset(part * FOUND))
            .arg_u32(N as u32)
            .arg_u32(HIDDEN as u32)
            .arg_u32((part * N) as u32)
            .launch(stream)?;
        let found = bench.words(pick.found.offset(part * FOUND), BLOCKS * 2)?;
        for (b, &token) in tokens.iter().enumerate() {
            let token = part * N + token as usize;
            assert_eq!(found[2 * b + 1] as usize, token, "part {part} block {b}");
            assert_eq!(
                found[2 * b],
                bf16_to_f32(logits[token]).to_bits(),
                "part {part}: token {token}'s logit"
            );
        }
    }
    Ok(())
}

/// The release's head, the hidden rows its reference run gave the head
/// (`final.safetensors` of the full-depth recording), and how long a pick
/// and a share of the head take.
///
///   ATLAS_DSV41_CKPT=<checkpoint> ATLAS_DSV41_REC=<recording> cargo test ... the_releases_head
#[test]
#[ignore = "needs a GB10 GPU, ATLAS_DSV41_CKPT and ATLAS_DSV41_REC"]
fn the_releases_head_is_picked_as_its_argmax() -> Result<()> {
    use crate::layers::deepseek_v41::test_support::checkpoint;
    use crate::weight_map::test_checkpoint::{read_checkpoint_tensor, read_safetensors_tensor};
    use std::path::Path;

    let gpu: &'static dyn GpuBackend = backend()?;
    let stream = gpu.default_stream();
    let (dtype, shape, head) = read_checkpoint_tensor(Path::new(&checkpoint()?), "head.weight")?;
    anyhow::ensure!(
        dtype == "BF16" && shape.len() == 2,
        "head.weight: {dtype} {shape:?}"
    );
    let (vocab, hidden) = (shape[0], shape[1]);
    let n = vocab / PARTS;
    let on_device = upload(gpu, &head)?;
    drop(head);
    let copy = Mxfp8Weight::alloc(gpu, vocab, hidden)?;
    let quantize = gpu.kernel("mxfp8_gemv", "mxfp8_quantize_bf16")?;
    mxfp8_quantize(
        gpu,
        quantize,
        on_device,
        copy.data,
        copy.scales,
        vocab,
        hidden,
        stream,
    )?;
    let pick = HeadPick::load(gpu, on_device, copy, Held::ALL, n, hidden)?;

    let rec = std::env::var("ATLAS_DSV41_REC")?;
    let rec = Path::new(&rec).join("final.safetensors");
    let mut rows = Vec::new();
    for name in ["prefill.final.normed", "decode.final.normed"] {
        let (dtype, _, bytes) = read_safetensors_tensor(&rec, name)?;
        anyhow::ensure!(dtype == "BF16", "{name}: {dtype}");
        rows.extend(bytes);
    }
    let count = rows.len() / (hidden * 2);
    let x = upload(gpu, &rows)?;
    let (gemv, argmax) = (
        gpu.kernel("gemv", "dense_gemv_bf16")?,
        gpu.kernel("argmax", "argmax_bf16")?,
    );
    let (logits, out) = (gpu.alloc(vocab * 2)?, gpu.alloc(4)?);
    let weight = DenseWeight { weight: on_device };
    let words = |p: DevicePtr, n: usize| -> Result<Vec<u32>> {
        let mut bytes = vec![0u8; n * 4];
        gpu.copy_d2h(p, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    };
    let mut kept = Vec::with_capacity(count);
    for first in (0..count).step_by(ROWS) {
        let some = (count - first).min(ROWS);
        let of = x.offset(first * hidden * 2);
        let picked = pick.pick(gpu, None, of, some, None, stream)?;
        let picked = words(picked, some)?;
        kept.extend(pick.kept(gpu, some)?);
        for (r, &token) in picked.iter().enumerate() {
            let row = of.offset(r * hidden * 2);
            dense_gemv(
                gpu,
                gemv,
                row,
                &weight,
                logits,
                vocab as u32,
                hidden as u32,
                stream,
            )?;
            argmax_bf16(gpu, argmax, logits, out, vocab as u32, stream)?;
            assert_eq!(token, words(out, 1)?[0], "row {}", first + r);
        }
    }
    kept.sort_unstable();
    let at = |q: f64| kept[((kept.len() - 1) as f64 * q) as usize];
    println!(
        "{count} rows of the reference run: every pick is the head's argmax; tokens kept of {vocab}: median {}, 90% {}, 99% {}, most {}",
        at(0.5),
        at(0.9),
        at(0.99),
        at(1.0)
    );

    // One row at a time, as a decode step: the pick over all four parts
    // against a part's share of the head, four times.
    const TIMES: usize = 200;
    let timed = |run: &dyn Fn(usize) -> Result<()>| -> Result<f64> {
        gpu.synchronize(stream)?;
        let start = std::time::Instant::now();
        for i in 0..TIMES {
            run(i)?;
        }
        gpu.synchronize(stream)?;
        Ok(start.elapsed().as_secs_f64() * 1e3 / TIMES as f64)
    };
    let a_row = |i: usize| x.offset((i * 7 % count) * hidden * 2);
    let picks = timed(&|i| pick.pick(gpu, None, a_row(i), 1, None, stream).map(|_| ()))?;
    let share = DenseWeight {
        weight: on_device.offset(n * hidden * 2),
    };
    let shares = timed(&|i| {
        dense_gemv(
            gpu,
            gemv,
            a_row(i),
            &share,
            logits,
            n as u32,
            hidden as u32,
            stream,
        )
    })?;
    println!(
        "a row, one GPU: the pick over four parts {picks:.3} ms ({:.3} a part); a part's share of the head {shares:.3} ms",
        picks / PARTS as f64
    );
    Ok(())
}

#[test]
fn a_rows_mark_is_read_as_its_step_wrote_it() {
    // `dsv41_head_pick`'s sixteen bytes for step 0x00050006, token 0x0001f4a0.
    let words: [u16; 8] = [
        0x7fc1, 0x7fc2, 0x7fc3, 0x7fc4, 0x0006, 0x0005, 0xf4a0, 0x0001,
    ];
    let mut bytes = [0u8; Mark::BYTES];
    for (two, word) in bytes.chunks_exact_mut(2).zip(words) {
        two.copy_from_slice(&word.to_le_bytes());
    }
    assert_eq!(Mark::read(&bytes, 0x0005_0006), Some(0x0001_f4a0));
    // Another step's mark, and logits, are not this step's pick.
    assert_eq!(Mark::read(&bytes, 0x0005_0007), None);
    bytes[2] = 0;
    assert_eq!(Mark::read(&bytes, 0x0005_0006), None);
    assert_eq!(Mark::read(&[0u8; Mark::BYTES], 0), None);
}
