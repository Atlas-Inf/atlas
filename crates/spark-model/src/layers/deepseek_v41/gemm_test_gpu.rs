// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]`): rows rounded to codes with their scales as they
//! come, the scales then tiled ([`GemmScratch::coded`]), against the operand
//! [`GemmScratch::quantize`] rounds whole; and some rows of the codes the
//! ranks gathered ([`GemmScratch::gathered_rows`]) against those rows of
//! every row's operand.
//!
//!   cargo test --release -p spark-model --lib deepseek_v41::gemm::test_gpu -- --ignored --nocapture --test-threads=1

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{GemmScratch, Input, Sent};
use crate::layers::deepseek_v41::parts::PARTS;
use crate::layers::deepseek_v41::test_support::{backend, upload};
use crate::layers::ops::{mx_lt_inner_pad, mx_lt_scale_bytes, mx_lt_sf_offset};

/// Rows that end inside a tile of scales, of the model's hidden width.
const ROWS: usize = 200;
const K: usize = 5120;

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn codes_with_their_scales_tiled_are_the_operand_rounded_whole() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    // Values of every size a block's scale can take, and zeros.
    let mut state = 0x2545_f491u32;
    let values: Vec<u8> = (0..ROWS * K)
        .flat_map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let size = [1e-6, 3e-3, 1.0, 40.0, 0.0][(i / 32 + i / K) % 5];
            (((unit * size).to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    let x = upload(gpu, &values)?;
    let scratch = GemmScratch::alloc(gpu, ROWS, K)?;
    let tiled = mx_lt_scale_bytes(ROWS, K);
    let read = |p: DevicePtr, bytes: usize| -> Result<Vec<u8>> {
        gpu.synchronize(stream)?;
        let mut out = vec![0u8; bytes];
        gpu.copy_d2h(p, &mut out)?;
        Ok(out)
    };
    // A tile's bytes past the rows are nobody's: the same before each.
    gpu.memset(scratch.sf, 0, tiled)?;
    let whole = scratch.quantize(gpu, Input::rows(x, K), ROWS, K, stream)?;
    let want = (read(whole.q, ROWS * K)?, read(whole.sf, tiled)?);
    assert!(
        want.1.iter().any(|&scale| scale != 0),
        "no scales to compare"
    );

    let (codes, scales) = (gpu.alloc(ROWS * K)?, gpu.alloc(ROWS * K / 32)?);
    gpu.memset(scratch.sf, 0, tiled)?;
    scratch.codes(gpu, x, codes, scales, ROWS, K, stream)?;
    let coded = scratch.coded(gpu, codes, scales, ROWS, K, stream)?;
    assert!(coded.q == codes, "the codes are read where they are");
    assert!(read(coded.q, ROWS * K)? == want.0, "the codes differ");
    assert!(read(coded.sf, tiled)? == want.1, "the tiled scales differ");
    println!("{} codes and {tiled} bytes of tiled scales equal", ROWS * K);
    for p in [x, codes, scales, scratch.q, scratch.sf] {
        gpu.free(p)?;
    }
    Ok(())
}

/// BF16 values of mixed sizes, `count` of them.
fn values(count: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .flat_map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let size = [3e-3, 1.0, 40.0][(i / 32) % 3];
            (((unit * size).to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

#[test]
#[ignore = "needs a GB10 GPU and the deepseek-v4.1-flash target"]
fn some_rows_of_gathered_codes_are_those_rows_of_the_operand() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    // Two groups of 64 values a part, as wo_a leaves its low-rank output.
    let (groups, piece) = (2usize, 64usize);
    let sent = Sent::new(ROWS, groups, piece);
    let k = PARTS * groups * piece;
    let scratch = GemmScratch::alloc(gpu, ROWS, k)?;
    let parts = gpu.alloc(PARTS * sent.bytes)?;
    let mut inputs = Vec::new();
    for part in 0..PARTS {
        for group in 0..groups {
            let x = upload(
                gpu,
                &values(ROWS * piece, (part * groups + group) as u32 + 7),
            )?;
            let at = parts.offset(part * sent.bytes);
            let codes = at.offset(group * ROWS * piece);
            let scales = at.offset(sent.scales + group * ROWS * piece / 32);
            scratch.codes(gpu, x, codes, scales, ROWS, piece, stream)?;
            inputs.push(x);
        }
    }
    let tiled = mx_lt_scale_bytes(ROWS, k);
    let read = |p: DevicePtr, bytes: usize| -> Result<Vec<u8>> {
        gpu.synchronize(stream)?;
        let mut out = vec![0u8; bytes];
        gpu.copy_d2h(p, &mut out)?;
        Ok(out)
    };
    let every = scratch.gathered(gpu, parts, sent, ROWS, stream)?;
    let every = (read(every.q, ROWS * k)?, read(every.sf, tiled)?);
    let of = 37..150;
    let some = scratch.gathered_rows(gpu, parts, sent, ROWS, of.clone(), stream)?;
    let some = (read(some.q, of.len() * k)?, read(some.sf, tiled)?);
    assert!(
        some.0 == every.0[of.start * k..of.end * k],
        "the rows' codes differ"
    );
    let pad = mx_lt_inner_pad(k);
    for row in 0..of.len() {
        for block in 0..k / 32 {
            let (mine, theirs) = (
                some.1[mx_lt_sf_offset(row, block, pad)],
                every.1[mx_lt_sf_offset(of.start + row, block, pad)],
            );
            assert!(
                mine == theirs && mine != 0,
                "row {row}, block {block}: scale {mine} for {theirs}"
            );
        }
    }
    println!(
        "{} rows of {ROWS}: their codes and scales are the operand's",
        of.len()
    );
    for p in inputs.into_iter().chain([parts, scratch.q, scratch.sf]) {
        gpu.free(p)?;
    }
    Ok(())
}
