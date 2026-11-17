// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`): rows rounded to codes with their scales as they
//! come, the scales then tiled ([`GemmScratch::coded`]), against the operand
//! [`GemmScratch::quantize`] rounds whole.
//!
//!   cargo test --release -p spark-model --lib deepseek_v41::gemm::test_gpu -- --ignored --nocapture --test-threads=1

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{GemmScratch, Input};
use crate::layers::deepseek_v41::test_support::{backend, upload};
use crate::layers::ops::mx_lt_scale_bytes;

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
