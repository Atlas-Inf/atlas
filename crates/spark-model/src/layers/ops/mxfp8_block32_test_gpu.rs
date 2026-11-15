// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]`; a GB10 and the deepseek-v4.1-flash kernel target)
//! for DeepSeek-V4.1's FP8 projections on the MXFP8 decode tiers.
//!
//! The release scales FP8 per 32 x 32 block. The loader repeats each scale
//! row (`fp8_block_scales_to_rows`) into the per-row layout the tiers read.
//! Here the reference weight is dequantized straight from the block grid, not
//! through that expansion, and the tiers must then pass the same parity and
//! row-invariance checks as checkpoint MXFP8 does.
//!
//!   cargo test --release -p spark-model --lib block32 -- --ignored --nocapture --test-threads=1
//!
//! and the same on the release's own tensors:
//!
//!   ATLAS_DSV41_CKPT=<checkpoint dir> cargo test --release -p spark-model --lib \
//!       block32 -- --ignored --nocapture --test-threads=1

use super::*;
use crate::weight_map::fp8_block_scales_to_rows;
use crate::weight_map::test_checkpoint::read_checkpoint_tensor;

/// `(name, n, k)`: FP8 projections of DeepSeek-V4.1-Flash-1017 (hidden 5120,
/// 64 heads of 512, q rank 1280, o rank 1024 in 8 groups, experts 2304 wide).
const SHAPES: [(&str, usize, usize); 9] = [
    ("wq_a", 1280, 5120),
    ("wq_b tp1", 32768, 1280),
    ("wq_b tp4", 8192, 1280),
    ("wkv", 512, 5120),
    ("wo_a", 8192, 4096),
    ("wo_b", 5120, 8192),
    ("indexer wq_b", 4096, 1280),
    ("shared w1/w3", 2304, 5120),
    ("shared w2", 5120, 2304),
];

/// BF16 bits of E4M3 `values` `[n, k]` under one UE8M0 scale per 32 x 32 block.
fn dequant_blocks(values: &[u8], grid: &[u8], k: usize) -> Vec<u16> {
    let per_row = k / MXFP8_BLOCK;
    values
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let (row, col) = (i / k, i % k);
            let e = grid[row / MXFP8_BLOCK * per_row + col / MXFP8_BLOCK];
            let scale = f32::from_bits(u32::from(e) << 23);
            atlas_core::numeric::f32_to_bf16(atlas_core::numeric::fp8_e4m3_to_f32(b) * scale)
        })
        .collect()
}

#[test]
#[ignore = "needs a GB10 GPU and a kernel target with mxfp8_gemv"]
fn deepseek_v41_block_scaled_fp8_runs_on_the_mxfp8_tiers() -> Result<()> {
    let mut rng = Rng(0x1017_d5f4_1e8c_0b32);
    for (name, n, k) in SHAPES {
        let (values, _, _) = weight(&mut rng, n, k);
        let grid: Vec<u8> = (0..n.div_ceil(MXFP8_BLOCK) * k / MXFP8_BLOCK)
            .map(|_| SCALE_LO + (rng.next() % 8) as u8)
            .collect();
        let rows = fp8_block_scales_to_rows(&grid, n, k)?;
        let bf16 = dequant_blocks(&values, &grid, k);
        assert!(
            bf16 == dequant(&values, &rows, k),
            "{name}: the row scales are not the block grid's weight"
        );
        parity(&mut rng, name, &values, &rows, &bf16, n, k)?;
    }
    Ok(())
}

/// FP8 projections of the release, one of each kind and from both ends of
/// the stack.
const REAL: [&str; 11] = [
    "layers.0.attn.wq_a",
    "layers.0.attn.wq_b",
    "layers.0.attn.wkv",
    "layers.0.attn.wo_a",
    "layers.0.attn.wo_b",
    "layers.0.ffn.shared_experts.w1",
    "layers.0.ffn.shared_experts.w2",
    "layers.20.attn.indexer.wq_b",
    "layers.14.engram.wkv",
    "layers.39.attn.wq_b",
    "mtp.0.main_proj",
];

#[test]
#[ignore = "needs a GB10 GPU, a kernel target with mxfp8_gemv and ATLAS_DSV41_CKPT"]
fn deepseek_v41_released_fp8_runs_on_the_mxfp8_tiers() -> Result<()> {
    let dir = std::env::var("ATLAS_DSV41_CKPT").context("ATLAS_DSV41_CKPT=<checkpoint dir>")?;
    let read = |name: String| read_checkpoint_tensor(std::path::Path::new(&dir), &name);
    let mut rng = Rng(0x1017_0b5e_55ed_f00d);
    for prefix in REAL {
        let (value_dtype, shape, values) = read(format!("{prefix}.weight"))?;
        let (scale_dtype, grid_shape, grid) = read(format!("{prefix}.scale"))?;
        let [n, k] = shape[..] else {
            anyhow::bail!("{prefix}: weight shape {shape:?}");
        };
        anyhow::ensure!(
            value_dtype == "F8_E4M3"
                && scale_dtype == "F8_E8M0"
                && grid_shape == [n.div_ceil(MXFP8_BLOCK), k / MXFP8_BLOCK],
            "{prefix}: {value_dtype} {shape:?} + {scale_dtype} {grid_shape:?} is not block FP8"
        );
        let rows = fp8_block_scales_to_rows(&grid, n, k)?;
        // `dequant` refuses a scale outside the range where BF16 is exact.
        let bf16 = dequant(&values, &rows, k);
        assert!(
            bf16 == dequant_blocks(&values, &grid, k),
            "{prefix}: the row scales are not the block grid's weight"
        );
        parity(&mut rng, prefix, &values, &rows, &bf16, n, k)?;
    }
    Ok(())
}
