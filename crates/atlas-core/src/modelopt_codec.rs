// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side decoders for the storage formats NVIDIA's ModelOpt "FP4-Local"
//! packs use and Atlas kernels do not read directly:
//!
//! * **CSF NVFP4 block scales** (`weight_scale_encoding: "csf"`, codec
//!   `byte-window4-fixed-stream-u24-exceptions/1`). Each E4M3 block-scale
//!   matrix `[rows, cols]` ships as two tensors:
//!   - `<name>.nvfp4_csf_fixed`, U8 `[rows / 16, 16 * (1 + cols / 2)]`: per
//!     16-row slab, one base byte per row, then those rows' 4-bit codes, two
//!     per byte, low nibble first. A scale byte is `base + code`.
//!   - `<name>.nvfp4_csf_exceptions`, U32: words whose low 24 bits are a flat
//!     `row * cols + col` position and whose high 8 bits replace the byte
//!     there.
//!
//!   The decode is lossless: the bytes equal the uncompressed E4M3 scales, so
//!   every NVFP4 kernel consumes them unchanged. Format as specified by
//!   vLLM's `csf_scales.py` (Apache-2.0, vllm-project/vllm#60646); this is an
//!   independent Rust implementation of that published layout.
//!
//! * **MXFP8** (OCP microscaling): an E4M3 weight `[out, in]` with one E8M0
//!   exponent per 32 inputs, `[out, in / 32]`. Expanded to BF16 EXACTLY: an
//!   E4M3 value has 4 significant bits and the scale is a power of two, so the
//!   product always fits BF16's 8.
//!
//! * **NVFP4 rows → BF16**, for a table gathered by row (the PLE n-gram
//!   embedding), where the block scales live in a separate tensor.
//!
//! Pure arithmetic, no CUDA, no environment lookups in the hot loops.

use crate::numeric::{FP4_E2M1, FP8_E4M3_LUT};
use anyhow::{Result, bail, ensure};

/// Rows per CSF slab; every slab stores 16 base bytes ahead of its codes.
pub const CSF_SLAB_ROWS: usize = 16;
/// Largest legal row base: a base above 240 cannot reach 255 with a nibble.
pub const CSF_MAX_BASE: u8 = 240;
/// Inputs per MXFP8 scale.
pub const MXFP8_BLOCK: usize = 32;

/// Shape `[rows, cols]` of the E4M3 scale matrix a CSF fixed stream encodes.
pub fn csf_scale_shape(fixed_rows: usize, fixed_cols: usize) -> Result<(usize, usize)> {
    ensure!(
        fixed_cols.is_multiple_of(CSF_SLAB_ROWS) && fixed_cols >= CSF_SLAB_ROWS,
        "CSF fixed stream width {fixed_cols} is not 16 * (1 + cols / 2)"
    );
    let cols = (fixed_cols / CSF_SLAB_ROWS - 1) * 2;
    ensure!(cols > 0, "CSF fixed stream encodes zero columns");
    Ok((fixed_rows * CSF_SLAB_ROWS, cols))
}

/// Rebuild a `[rows, cols]` E4M3 block-scale matrix (as raw bytes) from its
/// CSF fixed stream `[fixed_rows, fixed_cols]` and little-endian U32
/// exception words.
pub fn decode_csf_scales(
    fixed: &[u8],
    fixed_rows: usize,
    fixed_cols: usize,
    exceptions: &[u8],
) -> Result<Vec<u8>> {
    let (rows, cols) = csf_scale_shape(fixed_rows, fixed_cols)?;
    ensure!(
        fixed.len() == fixed_rows * fixed_cols,
        "CSF fixed stream is {} bytes, shape [{fixed_rows}, {fixed_cols}] needs {}",
        fixed.len(),
        fixed_rows * fixed_cols
    );
    ensure!(
        exceptions.len().is_multiple_of(4),
        "CSF exception stream is {} bytes, not whole U32 words",
        exceptions.len()
    );
    let half = cols / 2;
    let mut out = vec![0u8; rows * cols];
    for (slab, src) in fixed.chunks_exact(fixed_cols).enumerate() {
        let (bases, codes) = src.split_at(CSF_SLAB_ROWS);
        for (i, &base) in bases.iter().enumerate() {
            if base > CSF_MAX_BASE {
                bail!("CSF row base {base} exceeds {CSF_MAX_BASE}");
            }
            let row = slab * CSF_SLAB_ROWS + i;
            let packed = &codes[i * half..(i + 1) * half];
            let dst = &mut out[row * cols..(row + 1) * cols];
            for (pair, &byte) in dst.chunks_exact_mut(2).zip(packed) {
                pair[0] = base + (byte & 0x0F);
                pair[1] = base + (byte >> 4);
            }
        }
    }
    for word in exceptions.chunks_exact(4) {
        let w = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        let pos = (w & 0x00FF_FFFF) as usize;
        ensure!(
            pos < out.len(),
            "CSF exception position {pos} is outside the {rows}x{cols} scale matrix"
        );
        out[pos] = (w >> 24) as u8;
    }
    Ok(out)
}

/// The top 16 bits of an f32 whose low 16 bits are already zero (or whose
/// rounding the caller has accepted). Round-to-nearest-even; NaN stays NaN.
#[inline(always)]
fn bf16_rne(val: f32) -> u16 {
    let bits = val.to_bits();
    if val.is_nan() {
        return ((bits >> 16) as u16 & 0x8000) | 0x7FC0;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

/// `2^(e - 127)` for an E8M0 exponent byte. `0xFF` is NaN in OCP MX; it maps
/// to zero for the same reason the E4M3 table maps NaN to zero.
#[inline(always)]
fn e8m0_to_f32(e: u8) -> f32 {
    match e {
        0xFF => 0.0,
        // 2^-127 is an f32 subnormal: build it from its bits.
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits((e as u32) << 23),
    }
}

/// Expand an MXFP8 weight `[out, inp]` (E4M3 bytes) with its E8M0 scales
/// `[out, inp / 32]` into little-endian BF16 bytes `[out, inp]`. Exact.
pub fn mxfp8_to_bf16(weight: &[u8], scale: &[u8], out: usize, inp: usize) -> Result<Vec<u8>> {
    ensure!(
        inp.is_multiple_of(MXFP8_BLOCK),
        "MXFP8 input width {inp} is not a multiple of {MXFP8_BLOCK}"
    );
    let blocks = inp / MXFP8_BLOCK;
    ensure!(
        weight.len() == out * inp && scale.len() == out * blocks,
        "MXFP8 weight {} / scale {} bytes do not match [{out}, {inp}]",
        weight.len(),
        scale.len()
    );
    let mut dst = vec![0u8; out * inp * 2];
    for ((w, s), d) in weight
        .chunks_exact(MXFP8_BLOCK)
        .zip(scale)
        .zip(dst.chunks_exact_mut(MXFP8_BLOCK * 2))
    {
        let mult = e8m0_to_f32(*s);
        for (&b, o) in w.iter().zip(d.chunks_exact_mut(2)) {
            o.copy_from_slice(&bf16_rne(FP8_E4M3_LUT[b as usize] * mult).to_le_bytes());
        }
    }
    Ok(dst)
}

/// Dequantize NVFP4 rows to little-endian BF16. `packed` is `[rows, dim / 2]`
/// (low nibble first), `scale` is `[rows, dim / group]` E4M3, `global` is the
/// tensor's `weight_scale_2`. Writes `rows * dim` BF16 values into `dst`.
pub fn nvfp4_rows_to_bf16(
    packed: &[u8],
    scale: &[u8],
    global: f32,
    dim: usize,
    group: usize,
    dst: &mut [u8],
) -> Result<()> {
    ensure!(
        dim.is_multiple_of(2) && group > 0 && dim.is_multiple_of(group),
        "NVFP4 row width {dim} must be even and a multiple of group {group}"
    );
    let rows = packed.len() / (dim / 2);
    ensure!(
        packed.len() == rows * dim / 2
            && scale.len() == rows * dim / group
            && dst.len() == rows * dim * 2,
        "NVFP4 rows: packed {} / scale {} / dst {} bytes disagree on {rows} rows of {dim}",
        packed.len(),
        scale.len(),
        dst.len()
    );
    for (i, (&byte, o)) in packed.iter().zip(dst.chunks_exact_mut(4)).enumerate() {
        let elem = i * 2;
        // `(fp4 * block) * global`, the order `numeric::nvfp4_dequant` uses:
        // the first product is exact, so there is exactly one f32 rounding.
        let block = FP8_E4M3_LUT[scale[elem / group] as usize];
        let lo = bf16_rne(FP4_E2M1[(byte & 0x0F) as usize] * block * global);
        let hi = bf16_rne(FP4_E2M1[(byte >> 4) as usize] * block * global);
        o[..2].copy_from_slice(&lo.to_le_bytes());
        o[2..].copy_from_slice(&hi.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
#[path = "modelopt_codec_tests.rs"]
mod tests;
