// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time decoder for CSF-compressed NVFP4 block scales.
//!
//! CSF checkpoints (`quantized_layers.*.weight_scale_encoding = "csf"`, codec
//! `byte-window4-fixed-stream-u24-exceptions/1`) do not ship an NVFP4
//! projection's E4M3 `weight_scale [n, k/16]`. They ship two tensors instead:
//!
//! * `<name>.nvfp4_csf_fixed` — U8 `[rows/16, 16 * (1 + cols/2)]`. Per 16-row
//!   slab: one base byte per row, then each row's 4-bit codes, two per byte,
//!   LOW nibble first. A scale byte is `base + code`.
//! * `<name>.nvfp4_csf_exceptions` — U32, sorted words whose low 24 bits are a
//!   flat `row * cols + col` position and whose high 8 bits REPLACE the scale
//!   byte there (the bytes outside the row's 16-wide window).
//!
//! The decode is lossless: the output is the uncompressed checkpoint's E4M3
//! bytes exactly, so every NVFP4 consumer reads `<name>` unchanged and never
//! learns the checkpoint was compressed. That is why it lives here, at upload,
//! rather than in any model loader: Qwen3.8-Flash-Next-FP4-Local's 73,728
//! routed-expert scales use it, and so does the GLM CSF checkpoint.
//!
//! Format references (both read, neither copied — the arithmetic is the format):
//! * vLLM PR 60646, `vllm/model_executor/model_loader/csf_scales.py`
//!   (Apache-2.0), as carried by `eugr/spark-vllm-docker`'s
//!   `docker/patches/qwen3.8/vllm/14-pr60646.patch`. Its decoder is the golden
//!   reference in `scripts/nvfp4_csf_golden.py`.
//! * flashinfer `flashinfer/experimental/b12x/_lib/quant/nvfp4_csf.py`
//!   (Apache-2.0) — the GPU expander. Its `make_nvfp4_csf_batch` is where the
//!   stricter checks below come from: exception positions must be unique,
//!   sorted and in range, and the position field must be able to address the
//!   whole matrix.
//!
//! Only the byte-window codec exists in checkpoints. flashinfer also has two
//! resident-only codecs (`codec` 1/2, 8-row selector/mantissa planes) that are
//! never serialized; a tensor named like a CSF part but shaped like neither is
//! refused by the shape checks rather than guessed at. MXFP4 has no CSF form.
//!
//! The newer block encoding (`"csf_32x2_v1"`, five components per scale) is
//! decoded by [`super::nvfp4_csf32`]; [`CsfAssembler`] gathers and decodes
//! both, so a loader routes every part of either encoding through one place.

use anyhow::{Context, Result, bail, ensure};

use super::nvfp4_csf32::{CSF32_ELEM_BYTES, CSF32_SUFFIXES};

#[path = "nvfp4_csf_assembler.rs"]
mod assembler;
pub use assembler::{CsfAssembler, DecodedScale, decode_parts};

/// Suffix of the fixed stream.
pub const CSF_FIXED_SUFFIX: &str = ".nvfp4_csf_fixed";
/// Suffix of the exception stream.
pub const CSF_EXCEPTIONS_SUFFIX: &str = ".nvfp4_csf_exceptions";

/// Row bases above this would carry into the next byte when a code is added
/// (the reference packs four decoded bytes in one u32 add and relies on it).
/// Shared with `csf_32x2_v1`, whose NVFP4 bases have the same bound.
pub(crate) const MAX_BASE: u8 = 240;
/// Rows per fixed-stream slab.
const SLAB: usize = 16;
/// The exception position field is 24 bits wide.
const POSITION_BITS: u32 = 24;

/// How a compressed scale is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsfEncoding {
    /// `"csf"`: a fixed stream and an exception stream (this module).
    Slab,
    /// `"csf_32x2_v1"`: five components ([`super::nvfp4_csf32`]).
    Blocks32,
}

impl CsfEncoding {
    fn part_count(self) -> usize {
        match self {
            Self::Slab => 2,
            Self::Blocks32 => CSF32_SUFFIXES.len(),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Slab => "CSF",
            Self::Blocks32 => "csf_32x2_v1",
        }
    }
}

/// Which component of a compressed scale a tensor name is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsfPart {
    /// Slab: the U8 fixed stream.
    Fixed,
    /// Slab: the U32 exception words.
    Exceptions,
    /// `csf_32x2_v1`: U8 row bases.
    Bases,
    /// `csf_32x2_v1`: U8 packed 4-bit offsets.
    Codes,
    /// `csf_32x2_v1`: U32 page origins (bytes) in the exception arena.
    PageBases,
    /// `csf_32x2_v1`: U16 per-page record prefixes.
    Offsets,
    /// `csf_32x2_v1`: U16 exception records.
    BlockExceptions,
}

/// Every component suffix with its part.
const PARTS: [(&str, CsfPart); 7] = [
    (CSF_FIXED_SUFFIX, CsfPart::Fixed),
    (CSF_EXCEPTIONS_SUFFIX, CsfPart::Exceptions),
    (CSF32_SUFFIXES[0], CsfPart::Bases),
    (CSF32_SUFFIXES[1], CsfPart::Codes),
    (CSF32_SUFFIXES[2], CsfPart::PageBases),
    (CSF32_SUFFIXES[3], CsfPart::Offsets),
    (CSF32_SUFFIXES[4], CsfPart::BlockExceptions),
];

impl CsfPart {
    pub fn encoding(self) -> CsfEncoding {
        match self {
            Self::Fixed | Self::Exceptions => CsfEncoding::Slab,
            _ => CsfEncoding::Blocks32,
        }
    }

    /// Position among its encoding's parts.
    fn slot(self) -> usize {
        match self {
            Self::Fixed | Self::Bases => 0,
            Self::Exceptions | Self::Codes => 1,
            Self::PageBases => 2,
            Self::Offsets => 3,
            Self::BlockExceptions => 4,
        }
    }

    /// Bytes per element.
    pub fn elem_bytes(self) -> usize {
        match self.encoding() {
            CsfEncoding::Slab => [1, 4][self.slot()],
            CsfEncoding::Blocks32 => CSF32_ELEM_BYTES[self.slot()],
        }
    }
}

/// The component suffixes and parts of `encoding`, in slot order.
pub fn parts_of(encoding: CsfEncoding) -> impl Iterator<Item = (&'static str, CsfPart)> {
    PARTS
        .into_iter()
        .filter(move |(_, p)| p.encoding() == encoding)
}

/// `Some((decoded_name, part))` when `name` is a component of a compressed
/// scale, in either encoding. The decoded name is the tensor the components
/// stand in for (`...weight_scale`).
pub fn csf_part(name: &str) -> Option<(&str, CsfPart)> {
    PARTS
        .iter()
        .find_map(|&(suffix, part)| name.strip_suffix(suffix).map(|base| (base, part)))
}

/// What a component of this shape costs once its scale is decoded and
/// uploaded: the whole decoded matrix for the one part that sizes it (the
/// slab fixed stream, or the `csf_32x2_v1` codes at two scales per byte), 0
/// for the others. Used by the pre-flight estimate, which would otherwise
/// count the 3.6 GB compressed and miss 3.9 GB of the 7.5 GB that lands on
/// the device (Qwen FP4-Local). `None` for a shape whose size overflows.
pub fn decoded_len(part: CsfPart, shape: &[usize]) -> Option<usize> {
    match part {
        CsfPart::Fixed => geometry(shape).ok().map(|(_, _, total)| total),
        CsfPart::Codes => shape.iter().try_fold(2usize, |n, &d| n.checked_mul(d)),
        _ => Some(0),
    }
}

/// `(rows, cols, rows * cols)` of the decoded matrix, from the fixed stream's
/// shape. The shape comes from a file header, so every product is checked:
/// unchecked, a hostile header wraps in a release build.
fn geometry(fixed_shape: &[usize]) -> Result<(usize, usize, usize)> {
    ensure!(
        fixed_shape.len() == 2,
        "CSF fixed stream must be 2-D, got {fixed_shape:?}"
    );
    let (slabs, width) = (fixed_shape[0], fixed_shape[1]);
    ensure!(
        slabs > 0 && width % SLAB == 0 && width / SLAB >= 2,
        "CSF fixed stream {fixed_shape:?} is not [rows/16, 16 * (1 + cols/2)]"
    );
    let rows = slabs.checked_mul(SLAB);
    let cols = (width / SLAB - 1).checked_mul(2);
    let (Some(rows), Some(cols)) = (rows, cols) else {
        bail!("CSF fixed stream {fixed_shape:?}: decoded size overflows");
    };
    let total = rows
        .checked_mul(cols)
        .with_context(|| format!("CSF fixed stream {fixed_shape:?}: decoded size overflows"))?;
    Ok((rows, cols, total))
}

/// Decode one CSF pair to the E4M3 `[rows, cols]` scale matrix, as raw bytes.
///
/// `exceptions` is the U32 stream's raw little-endian bytes. Every check that
/// can fail does so loudly: a scale that decodes wrong produces an expert that
/// runs and is wrong, which is the failure this codebase most wants to avoid.
pub fn decode_nvfp4_csf(
    fixed: &[u8],
    fixed_shape: &[usize],
    exceptions: &[u8],
) -> Result<(Vec<u8>, [usize; 2])> {
    let (rows, cols, total) = geometry(fixed_shape)?;
    let width = fixed_shape[1];
    // `rows * cols` did not overflow, and the fixed stream is smaller.
    let need = fixed_shape[0] * width;
    ensure!(
        fixed.len() == need,
        "CSF fixed stream holds {} bytes but its shape {fixed_shape:?} needs {need}",
        fixed.len(),
    );
    ensure!(
        exceptions.len().is_multiple_of(4),
        "CSF exception stream is {} bytes, not a whole number of u32 words",
        exceptions.len()
    );
    ensure!(
        total <= 1 << POSITION_BITS,
        "CSF {rows}x{cols} matrix cannot be addressed by a {POSITION_BITS}-bit position"
    );

    let mut out = vec![0u8; total];
    let half = cols / 2;
    for (slab, src) in fixed.chunks_exact(width).enumerate() {
        let (bases, codes) = src.split_at(SLAB);
        for (r, (&base, row_codes)) in bases.iter().zip(codes.chunks_exact(half)).enumerate() {
            ensure!(
                base <= MAX_BASE,
                "CSF row {} base {base:#04x} exceeds {MAX_BASE} — corrupt fixed stream",
                slab * SLAB + r
            );
            let row = slab * SLAB + r;
            let dst = &mut out[row * cols..(row + 1) * cols];
            for (pair, &packed) in dst.chunks_exact_mut(2).zip(row_codes) {
                pair[0] = base + (packed & 0x0F);
                pair[1] = base + (packed >> 4);
            }
        }
    }

    let mut prev: Option<usize> = None;
    for (i, w) in exceptions.chunks_exact(4).enumerate() {
        let word = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        let pos = (word & ((1 << POSITION_BITS) - 1)) as usize;
        ensure!(
            pos < total,
            "CSF exception {i} addresses {pos}, past the {rows}x{cols} matrix"
        );
        ensure!(
            prev.is_none_or(|p| pos > p),
            "CSF exception {i} at {pos} is out of order (previous {prev:?}) — the \
             stream must be sorted and unique"
        );
        prev = Some(pos);
        out[pos] = (word >> POSITION_BITS) as u8;
    }

    check_decoded_scales(&out, cols, "CSF")?;
    Ok((out, [rows, cols]))
}

/// Refuse a decoded scale matrix holding a byte that is not a non-negative
/// finite E4M3 scale. NVFP4 block scales are amax-derived, so non-negative
/// and finite: an E4M3 byte with the sign bit set or the 0x7F NaN code cannot
/// come from a sound encoder. Checking costs one pass over bytes already in
/// cache and catches a flipped high bit that every structural check would let
/// through. Shared by both encodings.
pub(crate) fn check_decoded_scales(out: &[u8], cols: usize, what: &str) -> Result<()> {
    // A max over chunks vectorizes; `position` alone does not, and was most
    // of the slab decode's time. It runs only to name the bad byte.
    if out
        .chunks(4096)
        .any(|c| c.iter().fold(0u8, |m, &b| m.max(b)) >= 0x7F)
        && let Some(at) = out.iter().position(|&b| b >= 0x7F)
    {
        bail!(
            "{what} decoded scale {:#04x} at row {} col {} is negative or NaN — corrupt stream",
            out[at],
            at / cols,
            at % cols
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "nvfp4_csf_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "csf_checkpoint_tests.rs"]
mod checkpoint_tests;
