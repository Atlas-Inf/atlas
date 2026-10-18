// SPDX-License-Identifier: AGPL-3.0-only

//! Host dequant of NVFP4 n-gram rows (NVIDIA's Qwen3.8-Flash-Next-FP4-Local).
//!
//! That pack stores the PLE table as NVFP4: each shard is a U8 `[R, dim/2]`
//! weight (two E2M1 values per byte, low nibble first) with an E4M3
//! `[R, dim/16]` block-scale tensor beside it, and the table carries one F32
//! `weight_scale_2`. The main row cache holds the packed weight rows; a
//! second cache, segmented identically, holds the block-scale rows. A gather
//! resolves both, decodes on the host to the BF16 `[n, dim]` embedding the
//! PLE GEMM expects, and uploads it into `emb` — the same contract as the
//! EXL3 trellis path, so `gather_embed` skips its kernel for both.
//!
//! 26.8 GiB on disk against the NVFP4 pack's 47.7 GiB FP8 table, with the
//! same row count: fewer bytes per fault, more rows per page-cache GiB.

use anyhow::{Context, Result, ensure};
use atlas_core::modelopt_codec::nvfp4_rows_to_bf16;

/// Row geometry of an NVFP4 n-gram table.
#[derive(Debug, Clone, Copy)]
pub struct Nvfp4RowGeom {
    /// Decoded row width (160 for Flash-Next).
    pub dim: usize,
    /// Elements per E4M3 block scale (16).
    pub group: usize,
    /// The table's `weight_scale_2`.
    pub global: f32,
}

/// An NVFP4 n-gram table's geometry plus its block-scale row cache.
pub struct NgramNvfp4 {
    pub geom: Nvfp4RowGeom,
    /// Block-scale rows, `dim / group` bytes each, at the same row ids as the
    /// weight cache.
    #[cfg(feature = "cuda")]
    pub scales: std::sync::Mutex<spark_storage::NgramRowCache>,
}

impl Nvfp4RowGeom {
    pub fn packed_stride(&self) -> usize {
        self.dim / 2
    }

    pub fn scale_stride(&self) -> usize {
        self.dim / self.group
    }
}

/// The weight cache must hold packed rows and nothing else: a scale array
/// would select the FP8 gather, and a wrong stride reads the next row's bytes.
pub fn check_packed_stride(stride: usize, spec: &Nvfp4RowGeom, scaled: bool) -> Result<()> {
    ensure!(
        !scaled,
        "PLE: NVFP4 n-gram rows take their scales from a block-scale table, \
         not a cache scale array"
    );
    ensure!(
        stride == spec.packed_stride(),
        "PLE: NVFP4 row stride {stride} != {} packed bytes for {} dims",
        spec.packed_stride(),
        spec.dim
    );
    Ok(())
}

/// Decode gathered rows: `packed[i]` and `scales[i]` are row `i`'s bytes from
/// the weight and block-scale caches. Returns little-endian BF16 `[n, dim]`.
pub fn decode_rows_bf16(
    spec: &Nvfp4RowGeom,
    packed: &[Vec<u8>],
    scales: &[Vec<u8>],
) -> Result<Vec<u8>> {
    ensure!(
        packed.len() == scales.len(),
        "PLE nvfp4: {} weight rows but {} scale rows",
        packed.len(),
        scales.len()
    );
    let (pw, sw) = (spec.packed_stride(), spec.scale_stride());
    let row_bytes = spec.dim * 2;
    let mut out = vec![0u8; packed.len() * row_bytes];
    for (i, (p, s)) in packed.iter().zip(scales).enumerate() {
        ensure!(
            p.len() == pw && s.len() == sw,
            "PLE nvfp4: row {i} is {}+{} bytes, expected {pw}+{sw}",
            p.len(),
            s.len()
        );
        let dst = &mut out[i * row_bytes..(i + 1) * row_bytes];
        nvfp4_rows_to_bf16(p, s, spec.global, spec.dim, spec.group, dst)
            .with_context(|| format!("PLE nvfp4: row {i}"))?;
    }
    Ok(out)
}

/// Resolve `ids` in the block-scale cache, copy each row out, and release the
/// pins at once: the decode that consumes them runs synchronously on the host.
#[cfg(feature = "cuda")]
pub fn gather_scales(spec: &NgramNvfp4, ids: &[u64]) -> Result<Vec<Vec<u8>>> {
    let mut cache = spec
        .scales
        .lock()
        .map_err(|_| anyhow::anyhow!("PLE nvfp4 scale cache mutex poisoned"))?;
    let mut slots = Vec::with_capacity(ids.len());
    cache.resolve(ids, &mut slots)?;
    let rows = slots
        .iter()
        .map(|&s| cache.copy_slot(s))
        .collect::<Result<Vec<_>>>();
    cache.end_batch();
    rows
}

#[cfg(test)]
#[path = "ngram_nvfp4_tests.rs"]
mod tests;
