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
//!
//! That split costs two NVMe reads per miss. With `ATLAS_PLE_PACKED_DIR` the
//! loader instead interleaves the two planes once (`spark_storage::
//! ngram_pack`) into rows of
//!
//! ```text
//! [ dim/2 B codes | dim/group B scales | zero pad to a 16 B multiple ]
//!   80              10                   6        = 96 B at dim 160
//! ```
//!
//! and the main cache holds those whole rows, with no block-scale arena: one
//! read per miss, and the same bytes reach the same decode.

#[cfg(any(test, feature = "cuda"))]
use anyhow::Context as _;
use anyhow::{Result, ensure};
#[cfg(any(test, feature = "cuda"))]
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

/// An NVFP4 n-gram table's decode geometry; its block-scale rows fault in
/// through the WEIGHT cache's segmented scale source (`attach_segmented_
/// block_scales`), so one CLOCK serves both arenas.
pub struct NgramNvfp4 {
    pub geom: Nvfp4RowGeom,
    /// `(file, base_offset)` of each scale shard — handed to
    /// `NgramRowCache::attach_segmented_block_scales` by the loader.
    pub scale_shards: Vec<(std::path::PathBuf, u64)>,
}

impl Nvfp4RowGeom {
    pub fn packed_stride(&self) -> usize {
        self.dim / 2
    }

    pub fn scale_stride(&self) -> usize {
        self.dim / self.group
    }

    /// Bytes per interleaved row: codes, block scales, zero pad to 16.
    pub fn interleaved_stride(&self) -> usize {
        (self.packed_stride() + self.scale_stride()).next_multiple_of(16)
    }

    /// What `batched_embed_nvfp4` reads for a gather from `table_va`: the
    /// block-scale base, the code-row stride and the scale-row stride. Split
    /// rows find their scales in the cache's block-scale arena; an
    /// interleaved row (no arena) carries them right after its codes.
    pub fn gather_planes(&self, table_va: u64, block_scales: Option<u64>) -> (u64, usize, usize) {
        match block_scales {
            Some(va) => (va, self.packed_stride(), self.scale_stride()),
            None => (
                table_va + self.packed_stride() as u64,
                self.interleaved_stride(),
                self.interleaved_stride(),
            ),
        }
    }
}

/// The weight cache holds either split rows (codes only, with the segmented
/// block-scale arena attached) or interleaved rows (codes and scales, no
/// arena). Any other pairing gathers the wrong bytes.
pub fn check_packed_stride(stride: usize, spec: &Nvfp4RowGeom, scaled: bool) -> Result<()> {
    let (want, layout) = if scaled {
        (spec.packed_stride(), "codes beside the block-scale arena")
    } else {
        (
            spec.interleaved_stride(),
            "interleaved codes + scales, no arena",
        )
    };
    ensure!(
        stride == want,
        "PLE: NVFP4 row stride {stride} != {want} ({layout}) for {} dims",
        spec.dim
    );
    Ok(())
}

/// Decode gathered rows — the host ORACLE for the GPU microtest, the
/// host-side cache tests, and the `ATLAS_PLE_RESIDENT` boot self-check; the
/// serving path decodes in `batched_embed_nvfp4`. `packed[i]` and
/// `scales[i]` are row `i`'s bytes from the weight and block-scale caches.
/// Returns little-endian BF16.
#[cfg(any(test, feature = "cuda"))]
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

#[cfg(test)]
#[path = "ngram_nvfp4_tests.rs"]
mod tests;
