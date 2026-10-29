// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time decoder for `csf_32x2_v1` NVFP4 block scales.
//!
//! The successor of the slab encoding in [`super::nvfp4_csf`]
//! (`weight_scale_encoding: "csf_32x2_v1"`, nvidia GLM-5.3-Flash-FP4-Local's
//! routed experts). A logical E4M3 scale matrix `[R, C]` (`C = K/16` for a
//! packed weight `[R, K/2]`) ships as five 1-D components:
//!
//! | Suffix              | Dtype | Contents                                     |
//! |---------------------|-------|----------------------------------------------|
//! | `.csf32_bases`      | U8    | `[R]` one palette base per row, `0..=240`     |
//! | `.csf32_codes`      | U8    | `[R*C/2]` 4-bit offsets in block order        |
//! | `.csf32_page_bases` | U32   | `[P]` BYTE origin of each page in the arena   |
//! | `.csf32_offsets`    | U16   | `[B+P]` per-page record prefixes per block    |
//! | `.csf32_exceptions` | U16   | `[n]` `(value << 8) \| (2*local_row + local_col)` |
//!
//! Blocks are 32 rows x 2 columns (the last is one column wide when `C` is
//! odd), grouped into 128 x 4 tiles visited row-major; within a tile the first
//! column pair's (up to four) row blocks precede the second's. Each block's
//! offsets are row-major, LOW nibble first, so a full block is one code byte
//! per row. A page is 32 whole tiles (the last may be short); its prefix array
//! starts at 0 and ends with a terminal entry, so `B` blocks in `P` pages give
//! `B + P` prefixes. A scale byte is `base + offset`, then each exception
//! replaces the byte at its block-local position.
//!
//! The decode is lossless, so every NVFP4 consumer reads `weight_scale`
//! unchanged, exactly as for the slab encoding.
//!
//! Format reference (read, not copied — the arithmetic is the format): the
//! Local Inference Lab FlashInfer fork at commit 7e203806 (Apache-2.0),
//! `docs/csf-scale-blocks.md` and `scripts/encode_csf.py` (`_geometry`,
//! `encode_scales`, `decode_scales`). Its `decode_scales` is the golden
//! reference in `scripts/nvfp4_csf32_golden.py`; its `encode_scales` is
//! re-expressed in the tests for synthetic round trips.
//!
//! The spec's MXFP4 variant (1-bit offsets, bases `0..=254`) is not handled:
//! no MXFP4 checkpoint uses it here, and an MXFP4 set would fail the
//! weight-shape check (`C = K/32`) rather than decode as NVFP4.

use anyhow::{Context, Result, ensure};
use std::collections::HashMap;

use super::nvfp4_csf::{MAX_BASE, check_decoded_scales};

/// Component suffixes, in [`Csf32Parts`] order.
pub const CSF32_SUFFIXES: [&str; 5] = [
    ".csf32_bases",
    ".csf32_codes",
    ".csf32_page_bases",
    ".csf32_offsets",
    ".csf32_exceptions",
];
/// Element width of each component, in [`CSF32_SUFFIXES`] order.
pub(crate) const CSF32_ELEM_BYTES: [usize; 5] = [1, 1, 4, 2, 2];

/// Rows per block.
const BLOCK_ROWS: usize = 32;
/// Row blocks per tile (a tile is 128 rows).
const TILE_ROW_BLOCKS: usize = 4;
/// Column pairs per tile (a tile is four columns).
const TILE_COL_PAIRS: usize = 2;
/// Tiles (groups) per index page.
const PAGE_GROUPS: usize = 32;

/// The five components of one scale, as raw little-endian bytes.
#[derive(Clone, Copy)]
pub struct Csf32Parts<'a> {
    pub bases: &'a [u8],
    pub codes: &'a [u8],
    pub page_bases: &'a [u8],
    pub offsets: &'a [u8],
    pub exceptions: &'a [u8],
}

/// `[R, C]` of the decoded matrix: `R` is the row-base count and `C` follows
/// from the code stream's `R * C / 2` bytes. The format stores `C` nowhere
/// else; the offset count must agree with it ([`Geometry::decode`]) and the
/// packed weight pins it (`CsfAssembler::finish`).
pub fn shape(parts: &Csf32Parts<'_>) -> Result<[usize; 2]> {
    shape_of(parts.bases.len(), parts.codes.len())
}

/// [`shape`] from the component lengths alone: `rows` bases, `code_bytes`
/// code bytes (what a header gives for deferred components).
pub(crate) fn shape_of(rows: usize, code_bytes: usize) -> Result<[usize; 2]> {
    ensure!(
        rows > 0 && rows.is_multiple_of(BLOCK_ROWS),
        "csf32: {rows} row bases — R must be a positive multiple of {BLOCK_ROWS}"
    );
    let fields = code_bytes
        .checked_mul(2)
        .context("csf32: code stream length overflows")?;
    ensure!(
        fields > 0 && fields.is_multiple_of(rows),
        "csf32: {code_bytes} code bytes is not R * C / 2 for R = {rows}"
    );
    Ok([rows, fields / rows])
}

/// One 32-row block, in stream order.
#[derive(Clone, Copy)]
struct Block {
    /// Flat `row * C + col` of its top-left scale.
    first: usize,
    /// Its first row (the row base index of its local row 0).
    row: usize,
    /// 2, or 1 for the final column of an odd-`C` matrix.
    width: usize,
    /// Byte offset of its codes in the code stream.
    code: usize,
}

/// One row of tiles: up to 128 rows of the matrix.
struct Band {
    /// Its first row.
    row: usize,
    /// `(code offset, first column)` of each FULL tile — four row blocks of
    /// two full columns, i.e. 256 code bytes: column pair 0's 128 rows, then
    /// column pair 1's — decoded by the fast path.
    full: Vec<(usize, usize)>,
    /// The blocks of every other tile (a short band, or a final one-pair or
    /// single-column tile), decoded block by block.
    partial: Vec<usize>,
}

/// Block order and page boundaries for one `[R, C]`, computed once per shape
/// (GLM has two shapes across its 36,288 matrices).
pub struct Geometry {
    rows: usize,
    cols: usize,
    blocks: Vec<Block>,
    bands: Vec<Band>,
    /// Index of each page's first block, then the block count.
    pages: Vec<usize>,
}

impl Geometry {
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        ensure!(
            rows > 0 && rows.is_multiple_of(BLOCK_ROWS) && cols > 0,
            "csf32: [{rows}, {cols}] — R must be a positive multiple of {BLOCK_ROWS} and C positive"
        );
        rows.checked_mul(cols)
            .with_context(|| format!("csf32: [{rows}, {cols}] overflows"))?;
        let (row_blocks, col_pairs) = (rows / BLOCK_ROWS, cols.div_ceil(2));
        let mut blocks: Vec<Block> = Vec::with_capacity(row_blocks * col_pairs);
        let mut bands = Vec::new();
        let mut pages = Vec::new();
        let (mut code, mut tiles) = (0usize, 0usize);
        for tr in (0..row_blocks).step_by(TILE_ROW_BLOCKS) {
            let mut band = Band {
                row: tr * BLOCK_ROWS,
                full: Vec::new(),
                partial: Vec::new(),
            };
            for tc in (0..col_pairs).step_by(TILE_COL_PAIRS) {
                if tiles.is_multiple_of(PAGE_GROUPS) {
                    pages.push(blocks.len());
                }
                tiles += 1;
                let first_block = blocks.len();
                for cp in tc..col_pairs.min(tc + TILE_COL_PAIRS) {
                    for rb in tr..row_blocks.min(tr + TILE_ROW_BLOCKS) {
                        let (row, col) = (rb * BLOCK_ROWS, cp * 2);
                        let width = (cols - col).min(2);
                        blocks.push(Block {
                            first: row * cols + col,
                            row,
                            width,
                            code,
                        });
                        code += BLOCK_ROWS * width / 2;
                    }
                }
                let full = blocks.len() - first_block == TILE_ROW_BLOCKS * TILE_COL_PAIRS
                    && blocks[blocks.len() - 1].width == 2;
                if full {
                    let b = blocks[first_block];
                    band.full.push((b.code, b.first - b.row * cols));
                } else {
                    band.partial.extend(first_block..blocks.len());
                }
            }
            bands.push(band);
        }
        pages.push(blocks.len());
        Ok(Self {
            rows,
            cols,
            blocks,
            bands,
            pages,
        })
    }

    /// Page count `P`.
    fn page_count(&self) -> usize {
        self.pages.len() - 1
    }

    /// Decode one scale of this shape to its E4M3 bytes, row-major `[R, C]`.
    ///
    /// Every structural rule of the format is checked, and every decoded byte
    /// must be a non-negative finite E4M3 scale: a scale that decodes wrong
    /// produces an expert that runs and is wrong.
    pub fn decode(&self, parts: &Csf32Parts<'_>) -> Result<Vec<u8>> {
        let (rows, cols) = (self.rows, self.cols);
        let (nb, np) = (self.blocks.len(), self.page_count());
        // rows * cols fits (checked in `new`); every length below is smaller.
        let lens = [
            ("bases", parts.bases.len(), rows),
            ("codes", parts.codes.len(), rows * cols / 2),
            ("page_bases", parts.page_bases.len(), 4 * np),
            ("offsets", parts.offsets.len(), 2 * (nb + np)),
        ];
        for (what, got, want) in lens {
            ensure!(
                got == want,
                "csf32 {what}: {got} bytes, but a [{rows}, {cols}] matrix ({nb} blocks in \
                 {np} pages) needs {want}"
            );
        }
        ensure!(
            parts.exceptions.len().is_multiple_of(2),
            "csf32 exceptions: {} bytes is not a whole number of u16 records",
            parts.exceptions.len()
        );
        check_arena(parts.exceptions.len())?;
        if let Some(r) = parts.bases.iter().position(|&b| b > MAX_BASE) {
            anyhow::bail!(
                "csf32 row {r} base {:#04x} exceeds {MAX_BASE} — corrupt bases",
                parts.bases[r]
            );
        }

        let mut out = vec![0u8; rows * cols];
        self.expand_codes(parts.bases, parts.codes, &mut out);
        self.apply_exceptions(parts, &mut out)?;
        check_decoded_scales(&out, cols, "csf32")?;
        Ok(out)
    }

    /// `base + offset` for every scale. The lengths were checked by the
    /// caller, so every block's codes are there.
    fn expand_codes(&self, bases: &[u8], codes: &[u8], out: &mut [u8]) {
        const BAND_ROWS: usize = TILE_ROW_BLOCKS * BLOCK_ROWS;
        const RUN: usize = 4;
        let cols = self.cols;
        // A code byte's two nibbles as two bytes, low first.
        let spread = |c: u8| u32::from(c & 0x0F) | (u32::from(c >> 4) << 8);
        for band in &self.bands {
            // Full tiles: row i of a tile is byte i of column pair 0 and byte
            // i of pair 1 (128 bytes on), four scales in one u32. No byte
            // carries into its neighbour: base <= 240, offset <= 15. Words are
            // computed a tile at a time (vectorizes), then stored four tiles
            // per row, so each 128-row band costs one 16-byte store per row
            // per four tiles rather than a scattered store per tile.
            if !band.full.is_empty() {
                let rows = band.row..band.row + BAND_ROWS;
                let bases: [u32; BAND_ROWS] =
                    std::array::from_fn(|i| u32::from(bases[rows.start + i]) * 0x0101_0101);
                let dst = &mut out[rows.start * cols..rows.end * cols];
                let mut words = [[0u32; BAND_ROWS]; RUN];
                let mut rest = &band.full[..];
                while let Some(&(_, col)) = rest.first() {
                    let n = rest
                        .iter()
                        .take(RUN)
                        .zip((col..).step_by(4))
                        .take_while(|(t, c)| t.1 == *c)
                        .count();
                    for (w, &(code, _)) in words.iter_mut().zip(&rest[..n]) {
                        let src: &[u8; 2 * BAND_ROWS] =
                            codes[code..code + 2 * BAND_ROWS].try_into().unwrap();
                        for i in 0..BAND_ROWS {
                            let (c0, c1) = (src[i], src[BAND_ROWS + i]);
                            w[i] = (spread(c0) | (spread(c1) << 16)) + bases[i];
                        }
                    }
                    let rows = dst.chunks_exact_mut(cols).enumerate();
                    if n == RUN {
                        for (i, orow) in rows {
                            let mut v = [0u8; 4 * RUN];
                            for (o, w) in v.chunks_exact_mut(4).zip(&words) {
                                o.copy_from_slice(&w[i].to_le_bytes());
                            }
                            orow[col..col + 4 * RUN].copy_from_slice(&v);
                        }
                    } else {
                        for (i, orow) in rows {
                            for (k, w) in words[..n].iter().enumerate() {
                                orow[col + 4 * k..col + 4 * k + 4]
                                    .copy_from_slice(&w[i].to_le_bytes());
                            }
                        }
                    }
                    rest = &rest[n..];
                }
            }
            for b in band.partial.iter().map(|&i| &self.blocks[i]) {
                let bases = &bases[b.row..b.row + BLOCK_ROWS];
                if b.width == 2 {
                    // One byte per row: column 0 low nibble, column 1 high.
                    let src = &codes[b.code..b.code + BLOCK_ROWS];
                    for (i, (&code, &base)) in src.iter().zip(bases).enumerate() {
                        let o = b.first + i * cols;
                        out[o] = base + (code & 0x0F);
                        out[o + 1] = base + (code >> 4);
                    }
                } else {
                    // One column: rows 2j (low nibble) and 2j+1 (high) per byte.
                    let src = &codes[b.code..b.code + BLOCK_ROWS / 2];
                    for (j, &code) in src.iter().enumerate() {
                        let o = b.first + 2 * j * cols;
                        out[o] = bases[2 * j] + (code & 0x0F);
                        out[o + cols] = bases[2 * j + 1] + (code >> 4);
                    }
                }
            }
        }
    }

    /// Replace the bytes the exception records name, validating the page
    /// index as it goes.
    fn apply_exceptions(&self, parts: &Csf32Parts<'_>, out: &mut [u8]) -> Result<()> {
        let u16_at = |s: &[u8], i: usize| u16::from_le_bytes([s[2 * i], s[2 * i + 1]]);
        let cols = self.cols;
        let records = parts.exceptions.len() / 2;
        // Records consumed so far: pages tile the arena in order, so each
        // page must start exactly where the previous one ended.
        let mut cursor = 0usize;
        for p in 0..self.page_count() {
            let (b0, b1) = (self.pages[p], self.pages[p + 1]);
            let w = &parts.page_bases[4 * p..4 * p + 4];
            let origin = u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize;
            ensure!(
                origin.is_multiple_of(2),
                "csf32 page {p} origin byte {origin} is odd — records are u16"
            );
            ensure!(
                origin / 2 == cursor,
                "csf32 page {p} origin byte {origin} is not where the previous page's records \
                 end (byte {}) — pages overlap or leave a gap in the arena",
                2 * cursor
            );
            // This page's prefixes: one per block plus the terminal entry.
            let prefix = |i: usize| u16_at(parts.offsets, b0 + p + i) as usize;
            ensure!(
                prefix(0) == 0,
                "csf32 page {p} prefixes start at {}, not 0",
                prefix(0)
            );
            let count = prefix(b1 - b0);
            ensure!(
                cursor + count <= records,
                "csf32 page {p} holds records {cursor}..{} but the arena has {records}",
                cursor + count
            );
            for (i, b) in self.blocks[b0..b1].iter().enumerate() {
                let (lo, hi) = (prefix(i), prefix(i + 1));
                ensure!(
                    hi >= lo,
                    "csf32 page {p} prefixes decrease at block {} ({lo} -> {hi})",
                    b0 + i
                );
                // Checked before the records are read: a prefix past the
                // page's own count would read another page's records, or past
                // the arena, even if the prefixes come back down later.
                ensure!(
                    hi <= count,
                    "csf32 page {p} prefix {hi} at block {} exceeds the page's {count} records",
                    b0 + i + 1
                );
                let mut prev: Option<usize> = None;
                for r in cursor + lo..cursor + hi {
                    let rec = u16_at(parts.exceptions, r);
                    let local = (rec & 0xFF) as usize;
                    ensure!(
                        local < 2 * BLOCK_ROWS && local % 2 < b.width,
                        "csf32 exception {r} at local {local} is outside block {} \
                         ({BLOCK_ROWS} rows x {} column(s))",
                        b0 + i,
                        b.width
                    );
                    ensure!(
                        prev.is_none_or(|q| local > q),
                        "csf32 exception {r} at local {local} in block {} is out of order \
                         (previous {prev:?}) — positions must be sorted and unique",
                        b0 + i
                    );
                    prev = Some(local);
                    out[b.first + (local / 2) * cols + local % 2] = (rec >> 8) as u8;
                }
            }
            cursor += count;
        }
        ensure!(
            cursor == records,
            "csf32 arena holds {records} records but the pages index {cursor}"
        );
        Ok(())
    }
}

/// The exception arena must be smaller than 4 GiB: page origins are u32 byte
/// offsets into it.
fn check_arena(bytes: usize) -> Result<()> {
    ensure!(
        (bytes as u64) < 1 << 32,
        "csf32 exception arena is {bytes} bytes; the format requires < 4 GiB"
    );
    Ok(())
}

/// [`Geometry`] per shape, kept across a load.
#[derive(Default)]
pub struct GeometryCache(HashMap<[usize; 2], Geometry>);

impl GeometryCache {
    /// Decode one scale, reusing this shape's geometry.
    pub fn decode(&mut self, parts: &Csf32Parts<'_>) -> Result<(Vec<u8>, [usize; 2])> {
        let [rows, cols] = shape(parts)?;
        let geometry = match self.0.entry([rows, cols]) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(Geometry::new(rows, cols)?),
        };
        Ok((geometry.decode(parts)?, [rows, cols]))
    }
}

/// Decode one scale without a cache (tests, one-offs).
pub fn decode_nvfp4_csf32(parts: &Csf32Parts<'_>) -> Result<(Vec<u8>, [usize; 2])> {
    GeometryCache::default().decode(parts)
}

#[cfg(test)]
#[path = "nvfp4_csf32_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "nvfp4_csf32_refusal_tests.rs"]
mod refusal_tests;
