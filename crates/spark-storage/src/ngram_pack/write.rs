// SPDX-License-Identifier: AGPL-3.0-only

//! Writing a packed table: rows interleaved pass by pass, chunk checksums
//! computed on the way, and the header last.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use atlas_tier::pio::{read_exact_at, write_all_at};

use super::fs::{cached_open, drop_cache, nofollow};
use super::{FIXED, HEADER, MAGIC, PackSpec, interleave, packed_offset};

/// Write the packed form of `spec` to `out_path`: rows read `io_rows` at a
/// time, checksummed every `check_rows`, and the header LAST, so a file cut
/// short anywhere fails the magic. `out_path` must not exist: the file is
/// created exclusively, so neither a planted file nor a symlink is written.
pub(super) fn pack(
    spec: &PackSpec<'_>,
    out_path: &Path,
    io_rows: u64,
    check_rows: u64,
) -> Result<()> {
    let out = nofollow(OpenOptions::new().write(true).create_new(true))
        .open(out_path)
        .with_context(|| format!("ngram pack: create {}", out_path.display()))?;
    let identity = spec.identity()?;
    let rows_total = spec.rows_total();
    let chunks = rows_total.div_ceil(check_rows);
    // Refuse before 29 GB of writes, not after.
    header_fits(identity.len() as u64, chunks)?;

    let widths: Vec<usize> = spec.planes.iter().map(|p| p.row_bytes).collect();
    let chunk = io_rows.max(1);
    let mut srcs: Vec<Vec<u8>> = widths
        .iter()
        .map(|w| vec![0u8; chunk as usize * w])
        .collect();
    let mut buf = vec![0u8; chunk as usize * spec.stride];
    let mut files: Vec<(PathBuf, File)> = Vec::new();
    let mut sums = ChunkSums::new(check_rows, spec.stride);
    for shard in 0..spec.shards() {
        let mut local = 0u64;
        while local < spec.rows_per_shard {
            let n = (spec.rows_per_shard - local).min(chunk) as usize;
            let row = shard as u64 * spec.rows_per_shard + local;
            for (i, src) in srcs.iter_mut().enumerate() {
                let (path, off) = spec.source_loc(i, row);
                let f = cached_open(&mut files, path)?;
                let bytes = &mut src[..n * widths[i]];
                read_exact_at(f, bytes, off).with_context(|| {
                    format!(
                        "ngram pack: read plane {i} shard {shard} @ {off} of {}",
                        path.display()
                    )
                })?;
                drop_cache(f, off, bytes.len() as u64);
            }
            let planes: Vec<&[u8]> = srcs.iter().zip(&widths).map(|(s, w)| &s[..n * w]).collect();
            let dst = &mut buf[..n * spec.stride];
            interleave(&planes, &widths, spec.stride, dst);
            sums.feed(row, dst);
            let at = packed_offset(row, spec.stride);
            write_all_at(&out, dst, at).context("ngram pack: write rows")?;
            // Bound the dirty page cache: on GB10 it is the same memory the
            // GPU allocates from, and a 27 GB burst of dirty pages is not
            // something a loading model should have to compete with.
            out.sync_data().context("ngram pack: sync")?;
            drop_cache(&out, at, dst.len() as u64);
            local += n as u64;
        }
    }
    let sums = sums.finish(rows_total)?;
    let mut header = vec![0u8; HEADER as usize];
    header[..8].copy_from_slice(MAGIC);
    header[8..16].copy_from_slice(&(identity.len() as u64).to_le_bytes());
    header[16..24].copy_from_slice(&check_rows.to_le_bytes());
    header[24..32].copy_from_slice(&chunks.to_le_bytes());
    let id_end = FIXED as usize + identity.len();
    header[FIXED as usize..id_end].copy_from_slice(identity.as_bytes());
    for (i, sum) in sums.iter().enumerate() {
        header[id_end + 4 * i..id_end + 4 * (i + 1)].copy_from_slice(&sum.to_le_bytes());
    }
    write_all_at(&out, &header, 0).context("ngram pack: write header")?;
    out.sync_all().context("ngram pack: final sync")?;
    Ok(())
}

/// Refuse a header whose identity and `chunks` checksums would not fit.
/// Every term is bounded before it is added: these come from a file.
pub(super) fn header_fits(identity_len: u64, chunks: u64) -> Result<()> {
    ensure!(
        identity_len <= HEADER - FIXED,
        "header length {identity_len} out of range"
    );
    let room = (HEADER - FIXED - identity_len) / 4;
    ensure!(
        chunks <= room,
        "header holds {room} chunk checksums beside a {identity_len}-byte identity, \
         the table needs {chunks}"
    );
    Ok(())
}

/// Running per-chunk CRC32s over rows fed in increasing order.
///
/// CRC32, not a cryptographic hash: this catches a file damaged after it was
/// written (a missed bit has odds of 2^-32), and nothing stored beside the
/// data could stop someone who can rewrite the file anyway. It also has to be
/// cheap enough to run on every open: 7.8 GB/s on the ARMv8 CRC32
/// instructions, against 0.37 GB/s for the tree's software SHA-256.
struct ChunkSums {
    check_rows: u64,
    stride: usize,
    next_row: u64,
    hasher: crc32fast::Hasher,
    sums: Vec<u32>,
}

impl ChunkSums {
    fn new(check_rows: u64, stride: usize) -> Self {
        Self {
            check_rows,
            stride,
            next_row: 0,
            hasher: crc32fast::Hasher::new(),
            sums: Vec::new(),
        }
    }

    /// Rows `row..` of the table, packed (`bytes` is whole rows).
    fn feed(&mut self, row: u64, mut bytes: &[u8]) {
        debug_assert_eq!(row, self.next_row, "rows must arrive in order");
        while !bytes.is_empty() {
            let left_in_chunk = self.check_rows - self.next_row % self.check_rows;
            let take = (bytes.len() as u64 / self.stride as u64).min(left_in_chunk);
            let (now, rest) = bytes.split_at(take as usize * self.stride);
            self.hasher.update(now);
            self.next_row += take;
            if self.next_row.is_multiple_of(self.check_rows) {
                self.sums.push(std::mem::take(&mut self.hasher).finalize());
            }
            bytes = rest;
        }
    }

    fn finish(mut self, rows_total: u64) -> Result<Vec<u32>> {
        ensure!(
            self.next_row == rows_total,
            "ngram pack: checksummed {} of {rows_total} rows",
            self.next_row
        );
        if !rows_total.is_multiple_of(self.check_rows) {
            self.sums.push(self.hasher.finalize());
        }
        Ok(self.sums)
    }
}
