// SPDX-License-Identifier: AGPL-3.0-only

//! Reading a packed table back: the bounds-checked header, the per-shard row
//! spot check against the sources, and the chunk checksums.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use atlas_tier::pio::read_exact_at;

use super::fs::{cached_open, drop_cache, open_packed};
use super::write::header_fits;
use super::{FIXED, MAGIC, PackSpec, SAMPLE_CHUNKS, Verify, packed_offset};

/// A packed file's header, bounds-checked.
pub(super) struct Header {
    pub(super) identity: Vec<u8>,
    pub(super) check_rows: u64,
    pub(super) sums: Vec<u32>,
}

pub(super) fn read_header(f: &File) -> Result<Header> {
    let mut head = [0u8; FIXED as usize];
    read_exact_at(f, &mut head, 0).context("ngram pack: read header")?;
    ensure!(&head[..8] == MAGIC, "bad magic");
    let word = |i: usize| u64::from_le_bytes(head[i..i + 8].try_into().unwrap());
    let (len, check_rows, chunks) = (word(8), word(16), word(24));
    // Bounded on the u64s, with no addition: `16 + len` wrapped for a length
    // near u64::MAX in a release build, passed, and `vec![0; len]` aborted.
    header_fits(len, chunks)?;
    ensure!(check_rows > 0, "zero rows per checksum chunk");
    let mut rest = vec![0u8; (len + 4 * chunks) as usize];
    read_exact_at(f, &mut rest, FIXED).context("ngram pack: read identity")?;
    let sums = rest[len as usize..]
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    rest.truncate(len as usize);
    Ok(Header {
        identity: rest,
        check_rows,
        sums,
    })
}

/// Check that `path` is the packed form of `spec`: header identity, length,
/// one sampled row per shard against its sources, and the chunk checksums
/// `verify` asks for.
pub fn validate(spec: &PackSpec<'_>, path: &Path, verify: Verify) -> Result<()> {
    validate_file(spec, &open_packed(path)?, verify)
}

/// [`validate`] on an open file.
pub(super) fn validate_file(spec: &PackSpec<'_>, f: &File, verify: Verify) -> Result<()> {
    let hdr = read_header(f)?;
    ensure!(
        hdr.identity == spec.identity()?.as_bytes(),
        "identity differs from the sources"
    );
    let rows_total = spec.rows_total();
    let want = packed_offset(rows_total, spec.stride);
    let have = f.metadata()?.len();
    ensure!(have == want, "file is {have} bytes, expected {want}");
    ensure!(
        hdr.sums.len() as u64 == rows_total.div_ceil(hdr.check_rows),
        "{} chunk checksums for {rows_total} rows of {} per chunk",
        hdr.sums.len(),
        hdr.check_rows
    );

    let mut files: Vec<(PathBuf, File)> = Vec::new();
    let mut got = vec![0u8; spec.stride];
    let mut exp = vec![0u8; spec.stride];
    for shard in 0..spec.shards() as u64 {
        let row = shard * spec.rows_per_shard + sample_row(shard, spec.rows_per_shard);
        read_exact_at(f, &mut got, packed_offset(row, spec.stride))?;
        exp.fill(0);
        let mut at = 0;
        for (i, p) in spec.planes.iter().enumerate() {
            let (src, off) = spec.source_loc(i, row);
            let sf = cached_open(&mut files, src)?;
            read_exact_at(sf, &mut exp[at..at + p.row_bytes], off)?;
            at += p.row_bytes;
        }
        if got != exp {
            bail!("row {row} (shard {shard}) does not match its sources");
        }
    }

    let n = hdr.sums.len();
    let which: Vec<usize> = match verify {
        Verify::Full => (0..n).collect(),
        Verify::Sample => {
            rand::seq::index::sample(&mut rand::thread_rng(), n, SAMPLE_CHUNKS.min(n)).into_vec()
        }
    };
    verify_chunks(f, &hdr, rows_total, spec.stride, &which)
}

/// Re-hash chunks `which` of the body and compare with the header's sums.
pub(super) fn verify_chunks(
    f: &File,
    hdr: &Header,
    rows_total: u64,
    stride: usize,
    which: &[usize],
) -> Result<()> {
    // Read in 8 MiB pieces whatever the chunk size.
    let piece_rows = ((8u64 << 20) / stride as u64).max(1);
    let mut buf = vec![0u8; (piece_rows.min(hdr.check_rows)) as usize * stride];
    for &c in which {
        let first = c as u64 * hdr.check_rows;
        let end = (first + hdr.check_rows).min(rows_total);
        let mut h = crc32fast::Hasher::new();
        let mut r = first;
        while r < end {
            let n = (end - r).min(buf.len() as u64 / stride as u64);
            let bytes = &mut buf[..n as usize * stride];
            let at = packed_offset(r, stride);
            read_exact_at(f, bytes, at)?;
            drop_cache(f, at, bytes.len() as u64);
            h.update(&*bytes);
            r += n;
        }
        if h.finalize() != hdr.sums[c] {
            bail!("chunk {c} (rows {first}..{end}) fails its checksum: the packed file is damaged");
        }
    }
    Ok(())
}

/// A row of `shard` to spot-check, spread across the shard by a fixed hash so
/// the same rows are checked every open (first and last rows included).
pub(super) fn sample_row(shard: u64, rows: u64) -> u64 {
    match shard {
        0 => 0,
        s if s % 2 == 1 => rows - 1,
        s => s.wrapping_mul(0x9E37_79B9_7F4A_7C15) % rows,
    }
}
