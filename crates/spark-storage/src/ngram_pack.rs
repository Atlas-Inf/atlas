// SPDX-License-Identifier: AGPL-3.0-only

//! One-time repack of a SPLIT-ROW n-gram table into one row-major file.
//!
//! Some checkpoints store each table row in more than one tensor. The NVFP4
//! PLE table of Qwen3.8-Flash-Next-FP4-Local is the case this exists for:
//! every shard is `weight` U8 `[R, 80]` (the E2M1 codes) plus `weight_scale`
//! F8_E4M3 `[R, 10]` (the block scales), at unrelated offsets of the file. A
//! row-cache miss read in place would therefore cost TWO NVMe reads, and the
//! PLE stall on Spark is per-IO latency (the APST wake that
//! `ATLAS_PLE_NVME_KEEPALIVE_MS` exists for, and ~74 us per serial fault),
//! not bytes. So the planes are interleaved ONCE into a derived file, with
//! every row `[plane 0 | plane 1 | zero pad]` at a fixed stride, and the row
//! cache opens that file as an ordinary contiguous table: one read per miss.
//!
//! The derived file never goes into the checkpoint directory, which stays
//! byte-identical to the published one; the caller picks the directory.
//!
//! ```text
//! [0, 8)                 MAGIC
//! [8, 16)                u64 LE identity length L
//! [16, 24)               u64 LE rows per checksum chunk V
//! [24, 32)               u64 LE chunk count C = ceil(rows / V)
//! [32, 32 + L)           identity JSON
//! [32 + L, 32 + L + 4C)  per-chunk CRC32s, u32 LE
//! ... 64 KiB             zero pad
//! [64 KiB + r*stride)    row r: plane bytes in plane order, then zeros to stride
//! ```
//!
//! The identity JSON names every source (resolved file name, length, mtime,
//! inode, offset per shard), the geometry and a caller tag. Opening an
//! existing file requires the identity to match exactly, a sample of rows
//! (one per shard) to match the sources byte for byte, and a sample of the
//! chunk checksums to hold, so a re-published checkpoint with the same layout
//! but new values is caught rather than served stale, and so is a packed file
//! damaged after it was written. [`Verify::Full`] checks every chunk.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

mod check;
mod fs;
mod write;

pub use check::validate;
use check::validate_file;
use fs::{
    DirLock, ensure_free_space, file_id, file_stamp, fnv1a64, open_packed, remove_stale_temps,
};
use write::pack;

/// Header bytes ahead of row 0: a multiple of the 4 KiB O_DIRECT block, so
/// row 0 starts aligned, and room for the identity and chunk checksums of a
/// 128-shard table (~6 KB + ~5 KB; `identity_of_the_real_table_fits`).
pub const HEADER: u64 = 64 * 1024;
/// Version 2 added the chunk checksums and the sources' mtime and inode. A
/// version-1 file fails the magic and is rebuilt in place: its name, which
/// depends on the layout only, is unchanged.
const MAGIC: &[u8; 8] = b"ATLASPK\x02";
/// Bytes of the fixed header fields ahead of the identity.
const FIXED: u64 = 32;
/// Packed bytes per checksum chunk: 262144 rows of the 96-byte NVFP4 table,
/// 1221 chunks (4.9 KB of checksums) for its 320M rows.
const CHECK_BYTES: u64 = 24 << 20;
/// Chunks [`Verify::Sample`] re-hashes on every open: 16 of the NVFP4
/// table's 1221 (1.3%, 384 MiB), picked at random so successive opens cover
/// different parts of the file.
const SAMPLE_CHUNKS: usize = 16;

/// How much of a packed file's checksummed body an open re-reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verify {
    /// A fixed number of random chunks.
    Sample,
    /// Every chunk: the whole table is read once (`ATLAS_PLE_PACKED_VERIFY=full`
    /// for the PLE table).
    Full,
}

impl Verify {
    /// `full` is [`Verify::Full`]; anything else, unset included, samples.
    pub fn parse(v: Option<&str>) -> Self {
        if v == Some("full") {
            Self::Full
        } else {
            Self::Sample
        }
    }
}

/// One plane of a split-row table: where each shard's first row sits, and how
/// many bytes this plane holds per row. Shards may live in different files and
/// need not be laid out in order.
pub struct Plane<'a> {
    pub shards: &'a [(PathBuf, u64)],
    pub row_bytes: usize,
}

/// What to pack. `planes` are concatenated per row in the order given.
pub struct PackSpec<'a> {
    pub planes: Vec<Plane<'a>>,
    pub rows_per_shard: u64,
    /// Bytes per packed row, `>=` the planes' sum; the tail is zero.
    pub stride: usize,
    /// Format label, part of the identity and the file name (e.g. "nvfp4g16").
    pub tag: &'a str,
}

/// A packed table on disk, as [`open_or_pack`] validated it.
#[derive(Debug, Clone)]
pub struct PackedTable {
    pub path: PathBuf,
    pub base_offset: u64,
    pub rows_total: u64,
    pub stride: usize,
    /// `(st_dev, st_ino)` of the file that passed validation.
    file_id: (u64, u64),
}

impl PackedTable {
    fn new(spec: &PackSpec<'_>, path: PathBuf, validated: &File) -> Result<Self> {
        Ok(Self {
            path,
            base_offset: HEADER,
            rows_total: spec.rows_total(),
            stride: spec.stride,
            file_id: file_id(validated)?,
        })
    }

    /// Refuse `f` unless it is the very file [`open_or_pack`] validated: the
    /// row cache opens the table again by path (with O_DIRECT), and a file
    /// renamed over that path in between was never checked.
    pub fn check_same_file(&self, f: &File) -> Result<()> {
        let id = file_id(f)?;
        ensure!(
            id == self.file_id,
            "ngram pack: {} was replaced after it was validated (inode {:?}, validated {:?})",
            self.path.display(),
            id,
            self.file_id
        );
        Ok(())
    }

    /// The row cache over this table, opened on the validated file.
    #[cfg(feature = "cuda")]
    pub fn open_row_cache(&self, slots: usize) -> Result<crate::NgramRowCache> {
        let cache = crate::NgramRowCache::open_at(
            &self.path,
            self.base_offset,
            None,
            self.rows_total,
            self.stride,
            slots,
        )?;
        self.check_same_file(cache.backing_file())?;
        Ok(cache)
    }
}

impl PackSpec<'_> {
    fn shards(&self) -> usize {
        self.planes.first().map_or(0, |p| p.shards.len())
    }

    pub fn rows_total(&self) -> u64 {
        self.shards() as u64 * self.rows_per_shard
    }

    fn check(&self) -> Result<()> {
        ensure!(!self.planes.is_empty(), "ngram pack: no planes");
        ensure!(self.rows_per_shard > 0, "ngram pack: zero rows per shard");
        let n = self.shards();
        ensure!(n > 0, "ngram pack: no shards");
        for (i, p) in self.planes.iter().enumerate() {
            ensure!(
                p.shards.len() == n,
                "ngram pack: plane {i} has {} shards, plane 0 has {n}",
                p.shards.len()
            );
            ensure!(p.row_bytes > 0, "ngram pack: plane {i} has zero row bytes");
        }
        let used: usize = self.planes.iter().map(|p| p.row_bytes).sum();
        ensure!(
            used <= self.stride,
            "ngram pack: planes need {used} bytes per row, stride is {}",
            self.stride
        );
        Ok(())
    }

    /// The layout half of the identity: sources by resolved file name and
    /// length, offsets and geometry. It names the file, so a rebuild for a
    /// source that changed in place overwrites the stale table instead of
    /// leaving a 29 GB orphan beside it.
    ///
    /// Sources are named by their RESOLVED file name: in a Hugging Face cache
    /// the snapshot entry is a symlink to a blob named by its content hash, so
    /// a different revision of the same layout gets a different identity
    /// without anything having to read 27 GB.
    fn layout_identity(&self) -> Result<(String, Vec<PathBuf>)> {
        self.check()?;
        let mut files: Vec<&Path> = Vec::new();
        let mut real: Vec<PathBuf> = Vec::new();
        let mut desc = Vec::new();
        let mut planes = Vec::new();
        for p in &self.planes {
            let mut shards = Vec::with_capacity(p.shards.len());
            for (path, off) in p.shards {
                let idx = match files.iter().position(|f| *f == path.as_path()) {
                    Some(i) => i,
                    None => {
                        let r = std::fs::canonicalize(path)
                            .with_context(|| format!("ngram pack: resolve {}", path.display()))?;
                        let len = std::fs::metadata(&r)
                            .with_context(|| format!("ngram pack: stat {}", r.display()))?
                            .len();
                        let name = r.file_name().map(|n| n.to_string_lossy().into_owned());
                        desc.push(serde_json::json!({"name": name, "len": len}));
                        files.push(path.as_path());
                        real.push(r);
                        files.len() - 1
                    }
                };
                shards.push(serde_json::json!([idx, off]));
            }
            planes.push(serde_json::json!({"row_bytes": p.row_bytes, "shards": shards}));
        }
        let layout = serde_json::json!({
            "format": "atlas-ngram-pack/1",
            "tag": self.tag,
            "rows_per_shard": self.rows_per_shard,
            "stride": self.stride,
            "files": desc,
            "planes": planes,
        })
        .to_string();
        Ok((layout, real))
    }

    /// The JSON that a packed file must carry to be reused for this spec: the
    /// layout plus each source's mtime (ns) and inode, so a source rewritten
    /// in place with the same name and length is a different identity.
    pub fn identity(&self) -> Result<String> {
        let (layout, real) = self.layout_identity()?;
        let stat = real
            .iter()
            .map(|r| {
                let m = std::fs::metadata(r)
                    .with_context(|| format!("ngram pack: stat {}", r.display()))?;
                Ok(file_stamp(&m))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(format!(
            "{{\"layout\":{layout},\"stat\":{}}}",
            serde_json::Value::from(stat)
        ))
    }

    /// `ngram-{tag}-{fnv64(layout identity)}.bin`: one file per distinct layout.
    pub fn file_name(&self) -> Result<String> {
        Ok(format!(
            "ngram-{}-{:016x}.bin",
            self.tag,
            fnv1a64(self.layout_identity()?.0.as_bytes())
        ))
    }

    /// Rows per checksum chunk: [`CHECK_BYTES`] of packed rows.
    fn check_rows(&self) -> u64 {
        (CHECK_BYTES / self.stride.max(1) as u64).max(1)
    }

    /// Byte offset of row `row` of plane `plane` in its source file.
    pub fn source_loc(&self, plane: usize, row: u64) -> (&Path, u64) {
        let p = &self.planes[plane];
        let (path, base) = &p.shards[(row / self.rows_per_shard) as usize];
        (
            path,
            base + (row % self.rows_per_shard) * p.row_bytes as u64,
        )
    }
}

/// Byte offset of packed row `row`.
pub fn packed_offset(row: u64, stride: usize) -> u64 {
    HEADER + row * stride as u64
}

/// Interleave `rows` rows: `planes[i]` holds `rows * row_bytes[i]` bytes, and
/// `out` receives `rows * stride` bytes with each row's tail zeroed.
pub fn interleave(planes: &[&[u8]], row_bytes: &[usize], stride: usize, out: &mut [u8]) {
    let rows = out.len() / stride;
    for r in 0..rows {
        let dst = &mut out[r * stride..(r + 1) * stride];
        let mut at = 0;
        for (src, &w) in planes.iter().zip(row_bytes) {
            dst[at..at + w].copy_from_slice(&src[r * w..(r + 1) * w]);
            at += w;
        }
        dst[at..].fill(0);
    }
}

/// Return the packed table for `spec` under `dir`, building it first if no
/// valid one is there. `verify` is how much of an existing file's body the
/// open re-hashes ([`Verify`]).
///
/// A valid file is used without taking the lock, so a table packed at install
/// time can sit on a read-only mount: builds write a temp file and rename it,
/// so the final name only ever holds a complete file. A build holds an
/// exclusive lock on `<file>.lock`, so two processes on one host build once.
/// Nothing in `dir` is opened through a symlink, and the returned table names
/// the validated file by inode ([`PackedTable::check_same_file`]).
pub fn open_or_pack(spec: &PackSpec<'_>, dir: &Path, verify: Verify) -> Result<PackedTable> {
    let name = spec.file_name()?;
    let path = dir.join(&name);
    let existing = |verify| -> Result<PackedTable> {
        let f = open_packed(&path)?;
        validate_file(spec, &f, verify)?;
        PackedTable::new(spec, path.clone(), &f)
    };
    if let Ok(table) = existing(verify) {
        return Ok(table);
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("ngram pack: create {}", dir.display()))?;
    let _lock = DirLock::take(&dir.join(format!("{name}.lock")))?;
    // Re-check under the lock: another process may have just finished.
    if std::fs::symlink_metadata(&path).is_ok() {
        match existing(verify) {
            Ok(table) => return Ok(table),
            Err(e) => tracing::warn!("ngram pack: {} is stale, rebuilding: {e:#}", path.display()),
        }
    }
    let need = packed_offset(spec.rows_total(), spec.stride);
    ensure_free_space(dir, need)?;
    let tmp = dir.join(format!("{name}.tmp{}", std::process::id()));
    let t0 = std::time::Instant::now();
    tracing::info!(
        "ngram pack: building {} ({:.1} GB, {} rows x {} B) — one-time",
        path.display(),
        need as f64 / 1e9,
        spec.rows_total(),
        spec.stride
    );
    // Under the lock, a leftover temp file is a crashed build's: drop it.
    remove_stale_temps(dir, &name);
    if let Err(e) = pack(spec, &tmp, CHUNK_ROWS, spec.check_rows()) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("ngram pack: rename to {}", path.display()))?;
    let table =
        existing(Verify::Sample).context("ngram pack: freshly built file failed validation")?;
    tracing::info!(
        "ngram pack: built {} in {:.1}s",
        path.display(),
        t0.elapsed().as_secs_f64()
    );
    Ok(table)
}

/// Rows copied per pass. 262144 rows of the NVFP4 table are 21 MB + 2.6 MB in
/// and 25 MB out: big enough for sequential throughput, small enough that the
/// three buffers stay off the radar of a host whose GPU shares its memory.
const CHUNK_ROWS: u64 = 262_144;

#[cfg(test)]
mod tests;
