// SPDX-License-Identifier: AGPL-3.0-only
//
//! Streaming load of a segmented n-gram table straight into DEVICE memory —
//! the `ATLAS_PLE_RESIDENT=1` path. Each `(path, base_offset)` shard is one
//! contiguous `bytes_per_shard` block; shard `i` lands at
//! `dst + i * bytes_per_shard`. File bytes move through a reusable pinned
//! staging buffer (bounded: a few × 256 MiB, never the whole table), then
//! each backing file is `posix_fadvise(DONTNEED)`ed and closed so its page
//! cache does not eat unified memory.

use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};

#[cfg(feature = "cuda")]
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Pinned staging chunk size. Small enough that a handful of chunks is cheap,
/// large enough that pread + H2D stay in big sequential moves.
const STAGE_BYTES: usize = 256 * 1024 * 1024;
/// Pinned chunks used per load — two let a read overlap the previous copy.
const STAGE_CHUNKS: usize = 2;

/// Stream `shards` (each `bytes_per_shard` at its file offset) into the
/// device buffer `dst`, packed back-to-back in shard order. All file I/O
/// stays here (SBIO): the caller owns only the device allocation.
#[cfg(feature = "cuda")]
pub fn load_segments_to_device(
    shards: &[(PathBuf, u64)],
    bytes_per_shard: u64,
    dst: DevicePtr,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    ensure!(
        !shards.is_empty() && bytes_per_shard > 0,
        "resident n-gram load: empty shard list or zero shard size"
    );
    // One descriptor per distinct backing file (128 shards over ~7 files),
    // shard -> file index in first-use order.
    let mut paths: Vec<&PathBuf> = Vec::new();
    let mut shard_file: Vec<usize> = Vec::with_capacity(shards.len());
    for (path, _) in shards {
        let idx = paths.iter().position(|p| *p == path).unwrap_or_else(|| {
            paths.push(path);
            paths.len() - 1
        });
        shard_file.push(idx);
    }
    let mut files = Vec::with_capacity(paths.len());
    for path in &paths {
        files.push(
            std::fs::File::open(path)
                .with_context(|| format!("resident n-gram load: open {}", path.display()))?,
        );
    }
    let mut stage = Vec::with_capacity(STAGE_CHUNKS);
    for _ in 0..STAGE_CHUNKS {
        let p = gpu
            .alloc_host_pinned(STAGE_BYTES)
            .context("resident n-gram load: pinned staging")?;
        stage.push(p);
    }
    // Frees the pinned staging on every exit path, error included.
    struct StageGuard<'a> {
        gpu: &'a dyn GpuBackend,
        ptrs: Vec<*mut u8>,
    }
    impl Drop for StageGuard<'_> {
        fn drop(&mut self) {
            for &p in &self.ptrs {
                if !p.is_null() {
                    let _ = self.gpu.free_host_pinned(p, STAGE_BYTES);
                }
            }
        }
    }
    let stage = StageGuard { gpu, ptrs: stage };
    // `dirty[k]` = chunk k's last async copy has not been synchronized. A
    // chunk is synchronized only right before it is REUSED, so a pread for
    // the next chunk overlaps the previous chunk's H2D copy; one final sync
    // covers the tail.
    let mut dirty = [false; STAGE_CHUNKS];
    let mut total: u64 = 0;
    let mut chunk_idx = 0usize;
    for (i, (path, base)) in shards.iter().enumerate() {
        let file = &files[shard_file[i]];
        let mut done: u64 = 0;
        while done < bytes_per_shard {
            if dirty[chunk_idx] {
                gpu.synchronize(stream)?;
                dirty = [false; STAGE_CHUNKS];
            }
            let want = (bytes_per_shard - done).min(STAGE_BYTES as u64) as usize;
            // SAFETY: `p` is a live pinned alloc of STAGE_BYTES, synchronized
            // by the dirty-flag above before reuse.
            let buf = unsafe { std::slice::from_raw_parts_mut(stage.ptrs[chunk_idx], want) };
            file.read_exact_at(buf, base + done).with_context(|| {
                format!("resident n-gram load: shard {i} {} +{done}", path.display())
            })?;
            gpu.copy_h2d_async(
                buf,
                dst.offset((i as u64 * bytes_per_shard + done) as usize),
                stream,
            )?;
            dirty[chunk_idx] = true;
            done += want as u64;
            chunk_idx = (chunk_idx + 1) % STAGE_CHUNKS;
        }
        total += bytes_per_shard;
    }
    gpu.synchronize(stream)?;
    // Drop every backing file's page cache, then close.
    let dropped = files.len();
    for file in &files {
        dont_need(file)?;
    }
    drop(files);
    tracing::info!(
        "resident n-gram load: {} shards, {:.1} GB staged through {} x {} MiB pinned, \
         {dropped} file(s) fadvise(DONTNEED) + closed",
        shards.len(),
        total as f64 / 1e9,
        STAGE_CHUNKS,
        STAGE_BYTES / (1024 * 1024),
    );
    Ok(())
}

/// `posix_fadvise(DONTNEED)` a backing file — the bytes are on device now;
/// holding 30 GB of page cache would steal the same unified memory the table
/// was loaded into. Non-Linux unix is a no-op.
#[cfg(all(unix, not(target_os = "macos")))]
fn dont_need(file: &std::fs::File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if rc != 0 {
        bail!("posix_fadvise(DONTNEED) failed: {rc}");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn dont_need(file: &std::fs::File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    Ok(())
}

#[cfg(not(unix))]
fn dont_need(_file: &std::fs::File) -> Result<()> {
    Ok(())
}

/// Read `len` bytes of one shard region from disk into a Vec — the boot
/// self-check's host oracle input (a handful of rows, direct file reads).
#[cfg(feature = "cuda")]
pub fn read_shard_bytes(path: &PathBuf, offset: u64, len: usize) -> Result<Vec<u8>> {
    let f = std::fs::File::open(path)
        .with_context(|| format!("resident self-check: open {}", path.display()))?;
    let mut buf = vec![0u8; len];
    f.read_exact_at(&mut buf, offset)?;
    Ok(buf)
}
