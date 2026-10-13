// SPDX-License-Identifier: AGPL-3.0-only
//! Single-shard buffered/direct I/O and upload; extracted without changing order.
use super::{direct_io, header::parse_header};
use crate::gpu::GpuBackend;
use crate::weights::{WeightTensor, evict_page_cache, f16_to_bf16_bytes};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::mpsc::sync_channel;

/// Load a single shard with O_DIRECT + pipelined read/copy.
///
/// Pipeline:
///   reader thread: pread tensor N into aligned buffer → sync_channel ──▶
///   main thread:   recv → copy_h2d → store tensor
///
/// The channel has capacity 1, so at any time the reader is ≤1 tensor
/// ahead of the copier. Memory overhead per shard: 2 × max_tensor_bytes
/// (rounded up to O_DIRECT alignment).
#[allow(clippy::too_many_arguments)]
pub(super) fn load_shard_fast(
    shard_path: &Path,
    tensor_filter: Option<&[String]>,
    gpu: &dyn GpuBackend,
    skip_fn: &dyn Fn(&str) -> bool,
    defer_fn: &dyn Fn(&str) -> bool,
    try_direct_io: bool,
    direct_io_tensor_cap: usize,
    prefetch_shards: bool,
    out: &mut HashMap<String, WeightTensor>,
    deferred_out: &mut HashMap<String, crate::weights::DeferredTensor>,
    offload_logged: &mut bool,
) -> Result<()> {
    // Header parsing uses a buffered fd — header is a few KB, cache pollution
    // is negligible and buffered I/O handles short reads cleanly.
    let mut meta_file = File::open(shard_path)
        .with_context(|| format!("Failed to open {}", shard_path.display()))?;
    let mut tensors = parse_header(&mut meta_file)?;
    let file_size = meta_file.metadata()?.len();

    // Filter down to tensors we actually want (index filter + EP filter).
    if let Some(allow) = tensor_filter {
        let allow_set: std::collections::HashSet<&str> = allow.iter().map(|s| s.as_str()).collect();
        tensors.retain(|t| allow_set.contains(t.name.as_str()));
    }
    // The n-gram embedding TABLES are never uploaded with the checkpoint —
    // 63 GB (LongCat-Lite) to ~102 GB (Flash-Next) of BF16 would exhaust a
    // 121 GB unified box before any quantization could run, and the fallback
    // on GB10 is managed memory, i.e. Linux swap, i.e. a kernel freeze. They
    // are recorded with their on-disk location and served either by streaming
    // per-table quantize-on-load or straight off NVMe by the row cache.
    let mut deferred_here: Vec<(String, crate::weights::DeferredTensor)> = Vec::new();
    #[allow(clippy::items_after_statements)]
    tensors.retain(|t| {
        if crate::weights::is_ngram_table(&t.name) || (!skip_fn(&t.name) && defer_fn(&t.name)) {
            deferred_here.push((
                t.name.clone(),
                crate::weights::DeferredTensor {
                    path: shard_path.to_path_buf(),
                    offset: t.abs_offset,
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                },
            ));
            return false;
        }
        !skip_fn(&t.name)
    });
    if !deferred_here.is_empty() {
        tracing::info!(
            "Deferred {} tensor(s) in {} — served from disk, not uploaded",
            deferred_here.len(),
            shard_path.display()
        );
        deferred_out.extend(deferred_here);
    }

    // Per-shard heuristic: above `direct_io_tensor_cap` tensors, O_DIRECT's
    // per-tensor syscall + 4 KiB alignment overhead costs more than kernel
    // readahead on the buffered path saves. Skip the direct-open attempt
    // entirely in that case — keeps the log clean and avoids a wasted fd.
    let wants_direct = try_direct_io && tensors.len() <= direct_io_tensor_cap;
    if try_direct_io && !wants_direct {
        tracing::info!(
            "  Shard has {} tensors (> {} cap) — using buffered+pipelined path",
            tensors.len(),
            direct_io_tensor_cap
        );
    }

    // File for data reads. Try O_DIRECT; if it fails, fall through to buffered.
    let (direct_file, using_direct) = match wants_direct
        .then(|| direct_io::open_direct(shard_path))
        .transpose()
    {
        Ok(Some(f)) => (Some(f), true),
        Ok(None) => (None, false),
        Err(e) => {
            tracing::warn!(
                "O_DIRECT open failed for {} ({e}); falling back to buffered reads",
                shard_path.display()
            );
            (None, false)
        }
    };
    let buffered_file = File::open(shard_path)?;
    let data_fd = direct_file.as_ref().unwrap_or(&buffered_file);
    if prefetch_shards && !using_direct {
        advise_prefetch_shard(&buffered_file, shard_path, file_size);
    }

    // Pipelined reader: sends (tensor_index, aligned_buffer, slice_start) to main.
    type ReadMsg = (usize, direct_io::AlignedBuffer, usize);
    let (tx, rx) = sync_channel::<Result<ReadMsg>>(1);
    let tensors_for_reader: Vec<(u64, usize)> =
        tensors.iter().map(|t| (t.abs_offset, t.len)).collect();
    let raw_fd = {
        use std::os::unix::io::AsRawFd;
        data_fd.as_raw_fd()
    };

    let _ = file_size; // retained for future use (tail-fragment buffered read)
    let reader_handle = std::thread::spawn(move || {
        for (idx, (abs_offset, len)) in tensors_for_reader.iter().enumerate() {
            let msg = direct_io::read_tensor_aligned(raw_fd, *abs_offset, *len, using_direct)
                .map(|(buf, slice_start)| (idx, buf, slice_start));
            if tx.send(msg).is_err() {
                break; // receiver dropped
            }
        }
    });

    // Copier: drains the channel, does gpu alloc + copy_h2d, inserts into the map.
    for result in rx {
        let (idx, buf, slice_start) = result?;
        let meta = &tensors[idx];
        let raw = &buf.as_slice()[slice_start..slice_start + meta.len];
        // F16 shards: convert bytes to BF16 before upload (same length,
        // different bit layout — meta.dtype is already staged as BF16).
        let converted: Vec<u8>;
        let src: &[u8] = if meta.from_f16 {
            converted = f16_to_bf16_bytes(raw);
            &converted
        } else {
            raw
        };

        let ptr = match gpu.alloc(meta.len) {
            Ok(p) => {
                gpu.copy_h2d(src, p)?;
                p
            }
            Err(_) => {
                if !*offload_logged {
                    tracing::warn!(
                        "GPU alloc failed for {} ({} bytes) — switching to managed (UVM) memory",
                        meta.name,
                        meta.len
                    );
                    *offload_logged = true;
                }
                let p = gpu.alloc_managed(meta.len)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), p.0 as *mut u8, meta.len);
                }
                p
            }
        };

        out.insert(
            meta.name.clone(),
            WeightTensor {
                ptr,
                shape: meta.shape.clone(),
                dtype: meta.dtype,
            },
        );
    }

    reader_handle
        .join()
        .map_err(|_| anyhow::anyhow!("reader thread panicked"))?;

    // Release file handles, then advise the kernel to drop any pages we did
    // end up caching on the buffered fallback path. O_DIRECT reads never hit
    // the page cache, so the posix_fadvise is a no-op there but cheap.
    drop(direct_file);
    evict_page_cache(&buffered_file);
    drop(buffered_file);
    Ok(())
}

#[cfg(target_os = "linux")]
fn advise_prefetch_shard(file: &File, shard_path: &Path, file_size: u64) {
    use std::os::unix::io::AsRawFd;

    let fd = file.as_raw_fd();
    let seq_rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
    let willneed_rc = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_WILLNEED) };
    if seq_rc == 0 && willneed_rc == 0 {
        tracing::info!(
            "  NFS/shard prefetch requested for {} ({:.2} GB)",
            shard_path.display(),
            file_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        tracing::warn!(
            "  NFS/shard prefetch hint failed for {}: sequential_rc={}, willneed_rc={}",
            shard_path.display(),
            seq_rc,
            willneed_rc
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_prefetch_shard(_file: &File, _shard_path: &Path, _file_size: u64) {}
