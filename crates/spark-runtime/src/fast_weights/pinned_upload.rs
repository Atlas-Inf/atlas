// SPDX-License-Identifier: AGPL-3.0-only

//! Parallel read-and-upload of tensors into device memory allocated as the
//! reads go. Each worker `pread`s a piece into its own page-locked staging
//! buffer and copies it on its own stream, so the reads and copies of
//! different pieces run at once and no host buffer is allocated per tensor.
//! (Two buffers per worker, overlapping a worker's own read and copy, measured
//! slower on GB10: 2.7-2.9 s against 2.5 s for the same 18.8 GB.)
//!
//! The per-tensor path ([`super::shard_readers`]) hands every tensor to one
//! copier thread that copies it from pageable memory and drains the stream
//! before the next: on GB10 that loaded Qwen3.8-27B's 18.8 GB in 3.05 s
//! (6.2 GB/s), against ~10.5 GB/s the NVMe gives these files' O_DIRECT reads.

use super::direct_io::{O_DIRECT_ALIGN, aligned_window, pread_window};
use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Result, anyhow};
use std::os::unix::io::RawFd;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;

/// Workers, and the most bytes one of them reads at a time. Swept on GB10
/// over 4-32 workers and 8-64 MiB pieces: 16 x 16 MiB was at the front
/// (2.51 s for Qwen3.8-27B; 8 x 32 MiB 2.61 s, 16 x 64 MiB 2.77 s).
const WORKERS: usize = 16;
const PIECE_BYTES: usize = 16 << 20;

/// `len` bytes at file offset `file_off`, bound for `dst`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Piece {
    pub(super) file_off: u64,
    pub(super) len: usize,
    pub(super) dst: DevicePtr,
}

/// Split each `(file_off, len, dst)` tensor into pieces of at most `max`
/// bytes. Zero-length tensors yield nothing: there is nothing to copy.
pub(super) fn pieces(tensors: &[(u64, usize, DevicePtr)], max: usize) -> Vec<Piece> {
    let mut out = Vec::new();
    for &(file_off, len, dst) in tensors {
        let mut done = 0;
        while done < len {
            let n = (len - done).min(max);
            out.push(Piece {
                file_off: file_off + done as u64,
                len: n,
                dst: dst.offset(done),
            });
            done += n;
        }
    }
    out
}

/// One worker's staging buffer and stream.
struct Slot {
    /// Address of a page-locked buffer of `PinnedUploader::buf_len` bytes.
    buf: usize,
    stream: u64,
}

/// Page-locked staging buffers and streams for [`WORKERS`] workers, made
/// once per load and reused for every shard.
pub(super) struct PinnedUploader<'g> {
    gpu: &'g dyn GpuBackend,
    slots: Vec<Slot>,
    buf_len: usize,
}

impl<'g> PinnedUploader<'g> {
    pub(super) fn new(gpu: &'g dyn GpuBackend) -> Result<Self> {
        // A piece's aligned window can start up to one block before it and
        // end up to one block after it.
        let buf_len = PIECE_BYTES + 2 * O_DIRECT_ALIGN;
        let mut me = Self {
            gpu,
            slots: Vec::with_capacity(WORKERS),
            buf_len,
        };
        for _ in 0..WORKERS {
            let buf = gpu.alloc_host_pinned(buf_len)? as usize;
            let stream = match gpu.create_stream() {
                Ok(s) => s,
                Err(e) => {
                    gpu.free_host_pinned(buf as *mut u8, buf_len)?;
                    return Err(e);
                }
            };
            me.slots.push(Slot { buf, stream });
        }
        Ok(me)
    }

    /// Read every tensor `(file_off, len)` from `fd` into the device memory
    /// `alloc(i)` gives tensor `i`. `alloc` runs in tensor order on this
    /// thread while the workers read what it has already placed, so the
    /// allocations (the driver maps and zeroes pages: ~25 GB/s on GB10, from
    /// any number of threads) overlap the reads. A tensor `alloc` gives `None`
    /// is left alone. Returns every tensor's pointer once all have landed, or
    /// the first error.
    pub(super) fn upload(
        &self,
        fd: RawFd,
        using_direct: bool,
        tensors: &[(u64, usize)],
        mut alloc: impl FnMut(usize) -> Option<DevicePtr>,
    ) -> Result<Vec<Option<DevicePtr>>> {
        let (tx, rx) = std::sync::mpsc::channel::<Piece>();
        let rx = Mutex::new(rx);
        let failed = AtomicBool::new(false);
        std::thread::scope(|s| {
            let workers: Vec<_> = self
                .slots
                .iter()
                .map(|slot| {
                    let (rx, failed) = (&rx, &failed);
                    s.spawn(move || {
                        let r = self.work(slot, fd, using_direct, rx, failed);
                        if r.is_err() {
                            failed.store(true, Ordering::Release);
                        }
                        r
                    })
                })
                .collect();
            let mut ptrs = Vec::with_capacity(tensors.len());
            for (i, &(file_off, len)) in tensors.iter().enumerate() {
                if failed.load(Ordering::Acquire) {
                    break;
                }
                let ptr = alloc(i);
                if let Some(dst) = ptr {
                    for piece in pieces(&[(file_off, len, dst)], PIECE_BYTES) {
                        // Cannot fail: the receiver lives until this scope
                        // returns.
                        let _ = tx.send(piece);
                    }
                }
                ptrs.push(ptr);
            }
            drop(tx);
            let mut first = Ok(());
            for w in workers {
                let r = w
                    .join()
                    .unwrap_or_else(|_| Err(anyhow!("upload worker panicked")));
                if first.is_ok() {
                    first = r;
                }
            }
            first.map(|()| ptrs)
        })
    }

    fn work(
        &self,
        slot: &Slot,
        fd: RawFd,
        using_direct: bool,
        rx: &Mutex<Receiver<Piece>>,
        failed: &AtomicBool,
    ) -> Result<()> {
        self.gpu.bind_to_thread()?;
        let buf = slot.buf as *mut u8;
        while !failed.load(Ordering::Acquire) {
            let next = rx
                .lock()
                .map_err(|_| anyhow!("upload queue poisoned"))?
                .recv();
            let Ok(p) = next else {
                break; // every piece handed out
            };
            let (start, window, off) = aligned_window(p.file_off, p.len, using_direct);
            debug_assert!(window <= self.buf_len);
            // SAFETY: `buf` is this worker's own page-locked buffer of
            // `buf_len >= window` bytes (a piece is at most `PIECE_BYTES`, its
            // window at most two blocks more), and its last copy has finished.
            unsafe { pread_window(fd, buf, start, window, off + p.len) }?;
            // SAFETY: the buffer was zeroed at allocation and the read above
            // filled `off..off + p.len`; the slice lives until the
            // `synchronize` below, which is the retained copy's contract.
            let src = unsafe { std::slice::from_raw_parts(buf.add(off), p.len) };
            self.gpu.copy_h2d_async_retained(src, p.dst, slot.stream)?;
            self.gpu.synchronize(slot.stream)?;
        }
        Ok(())
    }
}

impl Drop for PinnedUploader<'_> {
    fn drop(&mut self) {
        for slot in self.slots.drain(..) {
            let _ = self.gpu.destroy_stream(slot.stream);
            let _ = self.gpu.free_host_pinned(slot.buf as *mut u8, self.buf_len);
        }
    }
}

#[cfg(all(test, unix))]
#[path = "pinned_upload_tests.rs"]
mod tests;
