// SPDX-License-Identifier: AGPL-3.0-only

//! [`SharedRecordFile`] — a fixed-stride record file whose I/O is positional
//! and lock-free, so a worker pool can keep several requests in flight
//! ([`ConcurrentSwapStore`]). `O_DIRECT` on Linux; buffered elsewhere (the
//! NVMe tier only runs on the Linux fleet, but the type must build everywhere).
//!
//! Unlike [`super::DirectSwapFile`] there is no shared bounce buffer — that is
//! what makes `&self` I/O sound. A caller buffer that is not 4 KiB-aligned is
//! staged through a per-call aligned copy (tests; production passes pinned,
//! page-aligned staging).
//!
//! A run is moved with ONE request: `pread`/`pwrite` in slot order, or
//! `preadv`/`pwritev` with the records listed back to front when the caller's
//! buffer holds the run highest slot first. The block layer splits either at
//! the device's max transfer size and submits the pieces together, so a
//! multi-megabyte run gets the drive's full queue depth from a single thread.

use std::fs::File;
use std::path::Path;

use anyhow::Result;

use crate::aligned::PageAlignedBuf;
use crate::direct_swap::validate_record_bytes;
use crate::pio;
use crate::traits::{ConcurrentSwapStore, run_records};

pub struct SharedRecordFile {
    file: File,
    record_bytes: usize,
}

impl SharedRecordFile {
    /// Create `path`; `record_bytes` must be a 4 KiB multiple. The file must
    /// not exist and is owner-only: records hold prompt-derived data, and an
    /// exclusive create can never open a pre-planted file or follow a symlink.
    pub fn create(path: &Path, record_bytes: usize) -> Result<Self> {
        validate_record_bytes(record_bytes)?;
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
            #[cfg(target_os = "linux")]
            opts.custom_flags(libc::O_DIRECT);
        }
        let file = opts
            .open(path)
            .map_err(|e| anyhow::anyhow!("open record file {}: {e}", path.display()))?;
        Ok(Self { file, record_bytes })
    }

    /// Allocate the file's first `bytes` now (`fallocate`). Two reasons to do
    /// it up front rather than let the file grow: the space is either there
    /// at startup or the caller hears about it at startup, and — what makes
    /// it matter for throughput — ext4 takes the inode lock EXCLUSIVELY for a
    /// direct write that has to allocate, so writers into a growing file
    /// queue up behind each other, while writes into allocated extents run
    /// concurrently (measured on a GB10: 8 writers of single 104 KB records,
    /// 0.36 GB/s into holes, 2.6 GB/s into reserved extents). A filesystem
    /// without `fallocate` is left sparse (`Ok(false)`); off Linux this is a
    /// no-op.
    pub fn reserve(&self, bytes: u64) -> Result<bool> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: a plain syscall on a descriptor this struct owns.
            let rc = unsafe { libc::fallocate(self.file.as_raw_fd(), 0, 0, bytes as libc::off_t) };
            if rc == 0 {
                return Ok(true);
            }
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() != Some(libc::EOPNOTSUPP) {
                anyhow::bail!("reserving {bytes} B for the record file: {e}");
            }
        }
        let _ = bytes;
        Ok(false)
    }

    fn offset(&self, slot: usize) -> u64 {
        slot as u64 * self.record_bytes as u64
    }
}

fn is_aligned(p: *const u8) -> bool {
    (p as usize) & 0xfff == 0
}

impl ConcurrentSwapStore for SharedRecordFile {
    fn record_bytes(&self) -> usize {
        self.record_bytes
    }

    fn read_run(&self, low_slot: usize, reversed: bool, out: &mut [u8]) -> Result<()> {
        let (rb, n) = run_records(self.record_bytes, out.len())?;
        if !is_aligned(out.as_ptr()) {
            let mut tmp = PageAlignedBuf::new(out.len());
            self.read_run(low_slot, reversed, tmp.as_mut_slice())?;
            out.copy_from_slice(tmp.as_slice());
            return Ok(());
        }
        let off = self.offset(low_slot);
        if reversed && n > 1 {
            return vectored::read_reversed(&self.file, out, rb, off);
        }
        pio::read_exact_at(&self.file, out, off)
            .map_err(|e| anyhow::anyhow!("read records {low_slot}..+{n}: {e}"))
    }

    fn write_run(&self, low_slot: usize, reversed: bool, bytes: &[u8]) -> Result<()> {
        let (rb, n) = run_records(self.record_bytes, bytes.len())?;
        if !is_aligned(bytes.as_ptr()) {
            let mut tmp = PageAlignedBuf::new(bytes.len());
            tmp.as_mut_slice().copy_from_slice(bytes);
            return self.write_run(low_slot, reversed, tmp.as_slice());
        }
        let off = self.offset(low_slot);
        if reversed && n > 1 {
            return vectored::write_reversed(&self.file, bytes, rb, off);
        }
        pio::write_all_at(&self.file, bytes, off)
            .map_err(|e| anyhow::anyhow!("write records {low_slot}..+{n}: {e}"))
    }
}

/// Back-to-front runs: one `preadv`/`pwritev` whose iovecs list the buffer's
/// records last to first, so the file sees ascending offsets.
#[cfg(target_os = "linux")]
mod vectored {
    use std::fs::File;
    use std::os::fd::AsRawFd;

    use anyhow::{Result, bail};

    /// `IOV_MAX` on Linux; longer runs go out as several requests.
    const MAX_IOV: usize = 1024;

    fn run(
        file: &File,
        base: *mut u8,
        len: usize,
        rb: usize,
        off: u64,
        io: unsafe extern "C" fn(i32, *const libc::iovec, i32, libc::off_t) -> isize,
        what: &str,
    ) -> Result<()> {
        let n = len / rb;
        // Record `k` of the FILE run is record `n - 1 - k` of the buffer.
        let iov: Vec<libc::iovec> = (0..n)
            .map(|k| libc::iovec {
                // SAFETY: `(n - 1 - k) * rb + rb <= len`, inside the buffer.
                iov_base: unsafe { base.add((n - 1 - k) * rb) } as *mut libc::c_void,
                iov_len: rb,
            })
            .collect();
        for (g, group) in iov.chunks(MAX_IOV).enumerate() {
            let at = off + (g * MAX_IOV * rb) as u64;
            // SAFETY: every iovec points at `rb` live bytes of the caller's
            // buffer, which outlives this synchronous call.
            let got = unsafe {
                io(
                    file.as_raw_fd(),
                    group.as_ptr(),
                    group.len() as i32,
                    at as libc::off_t,
                )
            };
            if got != (group.len() * rb) as isize {
                bail!(
                    "{what} of {} records at offset {at} returned {got}: {}",
                    group.len(),
                    std::io::Error::last_os_error()
                );
            }
        }
        Ok(())
    }

    pub(super) fn read_reversed(file: &File, out: &mut [u8], rb: usize, off: u64) -> Result<()> {
        run(
            file,
            out.as_mut_ptr(),
            out.len(),
            rb,
            off,
            libc::preadv,
            "preadv",
        )
    }

    pub(super) fn write_reversed(file: &File, bytes: &[u8], rb: usize, off: u64) -> Result<()> {
        // `pwritev` only reads through the iovecs; the cast is for the type.
        let base = bytes.as_ptr() as *mut u8;
        run(file, base, bytes.len(), rb, off, libc::pwritev, "pwritev")
    }
}

/// Portable fallback: one positional request per record.
#[cfg(not(target_os = "linux"))]
mod vectored {
    use std::fs::File;

    use super::{Result, pio};

    pub(super) fn read_reversed(file: &File, out: &mut [u8], rb: usize, off: u64) -> Result<()> {
        for (i, rec) in out.chunks_exact_mut(rb).rev().enumerate() {
            pio::read_exact_at(file, rec, off + (i * rb) as u64)?;
        }
        Ok(())
    }

    pub(super) fn write_reversed(file: &File, bytes: &[u8], rb: usize, off: u64) -> Result<()> {
        for (i, rec) in bytes.chunks_exact(rb).rev().enumerate() {
            pio::write_all_at(file, rec, off + (i * rb) as u64)?;
        }
        Ok(())
    }
}
