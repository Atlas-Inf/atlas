// SPDX-License-Identifier: AGPL-3.0-only

//! [`DirectSwapFile`] — the NVMe cold tier, one implementation per platform.
//!
//! The contract ([`crate::traits::SwapStore`]) and its invariants are identical
//! everywhere: fixed-stride records addressed by `disk_slot`, `record_bytes` a
//! non-zero 4 KiB multiple, the file growing sparsely as slots are allocated.
//! Only the I/O primitive differs, so only the I/O primitive is split:
//!
//!   * `unix` — `O_DIRECT` (Linux) + `pread`/`pwrite` on a raw fd, staging
//!     through a page-aligned bounce when the caller's buffer is not aligned.
//!     On non-Linux unix `O_DIRECT` does not exist and the file opens buffered,
//!     which is harmless: the NVMe cold tier only ever runs on the Linux fleet.
//!   * `windows` — buffered `seek_read`/`seek_write`. Windows' nearest
//!     equivalent, `FILE_FLAG_NO_BUFFERING`, imposes sector-alignment rules on
//!     every buffer and offset and buys nothing here for the same reason: this
//!     tier is not driven on Windows. Correctness over an unused fast path.
//!
//! Splitting by file rather than by `#[cfg]` attribute keeps each
//! implementation readable on its own and stops a change to one platform from
//! silently editing the other.

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::DirectSwapFile;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::DirectSwapFile;

mod shared;
pub use shared::SharedRecordFile;

/// Remove leftover swap files (`<prefix>*.swap`) from `dir`; returns how many
/// went. For tiers whose files carry no state across restarts: a file found at
/// startup was left by a process that died before unlinking it (or by a build
/// that never unlinked), and only wastes the disk budget. Unlinking cannot
/// hurt a live owner — it keeps its descriptor.
pub fn remove_stale_swap_files(dir: &std::path::Path, prefix: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with(prefix) && name.ends_with(".swap")
        })
        .filter(|e| std::fs::remove_file(e.path()).is_ok())
        .count()
}

/// `Some(kind)` when `dir` is on a filesystem that must not hold a swap tier:
/// memory-backed (a "disk" tier there spends the RAM it was meant to save —
/// and tmpfs accepts `O_DIRECT` on current kernels, so nothing else stops it)
/// or a container overlay (the records would land in the image store, not on
/// the mounted disk). `None` for everything else, off Linux, and when the
/// directory cannot be examined (the open that follows reports that).
pub fn unsuitable_swap_fs(dir: &std::path::Path) -> Option<&'static str> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        // SAFETY: `statfs` fills the zeroed struct it is handed; `path` is a
        // NUL-terminated string that outlives the call.
        let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(path.as_ptr(), &mut fs) } != 0 {
            return None;
        }
        // `f_type` is `i64` on glibc 64-bit and another width elsewhere.
        #[allow(clippy::unnecessary_cast)]
        fs_kind(fs.f_type as i64)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = dir;
        None
    }
}

/// Classify a Linux `statfs.f_type` (see [`unsuitable_swap_fs`]).
pub fn fs_kind(f_type: i64) -> Option<&'static str> {
    match f_type {
        0x0102_1994 => Some("tmpfs (host RAM)"),
        0x8584_58f6 => Some("ramfs (host RAM)"),
        0x794c_7630 => Some("overlayfs (the container layer, not a mounted disk)"),
        _ => None,
    }
}

/// Shared by both implementations so the error text of a misuse is identical
/// on every platform.
#[allow(dead_code)]
pub(crate) fn validate_record_bytes(record_bytes: usize) -> anyhow::Result<()> {
    if record_bytes == 0 || !record_bytes.is_multiple_of(4096) {
        anyhow::bail!(
            "DirectSwapFile: record_bytes ({record_bytes}) must be a non-zero 4 KiB multiple"
        );
    }
    Ok(())
}
