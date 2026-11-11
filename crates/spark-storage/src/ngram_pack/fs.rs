// SPDX-License-Identifier: AGPL-3.0-only

//! The packer's file-system edges: no-follow opens, file identity, the build
//! lock, free-space and page-cache hints.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

/// `(st_dev, st_ino)` of an open file (`(0, 0)` where there is no inode).
pub(super) fn file_id(f: &File) -> Result<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let m = f.metadata().context("ngram pack: fstat")?;
        Ok((m.dev(), m.ino()))
    }
    #[cfg(not(unix))]
    {
        let _ = f;
        Ok((0, 0))
    }
}

/// `opts` with O_NOFOLLOW, so a symlink planted at the path fails the open
/// (ELOOP) instead of redirecting it. For everything the packer creates or
/// trusts in its directory; the SOURCES are followed (a Hugging Face snapshot
/// is symlinks to blobs).
pub(super) fn nofollow(opts: &mut OpenOptions) -> &mut OpenOptions {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_NOFOLLOW);
    }
    opts
}

/// The packed file at `path`, opened for reading without following a link.
pub(super) fn open_packed(path: &Path) -> Result<File> {
    nofollow(OpenOptions::new().read(true))
        .open(path)
        .with_context(|| format!("ngram pack: open {}", path.display()))
}

/// What makes a source file this one: mtime (ns since the epoch) and inode.
pub(super) fn file_stamp(m: &std::fs::Metadata) -> serde_json::Value {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let ns = m
            .mtime()
            .saturating_mul(1_000_000_000)
            .saturating_add(m.mtime_nsec());
        serde_json::json!({"mtime_ns": ns, "ino": m.ino()})
    }
    #[cfg(not(unix))]
    {
        let ns = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos() as u64);
        serde_json::json!({"mtime_ns": ns})
    }
}

pub(super) fn remove_stale_temps(dir: &Path, name: &str) {
    let prefix = format!("{name}.tmp");
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with(&prefix) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}

pub(super) fn cached_open<'f>(
    files: &'f mut Vec<(PathBuf, File)>,
    path: &Path,
) -> Result<&'f File> {
    if let Some(i) = files.iter().position(|(p, _)| p == path) {
        return Ok(&files[i].1);
    }
    let f = File::open(path).with_context(|| format!("ngram pack: open {}", path.display()))?;
    files.push((path.to_path_buf(), f));
    Ok(&files.last().unwrap().1)
}

/// 64-bit FNV-1a: a stable name for an identity string (std's hasher is not
/// guaranteed stable across releases).
pub(super) fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(target_os = "linux")]
pub(super) fn drop_cache(f: &File, off: u64, len: u64) {
    use std::os::unix::io::AsRawFd;
    // Best effort: the copy is correct either way, only the page cache differs.
    unsafe {
        libc::posix_fadvise(
            f.as_raw_fd(),
            off as i64,
            len as i64,
            libc::POSIX_FADV_DONTNEED,
        )
    };
}

#[cfg(not(target_os = "linux"))]
pub(super) fn drop_cache(_f: &File, _off: u64, _len: u64) {}

#[cfg(unix)]
pub(super) fn ensure_free_space(dir: &Path, need: u64) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes())?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return Ok(()); // unknown: let the write itself report ENOSPC
    }
    // u64 on Linux, u32/c_ulong on macOS: the casts are only a no-op on one.
    #[allow(clippy::unnecessary_cast)]
    let free = st.f_bavail as u64 * st.f_frsize as u64;
    let margin = 1 << 30;
    ensure!(
        free >= need + margin,
        "ngram pack: {} has {:.1} GB free, the packed table needs {:.1} GB (+1 GB margin); \
         point ATLAS_PLE_PACKED_DIR at a larger volume",
        dir.display(),
        free as f64 / 1e9,
        need as f64 / 1e9
    );
    Ok(())
}

#[cfg(not(unix))]
pub(super) fn ensure_free_space(_dir: &Path, _need: u64) -> Result<()> {
    Ok(())
}

/// Exclusive advisory lock on a lock file, released on drop (by close).
pub(super) struct DirLock(#[allow(dead_code)] File);

impl DirLock {
    pub(super) fn take(path: &Path) -> Result<Self> {
        let f = nofollow(OpenOptions::new().create(true).truncate(false).write(true))
            .open(path)
            .with_context(|| format!("ngram pack: open lock {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } != 0 {
                bail!(
                    "ngram pack: lock {}: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                );
            }
        }
        Ok(Self(f))
    }
}
