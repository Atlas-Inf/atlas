// SPDX-License-Identifier: AGPL-3.0-only

//! Private seekable decoder input. Ordinary MP4 files keep their index at the
//! end; piping those bytes makes FFmpeg unable to seek back to the pictures.
use anyhow::{Context, Result, bail};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct SeekableInput(PathBuf);
impl SeekableInput {
    pub(super) fn new(bytes: &[u8]) -> Result<Self> {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        for _ in 0..100 {
            let path = std::env::temp_dir().join(format!(
                "atlas-video-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(mut file) => {
                    let input = Self(path);
                    file.write_all(bytes)
                        .context("staging seekable video input")?;
                    return Ok(input);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e).context("creating private video input"),
            }
        }
        bail!("could not create private video input after 100 collisions")
    }
    pub(super) fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for SeekableInput {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn input_is_seekable_private_and_removed_on_drop() {
        let input = SeekableInput::new(b"clip").unwrap();
        let path = input.path().to_owned();
        assert_eq!(std::fs::read(&path).unwrap(), b"clip");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        drop(input);
        assert!(!path.exists());
    }
}
