// SPDX-License-Identifier: AGPL-3.0-only

//! Device memory outside the system pool that a backend can lend
//! ([`super::GpuBackend::alloc_carveout`]).
//!
//! On GB10 (DGX Spark) the firmware reserves a 2 GiB display carveout
//! (`DISPLAY_FRM`) that Linux never sees and that the NVIDIA driver never
//! allocates from on this chip. `spark display-carveout` exports it as an RM
//! object fd and execs the server with that fd in [`FD_ENV`] and its size in
//! [`SIZE_ENV`]; the CUDA backend imports and maps it on first use. This
//! module is the backend-neutral half: the environment contract and the
//! arena that hands out pieces of the mapped range.
//!
//! Provenance: the observation that `DISPLAY_FRM` is unused on GB10 and can
//! be exported as an RM memory-list fd comes from kindling-spark-os's
//! `dispram` (github.com/kindlingai/kindling-spark-os, AGPL-3.0; idea only,
//! no code taken), which credits emihuang's earlier NVIDIA developer forum
//! post. The driver behaviour was re-checked against open-gpu-kernel-modules
//! 580.173.02 and 580.178.04.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

/// Inheritable fd of the exported carveout object.
pub const FD_ENV: &str = "ATLAS_DISPLAY_CARVEOUT_FD";
/// Byte size of the exported carveout object.
pub const SIZE_ENV: &str = "ATLAS_DISPLAY_CARVEOUT_SIZE";

/// Every piece starts on this boundary, so a piece occupies
/// [`CarveoutArena::footprint`] bytes of the range.
pub const CARVEOUT_ALIGN: usize = 64 << 10;

/// The exported fd and size from the environment, or `None` when the server
/// was not started under `spark display-carveout`. A malformed pair is an
/// error: the launcher sets both or neither.
pub fn from_env() -> Result<Option<(i32, usize)>> {
    match (std::env::var(FD_ENV), std::env::var(SIZE_ENV)) {
        (Err(_), Err(_)) => Ok(None),
        (Ok(fd), Ok(size)) => {
            let fd: i32 = fd
                .parse()
                .map_err(|e| anyhow::anyhow!("{FD_ENV}={fd}: {e}"))?;
            let size: usize = size
                .parse()
                .map_err(|e| anyhow::anyhow!("{SIZE_ENV}={size}: {e}"))?;
            if fd < 0 || size == 0 {
                bail!("{FD_ENV}={fd} {SIZE_ENV}={size}: need an fd and a non-zero size");
            }
            Ok(Some((fd, size)))
        }
        _ => bail!("{FD_ENV} and {SIZE_ENV} must be set together"),
    }
}

/// First-fit allocator over `[base, base + size)`. Host bookkeeping only.
#[derive(Debug)]
pub struct CarveoutArena {
    base: u64,
    size: usize,
    /// Live pieces: offset → footprint.
    live: BTreeMap<usize, usize>,
}

impl CarveoutArena {
    pub fn new(base: u64, size: usize) -> Self {
        Self {
            base,
            size,
            live: BTreeMap::new(),
        }
    }

    /// Bytes of the range a `bytes` request occupies.
    pub fn footprint(bytes: usize) -> usize {
        bytes.max(1).next_multiple_of(CARVEOUT_ALIGN)
    }

    pub fn capacity(&self) -> usize {
        self.size
    }

    pub fn used(&self) -> usize {
        self.live.values().sum()
    }

    /// The lowest gap that holds `bytes`.
    pub fn alloc(&mut self, bytes: usize) -> Result<u64> {
        let need = Self::footprint(bytes);
        let mut cursor = 0usize;
        for (&offset, &len) in &self.live {
            if offset - cursor >= need {
                break;
            }
            cursor = offset + len;
        }
        if self.size.saturating_sub(cursor) < need {
            bail!(
                "carveout cannot fit {bytes} bytes: {} of {} bytes in use",
                self.used(),
                self.size
            );
        }
        self.live.insert(cursor, need);
        Ok(self.base + cursor as u64)
    }

    /// Releases the piece starting at `ptr`; `false` if `ptr` is not one.
    pub fn free(&mut self, ptr: u64) -> bool {
        ptr.checked_sub(self.base)
            .and_then(|offset| self.live.remove(&(offset as usize)))
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    #[test]
    fn pieces_are_aligned_first_fit_and_reusable() {
        let mut a = CarveoutArena::new(0x1_0000_0000, 8 * MIB);
        let p0 = a.alloc(3 * MIB + 1).unwrap();
        let p1 = a.alloc(MIB).unwrap();
        assert_eq!(p0, 0x1_0000_0000);
        assert_eq!(p1, p0 + CarveoutArena::footprint(3 * MIB + 1) as u64);
        assert_eq!(p1 % CARVEOUT_ALIGN as u64, 0);
        assert!(a.free(p0));
        assert!(!a.free(p0), "a piece frees once");
        // The freed gap at the front is found first.
        assert_eq!(a.alloc(2 * MIB).unwrap(), p0);
        assert_eq!(a.used(), 3 * MIB);
    }

    #[test]
    fn a_request_past_the_end_is_refused_and_changes_nothing() {
        let mut a = CarveoutArena::new(0, 4 * MIB);
        a.alloc(3 * MIB).unwrap();
        assert!(a.alloc(MIB + 1).is_err());
        assert_eq!(a.used(), 3 * MIB);
        assert!(a.alloc(MIB).is_ok());
    }

    #[test]
    fn foreign_pointers_are_not_freed() {
        let mut a = CarveoutArena::new(0x4000_0000, 4 * MIB);
        let p = a.alloc(MIB).unwrap();
        assert!(!a.free(0x1000));
        assert!(!a.free(p + 1));
        assert!(a.free(p));
    }
}
