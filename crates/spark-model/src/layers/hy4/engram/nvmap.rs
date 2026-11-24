// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe-backed (mmap/page-cache) serving of one rank's bf16 Engram table
//! slice, as Flare does for its host-resident Engram n-gram tables.
//!
//! GB10 gives each process a coherent unified address space — host memory is
//! GPU-addressable as-is — so the Engram lookup kernels can read the rank's
//! table rows straight out of a read-only `mmap` of the safetensors shard.
//! The Linux page cache then IS the table's residency: hot rows stay in RAM,
//! cold rows fault in from NVMe on demand, and the rank's slice (half of
//! ~165M rows x 96 x 2 B = ~16 GB per layer across 4 layers) never occupies a
//! device allocation — which is what broke the 121 GB box when the bf16
//! slices were copied in alongside ~109 GB/rank of weights.
//!
//! No `cudaHostRegister`: the tree's only pinning door is
//! [`spark_runtime::gpu::GpuBackend::alloc_host_pinned`] (staging buffers,
//! tracked by `pinned_hosts` for copy-path soundness), and pinning the table
//! range would fault ALL of it into RAM for the process lifetime — the exact
//! residency this module exists to avoid. The coherent fabric makes the plain
//! mapping GPU-readable without registration; on backends that DO need a
//! registered pointer, [`GpuBackend::host_ptr_to_device`] is tried first and
//! its answer preferred.
//!
//! Gated to the bf16 path only (`engram_fp8_embedding == false`); the fp8
//! tables stay device-resident, byte-identical to before.

use anyhow::{Context, Result, ensure};
use memmap2::{Advice, Mmap, MmapOptions};
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend},
    weights::{DeferredTensor, WeightDtype},
};

/// `ATLAS_HY4_ENGRAM_NVMAP=0` forces the old device-resident bf16 slices
/// (A/B). Unset or any other value keeps the NVMe-mmap path.
pub(crate) const ENABLED_ENV: &str = "ATLAS_HY4_ENGRAM_NVMAP";
/// `ATLAS_HY4_ENGRAM_NVMAP_WILLNEED_MB`: how many MiB of the slice head get
/// `MADV_WILLNEED` so first requests fault in quickly (default 64; `0`
/// disables). The rest of the mapping is never prefaulted.
pub(crate) const WILLNEED_ENV: &str = "ATLAS_HY4_ENGRAM_NVMAP_WILLNEED_MB";
const DEFAULT_WILLNEED_MB: usize = 64;
/// `mmap` offsets must be page-aligned. Grace's arm64 kernel runs 64 KiB
/// pages, so the file offset is aligned DOWN to 64 KiB — a multiple of the
/// page size on every Linux Atlas ships (4/16/64 KiB) — and the skipped head
/// bytes stay inside the mapping.
const MMAP_ALIGN: u64 = 64 * 1024;

/// Pure decision from the env value, so tests need no process-global
/// `set_var` races; [`enabled`] reads the real environment. Only an exact
/// `0` opts out — everything else (including a typo) keeps the mmap path,
/// which is also what an unset variable selects.
pub(crate) fn enabled_from_env(enabled_var: Option<&str>) -> bool {
    enabled_var != Some("0")
}

/// Pure parse of the WILLNEED head: unset → 64 MiB; `0` → off; else the
/// parsed MiB count (an unparsable value logs once and keeps the default).
pub(crate) fn willneed_mb_from_env(var: Option<&str>) -> usize {
    var.and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_WILLNEED_MB)
}

/// Is the NVMe-mmap table path on for this load? Never for fp8 tables.
pub(crate) fn enabled(fp8_table: bool) -> bool {
    if fp8_table {
        return false;
    }
    #[cfg(test)]
    if let Some(o) = test_override_value() {
        return o;
    }
    enabled_from_env(std::env::var(ENABLED_ENV).ok().as_deref())
}

fn willneed_bytes() -> usize {
    let var = std::env::var(WILLNEED_ENV).ok();
    if let Some(v) = var.as_deref()
        && v.parse::<usize>().is_err()
    {
        tracing::warn!("ignoring invalid {WILLNEED_ENV}={v}, head stays {DEFAULT_WILLNEED_MB} MiB");
    }
    willneed_mb_from_env(var.as_deref()).saturating_mul(1024 * 1024)
}

/// One rank's table slice, read out of the safetensors shard by the page
/// cache. Held by `Hy4Engram` so the mapping outlives every kernel launch
/// that reads through [`Self::table_ptr`].
pub(crate) struct TableMap {
    map: Mmap,
    /// Bytes of the mapping before the slice start (mmap alignment slack).
    skip: usize,
    /// Slice byte length: this rank's rows only, never device-resident.
    bytes: usize,
}

impl TableMap {
    /// `mmap` the deferred tensor `name`'s bytes `[start, start + bytes)`
    /// read-only — its first element sits at `d.offset` in `d.path`, the
    /// metadata the fast loader already recorded — and return the base the
    /// lookup kernels read rank rows `0..rows.len()` from, so row addressing
    /// is byte-identical to the device-resident slices.
    pub(crate) fn open(
        d: &DeferredTensor,
        name: &str,
        dtype: WeightDtype,
        shape: &[usize],
        start: usize,
        bytes: usize,
    ) -> Result<Self> {
        ensure!(
            d.shape == shape && d.dtype == dtype,
            "Hy4 Engram deferred table contract: {name}"
        );
        let slice_at = d.offset + start as u64;
        let file = std::fs::File::open(&d.path)
            .with_context(|| format!("opening {name} shard {}", d.path.display()))?;
        ensure!(
            file.metadata()?.len() >= slice_at + bytes as u64,
            "Hy4 Engram deferred table {name} slice [{slice_at}, {}) overruns its shard",
            slice_at + bytes as u64
        );
        let aligned = slice_at - slice_at % MMAP_ALIGN;
        let skip = (slice_at - aligned) as usize;
        let map = unsafe {
            MmapOptions::new()
                .offset(aligned)
                .len(skip + bytes)
                .map(&file)?
        };
        // Read-ahead HINT for the head only — MADV_WILLNEED starts async
        // readahead but faults nothing in, so cold rows below the head stay
        // on NVMe. A filesystem that rejects the advice keeps the model load
        // alive; the pages simply fault on first touch instead.
        let head = willneed_bytes().min(bytes);
        if head > 0
            && let Err(e) = map.advise_range(Advice::WillNeed, skip, head)
        {
            tracing::debug!("{WILLNEED_ENV} head advice failed: {e}");
        }
        Ok(Self { map, skip, bytes })
    }

    /// Kernel base for this rank's slice row 0. `host_ptr_to_device` is
    /// preferred (a registered pointer on backends that need one); the raw
    /// host pointer is the GB10 answer, where the fabric is coherent.
    pub(crate) fn table_ptr(&self, gpu: &dyn GpuBackend) -> DevicePtr {
        let host = (self.map.as_ptr() as *mut u8).wrapping_add(self.skip);
        match gpu.host_ptr_to_device(host) {
            Ok(p) => p,
            Err(_) => DevicePtr(host as u64),
        }
    }

    /// This rank's slice byte length — the bytes kept out of device memory.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Test seam for the env knob: pin the decision on THIS thread without
/// process-global `set_var` races. The guard restores the env on drop.
#[cfg(test)]
#[must_use]
pub(crate) fn test_override(on: bool) -> TestOverrideGuard {
    TEST_OVERRIDE.set(Some(on));
    TestOverrideGuard {}
}

#[cfg(test)]
fn test_override_value() -> Option<bool> {
    TEST_OVERRIDE.get()
}

/// Releases the test-only override from [`test_override`].
#[cfg(test)]
pub(crate) struct TestOverrideGuard {}

#[cfg(test)]
impl Drop for TestOverrideGuard {
    fn drop(&mut self) {
        TEST_OVERRIDE.set(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The env contract: on by default, off only on an exact `0`.
    #[test]
    fn knob_is_on_by_default_and_off_only_on_exact_zero() {
        assert!(enabled_from_env(None));
        assert!(enabled_from_env(Some("1")));
        assert!(enabled_from_env(Some("yes")));
        assert!(!enabled_from_env(Some("0")));
    }

    /// The head hint defaults to 64 MiB, honours `0` and custom sizes.
    #[test]
    fn willneed_head_defaults_to_64mb_and_honours_zero() {
        assert_eq!(willneed_mb_from_env(None), 64);
        assert_eq!(willneed_mb_from_env(Some("0")), 0);
        assert_eq!(willneed_mb_from_env(Some("512")), 512);
        // A garbage value keeps the default instead of failing the load.
        assert_eq!(willneed_mb_from_env(Some("big")), 64);
    }
}
