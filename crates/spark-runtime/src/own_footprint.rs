// SPDX-License-Identifier: AGPL-3.0-only

//! This process's own memory, measured without reference to free memory.
//!
//! KV sizing takes "what Atlas holds" as free-at-init minus free-now. On
//! unified memory that difference moves whenever ANOTHER process frees or
//! allocates during the load: memory released by a co-tenant is credited to
//! Atlas as memory it never took, the pool is sized into it, and the host
//! runs out (a GB10 then hangs). The counters here belong to this process
//! alone, so nothing a co-tenant does can lower them.
//!
//! Measured on GB10 (driver 580.178, 2026-09-30, in a container; scripts and
//! output in `docs/campaigns/kv-sizing-2026-09`):
//!
//! * `cuMemAlloc` is NOT in the process's RSS (`RssAnon` +0 for 1 GiB) and is
//!   committed at allocation, before first touch. The driver's per-process
//!   accounting (NVML `usedGpuMemory`) counts what it really takes, rounding
//!   included: requests up to 2 MiB are carved from 2 MiB chunks (300
//!   requests of 1 MiB and one byte cost 600 MiB, 2000 of 100 KiB cost 200),
//!   larger ones round up to 64 KiB. It answers unprivileged under the
//!   in-container PID.
//! * Page-locked host memory (`cuMemAllocHost`, and `cuMemHostAlloc` with
//!   `PORTABLE|DEVICEMAP` as the RDMA pair region uses) is in `RssShmem`
//!   only: not in `RssAnon`, not in the driver's figure, so not counted twice.
//! * `RssFile` is page cache (the mmapped safetensors), which `MemAvailable`
//!   still counts as available, so it is left out.
//! * Managed memory (`cuMemAllocManaged`) moves NONE of these counters, even
//!   touched on the device. Atlas uses it only as the out-of-memory fallback
//!   for weights; there the tracked figure reads low.
//! * At 4 and 8 GiB the driver's growth matched the drop in `MemAvailable`
//!   within 0.5%. Not measured: the same at 100 GiB, and kernel-side memory
//!   that is in neither counter. `MemAvailable` itself moved by hundreds of
//!   MiB between samples on that (shared) host.
//!
//! So own = device (driver accounting, or the backend's allocation ledger
//! where that is unavailable or lower) + host (`RssAnon` + `RssShmem`), each
//! as growth since the backend was created, which is where the free-memory
//! baseline is taken too.

/// Where [`OwnFootprint::device`] came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceSource {
    /// The driver's per-process accounting: every device allocation in the
    /// process, whoever made it.
    Driver,
    /// Requested bytes of the backend's live allocations. A lower bound: it
    /// misses driver rounding (up to 2 MiB per small allocation) and the
    /// workspaces that call `cuMemAlloc` directly (CUTLASS, cuBLASLt,
    /// FlashInfer, NCCL). It cannot rule out a release of that size.
    Ledger,
}

impl DeviceSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Driver => "driver accounting",
            Self::Ledger => "allocation ledger",
        }
    }
}

/// Memory this process took since its GPU backend was created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnFootprint {
    pub device: usize,
    pub device_source: DeviceSource,
    pub host: usize,
}

impl OwnFootprint {
    pub fn total(&self) -> usize {
        self.device.saturating_add(self.host)
    }
}

/// One reading of the process-wide counters. `None` = not readable here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sample {
    /// Driver-accounted device bytes of this process.
    pub driver: Option<usize>,
    /// `RssAnon` + `RssShmem`.
    pub host: Option<usize>,
}

/// Growth from `base` to `now`, with `ledger` as the device floor.
///
/// A counter missing from either sample contributes nothing rather than a
/// guess, so the result can only under-state — and the caller takes the
/// larger of this and its free-memory delta.
pub fn since(base: Sample, now: Sample, ledger: usize) -> OwnFootprint {
    let grown = |b: Option<usize>, n: Option<usize>| Some(n?.saturating_sub(b?));
    let (device, device_source) = match grown(base.driver, now.driver) {
        Some(driver) if driver >= ledger => (driver, DeviceSource::Driver),
        _ => (ledger, DeviceSource::Ledger),
    };
    OwnFootprint {
        device,
        device_source,
        host: grown(base.host, now.host).unwrap_or(0),
    }
}

/// The backend's live device allocations: base pointer to bytes requested.
///
/// The sum is the device floor [`since`] falls back to. Managed allocations
/// are recorded with zero bytes: they can be paged out, so they are owned (for
/// the teardown sweep) without being footprint the ledger can vouch for.
#[derive(Debug, Default)]
pub struct AllocLedger(std::collections::HashMap<u64, usize>);

impl AllocLedger {
    pub fn record(&mut self, ptr: u64, bytes: usize) {
        self.0.insert(ptr, bytes);
    }

    /// Drop `ptr`, returning the bytes it was recorded with.
    pub fn forget(&mut self, ptr: u64) -> Option<usize> {
        self.0.remove(&ptr)
    }

    /// Requested bytes of everything still recorded.
    pub fn bytes(&self) -> usize {
        self.0.values().fold(0, |sum, &b| sum.saturating_add(b))
    }

    /// Empty the ledger, returning the pointers it held.
    pub fn drain(&mut self) -> Vec<u64> {
        self.0.drain().map(|(ptr, _)| ptr).collect()
    }
}

/// `RssAnon` + `RssShmem` out of a `/proc/<pid>/status` text.
pub fn parse_host_bytes(status: &str) -> Option<usize> {
    let kb = |key: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))?
            .split_whitespace()
            .next()?
            .parse::<usize>()
            .ok()
    };
    kb("RssAnon:")?
        .checked_add(kb("RssShmem:")?)?
        .checked_mul(1024)
}

/// This process's `RssAnon` + `RssShmem`; `None` off Linux.
pub fn host_bytes() -> Option<usize> {
    parse_host_bytes(&std::fs::read_to_string("/proc/self/status").ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: usize = 1 << 30;

    fn sample(driver: Option<usize>, host: Option<usize>) -> Sample {
        Sample { driver, host }
    }

    #[test]
    fn driver_growth_wins_over_a_smaller_ledger() {
        // Context + modules (0.3 GiB) predate the baseline and cancel out; the
        // driver also sees the 0.6 GiB of workspaces the ledger never did.
        let got = since(
            sample(Some(GIB * 3 / 10), Some(GIB / 10)),
            sample(Some(101 * GIB + GIB * 3 / 10), Some(GIB)),
            100 * GIB + GIB * 4 / 10,
        );
        assert_eq!(got.device, 101 * GIB);
        assert_eq!(got.device_source, DeviceSource::Driver);
        assert_eq!(got.host, GIB - GIB / 10);
        assert_eq!(got.total(), 101 * GIB + GIB - GIB / 10);
    }

    #[test]
    fn ledger_is_the_floor_when_the_driver_is_silent_or_lower() {
        for (base, now) in [
            (None, None),
            (None, Some(90 * GIB)),
            (Some(GIB), None),
            (Some(GIB), Some(50 * GIB)),
        ] {
            let got = since(sample(base, None), sample(now, None), 100 * GIB);
            assert_eq!(got.device, 100 * GIB);
            assert_eq!(got.device_source, DeviceSource::Ledger);
            assert_eq!(got.host, 0, "an unreadable host counter adds nothing");
        }
    }

    #[test]
    fn a_counter_that_shrank_reads_as_zero_not_as_a_wraparound() {
        let got = since(
            sample(Some(2 * GIB), Some(2 * GIB)),
            sample(Some(GIB), Some(GIB)),
            0,
        );
        assert_eq!((got.device, got.host), (0, 0));
    }

    #[test]
    fn host_bytes_are_anon_plus_shmem_and_never_file() {
        let status = "Name:\tspark\nVmRSS:\t  789576 kB\nRssAnon:\t  580508 kB\n\
                      RssFile:\t  101760 kB\nRssShmem:\t  107308 kB\nThreads:\t9\n";
        assert_eq!(parse_host_bytes(status), Some((580508 + 107308) * 1024));
    }

    #[test]
    fn a_status_without_the_split_rss_fields_is_unreadable() {
        assert_eq!(parse_host_bytes("VmRSS:\t  789576 kB\n"), None);
        assert_eq!(parse_host_bytes("RssAnon:\t  1 kB\n"), None);
        assert_eq!(parse_host_bytes("RssAnon:\tx kB\nRssShmem:\t1 kB\n"), None);
        assert_eq!(parse_host_bytes(""), None);
    }

    #[test]
    fn the_ledger_sums_requested_bytes_of_live_allocations_only() {
        let mut ledger = AllocLedger::default();
        assert_eq!(ledger.bytes(), 0);
        ledger.record(0x1000, 3 * GIB);
        ledger.record(0x2000, GIB + 1);
        // Managed memory: owned for the sweep, zero footprint.
        ledger.record(0x3000, 0);
        assert_eq!(ledger.bytes(), 4 * GIB + 1);

        assert_eq!(ledger.forget(0x1000), Some(3 * GIB));
        assert_eq!(ledger.forget(0x1000), None, "a pointer is forgotten once");
        assert_eq!(ledger.forget(0x9999), None, "never recorded");
        assert_eq!(ledger.bytes(), GIB + 1);

        // A free the driver refused puts the pointer back with its bytes, and
        // one the ledger never knew comes back as zero rather than a guess.
        for ptr in [0x2000, 0x9999] {
            let bytes = ledger.forget(ptr);
            ledger.record(ptr, bytes.unwrap_or(0));
        }
        assert_eq!(ledger.bytes(), GIB + 1);

        let mut swept = ledger.drain();
        swept.sort_unstable();
        assert_eq!(swept, vec![0x2000, 0x3000, 0x9999]);
        assert_eq!(ledger.bytes(), 0);
        assert!(ledger.drain().is_empty());
    }

    #[test]
    fn a_ledger_that_overflows_saturates() {
        let mut ledger = AllocLedger::default();
        ledger.record(1, usize::MAX);
        ledger.record(2, 1);
        assert_eq!(ledger.bytes(), usize::MAX);
    }
}
