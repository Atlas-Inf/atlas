// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side free-memory inputs for the backend's free-memory reports:
//! `/proc/meminfo` fields and the unified-vs-carve-out rule that decides
//! whether host MemAvailable may stand in for device free memory.

/// Device free memory, raised to the host's MemAvailable only when the device
/// draws from host RAM. On a unified box (GB10, a GTT-backed Strix Halo) the
/// device total is the host pool, and cuMemGetInfo's "free" excludes
/// reclaimable page cache, so MemAvailable is the truer number. On an APU
/// with a dedicated carve-out larger than host RAM (the 128 GB Framework at a
/// 96 GiB BIOS carve-out: 96 GiB device, 31 GiB host), host RAM cannot back a
/// device allocation; taking the max reported ~27 GB "free" with the carve-out
/// full, so the KV budget overran it and Flash-Next could not boot.
/// Unified = device total no more than ~10% above host MemTotal.
pub(crate) fn unified_free(
    free: usize,
    total: usize,
    mem_total: Option<usize>,
    mem_available: Option<usize>,
) -> usize {
    match (mem_total, mem_available) {
        (Some(host), Some(avail)) if total <= host + host / 10 => free.max(avail),
        _ => free,
    }
}

/// Read MemAvailable from /proc/meminfo (Linux only).
/// Returns None on non-Linux or if parsing fails.
pub(super) fn system_available_memory_bytes() -> Option<usize> {
    system_meminfo_bytes("MemAvailable:")
}

/// One `/proc/meminfo` field in bytes (Linux only).
pub(super) fn system_meminfo_bytes(field: &str) -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in contents.lines() {
        if line.starts_with(field) {
            let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
mod unified_free_tests {
    use super::unified_free;
    const GIB: usize = 1 << 30;

    #[test]
    fn unified_boxes_take_mem_available() {
        // GB10: device total ~= host total; page cache hides reclaimable memory.
        assert_eq!(
            unified_free(20 * GIB, 120 * GIB, Some(119 * GIB), Some(60 * GIB)),
            60 * GIB
        );
        // GTT-backed Strix Halo: 60 GB device view of a 61 GB host.
        assert_eq!(
            unified_free(10 * GIB, 60 * GIB, Some(61 * GIB), Some(40 * GIB)),
            40 * GIB
        );
    }

    #[test]
    fn a_carve_out_larger_than_host_ram_ignores_mem_available() {
        // Framework at a 96 GiB carve-out, 31 GiB host: host RAM can't back it.
        assert_eq!(
            unified_free(GIB / 2, 96 * GIB, Some(31 * GIB), Some(27 * GIB)),
            GIB / 2
        );
    }

    #[test]
    fn missing_meminfo_keeps_the_device_number() {
        assert_eq!(
            unified_free(5 * GIB, 60 * GIB, None, Some(40 * GIB)),
            5 * GIB
        );
        assert_eq!(
            unified_free(5 * GIB, 60 * GIB, Some(61 * GIB), None),
            5 * GIB
        );
    }
}
