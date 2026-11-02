// SPDX-License-Identifier: AGPL-3.0-only

//! Pins the RDMA pair proxy thread to the fastest core. Unpinned, the
//! scheduler moves the spinning proxy between GB10's big (3.9 GHz) and
//! little (2.8 GHz) cores, and every hop of a decode all-reduce waits on it.

use std::path::Path;

/// Pin the calling proxy thread to the fastest core this process may run on
/// ([`fastest_core`] of its affinity mask), unless `ATLAS_RDMA_PROXY_PIN=0`.
/// Best effort: logs and runs on unpinned if the cores cannot be read or set.
pub(super) fn pin_proxy() {
    if std::env::var("ATLAS_RDMA_PROXY_PIN").as_deref() == Ok("0") {
        return;
    }
    #[cfg(target_os = "linux")]
    pin_to_fastest(Path::new("/sys/devices/system/cpu"));
}

#[cfg(target_os = "linux")]
fn pin_to_fastest(root: &Path) {
    // SAFETY: a zeroed cpu_set_t is a valid empty set; the libc calls get
    // its size and write or read only it.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::cpu_set_t>();
    if unsafe { libc::sched_getaffinity(0, size, &mut set) } != 0 {
        tracing::warn!("RDMA proxy: sched_getaffinity failed; not pinned");
        return;
    }
    let allowed =
        (0..libc::CPU_SETSIZE as usize).filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &set) });
    let Some((freq, cpu)) = fastest_core(root, allowed) else {
        tracing::warn!("RDMA proxy: no core frequencies readable; not pinned");
        return;
    };
    // SAFETY: as above.
    unsafe {
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
    }
    if unsafe { libc::sched_setaffinity(0, size, &set) } == 0 {
        tracing::info!("RDMA proxy pinned to cpu {cpu} ({} MHz)", freq / 1000);
    } else {
        tracing::warn!("RDMA proxy: sched_setaffinity({cpu}) failed; not pinned");
    }
}

/// The `allowed` core with the highest `cpu<N>/cpufreq/cpuinfo_max_freq`
/// (kHz) under `root`, the last of equals, as `(kHz, N)`; `None` when no
/// allowed core's frequency reads.
fn fastest_core(root: &Path, allowed: impl IntoIterator<Item = usize>) -> Option<(u64, usize)> {
    let max_freq = |cpu: usize| {
        std::fs::read_to_string(root.join(format!("cpu{cpu}/cpufreq/cpuinfo_max_freq")))
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    };
    allowed
        .into_iter()
        .filter_map(|cpu| max_freq(cpu).map(|f| (f, cpu)))
        .max()
}

#[cfg(test)]
#[path = "pin_tests.rs"]
mod tests;
