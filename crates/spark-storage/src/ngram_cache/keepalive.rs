// SPDX-License-Identifier: AGPL-3.0-only

//! Keep the n-gram table's NVMe out of its autonomous power states while
//! decode is faulting rows from it (`ATLAS_PLE_NVME_KEEPALIVE_MS`).
//!
//! A decode step faults its novel n-gram rows in ONE batch, then leaves the
//! drive idle for the rest of the step. Linux's APST policy
//! (`nvme_core.apst_primary_timeout_ms=100`, the default) moves an idle
//! controller into a non-operational state after 100 ms, and a C8 verify step
//! is ~120-150 ms, so every step's first read paid the exit latency. Measured
//! on the pair's drive (SAMSUNG MZALC4T0HBL1, O_DIRECT 4 KiB
//! random reads after an idle gap; first read of a burst, p50):
//!
//! ```text
//!     gap <= 20 ms     245 us   (following reads 198 us)
//!     gap 25..95 ms    ~640 us
//!     gap >= 110 ms   9540 us
//! ```
//!
//! which is the ~10 ms `PLE gather ... resolve` mode in the pair's C8 logs
//! (8-63 misses resolving in 9.4-12.7 ms, against ~1-2 ms between closely
//! spaced gathers). Changing the host's APST policy needs root; this does it
//! from the engine instead: a background thread reads one 4 KiB block of the
//! table every `period` while the cache has resolved rows within the last
//! `idle` (so an idle server still lets the drive sleep).
//!
//! It never touches the cache: the read lands in the thread's own bounce
//! buffer, and the slot map, arena and CLOCK state are not reachable from
//! here. Every row the gather sees is therefore byte-for-byte what it was
//! without the ticker; only the latency of the faults that fill them moves.

use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::{AlignedBlock, BLOCK};

/// State shared between the cache (which marks activity) and the ticker.
struct Shared {
    /// Milliseconds since `epoch` of the last resolve.
    last_use_ms: AtomicU64,
    /// Ticks that issued a read (tests and the startup log read it).
    reads: AtomicU64,
    stop: AtomicBool,
    epoch: Instant,
}

impl Shared {
    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }
}

/// The running ticker. Dropping it stops and joins the thread (bounded by
/// one period plus one 4 KiB read).
pub(super) struct KeepAlive {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl KeepAlive {
    /// Start ticking against `file` (any file on the table's drive; the
    /// cache passes one of its own backing files, re-opened via `try_clone`
    /// so the descriptor keeps its O_DIRECT status flags).
    pub(super) fn spawn(file: File, period: Duration, idle: Duration) -> Result<Self> {
        let len = file
            .metadata()
            .context("NgramRowCache keepalive: stat backing file")?
            .len();
        let blocks = len / BLOCK as u64;
        anyhow::ensure!(
            blocks > 0,
            "NgramRowCache keepalive: backing file is shorter than one block"
        );
        let shared = Arc::new(Shared {
            // Start "idle": the first resolve arms the ticker.
            last_use_ms: AtomicU64::new(u64::MAX),
            reads: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            epoch: Instant::now(),
        });
        let worker = shared.clone();
        let thread = std::thread::Builder::new()
            .name("ple-nvme-keepalive".into())
            .spawn(move || run(file, blocks, period, idle, &worker))
            .context("NgramRowCache keepalive: spawn")?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// The cache resolved rows: keep the drive awake for the next `idle`.
    pub(super) fn touch(&self) {
        self.shared
            .last_use_ms
            .store(self.shared.now_ms(), Ordering::Relaxed);
    }

    /// Reads issued so far.
    pub(super) fn reads(&self) -> u64 {
        self.shared.reads.load(Ordering::Relaxed)
    }
}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

/// Whether a tick at `now_ms` should read: some resolve happened, and no
/// more than `idle_ms` ago.
fn active(last_use_ms: u64, now_ms: u64, idle_ms: u64) -> bool {
    last_use_ms != u64::MAX && now_ms.saturating_sub(last_use_ms) <= idle_ms
}

fn run(file: File, blocks: u64, period: Duration, idle: Duration, sh: &Shared) {
    use std::os::unix::fs::FileExt;
    let mut bounce = AlignedBlock::new();
    // xorshift64: scatter the reads so no device-side cache can absorb them
    // without touching the controller's power state machine.
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let idle_ms = idle.as_millis() as u64;
    while !sh.stop.load(Ordering::Relaxed) {
        std::thread::park_timeout(period);
        if sh.stop.load(Ordering::Relaxed) {
            break;
        }
        if !active(sh.last_use_ms.load(Ordering::Relaxed), sh.now_ms(), idle_ms) {
            continue;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let off = (x % blocks) * BLOCK as u64;
        match file.read_at(bounce.blocks(1), off) {
            Ok(_) => {
                sh.reads.fetch_add(1, Ordering::Relaxed);
            }
            Err(e) => {
                // Never fatal: without the ticker the faults are only slower.
                tracing::warn!("PLE NVMe keepalive: read at {off} failed ({e}); stopping");
                return;
            }
        }
    }
}

/// `ATLAS_PLE_NVME_KEEPALIVE_MS` (default 0 = off): the tick period, and
/// `ATLAS_PLE_NVME_KEEPALIVE_IDLE_MS` (default 2000): how long after the last
/// resolve it keeps ticking. 10 ms keeps the measured drive at its 245 us
/// first-read latency (a 25 ms gap already costs ~640 us).
pub fn keepalive_from_env() -> Option<(Duration, Duration)> {
    let ms = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok());
    let period = ms("ATLAS_PLE_NVME_KEEPALIVE_MS").filter(|&p| p > 0)?;
    let idle = ms("ATLAS_PLE_NVME_KEEPALIVE_IDLE_MS").unwrap_or(2000);
    Some((Duration::from_millis(period), Duration::from_millis(idle)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_window() {
        assert!(!active(u64::MAX, 5, 2000), "never used: idle");
        assert!(active(100, 100, 2000));
        assert!(active(100, 2100, 2000));
        assert!(!active(100, 2101, 2000));
        // A touch racing ahead of the tick's clock read is still active.
        assert!(active(200, 100, 2000));
    }

    fn temp_file(blocks: usize) -> (std::path::PathBuf, File) {
        let path =
            std::env::temp_dir().join(format!("ngram-keepalive-{}-{blocks}", std::process::id()));
        std::fs::write(&path, vec![0xA5u8; blocks * BLOCK]).unwrap();
        let f = File::open(&path).unwrap();
        (path, f)
    }

    /// Reads only while touched, stops after `idle`, and joins on drop.
    #[test]
    fn ticks_only_while_active() {
        let (path, f) = temp_file(4);
        let k = KeepAlive::spawn(f, Duration::from_millis(2), Duration::from_millis(40)).unwrap();
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(k.reads(), 0, "no resolve yet: the drive may sleep");
        k.touch();
        std::thread::sleep(Duration::from_millis(30));
        let during = k.reads();
        assert!(during > 0, "touched: ticking");
        std::thread::sleep(Duration::from_millis(120));
        let after = k.reads();
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(k.reads(), after, "idle past the window: stopped");
        drop(k);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_a_file_shorter_than_a_block() {
        let (path, f) = temp_file(0);
        assert!(KeepAlive::spawn(f, Duration::from_millis(5), Duration::from_millis(5)).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
