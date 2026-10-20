// SPDX-License-Identifier: AGPL-3.0-only

//! Persistent fault workers for the n-gram row cache
//! (`ATLAS_PLE_FAULT_POOL=1`, default off).
//!
//! `fetch_many` spawns a scoped thread per worker for EVERY fault batch. At
//! prefill scale that is noise, but a decode step faults 8-64 rows, and the
//! spawns are then most of the batch: measured on a GB10 against the
//! real PLE file, NVMe kept awake (`keepalive`), 16 misses at 16 spawned
//! workers resolved in 814 us p50 (one ~250 us read each), and 48 misses at
//! 48 spawned workers took 2.0 ms against 1.0 ms at 16. These workers are
//! spawned once, park on a condvar between batches, and pull faults off a
//! shared atomic cursor; the calling thread works the batch too. Same bench
//! (`examples/ngram_keepalive_bench.rs`, 140 ms between batches, keepalive on,
//! resolve p50): 16 misses 542 -> 338 us, 48 misses 1305 -> 534 us, and an
//! 8192-miss prefill batch 49.2 -> 44.1 ms.
//!
//! Every fault runs the same `fetch_row` as the scoped path into the slot its
//! decide pass chose, so the arena bytes are identical; only which thread
//! issues a read changes.

use std::fs::File;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use anyhow::{Context, Result, anyhow};

use super::fault::{ArenaPtrs, Fault, RowSource, fetch_row};
use super::{AlignedBlock, NgramRowCache, Segments};

/// The cache's immutable row geometry, owned (cloned descriptors) so 'static
/// workers can rebuild a `RowSource` from it.
struct OwnedSource {
    file: File,
    base_offset: u64,
    segments: Option<Segments>,
    row_stride: usize,
    scale_file: Option<File>,
}

impl OwnedSource {
    fn view(&self) -> RowSource<'_> {
        RowSource {
            file: &self.file,
            base_offset: self.base_offset,
            segments: self.segments.as_ref(),
            row_stride: self.row_stride,
            scale_file: self.scale_file.as_ref(),
        }
    }
}

/// The per-row read: `fetch_row`, swappable so the pool's own mechanics are
/// testable without an arena or NVMe.
type FetchFn = fn(&RowSource<'_>, &ArenaPtrs, &mut AlignedBlock, u64, u32) -> Result<()>;

/// One batch in flight. The raw arena pointers are valid for the whole batch:
/// `run` does not return until every claimed fault has finished.
struct Batch {
    faults: Vec<Fault>,
    rows: usize,
    scales: Option<usize>,
    next: AtomicUsize,
    finished: AtomicUsize,
    error: Mutex<Option<anyhow::Error>>,
}

impl Batch {
    /// Work faults until none are left to claim. Returns after this thread's
    /// last claimed fault completed. A panicking fetch is that row's error:
    /// every claimed fault must reach `finished`, or `run` waits forever.
    fn work(&self, fetch: FetchFn, src: &RowSource<'_>, bounce: &mut AlignedBlock) {
        let ptrs = ArenaPtrs {
            rows: self.rows as *mut u8,
            scales: self.scales.map(|p| p as *mut u8),
        };
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(f) = self.faults.get(i) else {
                return;
            };
            let r = catch_unwind(AssertUnwindSafe(|| fetch(src, &ptrs, bounce, f.id, f.slot)))
                .unwrap_or_else(|p| {
                    let msg = p
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| p.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("non-string payload");
                    Err(anyhow!("NgramRowCache: read row {} panicked: {msg}", f.id))
                });
            if let Err(e) = r {
                let mut g = self.error.lock().expect("fault pool error mutex");
                if g.is_none() {
                    *g = Some(e);
                }
            }
            self.finished.fetch_add(1, Ordering::Release);
        }
    }
}

#[derive(Default)]
struct Slot {
    /// Bumped per batch; a worker serves each generation once.
    generation: u64,
    batch: Option<Arc<Batch>>,
    stop: bool,
}

struct Shared {
    src: OwnedSource,
    fetch: FetchFn,
    slot: Mutex<Slot>,
    wake: Condvar,
    /// Signalled by the worker that finishes a batch's last fault.
    done: Condvar,
}

pub(super) struct FaultPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

/// Latched by the first failed build: the causes (fd or thread exhaustion)
/// are process-wide, so later resolves take the scoped path, not a retry.
static BUILD_FAILED: AtomicBool = AtomicBool::new(false);

/// `ATLAS_PLE_FAULT_POOL=1` (read once), until a build fails.
pub(super) fn pool_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_PLE_FAULT_POOL").ok().as_deref() == Some("1"))
        && !BUILD_FAILED.load(Ordering::Relaxed)
}

fn spawn_worker(i: usize, sh: Arc<Shared>) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name(format!("ple-fault-{i}"))
        .spawn(move || worker(&sh))
}

impl FaultPool {
    /// The pool, or `None` with one warning: an unbuildable pool costs the
    /// scoped path's spawns, never a failed resolve.
    pub(super) fn build(cache: &NgramRowCache, workers: usize) -> Option<Self> {
        Self::new(cache, workers)
            .inspect_err(|e| {
                if !BUILD_FAILED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        "PLE fault pool: build failed ({e:#}); scoped fault threads from now on"
                    );
                }
            })
            .ok()
    }

    fn new(cache: &NgramRowCache, workers: usize) -> Result<Self> {
        let segments = match &cache.segments {
            None => None,
            Some(s) => Some(Segments {
                bases: s.bases.clone(),
                rows_per: s.rows_per,
                files: s
                    .files
                    .iter()
                    .map(File::try_clone)
                    .collect::<std::io::Result<_>>()
                    .context("NgramRowCache fault pool: dup shard files")?,
                shard_file: s.shard_file.clone(),
            }),
        };
        let scale_file = match cache.scales.as_ref().and_then(|sc| sc.file.as_ref()) {
            Some(f) => Some(
                f.try_clone()
                    .context("NgramRowCache fault pool: dup scales")?,
            ),
            None => None,
        };
        let src = OwnedSource {
            file: cache
                .file
                .try_clone()
                .context("NgramRowCache fault pool: dup file")?,
            base_offset: cache.base_offset,
            segments,
            row_stride: cache.row_stride,
            scale_file,
        };
        Self::start(src, fetch_row, workers, spawn_worker)
    }

    /// Spawn `workers` threads. The pool owns each handle as it is spawned,
    /// so a failed spawn drops it and `Drop` stops and joins the rest.
    fn start(
        src: OwnedSource,
        fetch: FetchFn,
        workers: usize,
        mut spawn: impl FnMut(usize, Arc<Shared>) -> std::io::Result<JoinHandle<()>>,
    ) -> Result<Self> {
        let mut pool = Self {
            shared: Arc::new(Shared {
                src,
                fetch,
                slot: Mutex::new(Slot::default()),
                wake: Condvar::new(),
                done: Condvar::new(),
            }),
            workers: Vec::with_capacity(workers),
        };
        for i in 0..workers {
            let h = spawn(i, pool.shared.clone()).context("NgramRowCache fault pool: spawn")?;
            pool.workers.push(h);
        }
        Ok(pool)
    }

    /// Fault every row of `faults` into its slot; the caller's thread works
    /// too. Returns the first error after the WHOLE batch has settled, so no
    /// worker still writes the arena once this returns.
    pub(super) fn run(&self, faults: &[Fault], ptrs: &ArenaPtrs) -> Result<()> {
        let batch = Arc::new(Batch {
            faults: faults
                .iter()
                .map(|f| Fault {
                    id: f.id,
                    slot: f.slot,
                })
                .collect(),
            rows: ptrs.rows as usize,
            scales: ptrs.scales.map(|p| p as usize),
            next: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            error: Mutex::new(None),
        });
        {
            let mut s = self.shared.slot.lock().expect("fault pool mutex");
            s.generation += 1;
            s.batch = Some(batch.clone());
        }
        self.shared.wake.notify_all();
        let src = self.shared.src.view();
        let mut bounce = AlignedBlock::new();
        batch.work(self.shared.fetch, &src, &mut bounce);
        {
            let mut s = self.shared.slot.lock().expect("fault pool mutex");
            while batch.finished.load(Ordering::Acquire) < faults.len() {
                s = self.shared.done.wait(s).expect("fault pool mutex");
            }
            s.batch = None;
        }
        match batch.error.lock().expect("fault pool error mutex").take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

fn worker(sh: &Shared) {
    let src = sh.src.view();
    let mut bounce = AlignedBlock::new();
    let mut seen = 0u64;
    loop {
        let batch = {
            let mut s = sh.slot.lock().expect("fault pool mutex");
            while !s.stop && (s.generation == seen || s.batch.is_none()) {
                s = sh.wake.wait(s).expect("fault pool mutex");
            }
            if s.stop {
                return;
            }
            seen = s.generation;
            s.batch.clone().expect("checked above")
        };
        batch.work(sh.fetch, &src, &mut bounce);
        if batch.finished.load(Ordering::Acquire) == batch.faults.len() {
            // Taking the lock orders this notify after the caller's check.
            let _g = sh.slot.lock().expect("fault pool mutex");
            sh.done.notify_all();
        }
    }
}

impl Drop for FaultPool {
    fn drop(&mut self) {
        self.shared.slot.lock().expect("fault pool mutex").stop = true;
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
#[path = "fault_pool_tests.rs"]
mod tests;
