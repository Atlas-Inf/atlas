// SPDX-License-Identifier: AGPL-3.0-only

//! Worker pool behind the NVMe prefix tier's fast path (`nvme_fast.rs`).
//!
//! The serving thread owns the GPU side (gather / scatter between the KV
//! pools and pinned staging); these threads own everything that never touches
//! the GPU: checksums and disk I/O. A job names one staging CHUNK, which the
//! worker owns exclusively from `submit` until its outcome is collected.
//!
//! * **Write** — zero-pad and stamp each record (checksum), then write the
//!   chunk's slot runs.
//! * **Read** — read the chunk's slot runs and verify every trailer; reports
//!   how many leading records hold verified bytes.
//!
//! Jobs run concurrently and in no particular order, which is what gives a
//! fragmented tier (every record its own request) a queue depth above one.
//! The SUBMITTER therefore never has two jobs in flight that touch the same
//! slot (`nvme_fast.rs` drains first) — that, not an ordering here, is what
//! makes a re-issued slot end up holding its later record.
//!
//! A worker never panics the process: a panic inside a job is caught and
//! reported as that job's failure — the serving thread must never wait on a
//! job that cannot finish.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::thread::JoinHandle;

use anyhow::Result;
use atlas_tier::ConcurrentSwapStore;
use parking_lot::{Condvar, Mutex};

use super::nvme_spill::{note_write_failure, run_layout, stamp, verify};
use crate::prefix_cache::{DiskRef, SpillOrder};

/// What a worker does with one staging chunk (items in staging order).
pub(super) enum Work {
    Write(Vec<SpillOrder>),
    Read(Vec<DiskRef>),
}

pub(super) enum Outcome {
    /// Orders that did NOT reach the disk.
    Wrote(Vec<SpillOrder>),
    /// Leading records read AND verified (fewer than asked = the next failed).
    Read(usize),
}

struct Job {
    chunk: usize,
    work: Work,
}

struct State {
    jobs: VecDeque<Job>,
    done: Vec<(usize, Outcome)>,
    stop: bool,
}

struct Shared {
    store: Arc<dyn ConcurrentSwapStore>,
    staging: *mut u8,
    chunk_bytes: usize,
    record: usize,
    payload: usize,
    state: Mutex<State>,
    /// Workers: a job was queued, or the pool is stopping.
    work: Condvar,
    /// Owner: an outcome was posted.
    done: Condvar,
}

// SAFETY: `staging` is only dereferenced one chunk at a time, by the single
// worker holding that chunk's job (see the module doc); everything else is
// `Sync` already.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

pub(super) struct IoPool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl IoPool {
    /// `staging` holds chunks of `chunk_bytes`; it must outlive the pool.
    pub(super) fn new(
        store: Arc<dyn ConcurrentSwapStore>,
        staging: *mut u8,
        chunk_bytes: usize,
        (record, payload): (usize, usize),
        workers: usize,
    ) -> Result<Self> {
        let shared = Arc::new(Shared {
            store,
            staging,
            chunk_bytes,
            record,
            payload,
            state: Mutex::new(State {
                jobs: VecDeque::new(),
                done: Vec::new(),
                stop: false,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
        });
        let mut pool = Self {
            shared: shared.clone(),
            threads: Vec::with_capacity(workers),
        };
        for i in 0..workers {
            let shared = shared.clone();
            // On a spawn error `pool` drops and joins the workers started so far.
            pool.threads.push(
                std::thread::Builder::new()
                    .name(format!("atlas-kv-nvme-{i}"))
                    .spawn(move || worker(&shared))?,
            );
        }
        Ok(pool)
    }

    /// Hand `chunk` to a worker. The caller must not touch the chunk again
    /// until it has collected this job's outcome, and must not submit a job
    /// that shares a slot with one still in flight.
    pub(super) fn submit(&self, chunk: usize, work: Work) {
        let mut st = self.shared.state.lock();
        st.jobs.push_back(Job { chunk, work });
        drop(st);
        self.shared.work.notify_one();
    }

    /// Block until the job on `chunk` has finished; returns its outcome.
    pub(super) fn wait(&self, chunk: usize) -> Outcome {
        let mut st = self.shared.state.lock();
        loop {
            if let Some(i) = st.done.iter().position(|(c, _)| *c == chunk) {
                return st.done.swap_remove(i).1;
            }
            self.shared.done.wait(&mut st);
        }
    }

    /// `(chunk, failed orders)` of every finished write; with `block`, waits
    /// until there is at least one.
    pub(super) fn take_writes(&self, block: bool) -> Vec<(usize, Vec<SpillOrder>)> {
        let mut st = self.shared.state.lock();
        loop {
            let mut out = Vec::new();
            for (chunk, outcome) in std::mem::take(&mut st.done) {
                match outcome {
                    Outcome::Wrote(failed) => out.push((chunk, failed)),
                    read => st.done.push((chunk, read)),
                }
            }
            if !out.is_empty() || !block {
                return out;
            }
            self.shared.done.wait(&mut st);
        }
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        self.shared.state.lock().stop = true;
        self.shared.work.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// Runs queued jobs until the pool stops AND the queue is empty (a queued
/// write still reaches the disk).
fn worker(shared: &Shared) {
    loop {
        let job = {
            let mut st = shared.state.lock();
            loop {
                if let Some(job) = st.jobs.pop_front() {
                    break job;
                }
                if st.stop {
                    return;
                }
                shared.work.wait(&mut st);
            }
        };
        let items = match &job.work {
            Work::Write(o) => o.len(),
            Work::Read(r) => r.len(),
        };
        // A job never reaches past its chunk: an oversized one touches no
        // memory and fails as a whole (empty buffer → nothing read or written).
        let bytes = (items * shared.record).min(shared.chunk_bytes);
        let fits = bytes == items * shared.record;
        // SAFETY: this worker owns `job.chunk` until the outcome is posted,
        // and `bytes` lies inside that chunk.
        let buf = unsafe {
            std::slice::from_raw_parts_mut(
                shared.staging.add(job.chunk * shared.chunk_bytes),
                if fits { bytes } else { 0 },
            )
        };
        let outcome = match &job.work {
            Work::Write(orders) if !fits => Outcome::Wrote(orders.clone()),
            Work::Read(_) if !fits => Outcome::Read(0),
            Work::Write(orders) => Outcome::Wrote(
                catch_unwind(AssertUnwindSafe(|| {
                    stamp_all(shared, orders, buf);
                    write_runs(shared, orders, buf)
                }))
                .unwrap_or_else(|_| orders.clone()),
            ),
            Work::Read(refs) => Outcome::Read(
                catch_unwind(AssertUnwindSafe(|| read_verified(shared, refs, buf))).unwrap_or(0),
            ),
        };
        shared.state.lock().done.push((job.chunk, outcome));
        shared.done.notify_all();
    }
}

fn stamp_all(shared: &Shared, orders: &[SpillOrder], buf: &mut [u8]) {
    for (rec, o) in buf.chunks_exact_mut(shared.record).zip(orders) {
        rec[shared.payload..].fill(0);
        stamp(rec, shared.payload, o.tag);
    }
}

/// `true` when a run of `len` records starting at staging index `start`
/// holds its slots highest first.
fn reversed(slots: &[u32], start: usize, len: usize) -> bool {
    len > 1 && slots[start] > slots[start + 1]
}

/// One request per run of consecutive slots; returns the orders of every run
/// that failed.
fn write_runs(shared: &Shared, orders: &[SpillOrder], buf: &[u8]) -> Vec<SpillOrder> {
    let record = shared.record;
    let slots: Vec<u32> = orders.iter().map(|o| o.slot).collect();
    let mut failed = Vec::new();
    for (start, len, low) in run_layout(&slots) {
        let run = &buf[start * record..(start + len) * record];
        let rev = reversed(&slots, start, len);
        if let Err(e) = shared.store.write_run(low as usize, rev, run) {
            note_write_failure(low, &e);
            failed.extend_from_slice(&orders[start..start + len]);
        }
    }
    failed
}

/// Read every run into place (staging order = `refs` order), then verify;
/// returns how many leading records are read and verified.
fn read_verified(shared: &Shared, refs: &[DiskRef], buf: &mut [u8]) -> usize {
    let (record, payload) = (shared.record, shared.payload);
    let slots: Vec<u32> = refs.iter().map(|d| d.slot).collect();
    let mut read = 0;
    'runs: for (start, len, low) in run_layout(&slots) {
        let run = &mut buf[start * record..(start + len) * record];
        let rev = reversed(&slots, start, len);
        if shared.store.read_run(low as usize, rev, run).is_ok() {
            read += len;
            continue;
        }
        // Find the exact record that fails, keeping the ones before it.
        for (k, rec) in run.chunks_exact_mut(record).enumerate() {
            let slot = slots[start + k];
            if let Err(e) = shared.store.read_run(slot as usize, false, rec) {
                tracing::warn!("NVMe KV restore: read of slot {slot} failed ({e:#})");
                break 'runs;
            }
            read += 1;
        }
    }
    let ok = buf
        .chunks_exact(record)
        .zip(refs)
        .take(read)
        .take_while(|(rec, d)| verify(rec, payload, d.tag))
        .count();
    if ok < read {
        tracing::warn!(
            "NVMe KV restore: slot {} failed verification (tag/checksum) — recomputing",
            refs[ok].slot
        );
    }
    ok
}
