// SPDX-License-Identifier: AGPL-3.0-only

//! Read-ahead for the score pack store's NVMe tier (`score_store_disk.rs`):
//! the scheduler names the disk-only packs its queued jobs need
//! (`Model::score_packs_prefetch`) and one thread reads them into page-locked
//! host slots while the GPU runs other passes, so the pass that restores one
//! unpacks it from its slot (read in place by the GPU) instead of waiting on
//! the disk.
//!
//! The thread does disk I/O only; the scheduler thread owns the slots, the
//! GPU and the store. A slot belongs to the thread from `submit` until its
//! outcome is collected, and its records cannot be reused meanwhile (the
//! store keeps a pack that is being read on disk: `reading`). A panic in a
//! read is caught and reported as that read's failure, so the scheduler never
//! waits on a read that cannot finish.

use std::collections::{HashMap, HashSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::Result;
use atlas_tier::{ConcurrentSwapStore, SharedRecordFile};
use parking_lot::{Condvar, Mutex};
use spark_runtime::gpu::GpuBackend;

/// The largest pack read ahead (a slot holds one; 16 slots pin at most
/// 4 GiB): bigger ones (long text packs) are read by their pass.
pub(super) const MAX_SLOT_BYTES: usize = 256 << 20;

/// A pack's place in a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Ahead {
    Reading,
    Ready,
}

struct Entry {
    slot: usize,
    /// The stored pack's disk generation the bytes were read from.
    generation: u64,
    state: Ahead,
    /// When it was submitted (oldest Ready slots are reclaimed first).
    order: u64,
}

/// One read: runs of consecutive records (first record, byte offset in the
/// slot, bytes) into the slot at `dst`.
struct Job {
    slot: usize,
    dst: usize,
    runs: Vec<(usize, usize, usize)>,
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    /// (slot, read ok).
    done: Vec<(usize, bool)>,
    stop: bool,
}

#[derive(Default)]
struct Shared {
    q: Mutex<Queue>,
    work: Condvar,
    done: Condvar,
}

/// A page-locked host buffer.
struct Slot {
    ptr: *mut u8,
    cap: usize,
}

pub(super) struct ReadAhead {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
    slots: Vec<Slot>,
    free: Vec<usize>,
    packs: HashMap<u64, Entry>,
    /// Packs whose read ahead failed, left to their pass until it reads them.
    failed: HashSet<u64>,
    order: u64,
}

// SAFETY: the slot pointers are page-locked host memory this struct owns;
// the reader thread writes a slot only between `submit` and the collection
// of its outcome, during which the scheduler side does not touch it.
unsafe impl Send for ReadAhead {}

impl ReadAhead {
    /// `slots` packs at most in flight or ready; the thread reads `file`.
    pub(super) fn new(file: Arc<SharedRecordFile>, slots: usize) -> Result<Self> {
        let shared = Arc::new(Shared::default());
        let worker = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("score-read-ahead".into())
                .spawn(move || worker(&shared, file.as_ref()))?
        };
        Ok(Self {
            shared,
            worker: Some(worker),
            slots: (0..slots)
                .map(|_| Slot {
                    ptr: std::ptr::null_mut(),
                    cap: 0,
                })
                .collect(),
            free: (0..slots).rev().collect(),
            packs: HashMap::new(),
            failed: HashSet::new(),
            order: 0,
        })
    }

    /// Where pack `id` is, if it has a slot.
    pub(super) fn state(&self, id: u64) -> Option<Ahead> {
        self.packs.get(&id).map(|e| e.state)
    }

    /// Whether pack `id`'s read ahead failed (its pass reads it).
    pub(super) fn failed(&self, id: u64) -> bool {
        self.failed.contains(&id)
    }

    /// Whether any of `ids` is still being read.
    pub(super) fn reading(&self, ids: &[u64]) -> bool {
        ids.iter().any(|id| self.state(*id) == Some(Ahead::Reading))
    }

    /// Take finished reads: a read that failed loses its slot (the pass reads
    /// the pack itself, which drops it if the disk fails again).
    pub(super) fn collect(&mut self) {
        let done = std::mem::take(&mut self.shared.q.lock().done);
        for (slot, ok) in done {
            let Some((&id, _)) = self.packs.iter().find(|(_, e)| e.slot == slot) else {
                continue;
            };
            if ok {
                self.packs.get_mut(&id).expect("found").state = Ahead::Ready;
            } else {
                tracing::warn!("score pack store: reading pack {id:016x} ahead failed");
                self.packs.remove(&id);
                self.free.push(slot);
                self.failed.insert(id);
            }
        }
    }

    /// Wait up to `timeout` for a read to finish, then collect.
    pub(super) fn wait(&mut self, timeout: Duration) {
        {
            let mut q = self.shared.q.lock();
            let running = self.packs.values().any(|e| e.state == Ahead::Reading);
            if q.done.is_empty() && running {
                self.shared.done.wait_for(&mut q, timeout);
            }
        }
        self.collect();
    }

    /// Start reading pack `id` (`bytes` at `records`, disk generation
    /// `generation`) into a slot: a free one, else the oldest Ready slot whose
    /// pack is not in `wanted`. False when no slot can be had.
    pub(super) fn submit(
        &mut self,
        gpu: &dyn GpuBackend,
        id: u64,
        generation: u64,
        records: &[usize],
        record: usize,
        wanted: &[u64],
    ) -> Result<bool> {
        if self.packs.contains_key(&id) {
            return Ok(true);
        }
        let need = records.len() * record;
        if need > MAX_SLOT_BYTES || self.failed.contains(&id) {
            return Ok(false);
        }
        if self.free.is_empty() {
            let spare = self
                .packs
                .iter()
                .filter(|(id, e)| e.state == Ahead::Ready && !wanted.contains(id))
                .min_by_key(|(_, e)| e.order)
                .map(|(&id, _)| id);
            let Some(spare) = spare else {
                return Ok(false);
            };
            self.drop_pack(spare);
        }
        let slot = self.free.pop().expect("a free slot");
        let s = &mut self.slots[slot];
        if s.cap < need {
            if !s.ptr.is_null() {
                gpu.free_host_pinned(s.ptr, s.cap)?;
                (s.ptr, s.cap) = (std::ptr::null_mut(), 0);
            }
            match gpu.alloc_host_pinned(need) {
                Ok(ptr) => (s.ptr, s.cap) = (ptr, need),
                Err(e) => {
                    self.free.push(slot);
                    return Err(e);
                }
            }
        }
        let runs = super::score_store_disk::runs(records, record);
        self.order += 1;
        self.packs.insert(
            id,
            Entry {
                slot,
                generation,
                state: Ahead::Reading,
                order: self.order,
            },
        );
        self.shared.q.lock().jobs.push_back(Job {
            slot,
            dst: s.ptr as usize,
            runs,
        });
        self.shared.work.notify_one();
        Ok(true)
    }

    /// Pack `id`'s bytes if its slot holds `generation` (waiting for a read
    /// still running): `use_bytes` gets them, and must be done with them
    /// (GPU work on them synchronized) when it returns: the slot is freed.
    /// `None` when there is no such slot (or its read failed).
    pub(super) fn take<T>(
        &mut self,
        id: u64,
        generation: u64,
        bytes: usize,
        use_bytes: impl FnOnce(&[u8]) -> Result<T>,
    ) -> Result<Option<T>> {
        self.failed.remove(&id); // its pass reads it now, one way or the other
        while self.state(id) == Some(Ahead::Reading) {
            self.wait(Duration::from_millis(100));
        }
        let Some(e) = self.packs.get(&id) else {
            return Ok(None);
        };
        let (slot, current) = (e.slot, e.generation == generation);
        let fits = bytes <= self.slots[slot].cap;
        let out = if current && fits {
            // SAFETY: the slot's read finished (collected above), so the
            // thread no longer writes it, and it holds `cap >= bytes` bytes,
            // zeroed at allocation and written since.
            let src = unsafe { std::slice::from_raw_parts(self.slots[slot].ptr, bytes) };
            Some(use_bytes(src))
        } else {
            None
        };
        // Freed whether or not `use_bytes` succeeded (it is done with them).
        self.drop_pack(id);
        out.transpose()
    }

    /// Free the slots of packs that no longer match the store (`current`:
    /// the pack still needs a read and its disk generation is this one).
    pub(super) fn retain(&mut self, current: impl Fn(u64, u64) -> bool) {
        let stale: Vec<u64> = self
            .packs
            .iter()
            .filter(|&(&id, e)| e.state == Ahead::Ready && !current(id, e.generation))
            .map(|(&id, _)| id)
            .collect();
        for id in stale {
            self.drop_pack(id);
        }
    }

    fn drop_pack(&mut self, id: u64) {
        if let Some(e) = self.packs.remove(&id) {
            debug_assert!(
                e.state == Ahead::Ready,
                "a slot being read is never dropped"
            );
            self.free.push(e.slot);
        }
    }

    /// Stop the thread, then free the slots (model teardown).
    pub(super) fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.stop();
        let mut first = Ok(());
        for s in &mut self.slots {
            if !s.ptr.is_null() {
                let r = gpu.free_host_pinned(s.ptr, s.cap);
                (s.ptr, s.cap) = (std::ptr::null_mut(), 0);
                if first.is_ok() {
                    first = r;
                }
            }
        }
        self.packs.clear();
        self.failed.clear();
        first
    }

    fn stop(&mut self) {
        self.shared.q.lock().stop = true;
        self.shared.work.notify_all();
        if let Some(w) = self.worker.take()
            && w.join().is_err()
        {
            tracing::error!("score read-ahead thread panicked");
        }
    }
}

impl Drop for ReadAhead {
    fn drop(&mut self) {
        // The slots outlive the thread; `release` frees them with the GPU.
        self.stop();
    }
}

fn worker(shared: &Shared, file: &SharedRecordFile) {
    loop {
        let job = {
            let mut q = shared.q.lock();
            loop {
                if q.stop {
                    return;
                }
                if let Some(job) = q.jobs.pop_front() {
                    break job;
                }
                shared.work.wait(&mut q);
            }
        };
        let ok = catch_unwind(AssertUnwindSafe(|| {
            job.runs.iter().try_for_each(|&(first, at, len)| {
                // SAFETY: the slot is this thread's until its outcome is
                // collected, and the runs lie within it (`submit` sized it).
                let out = unsafe { std::slice::from_raw_parts_mut((job.dst + at) as *mut u8, len) };
                file.read_run(first, false, out)
            })
        }));
        let ok = matches!(ok, Ok(Ok(())));
        shared.q.lock().done.push((job.slot, ok));
        shared.done.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use atlas_tier::{ConcurrentSwapStore, SharedRecordFile};
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::{Ahead, ReadAhead};

    const R: usize = 4096;

    /// A record file of `records` records (record r all r + 1), in a fresh
    /// directory under the build tree (a container's `/tmp` may refuse
    /// `O_DIRECT`), removed again: the open file outlives its name.
    fn file(tag: &str, records: usize) -> Arc<SharedRecordFile> {
        let exe = std::env::current_exe().unwrap();
        let dir = exe
            .ancestors()
            .nth(3)
            .unwrap()
            .join("atlas-score-ahead-tests");
        let dir = dir.join(format!("{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let f = SharedRecordFile::create(&dir.join("packs.swap"), R).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        for r in 0..records {
            f.write_run(r, false, &vec![r as u8 + 1; R]).unwrap();
        }
        Arc::new(f)
    }

    #[test]
    fn a_read_ahead_lands_and_is_taken_once() {
        let gpu = MockGpuBackend::new();
        let mut ra = ReadAhead::new(file("lands", 8), 2).unwrap();
        assert!(ra.submit(&gpu, 7, 1, &[2, 3, 6], R, &[7]).unwrap());
        assert!(ra.reading(&[7]));
        while ra.state(7) == Some(Ahead::Reading) {
            ra.wait(std::time::Duration::from_millis(50));
        }
        assert_eq!(ra.state(7), Some(Ahead::Ready));
        let got = ra
            .take(7, 1, 3 * R - 5, |b| Ok(b.to_vec()))
            .unwrap()
            .unwrap();
        assert_eq!(
            (got[0], got[R], got[2 * R], got.len()),
            (3, 4, 7, 3 * R - 5)
        );
        // Taken: the slot is free again and the pack has none.
        assert_eq!(ra.state(7), None);
        assert!(ra.take(7, 1, R, |_| Ok(())).unwrap().is_none());
        ra.release(&gpu).unwrap();
    }

    #[test]
    fn a_stale_generation_is_never_used() {
        let gpu = MockGpuBackend::new();
        let mut ra = ReadAhead::new(file("stale", 4), 1).unwrap();
        ra.submit(&gpu, 1, 5, &[0], R, &[1]).unwrap();
        // Rewritten since (generation 6): the slot is dropped, not used.
        assert!(ra.take(1, 6, R, |_| Ok(())).unwrap().is_none());
        assert_eq!(ra.state(1), None);
        // `retain` frees ready slots the store no longer wants.
        ra.submit(&gpu, 2, 1, &[1], R, &[2]).unwrap();
        ra.wait(std::time::Duration::from_secs(5));
        ra.retain(|_, _| false);
        assert_eq!(ra.state(2), None);
        ra.release(&gpu).unwrap();
    }

    #[test]
    fn full_slots_reclaim_the_oldest_unwanted_ready_pack() {
        let gpu = MockGpuBackend::new();
        let mut ra = ReadAhead::new(file("reclaim", 4), 2).unwrap();
        ra.submit(&gpu, 1, 1, &[0], R, &[]).unwrap();
        ra.submit(&gpu, 2, 1, &[1], R, &[]).unwrap();
        while ra.reading(&[1, 2]) {
            ra.wait(std::time::Duration::from_millis(50));
        }
        // Both slots hold wanted packs: no room.
        assert!(!ra.submit(&gpu, 3, 1, &[2], R, &[1, 2, 3]).unwrap());
        // Pack 1 is no longer wanted: its slot goes to pack 3.
        assert!(ra.submit(&gpu, 3, 1, &[2], R, &[2, 3]).unwrap());
        assert_eq!((ra.state(1), ra.state(2)), (None, Some(Ahead::Ready)));
        let got = ra.take(3, 1, R, |b| Ok(b[0])).unwrap();
        assert_eq!(got, Some(3));
        ra.release(&gpu).unwrap();
    }

    #[test]
    fn a_failed_read_frees_its_slot() {
        let gpu = MockGpuBackend::new();
        let mut ra = ReadAhead::new(file("failed", 1), 1).unwrap();
        // Records past the end of the file read short: the read fails.
        ra.submit(&gpu, 9, 1, &[40], R, &[9]).unwrap();
        while ra.reading(&[9]) {
            ra.wait(std::time::Duration::from_millis(50));
        }
        assert_eq!(ra.state(9), None);
        // Remembered: not read ahead again, until its pass takes it.
        assert!(ra.failed(9));
        assert!(!ra.submit(&gpu, 9, 1, &[40], R, &[9]).unwrap());
        assert!(ra.take(9, 1, R, |_| Ok(())).unwrap().is_none());
        assert!(!ra.failed(9));
        assert!(ra.submit(&gpu, 8, 1, &[0], R, &[8]).unwrap());
        ra.release(&gpu).unwrap();
    }
}
