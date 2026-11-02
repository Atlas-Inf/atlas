// SPDX-License-Identifier: AGPL-3.0-only

//! The reader threads of one shard: tensors read ahead of the copier and
//! handed to it in tensor order. One reader waiting on one read at a time is
//! what bounds a checkpoint of many small tensors (~0.7 GB/s over a GLM-5 TP4
//! rank's shards).

use super::direct_io::{self, AlignedBuffer};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;
use std::os::unix::io::RawFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::thread::JoinHandle;

/// Reader threads per shard; how many tensors, and how many bytes past the
/// next one the upload needs, they may hold ahead of it.
const SHARD_READERS: usize = 8;
const READ_AHEAD: usize = 16;
const READ_AHEAD_BYTES: usize = 1 << 30;

/// One tensor's read: `(tensor_index, aligned_buffer, slice_start)`.
pub(super) type ReadMsg = (usize, AlignedBuffer, usize);

struct Shared {
    /// `(abs_offset, len)` per tensor.
    tensors: Vec<(u64, usize)>,
    /// The next tensor a reader may claim.
    next: AtomicUsize,
    /// Tensors handed to the copier.
    copied: AtomicUsize,
    /// Bytes read or being read, not yet handed to the copier.
    held: AtomicUsize,
    /// Raised when the copier is done, so no reader waits for it forever.
    stop: AtomicBool,
}

impl Shared {
    /// Wait until tensor `idx` of `len` bytes may be read; false on stop.
    fn admit(&self, idx: usize, len: usize) -> bool {
        loop {
            if self.stop.load(Ordering::Acquire) {
                return false;
            }
            let first = self.copied.load(Ordering::Acquire);
            if idx == first
                || (idx < first + READ_AHEAD
                    && self.held.load(Ordering::Acquire) + len <= READ_AHEAD_BYTES)
            {
                self.held.fetch_add(len, Ordering::AcqRel);
                return true;
            }
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }
}

/// Yields each tensor's read, in tensor order. The readers borrow `fd` by
/// number: declare this after the file it came from, so that it stops and
/// joins them (also on an early return) before the file closes.
pub(super) struct ShardReaders {
    shared: Arc<Shared>,
    rx: Option<Receiver<Result<ReadMsg>>>,
    /// Reads that overtook the one the copier wants.
    early: BTreeMap<usize, (AlignedBuffer, usize)>,
    want: usize,
    handles: Vec<JoinHandle<()>>,
}

impl ShardReaders {
    pub(super) fn spawn(fd: RawFd, using_direct: bool, tensors: Vec<(u64, usize)>) -> Self {
        let shared = Arc::new(Shared {
            tensors,
            next: AtomicUsize::new(0),
            copied: AtomicUsize::new(0),
            held: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
        });
        let (tx, rx) = sync_channel::<Result<ReadMsg>>(READ_AHEAD);
        let handles = (0..SHARD_READERS)
            .map(|_| {
                let (tx, shared) = (tx.clone(), shared.clone());
                std::thread::spawn(move || {
                    loop {
                        let idx = shared.next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(abs_offset, len)) = shared.tensors.get(idx) else {
                            break;
                        };
                        if !shared.admit(idx, len) {
                            break;
                        }
                        let msg = direct_io::read_tensor_aligned(fd, abs_offset, len, using_direct)
                            .map(|(buf, slice_start)| (idx, buf, slice_start));
                        if tx.send(msg).is_err() {
                            break; // receiver dropped
                        }
                    }
                })
            })
            .collect();
        Self {
            shared,
            rx: Some(rx),
            early: BTreeMap::new(),
            want: 0,
            handles,
        }
    }

    /// Stop and join the readers, reporting a panic in one of them.
    pub(super) fn join(mut self) -> Result<()> {
        self.stop_and_join()
    }

    fn stop_and_join(&mut self) -> Result<()> {
        self.shared.stop.store(true, Ordering::Release);
        // Dropping the receiver frees a reader blocked in `send`.
        self.rx = None;
        let mut panicked = 0;
        for reader in self.handles.drain(..) {
            panicked += usize::from(reader.join().is_err());
        }
        ensure!(panicked == 0, "{panicked} reader thread(s) panicked");
        Ok(())
    }
}

impl Iterator for ShardReaders {
    type Item = Result<ReadMsg>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((buf, slice_start)) = self.early.remove(&self.want) {
                let idx = self.want;
                self.shared
                    .held
                    .fetch_sub(self.shared.tensors[idx].1, Ordering::AcqRel);
                self.want += 1;
                self.shared.copied.store(self.want, Ordering::Release);
                return Some(Ok((idx, buf, slice_start)));
            }
            if self.want == self.shared.tensors.len() {
                return None;
            }
            match self.rx.as_ref()?.recv() {
                Ok(Ok((idx, buf, slice_start))) => {
                    self.early.insert(idx, (buf, slice_start));
                }
                Ok(Err(e)) => return Some(Err(e)),
                // Every reader is gone with a read outstanding: `join` says why.
                Err(_) => return None,
            }
        }
    }
}

impl Drop for ShardReaders {
    fn drop(&mut self) {
        let _ = self.stop_and_join();
    }
}

#[cfg(test)]
#[path = "shard_readers_tests.rs"]
mod tests;
