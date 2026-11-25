// SPDX-License-Identifier: AGPL-3.0-only

//! The score pack store's NVMe tier: a stored pack evicted from device
//! memory is written here (its FP8 bytes as they are) and read back on a
//! miss, so a photo library larger than the device budget still skips its
//! passes. A pack keeps its records when it goes back to the device, so one
//! evicted again costs no write; the disk drops its least recently used.
//!
//! `ATLAS_SCORE_STORE_DIR` (a directory on the node's NVMe) and
//! `ATLAS_SCORE_STORE_DISK_GB` (its budget) turn it on. One record file of
//! 1 MiB records, `O_DIRECT`, owner-only, its whole budget reserved up front
//! and unlinked once open, the same as the KV prefix tier: nothing outlives
//! the process, and the records are only ever this process's packs.
//! `ATLAS_SCORE_STORE_READ_AHEAD` packs (default 16, 0: off) are read ahead
//! of their passes into host memory (`score_store_ahead.rs`).

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, ensure};
use atlas_tier::{ConcurrentSwapStore, SharedRecordFile};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::score_store_ahead::ReadAhead;
use crate::model::pinned_upload::PinnedUpload;

const DIR_VAR: &str = "ATLAS_SCORE_STORE_DIR";
const GB_VAR: &str = "ATLAS_SCORE_STORE_DISK_GB";
const AHEAD_VAR: &str = "ATLAS_SCORE_STORE_READ_AHEAD";
const FILE_PREFIX: &str = "atlas-score-packs.";
/// A pack is a list of these (about 60 for a photo).
const RECORD: usize = 1 << 20;

pub(super) struct Disk {
    file: Arc<SharedRecordFile>,
    /// Free records below `next`; `next..total` were never written.
    free: BTreeSet<usize>,
    next: usize,
    total: usize,
    stage: PinnedUpload,
    /// Stamped on each write, so a read ahead of a rewritten pack is never used.
    generation: u64,
    pub(super) ahead: Option<ReadAhead>,
}

impl Disk {
    /// The tier the environment configures, or `None` when it is off.
    pub(super) fn open() -> Result<Option<Self>> {
        let Some(dir) = std::env::var_os(DIR_VAR)
            .filter(|d| !d.is_empty())
            .map(PathBuf::from)
        else {
            return Ok(None);
        };
        let gb: f64 = std::env::var(GB_VAR)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let total = (gb.max(0.0) * (1u64 << 30) as f64) as usize / RECORD;
        ensure!(
            total > 0,
            "{DIR_VAR} is set: {GB_VAR}=<GiB> (the disk budget) is required"
        );
        crate::factory::usable_disk_dir(DIR_VAR, &dir)?;
        atlas_tier::remove_stale_swap_files(&dir, FILE_PREFIX);
        // Owner-only, never through a pre-planted file or link (an exclusive
        // create), unlinked as soon as the space is reserved.
        let path = dir.join(format!("{FILE_PREFIX}{}.swap", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let file = SharedRecordFile::create(&path, RECORD)?;
        let reserved = file.reserve((total * RECORD) as u64);
        #[cfg(unix)]
        let _ = std::fs::remove_file(&path);
        if !reserved
            .map_err(|e| e.context(format!("{GB_VAR}: the disk budget does not fit {DIR_VAR}")))?
        {
            tracing::warn!(
                "score pack store: this filesystem cannot reserve space; the record file grows on demand"
            );
        }
        let slots: usize = std::env::var(AHEAD_VAR)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        tracing::info!(
            "score pack store NVMe tier ON: {} ({DIR_VAR}, O_DIRECT, unlinked), {gb:.1} GiB = {total} records of 1 MiB, \
             read-ahead {slots} packs ({AHEAD_VAR})",
            dir.display()
        );
        let file = Arc::new(file);
        let ahead = match slots {
            0 => None,
            n => Some(ReadAhead::new(Arc::clone(&file), n)?),
        };
        Ok(Some(Self {
            file,
            free: BTreeSet::new(),
            next: 0,
            total,
            stage: PinnedUpload::EMPTY,
            generation: 0,
            ahead,
        }))
    }

    /// Records a pack of `bytes` takes.
    pub(super) fn records(bytes: usize) -> usize {
        bytes.div_ceil(RECORD)
    }

    pub(super) fn free_records(&self) -> usize {
        self.free.len() + self.total - self.next
    }

    /// `n` free records, lowest first (so a pack's are mostly one run).
    fn take(&mut self, n: usize) -> Vec<usize> {
        let mut got = Vec::with_capacity(n);
        while got.len() < n {
            got.push(self.free.pop_first().unwrap_or_else(|| {
                self.next += 1;
                self.next - 1
            }));
        }
        got
    }

    pub(super) fn give_back(&mut self, records: Vec<usize>) {
        self.free.extend(records);
    }

    /// Write the device copy `buf` (`bytes`) to new records, which the
    /// caller has made room for: the records and the write's generation.
    pub(super) fn write(
        &mut self,
        gpu: &dyn GpuBackend,
        buf: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<(Vec<usize>, u64)> {
        let n = Self::records(bytes);
        ensure!(
            n <= self.free_records(),
            "score pack store: {n} records to write, {} free",
            self.free_records()
        );
        let records = self.take(n);
        let written = (|| -> Result<()> {
            let stage = self.stage.stage_as_left(gpu, n * RECORD)?;
            stage[bytes..].fill(0);
            gpu.copy_d2h_async(buf, &mut stage[..bytes], stream)?;
            gpu.synchronize(stream)?;
            for (first, at, len) in runs(&records, RECORD) {
                self.file.write_run(first, false, &stage[at..at + len])?;
            }
            Ok(())
        })();
        match written {
            Ok(()) => {
                self.generation += 1;
                Ok((records, self.generation))
            }
            Err(e) => {
                self.give_back(records);
                Err(e)
            }
        }
    }

    /// Read `records` (a pack of `bytes`) into the device buffer `buf`.
    pub(super) fn read(
        &mut self,
        gpu: &dyn GpuBackend,
        records: &[usize],
        buf: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let stage = self.stage.stage_as_left(gpu, records.len() * RECORD)?;
        for (first, at, len) in runs(records, RECORD) {
            self.file.read_run(first, false, &mut stage[at..at + len])?;
        }
        self.stage.send(gpu, bytes, buf, stream)
    }

    /// Start reading pack `id` ahead (see [`ReadAhead::submit`]); false when
    /// read-ahead is off or has no slot for it.
    pub(super) fn prefetch(
        &mut self,
        gpu: &dyn GpuBackend,
        id: u64,
        generation: u64,
        records: &[usize],
        wanted: &[u64],
    ) -> Result<bool> {
        match self.ahead.as_mut() {
            Some(ahead) => ahead.submit(gpu, id, generation, records, RECORD, wanted),
            None => Ok(false),
        }
    }

    /// Stop the read-ahead and free the host buffers (model teardown; the
    /// file closes on drop).
    pub(super) fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let ahead = self.ahead.as_mut().map_or(Ok(()), |a| a.release(gpu));
        self.stage.release(gpu).and(ahead)
    }
}

/// The runs of consecutive records in `records` (`record` bytes each):
/// (first record, byte offset in the pack, bytes).
pub(super) fn runs(records: &[usize], record: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < records.len() {
        let mut j = i + 1;
        while j < records.len() && records[j] == records[j - 1] + 1 {
            j += 1;
        }
        out.push((records[i], i * record, (j - i) * record));
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{RECORD, runs};

    #[test]
    fn records_group_into_runs() {
        assert_eq!(runs(&[], RECORD), vec![]);
        assert_eq!(runs(&[4, 5, 6], RECORD), vec![(4, 0, 3 * RECORD)]);
        assert_eq!(
            runs(&[0, 1, 7, 9, 10], RECORD),
            vec![
                (0, 0, 2 * RECORD),
                (7, 2 * RECORD, RECORD),
                (9, 3 * RECORD, 2 * RECORD)
            ]
        );
    }
}
