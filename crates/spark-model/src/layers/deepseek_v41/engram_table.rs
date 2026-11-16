// SPDX-License-Identifier: AGPL-3.0-only

//! An Engram layer's table where it sits, on disk: 384 million rows, of
//! which a token reads 24. The rows a token's n-grams hash to are faulted
//! into a pinned row cache, whose memory the GPU reads in place, and the
//! lookup is handed their slots.

use std::sync::Mutex;

use anyhow::{Context, Result, ensure};
use atlas_core::config::DeepseekV41Engram;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};
use spark_storage::NgramRowCache;

use super::engram::{ENGRAM_DIM, ENGRAM_SCALE_GROUP};

/// Rows of a table kept in memory: 64 MB of values and 2 MB of scales.
const CACHED_ROWS: usize = 1 << 18;

/// How one Engram layer hashes the n-grams ending at a token into table
/// rows (`NgramHashState` in the release's `inference/engram.py`, and
/// `DeepseekV41Engram::row_ids`, which this is held to).
pub struct EngramHash {
    primes: Vec<u64>,
    offsets: Vec<u64>,
    multipliers: Vec<u64>,
    pad: u32,
    heads: usize,
}

impl EngramHash {
    /// Engram layer `index` of `geometry`.
    pub fn new(geometry: &DeepseekV41Engram, index: usize) -> Self {
        Self {
            primes: geometry.primes(index),
            offsets: geometry.offsets(index),
            multipliers: geometry.multipliers(index),
            pad: geometry.pad_id,
            heads: geometry.n_heads,
        }
    }

    /// Tokens an n-gram can reach back over, the hashed one included.
    pub fn reach(&self) -> usize {
        self.multipliers.len()
    }

    /// Rows a token reads.
    pub fn columns(&self) -> usize {
        self.primes.len()
    }

    /// The rows of the n-grams ending at the last of `tokens`, n-gram size
    /// major. `tokens` are compressed ids: the last `reach()` of the
    /// sequence, or all of it while it is shorter. Shifts past its start
    /// take the pad.
    pub fn rows(&self, tokens: &[u32]) -> Vec<u64> {
        let mut rows = Vec::with_capacity(self.columns());
        let mut recent = tokens.iter().rev();
        let mut rolling = 0u64;
        for (shift, multiplier) in self.multipliers.iter().enumerate() {
            // Once the sequence has run out, every longer shift is the pad.
            let token = recent.next().copied().unwrap_or(self.pad);
            let product = u64::from(token) * multiplier;
            if shift == 0 {
                rolling = product;
                continue;
            }
            rolling ^= product;
            for head in 0..self.heads {
                let column = (shift - 1) * self.heads + head;
                rows.push(rolling % self.primes[column] + self.offsets[column]);
            }
        }
        rows
    }
}

/// Where a token's rows are, for the kernel that reads them.
pub struct Found {
    pub rows: DevicePtr,
    pub scales: DevicePtr,
    pub row_slots: DevicePtr,
    pub scale_slots: DevicePtr,
}

/// The value rows and their scale rows, cached apart: they are two tensors
/// of the checkpoint.
struct Caches {
    rows: NgramRowCache,
    scales: NgramRowCache,
}

pub struct EngramTable {
    pub hash: EngramHash,
    caches: Mutex<Caches>,
    /// `[2, columns]` u32: the value rows' slots, then the scale rows'.
    slots: DevicePtr,
}

impl EngramTable {
    /// The table of decoder layer `layer`, which the loader left on disk.
    pub fn open(
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        layer: usize,
        hash: EngramHash,
        rows_total: u64,
    ) -> Result<Self> {
        let open = |part: &str, dtype: WeightDtype, width: usize| -> Result<NgramRowCache> {
            let name = format!("layers.{layer}.engram.embed.{part}");
            let d = store
                .deferred(&name)
                .with_context(|| format!("{name}: not left on disk by the loader"))?;
            ensure!(
                d.dtype == dtype && d.shape == [rows_total as usize, width],
                "{name}: {:?} {:?}, expected {dtype:?} [{rows_total}, {width}]",
                d.dtype,
                d.shape
            );
            NgramRowCache::open_at(&d.path, d.offset, None, rows_total, width, CACHED_ROWS)
                .with_context(|| name)
        };
        let caches = Caches {
            rows: open("weight", WeightDtype::FP8E4M3, ENGRAM_DIM)?,
            scales: open(
                "scale",
                WeightDtype::FP8E8M0,
                ENGRAM_DIM / ENGRAM_SCALE_GROUP,
            )?,
        };
        Ok(Self {
            slots: gpu.alloc(2 * hash.columns() * 4)?,
            hash,
            caches: Mutex::new(caches),
        })
    }

    /// Bring `rows` into memory and say where they are. They stay there
    /// until [`Self::done`], which the caller calls once the kernels that
    /// read them have run.
    pub fn find(&self, gpu: &dyn GpuBackend, rows: &[u64], stream: u64) -> Result<Found> {
        let mut caches = self.caches.lock().expect("engram table");
        let (mut row_slots, mut scale_slots) = (Vec::new(), Vec::new());
        caches.rows.resolve(rows, &mut row_slots)?;
        caches.scales.resolve(rows, &mut scale_slots)?;
        let bytes: Vec<u8> = row_slots
            .iter()
            .chain(&scale_slots)
            .flat_map(|slot| slot.to_le_bytes())
            .collect();
        // The last token's lookup may still be reading the slots.
        gpu.synchronize(stream)?;
        gpu.copy_h2d(&bytes, self.slots)?;
        Ok(Found {
            rows: DevicePtr(caches.rows.table_dev_va()?),
            scales: DevicePtr(caches.scales.table_dev_va()?),
            row_slots: self.slots,
            scale_slots: self.slots.offset(rows.len() * 4),
        })
    }

    /// Let the rows of the last [`Self::find`] be evicted again.
    pub fn done(&self) {
        let mut caches = self.caches.lock().expect("engram table");
        caches.rows.end_batch();
        caches.scales.end_batch();
    }
}

#[cfg(test)]
#[path = "engram_table_tests.rs"]
mod tests;
