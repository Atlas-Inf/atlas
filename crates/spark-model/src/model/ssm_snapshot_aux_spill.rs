// SPDX-License-Identifier: AGPL-3.0-only

//! A spilled snapshot's AUX layer state: the host blobs a slot carries beside
//! its recurrent state (`SsmSnapshotPool::aux_blobs` — on qwen4_exp the PLE
//! token history and conv carry and the QSA indexer keys).
//!
//! The restore gate declines a slot without aux on a model whose layers hold
//! aux state (`requires_aux_state`), so a snapshot spilled without it could
//! never be restored from the tier: every fault-in was followed by a full
//! recompute. The aux now goes to the tier with the state and comes back with
//! it.
//!
//! The aux is variable-length (768 B a token of QSA keys on Qwen3.8-Flash-
//! Next, plus the PLE history) and every store is fixed-size, so it is framed
//! `[n u32] n × [layer u32][len u64][bytes]` (little-endian) and written as
//! blob-sized chunks under keys derived from the snapshot's ([`aux_key`]),
//! through the same staging buffer as the state blob. A small host map records
//! how many chunks each spilled key has; a key without an entry spilled no aux
//! and restores none, exactly as before (GLM, Holo).
//!
//! A chunk the tier no longer holds, or a frame that does not decode, is a
//! MISS for the whole snapshot (the caller retires the key), never a restore
//! with part of the aux.

use anyhow::Result;

use super::ssm_snapshot::SsmSnapshotPool;
use super::ssm_tier::SnapshotBlobStore;

/// Salts the derived keys away from snapshot prefix hashes.
const AUX_SALT: u64 = 0x5353_4d41_5558_5350; // "SSMAUXSP"

/// What a spilled key's aux occupies in the tier.
#[derive(Clone, Copy, Debug)]
pub(super) struct AuxRecord {
    chunks: usize,
    bytes: usize,
}

/// The tier key of chunk `chunk` of `key`'s aux.
pub(super) fn aux_key(key: u64, chunk: usize) -> u64 {
    atlas_tier::hash::mix64(key ^ AUX_SALT, chunk as u64)
}

impl SsmSnapshotPool {
    /// Put `snap_slot`'s aux under `key` (whose state blob the caller has
    /// put), through `staging`. `false` when the tier refused a chunk: what
    /// was written is removed and the caller treats the spill as refused.
    pub(super) fn spill_aux(
        &self,
        snap_slot: usize,
        key: u64,
        store: &dyn SnapshotBlobStore,
        staging: &mut [u8],
    ) -> Result<bool> {
        let prev = self.spilled_aux.lock().remove(&key).map_or(0, |r| r.chunks);
        let mut written = 0;
        // `None`: the slot has no aux; `Some(None)`: a chunk was refused.
        let frame = self
            .with_aux(snap_slot, |blobs| {
                write_frame(blobs, staging, |chunk| {
                    let ok = store.put(aux_key(key, written), chunk)?;
                    written += usize::from(ok);
                    Ok(ok)
                })
            })
            .transpose();
        // Chunks of an earlier spill of this key, and of a failed one, go.
        let keep = if let Ok(Some(Some(_))) = frame {
            written
        } else {
            0
        };
        for chunk in keep..prev.max(written) {
            store.remove(aux_key(key, chunk));
        }
        Ok(match frame? {
            None => true,
            Some(None) => false,
            Some(Some(bytes)) => {
                let rec = AuxRecord {
                    chunks: written,
                    bytes,
                };
                self.spilled_aux.lock().insert(key, rec);
                true
            }
        })
    }

    /// Read `key`'s aux back into `snap_slot` through `staging`. `true` when
    /// the key spilled no aux (nothing to restore) or its aux is restored;
    /// `false` is a miss.
    pub(super) fn fault_in_aux(
        &self,
        snap_slot: usize,
        key: u64,
        store: &dyn SnapshotBlobStore,
        staging: &mut [u8],
    ) -> Result<bool> {
        let Some(rec) = self.spilled_aux.lock().get(&key).copied() else {
            return Ok(true);
        };
        let mut reader = ChunkReader {
            store,
            key,
            staging,
            chunk: 0,
            pos: 0,
            len: 0,
            left: rec.bytes,
        };
        let Some(blobs) = reader.frame()? else {
            return Ok(false);
        };
        if reader.chunk != rec.chunks || reader.left != 0 {
            return Ok(false);
        }
        self.set_aux(snap_slot, blobs);
        Ok(true)
    }

    /// Remove `key`'s state blob and aux chunks from the tier.
    pub(super) fn remove_spilled(&self, store: &dyn SnapshotBlobStore, key: u64) {
        store.remove(key);
        if let Some(rec) = self.spilled_aux.lock().remove(&key) {
            for chunk in 0..rec.chunks {
                store.remove(aux_key(key, chunk));
            }
        }
    }
}

/// Write `blobs`' frame through `chunk`-sized `put`s. `Some(bytes)` is the
/// frame length, `None` a refused put (writing stops there). The last chunk's
/// unused tail is zeroed so no other snapshot's bytes reach the tier.
fn write_frame(
    blobs: &[(u32, Vec<u8>)],
    chunk: &mut [u8],
    mut put: impl FnMut(&[u8]) -> Result<bool>,
) -> Result<Option<usize>> {
    let mut pos = 0;
    let mut total = 0;
    let mut push = |mut bytes: &[u8], chunk: &mut [u8]| -> Result<bool> {
        while !bytes.is_empty() {
            let n = bytes.len().min(chunk.len() - pos);
            chunk[pos..pos + n].copy_from_slice(&bytes[..n]);
            (pos, total, bytes) = (pos + n, total + n, &bytes[n..]);
            if pos == chunk.len() {
                pos = 0;
                if !put(chunk)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    };
    let n = u32::try_from(blobs.len())?;
    if !push(&n.to_le_bytes(), chunk)? {
        return Ok(None);
    }
    for (layer, bytes) in blobs {
        let len = u64::try_from(bytes.len())?;
        for part in [&layer.to_le_bytes()[..], &len.to_le_bytes()[..], bytes] {
            if !push(part, chunk)? {
                return Ok(None);
            }
        }
    }
    if pos > 0 {
        chunk[pos..].fill(0);
        if !put(chunk)? {
            return Ok(None);
        }
    }
    Ok(Some(total))
}

/// Reads a frame back chunk by chunk; `None` from any read is a miss.
struct ChunkReader<'a> {
    store: &'a dyn SnapshotBlobStore,
    key: u64,
    staging: &'a mut [u8],
    /// Chunks fetched so far.
    chunk: usize,
    pos: usize,
    len: usize,
    /// Frame bytes not yet read.
    left: usize,
}

impl ChunkReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> Result<Option<()>> {
        if out.len() > self.left {
            return Ok(None);
        }
        let mut done = 0;
        while done < out.len() {
            if self.pos == self.len {
                if !self
                    .store
                    .get(aux_key(self.key, self.chunk), self.staging)?
                {
                    return Ok(None);
                }
                (self.chunk, self.pos, self.len) = (self.chunk + 1, 0, self.staging.len());
            }
            let n = (out.len() - done).min(self.len - self.pos);
            out[done..done + n].copy_from_slice(&self.staging[self.pos..self.pos + n]);
            (done, self.pos) = (done + n, self.pos + n);
        }
        self.left -= out.len();
        Ok(Some(()))
    }

    fn u32(&mut self) -> Result<Option<u32>> {
        let mut b = [0u8; 4];
        Ok(self.read(&mut b)?.map(|()| u32::from_le_bytes(b)))
    }

    fn frame(&mut self) -> Result<Option<Vec<(u32, Vec<u8>)>>> {
        let Some(n) = self.u32()? else {
            return Ok(None);
        };
        let mut blobs = Vec::with_capacity((n as usize).min(self.left / 12));
        for _ in 0..n {
            let Some(layer) = self.u32()? else {
                return Ok(None);
            };
            let mut len = [0u8; 8];
            if self.read(&mut len)?.is_none() {
                return Ok(None);
            }
            // Bounded by the frame, so a corrupt length cannot allocate more.
            let Some(len) = usize::try_from(u64::from_le_bytes(len))
                .ok()
                .filter(|&l| l <= self.left)
            else {
                return Ok(None);
            };
            let mut bytes = vec![0u8; len];
            if self.read(&mut bytes)?.is_none() {
                return Ok(None);
            }
            blobs.push((layer, bytes));
        }
        Ok(Some(blobs))
    }
}

#[cfg(test)]
#[path = "ssm_snapshot_aux_spill_tests.rs"]
mod tests;
