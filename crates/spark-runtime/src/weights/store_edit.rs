// SPDX-License-Identifier: AGPL-3.0-only

//! Post-load map edits on `WeightStore`: [`WeightStore::insert`] and
//! [`WeightStore::remove`], kept in a second impl block so `weights.rs` stays
//! under its LoC cap.
//!
//! The motivating use is EXL3 loading: each packed linear arrives as four
//! tensors (`<stem>.trellis/.suh/.svh/.mul1`); a later step builds a BF16
//! `<stem>.weight` on the GPU, `remove`s the packed four (freeing their
//! buffers) and `insert`s the replacement, so the existing loader code reads
//! the store unchanged.

use super::*;

impl WeightStore {
    /// Add a tensor the store did not have.
    ///
    /// Fails (and leaves the store unchanged) if `name` is already present:
    /// replacing a live tensor would drop its map entry without freeing the
    /// buffer — a leak the teardown `release` cannot see — so a caller that
    /// wants to swap a tensor must [`remove`](Self::remove) it first.
    pub fn insert(&mut self, name: String, tensor: WeightTensor) -> Result<()> {
        if let Some(existing) = self.weights.get(&name) {
            bail!(
                "Weight '{name}' is already in the store (ptr {}); \
                 remove it before inserting a replacement",
                existing.ptr
            );
        }
        self.weights.insert(name, tensor);
        Ok(())
    }

    /// Remove a tensor from the map and return it, transferring ownership of
    /// its buffer to the caller — who must free it (or keep it alive until
    /// something does), since `release` no longer sees it. Returns `None` if
    /// the name is absent.
    ///
    /// If the tensor had been [`reclaim`](Self::reclaim)ed, its memory is
    /// already gone: `remove` drops it from the `reclaimed` set as well (so
    /// the pointer cannot alias a future live tensor at teardown) and returns
    /// it with a **dead** pointer — the caller must NOT free it again. Ask
    /// [`was_reclaimed`](Self::was_reclaimed) before removing if you intend
    /// to free the buffer; a careful caller never double-frees.
    pub fn remove(&mut self, name: &str) -> Option<WeightTensor> {
        let tensor = self.weights.remove(name)?;
        // Best-effort: a poisoned set means the ptr is not recorded, so the
        // returned tensor is treated as live — the same trade `reclaim` makes
        // by failing loudly there.
        if let Ok(mut seen) = self.reclaimed.lock() {
            let _ = seen.remove(&tensor.ptr.0);
        }
        Some(tensor)
    }

    /// True if this name's tensor memory was already freed by
    /// [`reclaim`](Self::reclaim) (its map entry may still exist). Call this
    /// before [`remove`](Self::remove) to decide whether the returned buffer must be freed.
    pub fn was_reclaimed(&self, name: &str) -> bool {
        let Some(ptr) = self.weights.get(name).map(|t| t.ptr.0) else {
            return false;
        };
        self.reclaimed
            .lock()
            .map(|s| s.contains(&ptr))
            .unwrap_or(false)
    }
}
