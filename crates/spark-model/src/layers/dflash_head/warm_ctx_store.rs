// SPDX-License-Identifier: AGPL-3.0-only

//! The `ATLAS_DFLASH_WARM_CTX=restore` store (`warm_ctx` module docs).
//!
//! The slab holds `slots` windows of `window_rows` accumulator rows. A
//! prefill writes ONE window: its first checkpoint save copies the rows that
//! end there, each later checkpoint of the same prefill (the tail cut, the
//! tail block, the prompt leaf) appends only the rows past the previous one,
//! and every checkpoint registers a key that names the sub-range
//! `[window.lo, end)` it covers. So a slot is a conversation turn, not a
//! checkpoint. A window is evicted least recently used, with its keys; the
//! window a prefill restored from is demoted to the next victim once that
//! prefill has stored its own (the new window covers the conversation's next
//! turn; the old one only serves a branch back to an earlier turn).
//!
//! Keys are content keys: SHA-256 of the adapter and the prefix-cache key
//! tokens `[0, end)`. The rows at positions under `end` are a function of
//! exactly those tokens, as the KV radix's blocks are. The scheduler's
//! `session_hash` is not part of the key: it is derived from the prompt's
//! first 1024 tokens (`session_manager::compute_session_hash`), not a tenant
//! salt, and only gates restores of TAIL SSM snapshots, whose state bleeds
//! past the exact prefix; these rows never do.
//!
//! The slab is allocated by `build_model` before the KV pool is sized
//! ([`WarmCtxStore::for_build`]), so its bytes come out of the KV budget
//! instead of unified memory the budget already promised. Every copy into
//! or out of a window waits on the window's event and records it, so a
//! save and a restore on different streams are ordered on every path.

use std::collections::HashMap;

use anyhow::Result;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::warm_ctx::{WarmCtx, store_slots_from_env};

/// Content key of a checkpoint: the token prefix `[0, end)` it covers (the
/// prefix cache's key tokens) and the adapter. A collision-free digest, so
/// equal keys mean equal prefixes.
pub(crate) type StoreKey = [u8; 32];

pub(crate) fn store_key(adapter_id: u64, key_tokens: &[u32]) -> StoreKey {
    let mut h = Sha256::new();
    h.update(b"atlas/dflash-warm-ctx/v1");
    h.update(adapter_id.to_le_bytes());
    h.update((key_tokens.len() as u64).to_le_bytes());
    let mut buf = [0u8; 4096];
    for chunk in key_tokens.chunks(buf.len() / 4) {
        for (dst, t) in buf.chunks_exact_mut(4).zip(chunk) {
            dst.copy_from_slice(&t.to_le_bytes());
        }
        h.update(&buf[..chunk.len() * 4]);
    }
    h.finalize().into()
}

/// Bytes of a slab of `slots` windows: the startup reservation.
pub fn store_bytes(slots: usize, window_rows: usize, row_bytes: usize) -> Option<usize> {
    slots.checked_mul(window_rows)?.checked_mul(row_bytes)
}

/// A window a prefill wrote or restored from; stale once its slot is reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WindowRef {
    slot: usize,
    generation: u64,
}

/// A sequence's store handles for its current prefill.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StoreRefs {
    /// The window this prefill writes (later checkpoints extend it).
    pub own: Option<WindowRef>,
    /// The window it restored from (demoted once `own` exists).
    pub src: Option<WindowRef>,
}

#[derive(Debug, Clone, Copy)]
struct Window {
    /// Positions `[lo, hi)`; row 0 of the slot is position `lo`.
    lo: usize,
    hi: usize,
    generation: u64,
    used: u64,
}

#[derive(Debug, Clone, Copy)]
struct KeyRef {
    win: WindowRef,
    end: usize,
}

#[derive(Default)]
struct StoreInner {
    slab: Option<DevicePtr>,
    row_bytes: usize,
    window_rows: usize,
    windows: Vec<Option<Window>>,
    /// One event per slot (0 = the backend has none: copies sync instead).
    events: Vec<u64>,
    keys: HashMap<StoreKey, KeyRef>,
    tick: u64,
    generation: u64,
}

impl StoreInner {
    fn window(&self, r: WindowRef) -> Option<Window> {
        self.windows
            .get(r.slot)
            .copied()
            .flatten()
            .filter(|w| w.generation == r.generation)
    }

    /// The live key `key`: its window and end.
    fn key(&self, key: &StoreKey) -> Option<(WindowRef, Window, usize)> {
        let k = self.keys.get(key)?;
        Some((k.win, self.window(k.win)?, k.end))
    }

    fn touch(&mut self, slot: usize) {
        self.tick += 1;
        if let Some(w) = self.windows[slot].as_mut() {
            w.used = self.tick;
        }
    }

    /// A free slot, else the window `prefer` names (the one this prefill
    /// restored from: its conversation moves on to the new window), else the
    /// least recently used one; emptied with its keys.
    fn take_slot(&mut self, prefer: Option<WindowRef>) -> usize {
        let prefer = prefer.filter(|&r| self.window(r).is_some()).map(|r| r.slot);
        let slot = self
            .windows
            .iter()
            .position(Option::is_none)
            .or(prefer)
            .unwrap_or_else(|| {
                (0..self.windows.len())
                    .min_by_key(|&i| self.windows[i].map_or(0, |w| w.used))
                    .unwrap_or(0)
            });
        self.windows[slot] = None;
        self.keys.retain(|_, k| k.win.slot != slot);
        slot
    }

    /// `rows` rows from `src` into `slot` from its row `row0`, ordered after
    /// every earlier copy touching the slot, on any stream.
    fn copy_in(
        &self,
        gpu: &dyn GpuBackend,
        slot: usize,
        row0: usize,
        src: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let dst = self.slot_row(slot, row0);
        self.ordered(gpu, slot, stream, || {
            gpu.copy_d2d_async(src, dst, rows * self.row_bytes, stream)
        })
    }

    fn slot_row(&self, slot: usize, row: usize) -> DevicePtr {
        let slab = self.slab.expect("a live window has a slab");
        slab.offset((slot * self.window_rows + row) * self.row_bytes)
    }

    fn ordered(
        &self,
        gpu: &dyn GpuBackend,
        slot: usize,
        stream: u64,
        copy: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let event = self.events[slot];
        if event != 0 {
            gpu.stream_wait_event(stream, event)?;
        }
        copy()?;
        if event != 0 {
            gpu.record_event(event, stream)
        } else {
            gpu.synchronize(stream)
        }
    }
}

/// Device-resident drafter windows keyed by prefix content (module docs).
pub struct WarmCtxStore {
    inner: Mutex<StoreInner>,
}

impl WarmCtxStore {
    /// No store (every mode but `restore`, worker ranks, non-DFlash models).
    pub fn disabled() -> Self {
        Self {
            inner: Mutex::new(StoreInner::default()),
        }
    }

    /// A slab of `slots` windows, allocated now.
    pub fn reserve(
        gpu: &dyn GpuBackend,
        slots: usize,
        window_rows: usize,
        row_bytes: usize,
    ) -> Result<Self> {
        let bytes = store_bytes(slots, window_rows, row_bytes)
            .ok_or_else(|| anyhow::anyhow!("DFlash warm ctx store size overflows"))?;
        if bytes == 0 {
            return Ok(Self::disabled());
        }
        let slab = gpu.alloc(bytes)?;
        let events = (0..slots)
            .map(|_| gpu.create_event())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner: Mutex::new(StoreInner {
                slab: Some(slab),
                row_bytes,
                window_rows,
                windows: vec![None; slots],
                events,
                ..StoreInner::default()
            }),
        })
    }

    /// `build_model`, before the KV pool is sized: the slab when this rank
    /// owns the drafter context and `ATLAS_DFLASH_WARM_CTX=restore` (only
    /// then is `ATLAS_DFLASH_WARM_CTX_SLOTS` read); otherwise nothing.
    pub fn for_build(
        gpu: &dyn GpuBackend,
        capture_layers: usize,
        hidden_size: usize,
        ctx_window: Option<usize>,
        rank0: bool,
    ) -> Result<Self> {
        if capture_layers == 0 || !rank0 || !WarmCtx::from_env()?.restores() {
            return Ok(Self::disabled());
        }
        let slots = store_slots_from_env()?;
        let (window_rows, row_bytes) = (ctx_window.unwrap_or(0), capture_layers * hidden_size * 2);
        let mib = store_bytes(slots, window_rows, row_bytes).unwrap_or(usize::MAX) >> 20;
        match Self::reserve(gpu, slots, window_rows, row_bytes) {
            Ok(store) => {
                tracing::info!(
                    "Reserved before KV sizing: DFlash warm ctx store {mib} MiB \
                     ({slots} windows x {window_rows} rows x {row_bytes} B, \
                     ATLAS_DFLASH_WARM_CTX=restore)"
                );
                Ok(store)
            }
            Err(error) => {
                tracing::warn!(
                    "DFlash warm ctx: reserving the {mib} MiB store failed ({error}); \
                     warm turns fall back to the seeded window"
                );
                Ok(Self::disabled())
            }
        }
    }

    /// Windows currently held (tests, logs).
    pub fn len(&self) -> usize {
        self.inner.lock().windows.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Release the slab and the events (model teardown).
    pub fn free(&self, gpu: &dyn GpuBackend) {
        let mut inner = self.inner.lock();
        for event in inner.events.drain(..).filter(|&e| e != 0) {
            let _ = gpu.destroy_event(event);
        }
        if let Some(slab) = inner.slab.take()
            && let Err(error) = gpu.free(slab)
        {
            tracing::warn!("DFlash warm ctx: freeing the store slab failed ({error})");
        }
        inner.windows.clear();
        inner.keys.clear();
    }

    /// Store the checkpoint `key` = rows `[lo, end)`, read from `src_lo`
    /// (the accumulator row of position `lo`). `refs.own` is the window this
    /// prefill already wrote: when it starts at `lo` and ends at or under
    /// `end` only the new rows are copied into it. A new window takes a free
    /// slot, else `refs.src`'s, else the least recently used. Returns the
    /// window that now holds the rows (`refs.own` unchanged when the key
    /// already names a window with at least these rows, `None` when nothing
    /// was stored).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn save(
        &self,
        gpu: &dyn GpuBackend,
        key: StoreKey,
        refs: StoreRefs,
        src_lo: DevicePtr,
        lo: usize,
        end: usize,
        row_bytes: usize,
        stream: u64,
    ) -> Result<Option<WindowRef>> {
        let mut inner = self.inner.lock();
        if inner.slab.is_none()
            || row_bytes != inner.row_bytes
            || lo >= end
            || end - lo > inner.window_rows
        {
            return Ok(None);
        }
        // An equal key keeps the window with more rows (the smaller lo): a
        // seed-fallback turn must not shrink a full entry.
        if let Some((win, w, _)) = inner.key(&key)
            && w.lo <= lo
        {
            inner.touch(win.slot);
            return Ok(refs.own);
        }
        let win = match refs.own.and_then(|r| inner.window(r).map(|w| (r, w))) {
            Some((r, w)) if w.lo == lo && w.hi <= end => {
                let skip = w.hi - lo;
                if end > w.hi {
                    let src = src_lo.offset(skip * row_bytes);
                    inner.copy_in(gpu, r.slot, skip, src, end - w.hi, stream)?;
                }
                if let Some(w) = inner.windows[r.slot].as_mut() {
                    w.hi = end;
                }
                r
            }
            _ => {
                let slot = inner.take_slot(refs.src);
                inner.generation += 1;
                let r = WindowRef {
                    slot,
                    generation: inner.generation,
                };
                inner.copy_in(gpu, slot, 0, src_lo, end - lo, stream)?;
                inner.windows[slot] = Some(Window {
                    lo,
                    hi: end,
                    generation: r.generation,
                    used: 0,
                });
                r
            }
        };
        inner.keys.insert(key, KeyRef { win, end });
        inner.touch(win.slot);
        Ok(Some(win))
    }

    /// Copy the rows `[max(lo, want_lo), end)` of the checkpoint `key` (which
    /// must end at `end`) to `dst`. Returns that first position (`end` when
    /// the window holds nothing at or above `want_lo`) and the window, or
    /// `None` on a miss.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn restore(
        &self,
        gpu: &dyn GpuBackend,
        key: StoreKey,
        end: usize,
        want_lo: usize,
        row_bytes: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<Option<(usize, WindowRef)>> {
        let mut inner = self.inner.lock();
        let Some((win, w, key_end)) = inner.key(&key).filter(|_| inner.row_bytes == row_bytes)
        else {
            return Ok(None);
        };
        if key_end != end {
            return Ok(None);
        }
        inner.touch(win.slot);
        let lo = w.lo.max(want_lo).min(end);
        if lo < end {
            let src = inner.slot_row(win.slot, lo - w.lo);
            inner.ordered(gpu, win.slot, stream, || {
                gpu.copy_d2d_async(src, dst, (end - lo) * row_bytes, stream)
            })?;
        }
        Ok(Some((lo, win)))
    }

    /// Make `r` the next eviction victim (its keys stay valid until then).
    pub(crate) fn demote(&self, r: WindowRef) {
        let mut inner = self.inner.lock();
        if inner.window(r).is_some()
            && let Some(w) = inner.windows[r.slot].as_mut()
        {
            w.used = 0;
        }
    }
}
