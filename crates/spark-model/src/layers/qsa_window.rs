// SPDX-License-Identifier: AGPL-3.0-only

//! The raw indexer keys as a sliding window. Split from `qsa.rs` for the
//! 500-LoC cap.
//!
//! A raw key is read exactly once more after it is written: when its 4-token
//! block completes and `qsa_block_pool` folds it into a pooled block key.
//! Scoring, selection and the Marconi aux carry only ever need the POOLED
//! keys plus the unpooled tail. Keeping every raw key for the life of the
//! sequence cost 256 B per token per QSA layer (3 KiB per token across the
//! 12 layers, a quarter of the token's KV on a TP2 rank) to hold bytes
//! nothing reads; the window keeps the unpooled tail plus a rewind margin.
//!
//! The window is two buffers of `cap` positions. Sliding copies the kept
//! tail into the other buffer and switches, so the copy never overlaps its
//! source and no buffer is freed while the stream may still read it; only
//! growth (a slab wider than the window) reallocates, after a sync.
//!
//! Positions `[base, base + cap)` live at `bufs[cur]`. The pooling kernel
//! indexes raw keys by absolute position, so it is handed
//! [`QsaIndexer::raw_origin`]: the window pointer moved back by `base` rows.
//! It dereferences only positions `>= pooled * ratio >= base`, which are in
//! the window, so every byte it reads is the byte it read before.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{QsaIndexer, QsaSeqState};

/// Positions per window buffer (and the growth granule). Two buffers of
/// this at hd 128 cost 2 MiB per layer and sequence.
const RAW_WINDOW: usize = 4096;
/// Raw keys kept below the pooled frontier when the window slides. A
/// speculative verify ingests up to `VERIFY_ROW_CAP` rows and may then
/// rewind all but one; a block pooled during those rows is un-pooled and
/// re-pooled from its raw keys later, so they must survive a slide taken
/// mid-verify. 512 covers 128 verify rows plus a block with room to spare.
pub(super) const REWIND_MARGIN: usize = 512;

/// See the module doc.
#[derive(Debug)]
pub struct RawWindow {
    pub(super) bufs: [DevicePtr; 2],
    pub(super) cur: usize,
    pub(super) base: usize,
    pub(super) cap: usize,
}

impl RawWindow {
    pub(super) const EMPTY: Self = Self {
        bufs: [DevicePtr(0), DevicePtr(0)],
        cur: 0,
        base: 0,
        cap: 0,
    };

    /// First position the window holds.
    pub(super) fn base(&self) -> usize {
        self.base
    }

    /// Restart the window at `pos` with nothing kept (a reset or a restore).
    pub(super) fn restart_at(&mut self, pos: usize) {
        self.base = pos;
    }
}

impl QsaIndexer {
    /// Device address of the raw key at absolute position `pos`.
    pub(super) fn raw_slot(&self, st: &QsaSeqState, pos: usize) -> DevicePtr {
        debug_assert!(st.raw.base <= pos && pos < st.raw.base + st.raw.cap);
        st.raw.bufs[st.raw.cur].offset((pos - st.raw.base) * self.hd as usize * 2)
    }

    /// The window pointer moved back by `base` rows, for kernels that index
    /// raw keys by absolute position (module doc).
    pub(super) fn raw_origin(&self, st: &QsaSeqState) -> DevicePtr {
        let back = (st.raw.base * self.hd as usize * 2) as u64;
        DevicePtr(st.raw.bufs[st.raw.cur].0.wrapping_sub(back))
    }

    /// Make room to write the next `n` raw keys (positions `ingested..`).
    /// Keeps `[pooled * ratio - REWIND_MARGIN, ingested)`, block-aligned.
    pub(super) fn raw_room(
        &self,
        st: &mut QsaSeqState,
        n: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let end = st.ingested + n;
        if st.raw.cap > 0 && end <= st.raw.base + st.raw.cap {
            return Ok(());
        }
        let ratio = self.ratio as usize;
        let keep_from = if st.raw.cap == 0 {
            st.ingested // nothing held yet
        } else {
            ((st.pooled * ratio).saturating_sub(REWIND_MARGIN) / ratio * ratio)
                .max(st.raw.base)
                .min(st.ingested)
        };
        let keep = st.ingested - keep_from;
        let row = self.hd as usize * 2;
        let from = if keep > 0 {
            self.raw_slot(st, keep_from)
        } else {
            DevicePtr(0)
        };
        if st.raw.cap == 0 || keep + n > st.raw.cap {
            anyhow::ensure!(
                !gpu.stream_is_capturing(stream),
                "QSA: raw-key window growth inside a graph capture"
            );
            let cap = (keep + n).next_multiple_of(RAW_WINDOW).max(RAW_WINDOW);
            let a = gpu.alloc(cap * row)?;
            let b = match gpu.alloc(cap * row) {
                Ok(p) => p,
                Err(e) => {
                    let _ = gpu.free(a);
                    return Err(e);
                }
            };
            if keep > 0 {
                gpu.copy_d2d_async(from, a, keep * row, stream)?;
            }
            if st.raw.cap > 0 {
                gpu.synchronize(stream)?;
                self.free_raw(st, gpu)?;
            }
            st.raw = RawWindow {
                bufs: [a, b],
                cur: 0,
                base: keep_from,
                cap,
            };
        } else {
            let next = 1 - st.raw.cur;
            if keep > 0 {
                gpu.copy_d2d_async(from, st.raw.bufs[next], keep * row, stream)?;
            }
            st.raw.cur = next;
            st.raw.base = keep_from;
        }
        Ok(())
    }

    /// Release both window buffers; the window is empty afterwards.
    pub(super) fn free_raw(&self, st: &mut QsaSeqState, gpu: &dyn GpuBackend) -> Result<()> {
        for p in &mut st.raw.bufs {
            if p.0 != 0 {
                gpu.free(*p)?;
                *p = DevicePtr(0);
            }
        }
        st.raw = RawWindow::EMPTY;
        Ok(())
    }
}

#[cfg(test)]
#[path = "qsa_window_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "qsa_window_gpu_tests.rs"]
mod gpu_tests;
