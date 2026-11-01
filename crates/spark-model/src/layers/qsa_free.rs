// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence lifecycle of the QSA indexer carry: allocation, growth and
//! release, kept TOGETHER because the pair is the invariant that matters
//! (they drifted apart once already: alloc existed, free did not). Split from
//! `qsa.rs` for the 500-LoC cap. See `TransformerLayer::free_state` for why
//! release exists at all: `DevicePtr` has no `Drop` and the backend sweeps
//! only at process exit.
//!
//! The carry grows with the sequence instead of being sized for
//! `--max-seq-len` up front. At 262144 the up-front size was 80 MB per layer
//! and sequence (12 layers x 4 sequences = 3.8 GB) that no KV budget counted,
//! taken from the host's headroom the moment a request arrived. What grows
//! now is only the pooled block keys (64 B per token at hd 128, ratio 4);
//! raw keys live in a fixed window (`qsa_window.rs`). Growth copies the
//! pooled prefix into a larger buffer: the bytes every kernel reads are the
//! same, so the numerics are untouched.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::qsa_window::RawWindow;
use super::{QsaIndexer, QsaSeqState};

/// Growth granule in positions (a multiple of every pooling ratio in use).
const QSA_GRANULE: usize = 4096;

/// `ATLAS_QWEN4EXP_QSA_REUSE=1`: a released carry buffer goes to a per-thread
/// spare list and the next sequence's same-size request takes it, instead of
/// a `cuMemFree` at the end of one request and a `cuMemAlloc` at the start of
/// the next (36 of the ~40 allocations before a 16K prompt's first kernel,
/// ~3 ms of host time on the pair, nsys `sqpf-p3s7-r0`). A fresh allocation's
/// bytes are undefined anyway, and every reader stays inside what the carry
/// wrote, so the numerics cannot change; the buffers are only ever used on
/// the model stream, in order.
fn reuse_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_QSA_REUSE").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Spare buffers kept per thread by default (12 layers x 3 buffers, a
/// sequence and a bit).
const SPARE_MAX: usize = 64;

/// `ATLAS_QWEN4EXP_QSA_SPARE_MAX=N` (default [`SPARE_MAX`]): how many released
/// carry buffers a thread keeps. A sequence holds 3 per QSA layer (36 on
/// qwen4_exp), so 64 keeps under two sequences: a C8 wave frees 288, keeps
/// 64, and the next wave's admissions after the first two each pay 38
/// `cuMemAlloc` before their first kernel (~1.7 ms host with the GPU idle,
/// nsys `sqtp-Wprose-r0`) and the wave's retirements ~220 `cuMemFree`.
/// `max_batch x 36` keeps a whole wave. Only which allocation backs a
/// carry changes, never its bytes (see [`reuse_requested`]).
fn spare_max() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| parse_spare_max(std::env::var("ATLAS_QWEN4EXP_QSA_SPARE_MAX").ok()))
}

fn parse_spare_max(v: Option<String>) -> usize {
    v.and_then(|v| v.parse().ok()).unwrap_or(SPARE_MAX)
}

thread_local! {
    static SPARE: std::cell::RefCell<Vec<(DevicePtr, usize)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// A carry buffer of exactly `bytes`: a spare one, or a new allocation.
pub(super) fn carry_alloc(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    carry_alloc_via(gpu, bytes, reuse_requested())
}

/// Release a carry buffer of `bytes` (to the spares, or to the backend).
pub(super) fn carry_free(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<()> {
    carry_free_via(gpu, p, bytes, reuse_requested(), spare_max())
}

fn carry_alloc_via(gpu: &dyn GpuBackend, bytes: usize, reuse: bool) -> Result<DevicePtr> {
    let spare = reuse
        .then(|| {
            SPARE.with(|s| {
                let mut s = s.borrow_mut();
                let at = s.iter().position(|&(_, b)| b == bytes)?;
                Some(s.swap_remove(at).0)
            })
        })
        .flatten();
    spare.map_or_else(|| gpu.alloc(bytes), Ok)
}

fn carry_free_via(
    gpu: &dyn GpuBackend,
    p: DevicePtr,
    bytes: usize,
    reuse: bool,
    keep: usize,
) -> Result<()> {
    let kept = reuse
        && SPARE.with(|s| {
            let mut s = s.borrow_mut();
            (s.len() < keep).then(|| s.push((p, bytes))).is_some()
        });
    if kept { Ok(()) } else { gpu.free(p) }
}

impl QsaIndexer {
    /// Device bytes one token costs this layer's carry beyond the fixed raw
    /// window: its share of a pooled block key. What the KV budget charges
    /// per token.
    pub fn bytes_per_token(&self) -> usize {
        (self.hd as usize * 2).div_ceil(self.ratio as usize)
    }

    /// Bytes of a pooled-key buffer for `cap` positions.
    fn block_bytes(&self, cap: usize) -> usize {
        (cap / self.ratio as usize).max(1) * self.hd as usize * 2
    }

    /// The most positions a carry can hold (the served context).
    pub fn capacity(&self) -> usize {
        self.max_tokens
    }

    /// Free this sequence's buffers. Zeroed so a second teardown is a no-op.
    /// See `TransformerLayer::free_state`.
    pub fn free_seq_state(&self, st: &mut QsaSeqState, gpu: &dyn GpuBackend) -> Result<()> {
        if st.block_keys.0 != 0 {
            carry_free(gpu, st.block_keys, self.block_bytes(st.cap))?;
            st.block_keys = DevicePtr(0);
        }
        st.cap = 0;
        self.free_raw(st, gpu)
    }

    /// An empty carry; [`Self::reserve`] and the window give it room.
    pub fn new_seq_state(&self, _gpu: &dyn GpuBackend) -> Result<QsaSeqState> {
        Ok(QsaSeqState {
            ingested: 0,
            pooled: 0,
            table_len: 0,
            cap: 0,
            block_keys: DevicePtr(0),
            raw: RawWindow::EMPTY,
            share: super::qsa_decode_rows::DraftShare::Off,
        })
    }

    /// Make room for the pooled keys of positions `0..tokens` (and allocate
    /// the fixed raw-key window on first use). Grows by at least half again,
    /// in `QSA_GRANULE` steps, up to the served capacity, and keeps the
    /// pooled blocks. Never inside a graph capture: the buffer moves.
    pub fn reserve(
        &self,
        st: &mut QsaSeqState,
        tokens: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if st.raw.cap == 0 {
            self.raw_room(st, 0, gpu, stream)?;
        }
        if tokens <= st.cap {
            return Ok(());
        }
        anyhow::ensure!(
            tokens <= self.max_tokens,
            "QSA: {tokens} tokens exceeds the indexer capacity {} — it derives \
             from --max-seq-len (ATLAS_QSA_MAX_TOKENS overrides)",
            self.max_tokens
        );
        anyhow::ensure!(
            !gpu.stream_is_capturing(stream),
            "QSA: carry growth to {tokens} tokens inside a graph capture"
        );
        let cap = tokens
            .max(st.cap + st.cap / 2)
            .next_multiple_of(QSA_GRANULE)
            .min(self.max_tokens);
        let row = self.hd as usize * 2;
        let ratio = self.ratio as usize;
        let block = carry_alloc(gpu, self.block_bytes(cap))?;
        if st.cap > 0 {
            let pooled = st.pooled.min(st.cap / ratio);
            if pooled > 0 {
                gpu.copy_d2d_async(st.block_keys, block, pooled * row, stream)?;
            }
            // The copy (and any reader queued before it) must finish before
            // the old buffer can be released.
            gpu.synchronize(stream)?;
            carry_free(gpu, st.block_keys, self.block_bytes(st.cap))?;
        }
        st.block_keys = block;
        st.cap = cap;
        Ok(())
    }
}

#[cfg(test)]
#[path = "qsa_free_tests.rs"]
pub(super) mod tests;
