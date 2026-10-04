// SPDX-License-Identifier: AGPL-3.0-only

//! R11 fix 7: MTP drafter context on a warm (prefix-restored) turn, made
//! identical to a cold turn.
//!
//! A cold turn captures every prompt row's final-layer hidden into
//! `mtp_prefill_hidden` and, at the end of prefill, builds the drafter KV over
//! the whole prompt in one `prefill_drafter` call. A warm turn used to adopt
//! the previous turn's drafter KV (carry), whose rows were made by decode, so
//! the drafts — and with them the accepted tokens — differed from a cold run.
//!
//! Here each grid SSM checkpoint at token `a` also stores the hidden rows
//! `[0, a)` (host RAM, `a * hidden * 2` bytes). A warm hit restores them into
//! `mtp_prefill_hidden`, claims the capture generation, and the replay passes
//! append rows from `a`, so the end-of-prefill consume runs the SAME
//! whole-prompt `prefill_drafter` a cold run does. A hit whose rows are not
//! stored is declined (full recompute): correct, slower.
//!
//! Memory: one entry per checkpoint slot, bounded by
//! `ATLAS_MTP_ANCHOR_MAX_MB` (default 4096; oldest evicted first).
//! `ATLAS_MTP_ANCHOR=0` restores the old carry behaviour.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

struct Entry {
    hash: u64,
    rows: usize,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct Store {
    map: HashMap<usize, Entry>,
    order: VecDeque<usize>,
    total: usize,
}

static STORE: std::sync::OnceLock<Mutex<Store>> = std::sync::OnceLock::new();

fn store() -> &'static Mutex<Store> {
    STORE.get_or_init(|| Mutex::new(Store::default()))
}

fn cap_bytes() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("ATLAS_MTP_ANCHOR_MAX_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4096)
            * 1024
            * 1024
    })
}

fn prefix_hash(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl Store {
    fn remove(&mut self, slot: usize) {
        if let Some(e) = self.map.remove(&slot) {
            self.total -= e.bytes.len();
        }
        self.order.retain(|&s| s != slot);
    }
}

impl TransformerModel {
    /// Whether warm turns must reproduce the cold drafter context exactly.
    pub(in crate::model) fn mtp_anchor_on(&self) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        !self.mtp_prefill_hidden.is_null()
            && self.proposer.is_some()
            && self.prefill_grid() > 0
            && !*OFF.get_or_init(|| std::env::var("ATLAS_MTP_ANCHOR").as_deref() == Ok("0"))
    }

    /// Save hidden rows `[0, rows)` for checkpoint slot `snap_slot`. Only when
    /// this sequence still owns a capture that covers them.
    pub(in crate::model) fn mtp_anchor_save(
        &self,
        seq: &SequenceState,
        tokens: &[u32],
        snap_slot: usize,
        rows: usize,
        stream: u64,
    ) {
        let mut st = store().lock();
        st.remove(snap_slot);
        if !self.mtp_anchor_on() {
            return;
        }
        let captured = self.mtp_prefill_capture_len.load(Ordering::Relaxed);
        let owns = seq.mtp_capture_gen != 0
            && seq.mtp_capture_gen == self.mtp_prefill_capture_gen.load(Ordering::Relaxed);
        if !owns || captured < rows || rows > tokens.len() {
            return;
        }
        let n = rows * self.config.hidden_size * 2;
        if n > cap_bytes() {
            return;
        }
        while st.total + n > cap_bytes() {
            let Some(old) = st.order.pop_front() else { break };
            if let Some(e) = st.map.remove(&old) {
                st.total -= e.bytes.len();
            }
        }
        let mut bytes = vec![0u8; n];
        if let Err(e) = self.gpu.copy_d2h_on_stream(self.mtp_prefill_hidden, &mut bytes, stream) {
            tracing::warn!("MTP anchor save failed (anchor stays drafter-less): {e:#}");
            return;
        }
        st.total += n;
        st.order.push_back(snap_slot);
        st.map.insert(snap_slot, Entry { hash: prefix_hash(&tokens[..rows]), rows, bytes });
    }

    /// True when a warm restore from `snap_slot` at `rows` can rebuild the
    /// cold drafter context (or when exactness is off).
    pub(in crate::model) fn mtp_anchor_available(&self, snap_slot: usize, tokens: &[u32], rows: usize) -> bool {
        if !self.mtp_anchor_on() {
            return true;
        }
        let st = store().lock();
        let ok = st
            .map
            .get(&snap_slot)
            .is_some_and(|e| e.rows == rows && rows <= tokens.len() && e.hash == prefix_hash(&tokens[..rows]));
        if !ok {
            tracing::info!(
                "MTP anchor: no drafter rows stored for snapshot {snap_slot} @ {rows} — \
                 declining the warm restore (full recompute keeps MTP output equal to cold)"
            );
        }
        ok
    }

    /// Restore the rows into the capture buffer and claim the capture for
    /// `seq`, so the replay passes append from `rows` and the end-of-prefill
    /// consume runs the cold whole-prompt `prefill_drafter`.
    pub(in crate::model) fn mtp_anchor_restore(
        &self,
        seq: &mut SequenceState,
        snap_slot: usize,
        stream: u64,
    ) -> anyhow::Result<()> {
        if !self.mtp_anchor_on() {
            return Ok(());
        }
        let st = store().lock();
        let Some(e) = st.map.get(&snap_slot) else {
            anyhow::bail!("MTP anchor vanished between check and restore");
        };
        self.gpu.copy_h2d_async(&e.bytes, self.mtp_prefill_hidden, stream)?;
        // The H2D reads host memory owned by the store; finish it before the
        // lock is released so a concurrent eviction cannot free it mid-copy.
        self.gpu.synchronize(stream)?;
        let generation = self.mtp_prefill_capture_gen.fetch_add(1, Ordering::Relaxed) + 1;
        seq.mtp_capture_gen = generation;
        self.mtp_prefill_capture_len.store(e.rows, Ordering::Relaxed);
        let _ = DevicePtr::NULL;
        Ok(())
    }
}
