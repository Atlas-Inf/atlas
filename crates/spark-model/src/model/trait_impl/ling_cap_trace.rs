// SPDX-License-Identifier: AGPL-3.0-only

//! Captured-mode per-layer fingerprints for the Ling family's decode and
//! K=2 verify graphs (`ATLAS_LING_CAP_TRACE=1`) — the captured-phase
//! counterpart of the eager-only `ATLAS_LIGHTNING_VERIFY_LAYER_TRACE`.
//! That lever forces its steps eagerly, so it cannot see inside a replayed
//! graph; this one is the opposite pole. The lever answers: at the FIRST
//! step and LAYER where two replays of the same retained graph diverge,
//! which layer's post-layer hidden is the first to differ?
//!
//! How: one d2d shadow copy per layer runs INSIDE the capture
//! (capture-legal; the copies bake into the graph and replay with the
//! step) into a per-serve slab keyed by layer. AFTER the step — outside
//! the graph — the flush syncs, copies the slab (and the step's logits
//! rows) out, FNV-1a's each layer's shadowed rows, and prints one
//! `tracing::warn` line. EAGER steps shadow identically, so the two
//! `kind`s diff against each other too. Offline, two runs' lines join on
//! (kind, plen, slen); the first mismatching layer hash localizes the
//! divergence, the bt/slot fields carry the expected-identical request
//! fingerprint. With the flag off every entry point is one boolean check
//! and the slab is never allocated.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// FNV-1a over the shadowed bytes.
#[allow(clippy::missing_docs_in_private_items)]
pub(in crate::model) fn ling_cap_fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

impl TransformerModel {
    fn on(&self) -> bool {
        self.config.model_type == "bailing_moe_v3_vl" && crate::model::graph_flags::ling_cap_trace()
    }

    /// The per-serve slab, allocated once on first use. Layout: per layer
    /// `2 * hidden * 2` bytes (the K=2 verify worst case); the decode
    /// shadows row 0 of each layer and its flush reads only that row.
    fn slab_ptr(&self) -> Result<DevicePtr> {
        if let Some(p) = self.ling_cap_trace_slab.get() {
            return Ok(*p);
        }
        let bytes = self.layers.len() * 2 * self.config.hidden_size * 2;
        let p = self.gpu.alloc(bytes)?;
        let _ = self.ling_cap_trace_slab.set(p);
        Ok(p)
    }

    /// Shadow `rows` rows of `hidden` into layer `layer`'s slab slot. Runs
    /// inside the capture window on captured steps (the copies bake into
    /// the graph) and eagerly on eager steps.
    pub(super) fn ling_cap_trace_capture(
        &self,
        layer: usize,
        hidden: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.on() {
            return Ok(());
        }
        let row_bytes = self.config.hidden_size * 2;
        let slab = self.slab_ptr()?;
        self.gpu.copy_d2d_async(
            hidden,
            slab.offset(layer * 2 * row_bytes),
            rows * row_bytes,
            stream,
        )
    }

    /// Post-step flush: one warn line with the per-layer fingerprints,
    /// the block-table fingerprint, and (when `logits_row_bytes` is set)
    /// the step's logits rows.
    pub(super) fn ling_cap_trace_flush(
        &self,
        kind: &'static str,
        seq: &SequenceState,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.on() {
            return Ok(());
        }
        let row_bytes = self.config.hidden_size * 2;
        let layers = self.layers.len();
        let slab = self.slab_ptr()?;
        self.gpu.synchronize(stream)?;
        let mut host = vec![0u8; layers * 2 * row_bytes];
        self.gpu.copy_d2h(slab, &mut host)?;
        let mut hs = String::with_capacity(layers * 18);
        for l in 0..layers {
            if l > 0 {
                hs.push(',');
            }
            let off = l * 2 * row_bytes;
            hs.push_str(&format!(
                "{:016x}",
                ling_cap_fnv1a(&host[off..off + rows * row_bytes])
            ));
        }
        let lb = rows * self.config.vocab_size * 2;
        let mut logits_host = vec![0u8; lb];
        self.gpu.copy_d2h(self.buffers.logits(), &mut logits_host)?;
        let bt = seq.block_table.iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        });
        if kind == "ver" && rows == 2 {
            if let Some(l5) = self.layers.get(5) {
                l5.ling_cap_dump_meta(self.gpu.as_ref(), &seq.block_table, stream, "post-replay");
            }
        }
        tracing::warn!(
            "LING_CAP_TRACE kind={kind} plen={} slen={} slot={} rows={} bt={:016x} lg={:016x} hs=[{}]",
            seq.prompt_len,
            seq.seq_len,
            seq.slot_idx,
            rows,
            bt,
            ling_cap_fnv1a(&logits_host),
            hs
        );
        Ok(())
    }
}
