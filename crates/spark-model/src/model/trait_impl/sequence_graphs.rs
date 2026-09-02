// SPDX-License-Identifier: AGPL-3.0-only

//! Dropping a slot's captured CUDA graphs when the sequence whose
//! state they reference goes away. Split out of `sequence.rs` for
//! the 500-LoC cap; `free_sequence_dispatch` is the only caller.

use super::*;

impl TransformerModel {
    /// Drop every CUDA graph captured against `slot`.
    ///
    /// The decode and verify graph caches are keyed by POOL SLOT, and a slot
    /// is reused by the next sequence that lands on it. A captured graph bakes
    /// in the device addresses it was recorded with — including the ones that
    /// belong to the SEQUENCE rather than the slot: the PLE conv carry and the
    /// QSA indexer keys, both `gpu.alloc`ed per sequence and both released by
    /// `free_state` just above.
    ///
    /// While those allocations merely leaked, a stale graph replayed for the
    /// next sequence read the previous one's buffers — wrong, but memory that
    /// was still mapped. Now that they are actually freed, the same replay is
    /// a use-after-free: request 2 on a reused slot faulted with an illegal
    /// address (CUDA 700) inside the verify graph, which is sticky and takes
    /// the whole process down. So the graphs must die with the state they
    /// reference. The next sequence on this slot re-captures on its first
    /// step, which is what it would have done on a cold slot anyway.
    pub(super) fn invalidate_slot_graphs(&self, slot: usize) {
        let mut dead: Vec<spark_runtime::gpu::GraphHandle> = Vec::new();
        for cache in [
            &self.decode_graph,
            &self.verify2_graph,
            &self.verify3_graph,
            &self.verify4_graph,
        ] {
            if let Some(g) = cache.lock().remove(&slot) {
                dead.push(g);
            }
        }
        // Composite keys are `(slot, k)` — drop every width for this slot.
        self.verify_kgamma_graph.lock().retain(|k, g| {
            if k.0 == slot {
                dead.push(*g);
                false
            } else {
                true
            }
        });
        self.fused_graph.lock().retain(|k, g| {
            if k.0 == slot {
                dead.push(*g);
                false
            } else {
                true
            }
        });
        // Slot-VECTOR keyed: a batched graph spanning this slot is equally
        // stale, and its key is the whole batch's slot list.
        self.batch_decode_graphs.lock().0.retain(|k, (g, _)| {
            if k.contains(&(slot as u32)) {
                dead.push(*g);
                false
            } else {
                true
            }
        });
        for g in dead {
            if g.0 != 0
                && let Err(e) = self.gpu.destroy_graph(g)
            {
                tracing::warn!("free_sequence: destroy graph for slot {slot}: {e:#}");
            }
        }
    }
}
