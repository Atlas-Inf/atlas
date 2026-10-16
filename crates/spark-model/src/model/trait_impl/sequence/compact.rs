// SPDX-License-Identifier: AGPL-3.0-only

//! `compact_sequence` (split from `sequence.rs` for the 500-LoC cap).

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::super::ssm_pool::SsmStatePool;
use super::super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(in super::super) fn compact_sequence_dispatch(
        &self,
        seq: &mut SequenceState,
        new_slot: usize,
    ) -> Result<bool> {
        let old_slot = seq.slot_idx;
        if old_slot == new_slot {
            return Ok(true);
        }

        // Claim the NEW slot EXCLUSIVELY, BEFORE any copy. A target not on the
        // pool free list is OWNED by a live sequence (e.g. one still PREFILLING,
        // which the scheduler's active list does not see): refuse, copy nothing.
        if !self.ssm_pool.claim_specific(new_slot) {
            return Ok(false);
        }
        let stream = self.gpu.default_stream();
        self.ssm_pool
            .copy_slot(old_slot, new_slot, self.gpu.as_ref(), stream)
            .inspect_err(|_| self.ssm_pool.release_slot(new_slot))?; // un-claim

        // Update ALL SsmLayerState pool pointers to point at the new slot.
        // BUG FIX: previously only h_state and conv_state were repointed, leaving
        // the MTP checkpoint and intermediate pointers aimed at the OLD slot.
        // After release_slot, that old slot is reallocatable to a NEW sequence,
        // and any subsequent MTP save_hidden / start_checkpoint_async on this seq
        // would write into the new occupant's pool memory — cross-seq corruption.
        let has_mtp = self.ssm_pool.has_mtp;
        // Tiered pools: the H-intermediate count is a property of the SLOT
        // (h_inter_count), the conv count is uniform (num_intermediates).
        let num_intermediates = self.ssm_pool.num_intermediates;
        let h_intermediates = self.ssm_pool.h_inter_count(new_slot);
        let mut ssm_layer_idx = 0usize;
        for (i, state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                if let Some(ssm) = state.as_any_mut().downcast_mut::<SsmLayerState>() {
                    ssm.h_state = self.ssm_pool.h_state(ssm_layer_idx, new_slot);
                    ssm.conv_state = self.ssm_pool.conv_state(ssm_layer_idx, new_slot);
                    // Deferred-commit staging is slot-keyed (pool-stable) —
                    // repoint on rebind like every other pool buffer.
                    ssm.gdn_commit_qkv = self.ssm_pool.commit_qkv(ssm_layer_idx, new_slot);
                    ssm.gdn_commit_gb = self.ssm_pool.commit_gb(ssm_layer_idx, new_slot);
                    // Stage-3 f16-SIZED pool: the FP32 prefill staging blob is
                    // per-SLOT, so compaction must repoint it for the same
                    // reason the checkpoint/intermediate families below are
                    // repointed — the old slot becomes reallocatable, and a
                    // continuation chunk staging through the new occupant's
                    // blob is cross-sequence corruption. `None` stays `None`
                    // (FP32-sized pool: no staging exists).
                    ssm.h_prefill_stage = self.ssm_pool.h_prefill_stage(new_slot);
                    if has_mtp {
                        if ssm.h_state_checkpoint.is_some() {
                            ssm.h_state_checkpoint =
                                Some(self.ssm_pool.h_checkpoint(ssm_layer_idx, new_slot));
                        }
                        if ssm.conv_state_checkpoint.is_some() {
                            ssm.conv_state_checkpoint =
                                Some(self.ssm_pool.conv_checkpoint(ssm_layer_idx, new_slot));
                        }
                        if !ssm.h_state_intermediates.is_empty() {
                            ssm.h_state_intermediates.clear();
                            for t in 0..h_intermediates {
                                ssm.h_state_intermediates.push(self.ssm_pool.h_intermediate(
                                    ssm_layer_idx,
                                    new_slot,
                                    t,
                                ));
                            }
                        }
                        if !ssm.conv_state_intermediates.is_empty() {
                            ssm.conv_state_intermediates.clear();
                            for t in 0..num_intermediates {
                                ssm.conv_state_intermediates
                                    .push(self.ssm_pool.conv_intermediate(
                                        ssm_layer_idx,
                                        new_slot,
                                        t,
                                    ));
                            }
                        }
                    }
                }
                ssm_layer_idx += 1;
            }
        }

        seq.slot_idx = new_slot;
        // BUG FIX: synchronize before releasing the old slot. copy_slot is async
        // (queued D2D), so without this barrier, claim_slot() in the next request
        // could hand the old_slot back to a new sequence while the copy's source
        // reads are still in flight — cross-seq race that produces partial data.
        // Surfaced AFTER the bookkeeping: the guard must follow the repoint.
        let synced = self.gpu.synchronize(stream);
        // Slot-migration is an ownership TRANSFER, not a free: this sequence
        // keeps a live slot (the NEW one). Take the old idx out of the guard so
        // its Drop won't re-release it, release the old slot exactly once, then
        // re-point the guard at the new slot it now owns. This preserves the
        // exactly-once invariant: old_slot is pushed below (once) and new_slot
        // will be pushed by whichever path later frees THIS sequence (once).
        if let Some(g) = seq.ssm_slot.as_mut() {
            // Guard owned `old_slot`; drop that ownership before releasing.
            let owned = g.take();
            debug_assert_eq!(
                owned,
                Some(old_slot),
                "compact_sequence: guard owned {owned:?}, expected old_slot {old_slot}"
            );
            g.migrate(new_slot);
        }
        // A failed sync returns BEFORE the release: the copy may still be reading
        // old_slot, so leak it on the (dead) context rather than hand it out.
        synced?;
        // Released exactly once, guard or no guard (mock model with no SSM pool).
        self.ssm_pool.release_slot(old_slot);
        Ok(true)
    }
}
