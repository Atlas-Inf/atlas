// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::dflash_head::{DflashProposerState, SequenceGeneration};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn run_mtp_propose_inner(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        let proposer = match &self.proposer {
            Some(p) => p.as_ref(),
            None => return Ok(Vec::new()),
        };
        // ATLAS_DFLASH_DEBUG_DUMP_FULL: emit the full token sequence ONCE
        // so a Python reference can run the SAME tokens through HF
        // transformers and dump matching hidden-state captures.
        // Per-model latch: a static would let the previous model swallow this
        // one's dump. DFlash proposers carry the startup-frozen switch; other
        // proposers keep the legacy environment read (diagnostic-only).
        let dump_full = proposer
            .startup_diagnostics()
            .map(|diagnostics| diagnostics.dump_full)
            .unwrap_or_else(|| {
                std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                    .ok()
                    .as_deref()
                    == Some("1")
            });
        if dump_full && self.stats.dumped.keyed("dump:dflash_tokens") {
            let tokens_json = serde_json::json!({
                "prompt_len": position - seq.tokens.len() + seq.tokens.len(),
                "position": position,
                "last_token": token,
                "all_tokens": seq.tokens.clone(),
                "generated_tokens": seq.tokens.iter().skip(seq.prompt_len).copied().collect::<Vec<u32>>(),
            });
            if let Err(e) = std::fs::write(
                "/tmp/atlas_tokens.json",
                serde_json::to_string_pretty(&tokens_json).unwrap_or_default(),
            ) {
                tracing::warn!("DFLASH DUMP_FULL: tokens write failed: {e}");
            } else {
                tracing::info!(
                    "DFLASH DUMP_FULL: wrote /tmp/atlas_tokens.json (position={}, all_tokens.len()={}, prompt_len={})",
                    position,
                    seq.tokens.len(),
                    seq.prompt_len,
                );
            }
        }
        let stream = self.gpu.default_stream();
        let draft_embed_target = None;
        // MTP loads ALL experts on every rank (no EP filtering), so its MoE
        // output is already complete — no all_reduce needed. Passing comm: None
        // prevents MoeLayer::forward() from doubling the output via SUM.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None, // #30: MTP/draft decode never routes prefill.
            midchunk_capture: None,
            moe_lora_route: self.decode_moe_route(), // route-aware: base(Skip) skips fold, adapter folds (single-seq reject lifted)
        };
        // Give the drafter its prompt context on the first propose of this
        // sequence: whole-prompt prefill on a COLD turn, carried rows + a
        // short append on a WARM one. See `ensure_drafter_context`.
        self.ensure_drafter_context(proposer, seq, &ctx, stream);
        let expected_owner = seq.expected_dspark_owner()?;
        let prop_state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No proposer state for sequence"))?;
        // ATLAS_MTP_CATCHUP: before proposing, feed pairs the drafter missed
        // during a serial-decode stretch. Coordinates (measured 2026-07-20 on
        // the 27B rig): at propose entry `position == seq.tokens.len()` and
        // the imminent forward_one writes the pair for sequence key
        // `position - 1`; pair key k = (embed(tokens[k+1]), hidden_k), RoPE
        // k+1. The serial-decode ring stores, under label n, the hidden of
        // the step that COMMITTED token n — i.e. hidden_{n-1} — so pair key k
        // reads ring label k+1. Drafter KV slots are compacted (append-only)
        // while RoPE stays sequence-space, so RoPE gaps are already the norm:
        // partial feeds (clipped to ring coverage) are safe, and wrong feeds
        // cannot corrupt output (verify rejects bad drafts).
        if crate::speculative::mtp_catchup_enabled() && !self.mtp_catchup_ring.is_null() {
            let rows = proposer.drafter_rows(prop_state.as_mut());
            let last_key = proposer.last_pair_key(prop_state.as_mut());
            let (start, count) = *self.mtp_catchup_meta.lock();
            // ATLAS_MTP_REFEED_DEBUG: the ring round-trip check. The pair key
            // this propose is ABOUT to write is `position - 1`, and it reads
            // its hidden from `mtp_hidden_save`; under the label convention
            // that same hidden is ring label `position`. So
            // `fp(ring[position % rows]) == fp(mtp_hidden_save)` iff the
            // ring's write-side slot arithmetic, the D2D plumbing and the
            // read-side slot arithmetic all agree. It does NOT prove the
            // label convention itself (see `mtp_refeed_shift`).
            if crate::speculative::mtp_refeed_debug() {
                let ring_rows = super::types::MTP_CATCHUP_RING_ROWS;
                let h = self.config.hidden_size;
                let fp_save = crate::speculative::hidden_fingerprint(
                    self.gpu.as_ref(),
                    self.mtp_hidden_save,
                    h,
                );
                let fp_ring = crate::speculative::hidden_fingerprint(
                    self.gpu.as_ref(),
                    self.mtp_catchup_ring.offset((position % ring_rows) * h * 2),
                    h,
                );
                let covered = count > 0 && position >= start && position < start + count;
                tracing::info!(
                    "REFEED_DBG propose position={position} rows={rows} \
                     last_key={last_key:?} ring=[{start},+{count}) \
                     fp_save={fp_save:016x} fp_ring[{position}]={fp_ring:016x} \
                     covered={covered} roundtrip_ok={}",
                    covered && fp_save == fp_ring,
                );
            }
            if let Some(last) = last_key
                && rows > 0
                && count > 0
            {
                // Missing pair keys: (last .. position-1); the propose itself
                // covers position-1. Clip to ring coverage [start, start+count)
                // in label space (label = key + 1).
                let mut k0 = (last + 1).max(start.saturating_sub(1));
                let k1 = (position.saturating_sub(2)).min((start + count).saturating_sub(2));
                let want = (position.saturating_sub(1)).saturating_sub(last + 1);
                if k0 <= k1 && want > 0 {
                    let ring_rows = super::types::MTP_CATCHUP_RING_ROWS;
                    let h = self.config.hidden_size;
                    let bf16 = 2usize;
                    let fed_from = k0;
                    while k0 <= k1 {
                        // Ring-contiguous segment: labels k0+1 .. until wrap.
                        let slot = (k0 + 1) % ring_rows;
                        let seg_last = k1.min(k0 + (ring_rows - slot) - 1);
                        let n_rows = seg_last - k0 + 1;
                        // Row r feeds pair key k0+r = embed(tokens[k0+r+1]):
                        // the impl reads prompt_tokens[r+1], so pass the
                        // window starting at index k0 (n_rows + 1 tokens).
                        let toks = &seq.tokens[k0..=seg_last + 1];
                        let hid = self.mtp_catchup_ring.offset(slot * h * bf16);
                        if crate::speculative::mtp_refeed_debug() {
                            for r in 0..n_rows {
                                let fp = crate::speculative::hidden_fingerprint(
                                    self.gpu.as_ref(),
                                    hid.offset(r * h * bf16),
                                    h,
                                );
                                tracing::info!(
                                    "REFEED_DBG feed key={} label={} tok={} rope={} fp={fp:016x}",
                                    k0 + r,
                                    k0 + r + 1,
                                    toks[r + 1],
                                    k0 + r + 1,
                                );
                            }
                        }
                        let row_base = proposer.drafter_rows(prop_state.as_mut());
                        match proposer.catchup_drafter(
                            toks,
                            hid,
                            row_base,
                            k0 + 1,
                            prop_state.as_mut(),
                            &ctx,
                            stream,
                        ) {
                            Ok(w) if w == n_rows => k0 = seg_last + 1,
                            Ok(w) => {
                                tracing::debug!(
                                    "MTP catch-up: short feed ({w}/{n_rows} rows) — degrading"
                                );
                                break;
                            }
                            Err(e) => {
                                tracing::debug!("MTP catch-up: feed failed ({e:#}) — degrading");
                                break;
                            }
                        }
                    }
                    if k0 > k1 {
                        tracing::debug!(
                            "MTP catch-up: fed pair keys {fed_from}..={k1} \
                             (missed {want}, position {position})"
                        );
                    }
                } else if want > 0 {
                    tracing::debug!(
                        "MTP catch-up: gap of {want} pairs outside ring coverage \
                         (last_key={last} position={position} ring=[{start},+{count}))"
                    );
                }
            }
        }
        let drafts = proposer.propose(
            token,
            self.mtp_hidden_save,
            position,
            num_drafts,
            prop_state.as_mut(),
            Some(expected_owner),
            &ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            self.dflash_hidden_save,
        )?;
        // Confidence clamp (ATLAS_MTP_DRAFT_CONF, staged off by default):
        // when the drafter's chain confidence is below tau, discard the
        // drafts — the next step decodes serially instead of paying a
        // verify that would most likely reject (break-even acceptance at
        // K=1 on the 35B MoE is ~0.66). The drafter KV rows written by
        // this propose MUST be trimmed exactly as a full rejection would
        // (after_verify(0)), or the drafter desyncs from the target.
        let tau = crate::speculative::draft_conf_tau();
        if tau > 0.0
            && !drafts.is_empty()
            && let Some(conf) = proposer.last_confidence()
            && conf < tau
        {
            tracing::debug!(
                "MTP draft skipped: chain confidence {conf:.3} < tau {tau:.3}                  (pos {position}, {} drafts trimmed)",
                drafts.len(),
            );
            proposer.after_verify(0, Some(expected_owner), prop_state.as_mut(), stream)?;
            return Ok(Vec::new());
        }
        Ok(drafts)
    }

    // Proposer-wiring accessors live in impl_b3_accessors.rs (500-LoC cap).
}
