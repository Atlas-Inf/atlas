// SPDX-License-Identifier: AGPL-3.0-only

//! Predictor snapshots paired with exact canonical target-prefix boundaries.
//! Each entry owns refs independently of the originating request/slot generation.
use super::types::TransformerModel;
use crate::speculative::ProposerState;
use crate::traits::SequenceState;
use anyhow::{Context, Result};

mod admission;
pub(in crate::model) use admission::EP_CMD_NATIVE_PREFIX_PROBE;

pub(super) struct NativeMtpPrefix {
    tokens: Vec<u32>,
    adapter: u64,
    state: Box<dyn ProposerState>,
    ready: u64,
}

fn matches(entry: &NativeMtpPrefix, tokens: &[u32], end: usize, adapter: u64) -> bool {
    entry.adapter == adapter
        && entry.tokens.len() == end
        && tokens.get(..end) == Some(entry.tokens.as_slice())
}

impl TransformerModel {
    fn release_native_mtp_prefix(&self, mut entry: NativeMtpPrefix) -> Result<()> {
        let stream = self.gpu.default_stream();
        // Block refs and hidden storage cannot be reused while snapshot copies
        // or a subsequent restore copy are still queued on another stream.
        self.gpu.stream_wait_event(stream, entry.ready)?;
        self.gpu.synchronize(stream)?;
        self.gpu.destroy_event(entry.ready)?;
        if let Some(proposer) = self.proposer.as_ref() {
            proposer.free_state(self.gpu.as_ref(), None, entry.state.as_mut())?;
        }
        Ok(())
    }

    pub(in crate::model) fn clear_native_mtp_prefix(&self) -> Result<()> {
        let entries = std::mem::take(&mut *self.native_mtp_prefix.lock());
        let mut first = None;
        for entry in entries {
            if let Err(error) = self.release_native_mtp_prefix(entry) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Prefix refs are expendable; active requests always retain ownership.
    pub(in crate::model) fn evict_native_mtp_prefix_for_rows(&self, rows: usize) -> Result<()> {
        if let Some(proposer) = self.proposer.as_ref()
            && proposer
                .native_prefix_free_blocks()
                .is_some_and(|free| free < rows.div_ceil(16) + 2)
        {
            let count = self.native_mtp_prefix.lock().len();
            if count > 0 {
                tracing::info!(count, "native MTP prefix eviction: private pool pressure");
                self.clear_native_mtp_prefix()?;
            }
        }
        Ok(())
    }

    pub(in crate::model) fn save_native_mtp_prefix(
        &self,
        seq: &SequenceState,
        tokens: &[u32],
        end: usize,
        stream: u64,
    ) -> Result<()> {
        if !self.native_glm_mtp
            || seq.disable_mtp
            || !self.prefix_cache.is_active()
            || !self.ssm_snapshots.is_enabled()
            || end == 0
        {
            return Ok(());
        }
        let Some(proposer) = self.proposer.as_ref() else {
            return Ok(());
        };
        let Some(tokens) = tokens
            .get(..end)
            .and_then(|tokens| self.pc_key(seq, tokens))
        else {
            return Ok(());
        };
        let Some(state) = seq.proposer_state.as_ref() else {
            return Ok(());
        };
        self.evict_native_mtp_prefix_for_rows(0)?;
        // No spare block means retaining a snapshot would compete with live KV.
        if proposer.native_prefix_free_blocks() == Some(0) {
            return Ok(());
        }
        let mut entries = self.native_mtp_prefix.lock();
        if let Some(index) = entries
            .iter()
            .position(|entry| matches(entry, &tokens, end, seq.adapter_id))
        {
            self.release_native_mtp_prefix(entries.remove(index))?;
        }
        while entries.len() >= self.ssm_snapshots.num_slots {
            self.release_native_mtp_prefix(entries.remove(0))?;
        }
        let Some(mut snapshot) =
            proposer.clone_prefix_state(state.as_ref(), end - 1, self.gpu.as_ref(), stream)?
        else {
            return Ok(());
        };
        let ready = match self.gpu.create_event() {
            Ok(event) => event,
            Err(error) => {
                self.gpu.synchronize(stream)?;
                proposer.free_state(self.gpu.as_ref(), None, snapshot.as_mut())?;
                return Err(error);
            }
        };
        if let Err(error) = self.gpu.record_event(ready, stream) {
            self.gpu.synchronize(stream)?;
            let _ = self.gpu.destroy_event(ready);
            proposer.free_state(self.gpu.as_ref(), None, snapshot.as_mut())?;
            return Err(error);
        }
        entries.push(NativeMtpPrefix {
            tokens: tokens.into_owned(),
            adapter: seq.adapter_id,
            state: snapshot,
            ready,
        });
        Ok(())
    }

    /// Every rank calls this once for a lookup, including an ineligible target
    /// candidate (`end=0`). Rank zero owns predictor snapshots; peers agree on
    /// its hit/miss before any rank restores target state or skips target rows.
    pub(in crate::model) fn restore_native_mtp_prefix(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        end: usize,
        stream: u64,
    ) -> Result<bool> {
        if !self.native_glm_mtp {
            return Ok(true);
        }
        let key = self.pc_key(seq, tokens);
        let mut entries = self.native_mtp_prefix.lock();
        let found = key.as_ref().and_then(|key| {
            entries
                .iter()
                .position(|entry| matches(entry, key, end, seq.adapter_id))
        });
        let local = end > 0 && (seq.disable_mtp || self.proposer.is_none() || found.is_some());
        let agreed = if self.multi_rank_protocol_active() {
            self.ep_min_u32(if local {
                u32::try_from(end).context("native prefix length overflow")?
            } else {
                0
            })? as usize
        } else if local {
            end
        } else {
            0
        };
        let ready = agreed > 0 && agreed == end;
        let ready = if self.multi_rank_protocol_active() {
            self.ep_min_u32(u32::from(ready))? == 1
        } else {
            ready
        };
        if !ready {
            if end > 0 {
                tracing::info!(
                    end,
                    slot = seq.slot_idx,
                    generation = seq.mtp_capture_gen,
                    "native MTP prefix miss: recomputing target and predictor context"
                );
            }
            return Ok(false);
        }
        if seq.disable_mtp {
            return Ok(true);
        }
        let Some(proposer) = self.proposer.as_ref() else {
            return Ok(true);
        };
        let index = found.context("agreed native prefix is missing")?;
        let entry = entries.remove(index);
        if let Err(error) = self.gpu.stream_wait_event(stream, entry.ready) {
            entries.push(entry);
            return Err(error);
        }
        // If all spare blocks are used, evict other snapshots before cloning
        // the mutable tail. This never releases the chosen prefix or live KV.
        while proposer.native_prefix_free_blocks() == Some(0) && !entries.is_empty() {
            if let Err(error) = self.release_native_mtp_prefix(entries.remove(0)) {
                entries.push(entry);
                return Err(error);
            }
        }
        let restored =
            proposer.clone_prefix_state(entry.state.as_ref(), end - 1, self.gpu.as_ref(), stream);
        // Fence reads of the cached tail as well as its original writes. Keep
        // the owning entry even on an error so teardown releases every ref.
        let fence = self.gpu.record_event(entry.ready, stream);
        if fence.is_err() {
            let _ = self.gpu.synchronize(stream);
        }
        entries.push(entry);
        let mut restored =
            restored?.context("native proposer does not support prefix snapshots")?;
        if let Err(error) = fence {
            proposer.free_state(self.gpu.as_ref(), None, restored.as_mut())?;
            return Err(error);
        }
        if let Some(mut previous) = seq.proposer_state.replace(restored) {
            proposer.free_state(self.gpu.as_ref(), None, previous.as_mut())?;
        }
        tracing::info!(
            end,
            slot = seq.slot_idx,
            generation = seq.mtp_capture_gen,
            "native MTP prefix hit: restored predictor KV and target boundary"
        );
        Ok(true)
    }
}
