// SPDX-License-Identifier: AGPL-3.0-only

use super::wire::{ALLOC, FREE, STOP};
use super::*;
use crate::traits::{Model, SequenceState};
use spark_runtime::kernel_args::KernelLaunch;

impl Model for Hy4Model {
    fn teardown(&mut self) -> Result<()> {
        if let Some(cell) = &self.cell {
            let owner = cell.borrow_owner();
            owner.gpu.synchronize(owner.gpu.default_stream())?;
        }
        self.cell.take();
        Ok(())
    }
    fn generate_speculative(
        &self,
        _t: &[u32],
        _p: &spark_runtime::sampler::SamplingParams,
        _n: usize,
    ) -> Result<crate::engine::GenerateResult> {
        Self::unavailable()
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        Self::unavailable()
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _g: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Self::unavailable()
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        Self::unavailable()
    }
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let ptr = self.forward_rows(tokens, seq, super::batch::VERIFY, stream)?;
        self.argmax_batch(ptr, tokens.len(), stream)
    }
    fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()> {
        let stream = self.eager_stream(self.default_stream())?;
        self.send([
            super::speculation::CHECKPOINT,
            u32::try_from(seq.slot_idx)?,
            u32::try_from(seq.seq_len)?,
            0,
        ])?;
        self.borrow_dependent()
            .checkpoint(seq.slot_idx, seq.seq_len, stream)
    }
    fn commit_accepted_prefix(
        &self,
        seq: &mut SequenceState,
        accepted: usize,
        rows: usize,
    ) -> Result<()> {
        let r = self.borrow_dependent();
        r.validate_commit(seq.slot_idx, accepted, rows, Some(seq))?;
        let stream = self.eager_stream(self.default_stream())?;
        self.send([
            super::speculation::COMMIT,
            u32::try_from(seq.slot_idx)?,
            u32::try_from(accepted)?,
            u32::try_from(rows)?,
        ])?;
        r.commit(self.borrow_owner(), seq.slot_idx, accepted, rows, stream)
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        Self::unavailable()
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn eager_prompt_lookup(&self) -> bool {
        true
    }
    fn eager_draft(
        &self,
        bonus: u32,
        count: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Option<Vec<u32>>> {
        if self.borrow_dependent().drafter.is_none() {
            return Ok(None);
        }
        ensure!(
            (1..=8).contains(&count) && (bonus as usize) < self.vocab_size(),
            "Hy4 DFly invalid count/bonus"
        );
        let stream = self.eager_stream(stream)?;
        self.checkpoint_ssm_states(seq)?;
        self.send([
            draft::PROPOSE,
            u32::try_from(seq.slot_idx)?,
            bonus,
            u32::try_from(count)?,
        ])?;
        self.borrow_dependent()
            .propose(self.borrow_owner(), seq.slot_idx, bonus, count, stream)
            .map(Some)
    }
    fn verify_max_drafts(&self) -> Option<usize> {
        Some(8)
    }
    fn verify_context_limit(&self) -> Option<usize> {
        Some(self.borrow_owner().max_seq)
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        Self::unavailable()
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        Self::unavailable()
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        Self::unavailable()
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        Self::unavailable()
    }
    fn save_hidden_for_mtp(&self, _t: usize, _st: u64) -> Result<()> {
        Self::unavailable()
    }
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.prefill_chunk(tokens, seq, 0, tokens.len(), true, stream)
    }
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        start: usize,
        len: usize,
        last: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let end = start.checked_add(len).context("Hy4 chunk overflow")?;
        ensure!(
            len > 0
                && end <= tokens.len()
                && seq.seq_len == start
                && end <= self.borrow_owner().max_seq
                && last == (end == tokens.len())
                && tokens[start..end]
                    .iter()
                    .all(|t| (*t as usize) < self.vocab_size()),
            "Hy4 invalid/empty/discontinuous prefill chunk"
        );
        if start == 0 {
            self.maybe_verify_benchmark(tokens, seq, stream)?;
        }
        let mut logits = DevicePtr::NULL;
        for chunk in tokens[start..end].chunks(self.borrow_dependent().prefill.capacity) {
            let mode = if last && seq.seq_len + chunk.len() == end {
                super::batch::LAST
            } else {
                super::batch::BATCH
            };
            logits = self.forward_rows(chunk, seq, mode, stream)?;
        }
        seq.prompt_len = tokens.len();
        Ok(if last { logits } else { DevicePtr::NULL })
    }
    fn decode(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.forward_token(token, seq, stream)
    }
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        let s = self.borrow_owner();
        let r = self.borrow_dependent();
        ensure!(
            !tokens.is_empty() && tokens.len() == seqs.len() && tokens.len() <= r.max_batch,
            "Hy4 decode batch geometry"
        );
        let stream = self.eager_stream(stream)?;
        for (i, (&token, seq)) in tokens.iter().zip(seqs).enumerate() {
            let ptr = self.forward_token(token, seq, stream)?;
            s.gpu.copy_d2d_async(
                ptr,
                r.batch_logits.offset(i * s.config.vocab_size * 4),
                s.config.vocab_size * 4,
                stream,
            )?;
        }
        s.gpu.synchronize(stream)?;
        Ok(r.batch_logits)
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        let s = self.borrow_owner();
        let r = self.borrow_dependent();
        let requests = r.requests.lock();
        let slot = (0..r.max_batch)
            .find(|i| !requests.contains_key(i))
            .context("Hy4 no free request slots")?;
        drop(requests);
        if s.config.tp_rank == 0 {
            self.send([ALLOC, slot as u32, 0, 0])?;
        }
        r.allocate(s, slot)?;
        Ok(SequenceState::host_only(slot))
    }
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        if seq.slot_idx == usize::MAX {
            return Ok(());
        }
        if self.borrow_owner().config.tp_rank == 0 {
            self.send([FREE, seq.slot_idx as u32, 0, 0])?;
        }
        self.borrow_owner().gpu.synchronize(self.default_stream())?;
        self.borrow_dependent()
            .requests
            .lock()
            .remove(&seq.slot_idx);
        seq.slot_idx = usize::MAX;
        Ok(())
    }
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<bool> {
        Ok(seq.slot_idx == new_slot)
    }
    fn detach_slot_for_reuse(&self, _seq: &mut SequenceState) {}
    fn cache_sequence(&self, _seq: &SequenceState) {}
    fn vocab_size(&self) -> usize {
        self.borrow_owner().config.vocab_size
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.borrow_owner().gpu.bind_to_thread()
    }
    fn copy_logits_to_host(&self, ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.borrow_owner().gpu.copy_d2h(ptr, dst)
    }
    fn logits_ptr_is_fp32(&self, _ptr: DevicePtr) -> bool {
        true
    }
    fn decode_logits_fp32(&self) -> bool {
        true
    }
    fn decode_logits_ptr(&self) -> DevicePtr {
        self.logits_buffer_ptr()
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        let r = self.borrow_dependent();
        if r.verify_logits.load(std::sync::atomic::Ordering::Relaxed) {
            r.multi.logits
        } else {
            r.logits
        }
    }
    fn argmax_on_device(&self, ptr: DevicePtr, stream: u64) -> Result<u32> {
        let s = self.borrow_owner();
        let r = self.borrow_dependent();
        KernelLaunch::new(s.gpu.as_ref(), r.kernels[3])
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ptr)
            .arg_ptr(r.post)
            .arg_u32(s.config.vocab_size as u32)
            .launch(stream)?;
        s.gpu.synchronize(stream)?;
        let mut b = [0; 4];
        s.gpu.copy_d2h(r.post, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn argmax_batch(&self, ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>> {
        let s = self.borrow_owner();
        let r = self.borrow_dependent();
        ensure!(
            n <= if ptr == r.multi.logits {
                r.multi.capacity
            } else {
                r.max_batch
            },
            "Hy4 argmax batch capacity"
        );
        if ptr != r.multi.logits || n == 1 {
            return (0..n)
                .map(|i| self.argmax_on_device(ptr.offset(i * self.vocab_size() * 4), stream))
                .collect();
        }
        KernelLaunch::new(s.gpu.as_ref(), r.kernels[4])
            .grid([n as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(ptr)
            .arg_ptr(r.multi.post)
            .arg_u32(self.vocab_size() as u32)
            .arg_u32(n as u32)
            .launch(stream)?;
        s.gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; n * 4];
        s.gpu.copy_d2h(r.multi.post, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("u32 chunk")))
            .collect())
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        let r = self.borrow_dependent();
        let rows = r.head_rows.load(std::sync::atomic::Ordering::Relaxed);
        if rows > 0 {
            r.multi
                .normed
                .offset((rows - 1) * self.borrow_owner().config.hidden_size * 2)
        } else {
            r.normed
        }
    }
    fn ep_worker_step(&self, _slots: &mut [Option<SequenceState>]) -> Result<bool> {
        self.worker()
    }
    fn is_ep(&self) -> bool {
        self.borrow_owner().config.tp_world_size > 1
    }
    fn ep_broadcast_cmd(&self, cmd: u32) -> Result<()> {
        if cmd == STOP {
            self.send([STOP, 0, 0, 0])?;
        }
        Ok(())
    }
    fn ep_broadcast_cmd_for_seq(&self, _seq_id: u32, cmd: u32) -> Result<()> {
        self.ep_broadcast_cmd(cmd)
    }
    fn is_mla(&self) -> bool {
        true
    }
    fn supports_chunked_mla(&self) -> bool {
        true
    }
    fn hc_mult(&self) -> usize {
        4
    }
    fn kv_block_size(&self) -> Option<usize> {
        Some(self.borrow_owner().page_size)
    }
    fn num_free_blocks(&self) -> usize {
        let r = self.borrow_dependent();
        (r.max_batch - r.requests.lock().len())
            * self
                .borrow_owner()
                .max_seq
                .div_ceil(self.borrow_owner().page_size)
    }
    fn num_total_blocks(&self) -> usize {
        self.borrow_dependent().max_batch
            * self
                .borrow_owner()
                .max_seq
                .div_ceil(self.borrow_owner().page_size)
    }
    fn default_stream(&self) -> u64 {
        self.borrow_owner().gpu.default_stream()
    }
    fn create_stream(&self) -> Result<u64> {
        Ok(self.default_stream())
    }
    fn create_event(&self) -> Result<u64> {
        self.borrow_owner().gpu.create_event()
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.borrow_owner().gpu.record_event(event, stream)
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.borrow_owner().gpu.stream_wait_event(stream, event)
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.borrow_owner().gpu.synchronize(stream)
    }
}
