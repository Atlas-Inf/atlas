// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for decode-time KV preemption with resume (`preempt.rs`).
//!
//! The bug class under test is client-facing: the pre-resume code called
//! `send_error` on the victim, emitting a mid-stream SSE error frame and
//! ending the stream with no finish chunk — an HTTP-200 "success" with
//! silently truncated content. These tests drive the REAL retry loop
//! (`decode_batch_with_preemption`) against a scripted `Model` stub and
//! assert on the transport seam the API layer consumes (the per-request
//! `StreamEvent` channel): a preempted victim's channel must stay OPEN and
//! EMPTY — no `StreamEvent::Error`, no premature `Done`.

use super::lifecycle::resume_swapped_seq;
use super::preempt::{
    PREEMPT_IMMUNITY_TOKENS, decode_batch_with_preemption, preempt_requeue, resume_preempted_seq,
    resume_preempted_seqs, spill_out_sequence, spill_pool_enabled,
};
use super::test_support::test_seq;
use super::types::{ActiveSeq, ResponseSink};
use anyhow::Result;
use spark_model::model::kv_admission::{KvAdmissionRefused, after_partial_step};
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_spill::KvSpillManager;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

mod chunk_retry;
mod kv_refusal;
mod min_tokens_ban;
mod resume;
mod victim_policy;

/// Scripted stub: `decode_batch` fails with the KV-exhausted error for the
/// first `fail_decodes` calls, then succeeds. Records every free/cache/
/// prefill so the tests can assert the preemption side effects.
#[derive(Default)]
struct PreemptStubModel {
    fail_decodes: AtomicUsize,
    /// When set, `decode_batch` always fails with this message instead.
    hard_error: Option<&'static str>,
    decode_calls: AtomicUsize,
    freed_slots: Mutex<Vec<usize>>,
    cached_seqs: AtomicUsize,
    prefilled: Mutex<Vec<Vec<u32>>>,
    vision_pad: Option<u32>,
    free_blocks: AtomicUsize,
    total_blocks: usize,
    reclaimable: AtomicUsize,
    /// `prefill_chunk` records `(start, len)` and refuses (agreed) the first
    /// `refuse_chunks` calls, completing a split half up to `split_cut` first;
    /// `refusal_is_final` makes the refusal non-retryable.
    chunk_calls: Mutex<Vec<(usize, usize)>>,
    refuse_chunks: AtomicUsize,
    split_cut: usize,
    refusal_is_final: bool,
    /// Every EP command word the head sends.
    wire: Mutex<Vec<u32>>,
    /// `decode_batch` fails with an agreed, retryable refusal raised after
    /// this many sequences of a per-sequence loop already ran.
    partial_refusal_after: Option<usize>,
}

impl PreemptStubModel {
    fn failing(n: usize) -> Self {
        Self {
            fail_decodes: AtomicUsize::new(n),
            ..Default::default()
        }
    }
}

impl Model for PreemptStubModel {
    fn prefill(&self, t: &[u32], s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        self.prefilled.lock().unwrap().push(t.to_vec());
        // Mirror the real contract: prefill populates tokens/seq_len/prompt_len.
        s.tokens.extend_from_slice(t);
        s.seq_len = s.tokens.len();
        s.prompt_len = t.len();
        Ok(DevicePtr::NULL)
    }
    fn decode(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in preempt tests")
    }
    fn prefill_chunk(
        &self,
        t: &[u32],
        s: &mut SequenceState,
        cs: usize,
        cl: usize,
        _last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        self.chunk_calls.lock().unwrap().push((cs, cl));
        if let Some(msg) = self.hard_error {
            anyhow::bail!("{msg}");
        }
        let refuse = (self.refuse_chunks)
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok();
        let split = cs < self.split_cut && self.split_cut < cs + cl;
        let stop = match (refuse, split) {
            (false, _) => cs + cl,
            (true, true) => self.split_cut,
            (true, false) => cs,
        };
        s.tokens.extend_from_slice(&t[cs..stop]);
        (s.seq_len, s.prompt_len) = (stop, t.len());
        let (by_peer, retryable) = (true, !self.refusal_is_final);
        anyhow::ensure!(
            !refuse,
            KvAdmissionRefused {
                by_peer,
                retryable,
                decode: false
            }
        );
        Ok(DevicePtr::NULL)
    }
    fn ep_broadcast_cmd(&self, word: u32) -> Result<()> {
        self.wire.lock().unwrap().push(word);
        Ok(())
    }
    fn ep_broadcast_cmd_for_seq(&self, _seq: u32, cmd: u32) -> Result<()> {
        self.ep_broadcast_cmd(cmd)
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        s: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        self.decode_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(msg) = self.hard_error {
            anyhow::bail!("{msg}");
        }
        if let Some(advanced) = self.partial_refusal_after {
            let refused = KvAdmissionRefused {
                by_peer: false,
                retryable: true,
                decode: true,
            };
            return Err(after_partial_step(refused.into(), advanced, s.len()));
        }
        if self
            .fail_decodes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            anyhow::bail!("KV cache exhausted: no free blocks");
        }
        Ok(DevicePtr::NULL)
    }
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _n: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        anyhow::bail!("unused in preempt tests")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        anyhow::bail!("unused in preempt tests")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        _n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _bm: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn vocab_size(&self) -> usize {
        0
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        Ok(())
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        Ok(SequenceState::host_only(0))
    }
    fn copy_logits_to_host(&self, _p: DevicePtr, _d: &mut [u8]) -> Result<()> {
        Ok(())
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _p: DevicePtr, _st: u64) -> Result<u32> {
        anyhow::bail!("unused in preempt tests")
    }
    fn argmax_batch(&self, _p: DevicePtr, _n: usize, _st: u64) -> Result<Vec<u32>> {
        anyhow::bail!("unused in preempt tests")
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn checkpoint_ssm_states(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn rollback_ssm_states(&self, _s: &mut SequenceState, _n: usize) -> Result<()> {
        Ok(())
    }
    fn has_proposer(&self) -> bool {
        false
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        anyhow::bail!("unused in preempt tests")
    }
    fn cache_sequence(&self, _s: &SequenceState) {
        self.cached_seqs.fetch_add(1, Ordering::SeqCst);
    }
    fn save_sequence_state(
        &self,
        _s: &SequenceState,
        _writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        Ok(())
    }
    fn restore_sequence_state(
        &self,
        _s: &mut SequenceState,
        _num_blocks: usize,
        _reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        Ok(())
    }
    fn swap_resumable(&self) -> bool {
        true
    }
    fn free_sequence(&self, s: &mut SequenceState) -> Result<()> {
        self.freed_slots.lock().unwrap().push(s.slot_idx);
        Ok(())
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _slot: usize) -> Result<bool> {
        Ok(true)
    }
    fn detach_slot_for_reuse(&self, _s: &mut SequenceState) {}
    fn save_hidden_for_mtp(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn tokens_contain_vision_pad(&self, tokens: &[u32]) -> bool {
        self.vision_pad
            .map(|pad| tokens.contains(&pad))
            .unwrap_or(false)
    }
    fn num_free_blocks(&self) -> usize {
        self.free_blocks.load(Ordering::SeqCst)
    }
    fn num_total_blocks(&self) -> usize {
        self.total_blocks
    }
    fn reclaim_prefix_blocks(&self, num_blocks: usize) -> usize {
        let take = num_blocks.min(self.reclaimable.load(Ordering::SeqCst));
        self.reclaimable.fetch_sub(take, Ordering::SeqCst);
        self.free_blocks.fetch_add(take, Ordering::SeqCst);
        take
    }
}

/// An unfinished decode-active sequence at `slot` with `n_out` generated
/// tokens and a known prompt in `seq.tokens`.
fn active_seq(slot: usize, n_out: usize) -> (ActiveSeq, super::test_support::RespRx) {
    let out: Vec<u32> = (100..100 + n_out as u32).collect();
    let (mut a, rx) = test_seq(out, 50, None, 4 + n_out);
    a.finished = false;
    a.seq.slot_idx = slot;
    // prompt [1,2,3,4] + all PROCESSED outputs (everything but last_token).
    a.seq.tokens = vec![1, 2, 3, 4];
    let n = a.output_tokens.len();
    a.seq
        .tokens
        .extend_from_slice(&a.output_tokens[..n.saturating_sub(1)]);
    (a, rx)
}

fn streaming_seq(
    slot: usize,
    n_out: usize,
) -> (
    ActiveSeq,
    tokio::sync::mpsc::Receiver<crate::api::StreamEvent>,
) {
    let (a, _rx) = active_seq(slot, n_out);
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let mut a = a;
    a.sink = ResponseSink::Streaming(tx);
    (a, rx)
}

// ── decode_batch_with_preemption ─────────────────────────────────────────

#[test]
fn kv_exhaustion_requeues_least_progress_victim_and_sends_nothing() {
    let model = PreemptStubModel::failing(1);
    let (a0, _rx0) = active_seq(0, 5);
    let (victim, mut victim_rx) = streaming_seq(1, 2); // least progress
    let (a2, _rx2) = active_seq(2, 9);
    let mut active = vec![a0, victim, a2];
    let mut swapped = Vec::new();
    let mut preempted = Vec::new();

    let logits =
        decode_batch_with_preemption(&model, &mut active, None, &mut swapped, &mut preempted);

    // The batch survived: one retry after one preemption.
    assert!(logits.is_some());
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 2);
    assert_eq!(active.len(), 2);
    // Least-progress victim (slot 1, 2 tokens) was requeued, not killed.
    assert_eq!(preempted.len(), 1);
    assert!(swapped.is_empty());
    assert_eq!(preempted[0].a.output_tokens.len(), 2);
    // Its GPU state was freed, its KV offered to the prefix cache first.
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![1]);
    assert_eq!(model.cached_seqs.load(Ordering::SeqCst), 1);
    // STREAM CONTRACT: the victim's channel is OPEN and EMPTY — no
    // mid-stream Error frame, no Done. (The old code sent
    // StreamEvent::Error here: an HTTP-200 with silent truncation.)
    assert!(matches!(
        victim_rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    // History retained for the re-prefill: prompt + processed outputs.
    assert_eq!(preempted[0].tokens, vec![1, 2, 3, 4, 100]);
    assert_eq!(preempted[0].a.last_token, 101);
}

#[test]
fn non_kv_error_still_fails_the_whole_batch() {
    let model = PreemptStubModel {
        hard_error: Some("CUDA error 700: illegal memory access"),
        ..Default::default()
    };
    let (a0, mut rx0) = active_seq(0, 3);
    let (a1, mut rx1) = active_seq(1, 4);
    let mut active = vec![a0, a1];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    let logits =
        decode_batch_with_preemption(&model, &mut active, None, &mut swapped, &mut preempted);
    assert!(logits.is_none());
    assert!(active.is_empty() && preempted.is_empty() && swapped.is_empty());
    // Non-recoverable errors still reach the clients.
    assert!(rx0.try_recv().expect("response sent").is_err());
    assert!(rx1.try_recv().expect("response sent").is_err());
}

#[test]
fn a_refusal_after_part_of_the_batch_ran_fails_it_without_a_retry() {
    // The per-sequence highway loop: sequence 0 already ran this step when
    // sequence 1's block was refused. Preempting a victim and re-running the
    // batch would feed sequence 0 its last token a second time.
    let model = PreemptStubModel {
        partial_refusal_after: Some(1),
        ..Default::default()
    };
    let (a0, mut rx0) = active_seq(0, 3);
    let (a1, mut rx1) = active_seq(1, 4);
    let mut active = vec![a0, a1];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    let logits =
        decode_batch_with_preemption(&model, &mut active, None, &mut swapped, &mut preempted);
    assert!(logits.is_none());
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 1);
    assert!(active.is_empty() && preempted.is_empty() && swapped.is_empty());
    assert!(rx0.try_recv().expect("response sent").is_err());
    assert!(rx1.try_recv().expect("response sent").is_err());
}

#[test]
fn single_sequence_exhaustion_is_not_preemptible() {
    // With one sequence, preemption cannot free anything the survivor
    // needs — the existing terminal path is kept.
    let model = PreemptStubModel {
        hard_error: Some("KV cache exhausted: no free blocks"),
        ..Default::default()
    };
    let (a0, mut rx0) = active_seq(0, 3);
    let mut active = vec![a0];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    let logits =
        decode_batch_with_preemption(&model, &mut active, None, &mut swapped, &mut preempted);
    assert!(logits.is_none());
    assert!(preempted.is_empty());
    assert!(rx0.try_recv().expect("response sent").is_err());
}

#[test]
fn spill_pool_runs_only_for_models_that_resume_a_swap() {
    let resumable = PreemptStubModel::default();
    assert!(spill_pool_enabled(&resumable, 3));
    assert!(!spill_pool_enabled(&resumable, 0));
    // The trait default, which TransformerModel also reports for a GLM
    // semantic-index cache: its spill image carries no index pools or tails,
    // so a swapped-in sequence would select over another owner's keys.
    // Admission then waits for blocks and decode preemption requeues.
    let cannot = super::cancel_test_model::TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    };
    assert!(!spill_pool_enabled(&cannot, 3));
}
