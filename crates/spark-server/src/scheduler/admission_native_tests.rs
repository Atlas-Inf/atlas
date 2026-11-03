// SPDX-License-Identifier: AGPL-3.0-only

//! Reproduce the near-1M shared-prefix queueing bug through the real gate.
use super::*;
use crate::scheduler::test_support::{RespRx, test_owned_seq, test_request};
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use std::sync::Arc;

const BS: usize = 16;
const PREFIX: usize = 1_046_976;
const PROMPT: usize = PREFIX + 12;
const BUDGET: usize = 384;
const LIMIT: usize = 1_048_576;
const POOL: usize = 65_700;

struct NativePool {
    enabled: bool,
    private_free: usize,
}

impl Model for NativePool {
    fn prefill_chunk(
        &self,
        _: &[u32],
        _: &mut SequenceState,
        _: usize,
        _: usize,
        _: bool,
        _: u64,
    ) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode_batch(&self, _: &[u32], _: &mut [&mut SequenceState], _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode_verify(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn generate_speculative(
        &self,
        _: &[u32],
        _: &spark_runtime::sampler::SamplingParams,
        _: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        unreachable!()
    }
    fn decode_verify_graphed(
        &self,
        _: &[u32; 2],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 2]> {
        unreachable!()
    }
    fn decode_verify_graphed_k3(
        &self,
        _: &[u32; 3],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 3]> {
        unreachable!()
    }
    fn decode_verify_graphed_k4(
        &self,
        _: &[u32; 4],
        _: &mut SequenceState,
        _: u64,
    ) -> Result<[u32; 4]> {
        unreachable!()
    }
    fn run_mtp_propose(
        &self,
        _: u32,
        _: usize,
        _: &mut SequenceState,
        _: u64,
    ) -> Result<Option<u32>> {
        unreachable!()
    }
    fn run_mtp_propose_multi(
        &self,
        _: u32,
        _: usize,
        _: usize,
        _: &mut SequenceState,
        _: u64,
        _: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn trim_proposer_state(&self, _: &mut SequenceState, _: usize, _: u64) -> Result<()> {
        unreachable!()
    }

    fn native_mtp_enabled(&self) -> bool {
        self.enabled
    }
    fn is_ep(&self) -> bool {
        true
    }
    fn num_total_blocks(&self) -> usize {
        POOL
    }
    fn native_prefix_admission_blocks(
        &self,
        owner: &SequenceState,
        tokens: &[u32],
        _: u64,
        need: usize,
    ) -> Result<Option<usize>> {
        let shared = owner
            .prefix_ref_tokens
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count()
            / BS;
        Ok((need <= self.private_free && shared * BS >= owner.marconi_skip_to).then_some(shared))
    }
    fn prefill(&self, _: &[u32], _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn decode(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn vocab_size(&self) -> usize {
        1
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        unreachable!()
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        unreachable!()
    }
    fn copy_logits_to_host(&self, _: DevicePtr, _: &mut [u8]) -> Result<()> {
        unreachable!()
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn argmax_on_device(&self, _: DevicePtr, _: u64) -> Result<u32> {
        unreachable!()
    }
    fn argmax_batch(&self, _: DevicePtr, _: usize, _: u64) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        DevicePtr::NULL
    }
    fn checkpoint_ssm_states(&self, _: &mut SequenceState) -> Result<()> {
        unreachable!()
    }
    fn rollback_ssm_states(&self, _: &mut SequenceState, _: usize) -> Result<()> {
        unreachable!()
    }
    fn has_proposer(&self) -> bool {
        true
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _: u32, _: &mut SequenceState, _: u64) -> Result<DevicePtr> {
        unreachable!()
    }
    fn cache_sequence(&self, _: &SequenceState) {
        unreachable!()
    }
    fn free_sequence(&self, _: &mut SequenceState) -> Result<()> {
        unreachable!()
    }
    fn compact_sequence(&self, _: &mut SequenceState, _: usize) -> Result<bool> {
        unreachable!()
    }
    fn detach_slot_for_reuse(&self, _: &mut SequenceState) {
        unreachable!()
    }
    fn save_hidden_for_mtp(&self, _: usize, _: u64) -> Result<()> {
        unreachable!()
    }
}

fn owner(slot: usize) -> (ActiveSeq, RespRx) {
    let mut seq = SequenceState::host_only(slot);
    seq.tokens = vec![7; PROMPT];
    seq.seq_len = PROMPT;
    seq.prompt_len = PROMPT;
    seq.cached_prefix_tokens = PREFIX;
    seq.cached_prefix_blocks = PREFIX / BS;
    seq.prefix_ref_tokens = vec![7; PREFIX];
    seq.prefix_lookup_applied = true;
    seq.prefix_lookup_skip = true;
    seq.marconi_skip_to = PREFIX - BS;
    seq.block_table = (0..(PREFIX / BS) as u32).collect();
    seq.block_table.push((PREFIX / BS + slot) as u32);
    let (mut active, rx) = test_owned_seq(seq, vec![9], BUDGET, None);
    active.finished = false;
    (active, rx)
}

fn request() -> (InferenceRequest, RespRx) {
    let (response_tx, rx) = tokio::sync::oneshot::channel();
    let mut request = test_request!(Blocking, response_tx,);
    if let InferenceRequest::Blocking {
        prompt_tokens,
        max_tokens,
        ..
    } = &mut request
    {
        *prompt_tokens = Arc::new(vec![7; PROMPT]);
        *max_tokens = BUDGET;
    }
    (request, rx)
}

fn gate(
    model: &NativePool,
    active: &[ActiveSeq],
    requests: Vec<InferenceRequest>,
) -> (usize, usize) {
    let pending = Arc::new((
        Mutex::new(PendingQueue {
            requests: vec![],
            closed: false,
            rotations: vec![],
        }),
        Condvar::new(),
    ));
    let admitted = gate_admissions(
        model,
        &pending,
        requests,
        active,
        &[],
        &[],
        &[],
        LIMIT,
        LIMIT,
        BS,
    );
    let queued = pending.0.lock().requests.len();
    (admitted.len(), queued)
}

#[test]
fn warm_native_near_1m_reaches_c4_one_follower_per_tick() {
    let model = NativePool {
        enabled: true,
        private_free: 256,
    };
    let mut active = Vec::new();
    let mut receivers = Vec::new();
    let (first, rx) = owner(0);
    active.push(first);
    receivers.push(rx);
    for n in 1..4 {
        let (requests, rx): (Vec<_>, Vec<_>) = (n..4).map(|_| request()).unzip();
        receivers.extend(rx);
        assert_eq!(gate(&model, &active, requests), (1, 3 - n));
        let (next, rx) = owner(n);
        active.push(next);
        receivers.push(rx);
    }
}

#[test]
fn cold_identical_prompts_do_not_get_speculative_sharing_credit() {
    let model = NativePool {
        enabled: true,
        private_free: 256,
    };
    let (requests, _rx): (Vec<_>, Vec<_>) = (0..4).map(|_| request()).unzip();
    assert_eq!(gate(&model, &[], requests), (1, 3));
}

#[test]
fn rejected_oversized_ep_request_never_enters_native_follower_probe() {
    let model = NativePool {
        enabled: true,
        private_free: 256,
    };
    let (mut req, mut rx) = request();
    if let InferenceRequest::Blocking { prompt_tokens, .. } = &mut req {
        *prompt_tokens = Arc::new(vec![7; POOL * BS + 1]);
    }
    assert_eq!(gate(&model, &[], vec![req]), (0, 0));
    assert!(
        rx.try_recv()
            .unwrap()
            .err()
            .expect("oversized request must fail")
            .to_string()
            .contains("pool has")
    );
}

#[test]
fn missing_private_continuation_room_keeps_follower_queued() {
    let model = NativePool {
        enabled: true,
        private_free: 1,
    };
    let (first, _rx) = owner(0);
    let (req, _rrx) = request();
    assert_eq!(gate(&model, &[first], vec![req]), (0, 1));
}

#[test]
fn private_pool_reserves_existing_owners_future_not_only_candidate_tail() {
    // Candidate needs29blocks, but the owner also reserves27. Fiftyfive
    // spare blocks cannot promise both continuations; fiftysix can.
    for (free, expected) in [(55, (0, 1)), (56, (1, 0))] {
        let model = NativePool {
            enabled: true,
            private_free: free,
        };
        let (first, _rx) = owner(0);
        let (req, _rrx) = request();
        assert_eq!(gate(&model, &[first], vec![req]), expected);
    }
}

#[test]
fn non_native_model_keeps_conservative_admission() {
    let model = NativePool {
        enabled: false,
        private_free: 256,
    };
    let (first, _rx) = owner(0);
    let (req, _rrx) = request();
    assert_eq!(gate(&model, &[first], vec![req]), (0, 1));
}

#[test]
fn different_prompt_or_logprob_collection_gets_no_credit() {
    let model = NativePool {
        enabled: true,
        private_free: 256,
    };
    let (first, _rx) = owner(0);
    for logprobs in [false, true] {
        let (mut req, _rrx) = request();
        if let InferenceRequest::Blocking {
            prompt_tokens,
            prompt_logprobs,
            ..
        } = &mut req
        {
            if logprobs {
                *prompt_logprobs = Some(1);
            } else {
                Arc::make_mut(prompt_tokens)[0] = 99;
            }
        }
        assert_eq!(
            gate(&model, std::slice::from_ref(&first), vec![req]),
            (0, 1)
        );
    }
}

#[test]
fn distinct_short_branch_tail_can_share_the_proven_causal_prefix() {
    let model = NativePool {
        enabled: true,
        private_free: 256,
    };
    let (first, _rx) = owner(0);
    let (mut req, _rrx) = request();
    if let InferenceRequest::Blocking { prompt_tokens, .. } = &mut req {
        Arc::make_mut(prompt_tokens)[PROMPT - 1] = 99;
    }
    assert_eq!(gate(&model, &[first], vec![req]), (1, 0));
}

#[test]
fn shared_prefix_keeps_each_writable_tail_and_future_reservation() {
    let (a, _rx) = owner(0);
    let (b, _rx2) = owner(1);
    let demand = || SeqDemand {
        current_tokens: PROMPT + 1,
        budget_tokens: BUDGET,
    };
    let demands = [demand(), demand()];
    let want = PREFIX / BS + 2 * (seq_commitment_blocks(&demand(), LIMIT, LIMIT, BS) - PREFIX / BS);
    assert_eq!(
        native::resident_commitment(&demands, [&a.seq, &b.seq].into_iter(), LIMIT, LIMIT, BS),
        want
    );
}

#[test]
fn shared_prefix_does_not_erase_parked_sequence_reservations() {
    let (a, _rx) = owner(0);
    let demand = || SeqDemand {
        current_tokens: PROMPT + 1,
        budget_tokens: BUDGET,
    };
    let demands = [demand(), demand()];
    assert_eq!(
        native::resident_commitment(&demands, [&a.seq].into_iter(), LIMIT, LIMIT, BS),
        2 * seq_commitment_blocks(&demand(), LIMIT, LIMIT, BS)
    );
}

fn prefill(slot: usize) -> (PrefillInProgress, RespRx) {
    let (mut active, rx) = owner(slot);
    active.seq.seq_len = 8192; // cached chunk traversal, predictor is already restored
    let p = super::super::prefill_a_step_params::build_prefill_in_progress(
        Arc::new(vec![7; PROMPT]),
        0,
        active.seq,
        8192,
        BUDGET,
        0,
        vec![2],
        active.sink,
        None,
        std::time::Instant::now(),
        0.0,
        0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        vec![],
        false,
        None,
        None,
        0,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        None,
    );
    (p, rx)
}

#[test]
fn already_restored_prefill_can_acquire_the_next_follower_without_serializing_decode() {
    let model = NativePool {
        enabled: true,
        private_free: 256,
    };
    let (p, _rx) = prefill(0);
    let (req, _rrx) = request();
    let demands = [SeqDemand {
        current_tokens: PROMPT,
        budget_tokens: BUDGET,
    }];
    let committed = native::resident_commitment(&demands, [&p.seq].into_iter(), LIMIT, LIMIT, BS);
    assert!(native::admit_follower(
        &model,
        &req,
        &[],
        &[p],
        committed,
        POOL,
        LIMIT,
        LIMIT,
        BS
    ));
}

#[test]
fn uninitialized_prefill_cannot_promise_target_or_predictor_sharing() {
    let model = NativePool {
        enabled: true,
        private_free: 256,
    };
    let (mut p, _rx) = prefill(0);
    p.seq.prefix_lookup_applied = false;
    let (req, _rrx) = request();
    assert!(!native::admit_follower(
        &model,
        &req,
        &[],
        &[p],
        65_461,
        POOL,
        LIMIT,
        LIMIT,
        BS
    ));
}
