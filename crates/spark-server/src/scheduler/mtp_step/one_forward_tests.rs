// SPDX-License-Identifier: AGPL-3.0-only

//! One forward per MTP step (`one_forward`): `step_mtp` driven against a
//! stub model that records every target forward it is asked for.

use super::super::step_mtp;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::{RespRx, test_seq};
use crate::scheduler::types::ActiveSeq;
use anyhow::{Result, bail};
use spark_model::traits::{Model, SequenceState};
use spark_runtime::gpu::DevicePtr;
use std::sync::Mutex;

/// What the stub saw.
#[derive(Default)]
struct Seen {
    /// `decode_verify_batched` calls: `(ks, tokens)`.
    batched: Vec<(Vec<usize>, Vec<u32>)>,
    /// Every other target forward, by name.
    other: Vec<&'static str>,
    /// `trim_proposer_state` calls.
    trims: usize,
    /// `commit_accepted_prefix` calls as `(num_accepted, k)`.
    commits: Vec<(usize, usize)>,
}

/// Bonus pick of batch member `i`; the scripted verify accepts every draft.
const BONUS: u32 = 900;
/// The drafter proposes this token at every depth.
const DRAFT: u32 = 700;
/// What a standalone decode picks.
const DECODED: u32 = 800;

struct Stub {
    /// Admit decode rows (`ks[i] == 1`), windows of up to 8 rows and a lone
    /// window past 4 rows, as qwen4_exp's exact lane does.
    decode_rows: bool,
    seen: Mutex<Seen>,
}

impl Stub {
    fn new(decode_rows: bool) -> Self {
        Self {
            decode_rows,
            seen: Mutex::new(Seen::default()),
        }
    }
    fn other(&self, name: &'static str) {
        self.seen.lock().unwrap().other.push(name);
    }
}

impl Model for Stub {
    fn can_batch_verify(&self, ks: &[usize]) -> bool {
        let (min, max) = if self.decode_rows { (1, 8) } else { (2, 4) };
        let lone_ok = ks.len() >= 2 || (self.decode_rows && ks.iter().all(|&k| k > 4));
        (1..=32).contains(&ks.len())
            && lone_ok
            && ks.iter().all(|k| (min..=max).contains(k))
            && ks.iter().any(|&k| k >= 2)
    }
    fn decode_verify_batched(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<Vec<u32>> {
        let mut picks = Vec::with_capacity(tokens.len());
        let mut off = 0;
        for (i, (seq, &k)) in seqs.iter_mut().zip(ks).enumerate() {
            let window = &tokens[off..off + k];
            seq.tokens.extend_from_slice(window);
            seq.seq_len += k;
            picks.extend_from_slice(&window[1..]);
            picks.push(BONUS + i as u32);
            off += k;
        }
        let mut seen = self.seen.lock().unwrap();
        seen.batched.push((ks.to_vec(), tokens.to_vec()));
        Ok(picks)
    }
    fn stash_verify_hidden_rows(&self, _rows: &[usize], _st: u64) -> Result<()> {
        Ok(())
    }
    fn save_hidden_for_mtp_from_stash(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn mtp_propose_batch_max(&self) -> usize {
        8
    }
    fn run_mtp_propose_batched(
        &self,
        tokens: &[u32],
        _positions: &[usize],
        _stash_idx: &[usize],
        num_drafts: usize,
        _seqs: &mut [&mut SequenceState],
        _st: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
        _masks: Option<&[Option<Vec<i32>>]>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        if let Some(c) = out_conf {
            *c = vec![vec![0.0; num_drafts]; tokens.len()];
        }
        Ok(Some(vec![vec![DRAFT; num_drafts]; tokens.len()]))
    }
    fn prefill(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        bail!("unused")
    }
    fn decode(&self, _t: u32, s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        self.other("decode");
        s.seq_len += 1;
        Ok(DevicePtr::NULL)
    }
    fn prefill_chunk(
        &self,
        _t: &[u32],
        _s: &mut SequenceState,
        _cs: usize,
        _cl: usize,
        _last: bool,
        _st: u64,
    ) -> Result<DevicePtr> {
        bail!("unused")
    }
    fn decode_batch(
        &self,
        _t: &[u32],
        seqs: &mut [&mut SequenceState],
        _st: u64,
    ) -> Result<DevicePtr> {
        self.other("decode_batch");
        for s in seqs.iter_mut() {
            s.seq_len += 1;
        }
        Ok(DevicePtr::NULL)
    }
    fn decode_verify(&self, _t: &[u32], _s: &mut SequenceState, _st: u64) -> Result<Vec<u32>> {
        self.other("decode_verify");
        bail!("a second forward")
    }
    fn generate_speculative(
        &self,
        _p: &[u32],
        _params: &spark_runtime::sampler::SamplingParams,
        _n: usize,
    ) -> Result<spark_model::engine::GenerateResult> {
        bail!("unused")
    }
    fn decode_verify_graphed(
        &self,
        _t: &[u32; 2],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 2]> {
        self.other("verify_k2");
        bail!("a second forward")
    }
    fn decode_verify_graphed_k3(
        &self,
        _t: &[u32; 3],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 3]> {
        self.other("verify_k3");
        bail!("a second forward")
    }
    fn decode_verify_graphed_k4(
        &self,
        _t: &[u32; 4],
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<[u32; 4]> {
        self.other("verify_k4");
        bail!("a second forward")
    }
    fn run_mtp_propose(
        &self,
        _t: u32,
        _p: usize,
        _s: &mut SequenceState,
        _st: u64,
    ) -> Result<Option<u32>> {
        bail!("unused")
    }
    fn run_mtp_propose_multi(
        &self,
        _t: u32,
        _p: usize,
        n: usize,
        _s: &mut SequenceState,
        _st: u64,
        _bm: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        Ok(vec![DRAFT; n])
    }
    fn trim_proposer_state(&self, _s: &mut SequenceState, _n: usize, _st: u64) -> Result<()> {
        self.seen.lock().unwrap().trims += 1;
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
        Ok(DECODED)
    }
    fn argmax_batch(&self, _p: DevicePtr, n: usize, _st: u64) -> Result<Vec<u32>> {
        Ok(vec![DECODED; n])
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
        true
    }
    fn has_self_speculative(&self) -> bool {
        false
    }
    fn decode_draft(&self, _t: u32, _s: &mut SequenceState, _st: u64) -> Result<DevicePtr> {
        bail!("unused")
    }
    fn cache_sequence(&self, _s: &SequenceState) {}
    fn free_sequence(&self, _s: &mut SequenceState) -> Result<()> {
        Ok(())
    }
    fn compact_sequence(&self, _s: &mut SequenceState, _slot: usize) -> Result<bool> {
        Ok(false)
    }
    fn detach_slot_for_reuse(&self, _s: &mut SequenceState) {}
    fn save_hidden_for_mtp(&self, _i: usize, _st: u64) -> Result<()> {
        Ok(())
    }
    fn commit_accepted_prefix(&self, _s: &mut SequenceState, n: usize, k: usize) -> Result<()> {
        self.seen.lock().unwrap().commits.push((n, k));
        Ok(())
    }
}

/// Live, greedy, penalty-free sequences, one per `(last token, drafts)`, on
/// pool slots 0.., with their response receivers (a dropped receiver reads
/// as a client gone, and the emit would finish the sequence).
fn seqs(spec: Vec<(u32, Vec<u32>)>) -> (Vec<ActiveSeq>, Vec<RespRx>) {
    spec.into_iter()
        .enumerate()
        .map(|(slot, (last, drafts))| {
            let (mut a, rx) = test_seq(vec![last], 50, None, 40);
            a.finished = false;
            a.min_tokens = 0;
            a.lz_penalty = 0.0;
            a.seq.slot_idx = slot;
            a.seq.tokens = (0..40).collect();
            a.pending_drafts = drafts;
            (a, rx)
        })
        .unzip()
}

fn step(model: &Stub, active: &mut [ActiveSeq]) {
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    step_mtp(model, active, &sched, 3, &ctx, false);
}

/// The C=8 shape that measured `serial=0.60`: one sequence at full depth,
/// one the confidence stop left with a single draft, one with none. All
/// three ride ONE batched verify; nothing else touches the target.
#[test]
fn every_sequence_rides_one_forward() {
    let model = Stub::new(true);
    let (mut active, _rx) = seqs(vec![
        (10, vec![11, 12, 13]),
        (20, vec![21]),
        (30, Vec::new()),
    ]);
    step(&model, &mut active);
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.other, Vec::<&str>::new(), "no second forward");
    assert_eq!(
        seen.batched,
        vec![(vec![4, 2, 1], vec![10, 11, 12, 13, 20, 21, 30])]
    );
    // Each sequence emitted its accepted drafts and the bonus row.
    assert_eq!(active[0].output_tokens, vec![10, 11, 12, 13, BONUS]);
    assert_eq!(active[1].output_tokens, vec![20, 21, BONUS + 1]);
    assert_eq!(active[2].output_tokens, vec![30, BONUS + 2]);
    assert_eq!(active[2].seq.seq_len, 41, "the decode row advanced one");
    // The decode row commits its one row and trims no drafter row.
    assert_eq!(seen.commits, vec![(4, 4), (2, 2), (1, 1)]);
    assert_eq!(seen.trims, 2);
    // And every sequence proposes for the next step, the decode row too.
    for a in &active {
        assert_eq!(a.pending_drafts, vec![DRAFT; 3]);
    }
}

/// A single drafted sequence plus a draftless one is still one forward
/// (it used to be a bootstrap decode plus a per-sequence verify).
#[test]
fn one_drafted_sequence_carries_the_decode_row() {
    let model = Stub::new(true);
    let (mut active, _rx) = seqs(vec![(10, vec![11, 12]), (20, Vec::new())]);
    step(&model, &mut active);
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.other, Vec::<&str>::new());
    assert_eq!(seen.batched, vec![(vec![3, 1], vec![10, 11, 12, 20])]);
}

/// Where the model does not verify decode rows, the conf-stopped sequence
/// still rides the batched verify and the draftless one bootstraps on its
/// own, as before.
#[test]
fn without_decode_rows_the_draftless_sequence_bootstraps() {
    let model = Stub::new(false);
    let (mut active, _rx) = seqs(vec![
        (10, vec![11, 12, 13]),
        (20, vec![21]),
        (30, Vec::new()),
    ]);
    step(&model, &mut active);
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.other, vec!["decode"]);
    assert_eq!(
        seen.batched,
        vec![(vec![4, 2], vec![10, 11, 12, 13, 20, 21])]
    );
    assert_eq!(active[2].output_tokens, vec![30, DECODED]);
}

/// Sequences without drafts alone are a bootstrap step, never a verify.
#[test]
fn decode_rows_never_make_a_verify_on_their_own() {
    let model = Stub::new(true);
    let (mut active, _rx) = seqs(vec![(10, Vec::new()), (20, Vec::new())]);
    step(&model, &mut active);
    let seen = model.seen.lock().unwrap();
    assert!(seen.batched.is_empty());
    assert_eq!(seen.other, vec!["decode_batch"], "the batched bootstrap");
    assert_eq!(active[0].output_tokens, vec![10, DECODED]);
}

/// A step under `ATLAS_MTP_DYNAMIC_DEPTH` with a 7-draft ceiling.
fn deep_step(model: &Stub, active: &mut [ActiveSeq]) {
    let mut levers = crate::scheduler::levers::SchedLevers::defaults();
    levers.mtp_dynamic_depth = true;
    let sched = SchedCtx::new(
        crate::scheduler::vocab_masks::VocabMasks::default(),
        std::sync::Arc::new(levers),
        std::sync::Arc::new(crate::scheduler::snapshot::SnapshotCell::default()),
        crate::scheduler::limits::SchedLimits::NONE,
        crate::scheduler::helpers::WatchdogParams::default(),
    );
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    step_mtp(model, active, &sched, 7, &ctx, false);
}

/// Promote a request's dynamic ceiling to `depth` (from the floor 3, +2 a
/// window at full acceptance).
fn promote(a: &mut ActiveSeq, windows: usize) {
    let levers = crate::scheduler::mtp_deep_depth::DepthLevers {
        adapt: false,
        deep: true,
    };
    for _ in 0..windows {
        let d = a.mtp_acct.depth_drafts(7, levers);
        for _ in 0..crate::scheduler::mtp_deep_depth::WINDOW {
            a.mtp_acct.record_depth_verify(d, d, 7, levers);
        }
    }
}

/// A lone request whose ceiling the controller raised to 5 verifies 6 rows
/// in ONE batched forward (the only verify serving more than 4 rows), and
/// refills to its ceiling.
#[test]
fn a_lone_deep_window_rides_the_batched_verify() {
    let model = Stub::new(true);
    let (mut active, _rx) = seqs(vec![(10, vec![11, 12, 13, 14, 15, 16, 17])]);
    promote(&mut active[0], 1);
    deep_step(&model, &mut active);
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.other, Vec::<&str>::new());
    assert_eq!(
        seen.batched,
        vec![(vec![6], vec![10, 11, 12, 13, 14, 15])],
        "cut to its ceiling of 5 drafts"
    );
    assert_eq!(active[0].output_tokens, vec![10, 11, 12, 13, 14, 15, BONUS]);
    assert_eq!(seen.commits, vec![(6, 6)]);
}

/// At its floor a lone request keeps the 4-row verify of its own.
#[test]
fn a_lone_request_at_the_floor_keeps_its_own_verify() {
    let model = Stub::new(true);
    let (mut active, _rx) = seqs(vec![(10, vec![11, 12, 13, 14, 15])]);
    deep_step(&model, &mut active);
    let seen = model.seen.lock().unwrap();
    assert!(seen.batched.is_empty());
    assert_eq!(seen.other, vec!["verify_k4"]);
}

/// Each member verifies its own ceiling, and the batch fits the deep row
/// budget (5 rows a sequence at n=2): 8 + 4 rows trimmed to 6 + 4, which
/// D-Cut (n <= 8, ratio 0.75) may prune further.
#[test]
fn deep_members_verify_their_own_ceiling_within_the_row_budget() {
    let model = Stub::new(true);
    let (mut active, _rx) = seqs(vec![
        (10, vec![11, 12, 13, 14, 15, 16, 17]),
        (20, vec![21, 22, 23, 24, 25, 26, 27]),
    ]);
    promote(&mut active[0], 2); // ceiling 7
    deep_step(&model, &mut active);
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.other, Vec::<&str>::new());
    let (ks, _) = &seen.batched[0];
    assert!(
        ks.iter().sum::<usize>() <= 10,
        "the n=2 deep row budget: {ks:?}"
    );
    assert!(
        ks[0] > 4,
        "the promoted member verifies past 4 rows: {ks:?}"
    );
    assert!(ks[1] <= 4, "the floor member stays at its ceiling: {ks:?}");
}
