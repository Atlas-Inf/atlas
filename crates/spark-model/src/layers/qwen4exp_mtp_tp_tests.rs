// SPDX-License-Identifier: AGPL-3.0-only

//! Draft TP lockstep: whatever the head's propose does after the announce,
//! the two ranks issue the same swaps, in the same order, of the same sizes.
//! Each rank runs on its own mock device; the mock pair records every
//! exchange a rank issues.

use std::sync::Mutex;

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;
use crate::weight_map::QuantizedWeight;

const VOCAB: usize = 4101;
const HIDDEN: usize = 64;

/// One rank of the pair: the bytes of every exchange it issued, in order.
struct Pair {
    rank: usize,
    /// The one-shot cap the logits exchange chunks by (0: one exchange).
    max: usize,
    sizes: Mutex<Vec<usize>>,
}

impl Pair {
    fn new(rank: usize, max: usize) -> Self {
        Self {
            rank,
            max,
            sizes: Mutex::default(),
        }
    }
    fn sizes(&self) -> Vec<usize> {
        self.sizes.lock().unwrap().clone()
    }
}

impl CommBackend for Pair {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn peer_exchange_async(&self, _: u64, _: u64, bytes: usize, _: u64) -> Result<()> {
        self.sizes.lock().unwrap().push(bytes);
        Ok(())
    }
    fn supports_peer_exchange_async(&self) -> bool {
        true
    }
    fn capturable_all_reduce_max_bytes(&self) -> usize {
        self.max
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        bail!("unexpected broadcast")
    }
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected all-gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        bail!("unexpected reduce-scatter")
    }
    fn barrier(&self) -> Result<()> {
        bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        bail!("unexpected receive")
    }
}

fn lm_head(gpu: &MockGpuBackend) -> DenseWeight {
    DenseWeight {
        weight: gpu.alloc(VOCAB * HIDDEN * BF16).unwrap(),
    }
}

/// The head's side: its draft head, split state and the buffers a propose
/// position reads and writes.
struct Head {
    gpu: MockGpuBackend,
    draft: DraftHead,
    tp: DraftTp,
    h_out: DevicePtr,
    logits: DevicePtr,
}

impl Head {
    fn new() -> Self {
        let gpu = MockGpuBackend::new();
        let draft = DraftHead::build(&lm_head(&gpu), VOCAB, HIDDEN, 0, &gpu).unwrap();
        let tp = DraftTp::new(&draft, HIDDEN, 0, &gpu).unwrap();
        let h_out = gpu.alloc(PROPOSE_BATCH_MAX * HIDDEN * BF16).unwrap();
        let logits = gpu.alloc(PROPOSE_BATCH_MAX * VOCAB * BF16).unwrap();
        Self {
            gpu,
            draft,
            tp,
            h_out,
            logits,
        }
    }

    fn run<'a>(&'a self, comm: &'a Pair, plan: Plan) -> TpRun<'a> {
        TpRun {
            tp: &self.tp,
            comm,
            gpu: &self.gpu,
            plan,
            stream: 0,
            issued: Cell::new(0),
        }
    }

    /// `positions` positions of the draft head, then the run ends (dropped).
    fn propose(&self, comm: &Pair, plan: Plan, positions: usize) -> Result<()> {
        let run = self.run(comm, plan);
        for _ in 0..positions {
            run.head(&self.draft, KernelHandle(0xDEAD), self.h_out, self.logits)?;
        }
        Ok(())
    }
}

/// The worker's walk of `plan` over a pair capped at `max`.
fn worker_sizes(plan: Plan, max: usize) -> Vec<usize> {
    let gpu = MockGpuBackend::new();
    let assist = Qwen4ExpDraftAssist::new(&lm_head(&gpu), VOCAB, HIDDEN, 0, &gpu).unwrap();
    let comm = Pair::new(1, max);
    assist.serve(&gpu, &comm, 0, plan.word().unwrap()).unwrap();
    comm.sizes()
}

#[test]
fn the_switch_parses_on_off_and_refuses_the_rest() {
    assert!(!parse(None).unwrap());
    assert!(!parse(Some(" ")).unwrap());
    assert!(!parse(Some("0")).unwrap());
    assert!(parse(Some("1")).unwrap());
    assert!(parse(Some("on")).is_err());
}

#[test]
fn a_plan_survives_its_word_and_a_stray_word_is_refused() {
    for n in 1..=PROPOSE_BATCH_MAX {
        for drafts in 1..=PROPOSE_BATCH_MAX_DRAFTS {
            let plan = Plan { n, drafts };
            assert_eq!(Plan::from_word(plan.word().unwrap()).unwrap(), plan);
        }
    }
    assert!(Plan { n: 0, drafts: 3 }.word().is_err());
    assert!(Plan { n: 9, drafts: 3 }.word().is_err());
    assert!(Plan { n: 8, drafts: 9 }.word().is_err());
    // A GLM announce word (rows | ctx << 12), a token, a zero.
    for stray in [8, 8 | 3 << 12, 151_000, 0] {
        assert!(Plan::from_word(stray).is_err(), "{stray:#x}");
    }
    let word = Plan { n: 8, drafts: 3 }.word().unwrap();
    assert!(Plan::from_word(word | 1 << 16).is_err());
}

#[test]
fn every_position_swaps_its_rows_then_its_logits() {
    let swaps: Vec<_> = Plan { n: 4, drafts: 3 }.swaps().collect();
    use Swap::*;
    assert_eq!(swaps, [Hidden, Logits, Hidden, Logits, Hidden, Logits]);
}

#[test]
fn both_ranks_issue_the_same_swaps_for_a_whole_propose() {
    let width = Split::new(VOCAB).unwrap().width();
    // No cap: one exchange a swap. A cap of 3 rows: the logits in chunks.
    for (max, chunks) in [(0, 1), (3 * width * BF16, 3)] {
        let plan = Plan { n: 8, drafts: 3 };
        let head = Head::new();
        let comm = Pair::new(0, max);
        head.propose(&comm, plan, plan.drafts).unwrap();
        let sizes = comm.sizes();
        assert_eq!(sizes, worker_sizes(plan, max), "cap {max}");
        assert_eq!(sizes.len(), plan.drafts * (1 + chunks));
        assert_eq!(sizes[0], 8 * HIDDEN * BF16);
        assert_eq!(sizes[1..=chunks].iter().sum::<usize>(), 8 * width * BF16);
    }
}

#[test]
fn a_propose_that_stops_early_still_issues_every_swap() {
    let plan = Plan { n: 5, drafts: 3 };
    let want = worker_sizes(plan, 0);
    // Every exit: the fallback before any position, an error after one or
    // two positions, an error between a position's two swaps.
    for positions in 0..plan.drafts {
        let head = Head::new();
        let comm = Pair::new(0, 0);
        head.propose(&comm, plan, positions).unwrap();
        assert_eq!(comm.sizes(), want, "stopped after {positions} positions");
    }
    let head = Head::new();
    let comm = Pair::new(0, 0);
    {
        let run = head.run(&comm, plan);
        run.issue(Swap::Hidden, head.h_out).unwrap();
    }
    assert_eq!(comm.sizes(), want, "stopped between a position's swaps");
}

#[test]
fn a_swap_out_of_order_is_refused_and_the_plan_still_completes() {
    let plan = Plan { n: 2, drafts: 2 };
    let head = Head::new();
    let comm = Pair::new(0, 0);
    {
        let run = head.run(&comm, plan);
        assert!(run.issue(Swap::Logits, DevicePtr::NULL).is_err());
        run.head(&head.draft, KernelHandle(0xDEAD), head.h_out, head.logits)
            .unwrap();
        run.finish().unwrap();
    }
    assert_eq!(comm.sizes(), worker_sizes(plan, 0));
}

#[test]
fn the_worker_refuses_a_word_it_cannot_walk() {
    let gpu = MockGpuBackend::new();
    let assist = Qwen4ExpDraftAssist::new(&lm_head(&gpu), VOCAB, HIDDEN, 0, &gpu).unwrap();
    let comm = Pair::new(1, 0);
    assert!(assist.serve(&gpu, &comm, 0, 8).is_err());
    assert!(comm.sizes().is_empty());
    // Only rank 1 serves.
    let head_rank = Pair::new(0, 0);
    let word = Plan { n: 2, drafts: 1 }.word().unwrap();
    assert!(assist.serve(&gpu, &head_rank, 0, word).is_err());
}

#[test]
fn the_ranks_project_complementary_rows_of_the_draft_head() {
    let geom = Split::new(VOCAB).unwrap();
    assert_eq!(geom.start(0), 0);
    assert!(geom.start(1) <= geom.width());
    assert_eq!(geom.start(1) + geom.width(), VOCAB);
}

#[test]
fn a_row_view_offsets_every_per_row_pointer() {
    let base = QuantizedWeight {
        weight: DevicePtr(0x1000),
        weight_scale: DevicePtr(0x9000),
        weight_scale_2: 0.25,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    };
    let v = base.rows_from(10, 2560);
    assert_eq!(v.weight, DevicePtr(0x1000 + 10 * 1280));
    assert_eq!(v.weight_scale, DevicePtr(0x9000 + 10 * 160));
    assert_eq!(v.weight_scale_2, 0.25);
    assert!(v.input_scale.is_null() && v.weight_scale_2_vec.is_null());
    let prs = QuantizedWeight {
        weight_scale_2_vec: DevicePtr(0x20),
        ..base
    };
    assert_eq!(
        prs.rows_from(3, 2560).weight_scale_2_vec,
        DevicePtr(0x20 + 12)
    );
}
