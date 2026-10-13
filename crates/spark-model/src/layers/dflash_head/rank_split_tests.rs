// SPDX-License-Identifier: AGPL-3.0-only

//! The rank split's plan, walk and swaps on a host: both ranks issue the
//! same swaps in the same order, the halves cover every row once, and the
//! joined halves land where the unsplit launch writes whole rows.

use std::cell::RefCell;
use std::collections::VecDeque;

use anyhow::bail;
use spark_runtime::gpu::mock::MockGpuBackend;

use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

use super::super::DflashScratch;
use super::super::free_state_tests::zero_scratch;
use super::exec::row_range;
use super::*;
use crate::weight_map::QuantizedWeight;

fn geometry(parts: Parts) -> Geometry {
    Geometry {
        gamma: 3,
        hidden: 64,
        inter: 160,
        vocab: 98,
        layers: 2,
        parts,
        ctx_in: 0,
        ctx_kv: 0,
        ctx: CtxRows::default(),
    }
}

const ALL: Parts = Parts {
    mlp: true,
    head: true,
};
const MLP: Parts = Parts {
    mlp: true,
    head: false,
};
const HEAD: Parts = Parts {
    mlp: false,
    head: true,
};

#[test]
fn the_switch_names_its_parts_and_refuses_anything_else() {
    for off in [None, Some(""), Some("0"), Some(" ")] {
        assert_eq!(Parts::parse(off).unwrap(), None);
    }
    assert_eq!(Parts::parse(Some("1")).unwrap(), Some(ALL));
    assert_eq!(Parts::parse(Some("mlp")).unwrap(), Some(MLP));
    assert_eq!(Parts::parse(Some("head")).unwrap(), Some(HEAD));
    assert_eq!(Parts::parse(Some("head, mlp")).unwrap(), Some(ALL));
    for bad in ["2", "true", "mlp,tail", "on"] {
        assert!(Parts::parse(Some(bad)).is_err(), "{bad}");
    }
    // Every value both ranks may disagree on reads differently.
    let words: Vec<u64> = [None, Some(MLP), Some(HEAD), Some(ALL)]
        .into_iter()
        .map(Parts::word)
        .collect();
    assert_eq!(words, [0, 1, 2, 3]);
}

#[test]
fn both_ranks_issue_the_same_swaps_in_the_same_order() {
    for parts in [ALL, MLP, HEAD] {
        let g = geometry(parts);
        let (head, worker) = (g.steps(0), g.steps(1));
        assert_eq!(swaps_of(&head), swaps_of(&worker), "{parts:?}");
        assert_eq!(swaps_of(&head), g.swaps());
        let expect = parts.mlp as usize * 3 * g.layers + parts.head as usize * 2;
        assert_eq!(g.swaps().len(), expect);
        assert_eq!(run_count(&head), expect + 1);
        // The worker has nothing to run before the head's first payload or
        // after its own last one.
        assert!(matches!(worker.first(), Some(Step::Swap(_))));
        assert!(matches!(worker.last(), Some(Step::Swap(_))));
    }
}

#[test]
fn the_worker_runs_only_the_shared_pieces_and_the_head_keeps_its_order() {
    use Piece::*;
    let pieces = |steps: Vec<Step>| -> Vec<Piece> {
        steps
            .into_iter()
            .filter_map(|s| match s {
                Step::Run(p) => Some(p),
                Step::Swap(_) => None,
            })
            .collect()
    };
    let g = geometry(ALL);
    assert_eq!(
        pieces(g.steps(1)),
        [GateUp(0), Down(0), GateUp(1), Down(1), Vocab]
    );
    // The unsplit order with each split projection in its place.
    assert_eq!(
        pieces(g.steps(0)),
        [
            Attention(0),
            Project(0),
            GateUp(0),
            Down(0),
            Residual(0),
            Attention(1),
            Project(1),
            GateUp(1),
            Down(1),
            Residual(1),
            Norm,
            Vocab,
            Select
        ]
    );
    assert_eq!(
        pieces(geometry(HEAD).steps(0)),
        [
            Attention(0),
            Post(0),
            Attention(1),
            Post(1),
            Norm,
            Vocab,
            Select
        ]
    );
    assert_eq!(pieces(geometry(MLP).steps(0)).last().copied(), Some(Tail));
    assert_eq!(pieces(geometry(HEAD).steps(1)), [Vocab]);
}

#[test]
fn the_shares_cover_every_output_row_once_and_start_on_a_cta() {
    let g = geometry(ALL);
    for n in [g.inter, g.hidden, g.vocab, 154_856, 12_288, 4096] {
        let (f0, r0) = half(n, 0);
        let (f1, r1) = half(n, 1);
        assert_eq!((f0, f0 + r0, f1 + r1), (0, f1, n));
        assert!(r0 > 0 && r1 >= r0 && f1 % 16 == 0, "{n}");
    }
    // The GLM-5.3 drafter: equal MLP halves, a head cut at CTA 4839.
    assert_eq!(half(12_288, 1), (6144, 6144));
    assert_eq!(half(154_856, 0), (0, 77_424));
    assert_eq!(half(154_856, 1), (77_424, 77_432));
    // A swap carries the larger share's width both ways.
    assert_eq!(g.bytes(Swap::Input(1)), 3 * 64 * 2);
    assert_eq!(g.bytes(Swap::Activation(0)), 3 * 80 * 2);
    assert_eq!(g.bytes(Swap::Output(0)), 3 * 32 * 2);
    assert_eq!(g.bytes(Swap::Hidden), 3 * 64 * 2);
    assert_eq!(half(g.vocab, 1), (48, 50));
    assert_eq!(g.bytes(Swap::Logits), 3 * 50 * 2);
}

#[test]
fn a_shape_the_shares_cannot_serve_is_refused() {
    assert!(geometry(ALL).validate().is_ok());
    for bad in [
        Geometry {
            vocab: 31,
            ..geometry(ALL)
        },
        Geometry {
            inter: 168,
            ..geometry(ALL)
        },
        Geometry {
            hidden: 72,
            ..geometry(ALL)
        },
        Geometry {
            parts: Parts {
                mlp: false,
                head: false,
            },
            ..geometry(ALL)
        },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
}

#[test]
fn a_row_range_starts_at_its_first_row_of_the_packed_twin() {
    let w = QuantizedWeight {
        weight: DevicePtr(0x1000),
        weight_scale: DevicePtr(0x9000),
        weight_scale_2: 0.25,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    };
    // 4096 columns: 2048 packed bytes and 256 group scales a row.
    let r = row_range(&w, 77428, 4096);
    assert_eq!(r.weight.0, 0x1000 + 77428 * 2048);
    assert_eq!(r.weight_scale.0, 0x9000 + 77428 * 256);
    assert_eq!(r.weight_scale_2, 0.25);
    // 16-byte loads stay aligned on any row of a K % 32 == 0 weight.
    assert_eq!((r.weight.0 - w.weight.0) % 16, 0);
}

/// Records what a walk asks for.
#[derive(Default)]
struct Log(RefCell<Vec<Step>>);

impl SplitOps for Log {
    fn piece(&self, piece: Piece) -> Result<()> {
        self.0.borrow_mut().push(Step::Run(piece));
        Ok(())
    }
    fn swap(&self, swap: Swap) -> Result<()> {
        self.0.borrow_mut().push(Step::Swap(swap));
        Ok(())
    }
}

#[test]
fn a_walk_issues_every_swap_once_whether_its_runs_are_eager_or_graphs() {
    let gpu = MockGpuBackend::new();
    let steps = geometry(ALL).steps(0);
    let swaps = swaps_of(&steps);

    let eager = Log::default();
    walk(&steps, Graphs::Eager, &gpu, 0, &eager).unwrap();
    assert_eq!(*eager.0.borrow(), steps, "eager is the plan itself");

    // Captured graphs replace the pieces; the swaps stay where they were.
    let handles = vec![GraphHandle(7); run_count(&steps)];
    let replay = Log::default();
    walk(&steps, Graphs::Replay(&handles), &gpu, 0, &replay).unwrap();
    assert_eq!(swaps_of(&replay.0.borrow()), swaps);
    assert_eq!(replay.0.borrow().len(), swaps.len());

    // A run whose capture came back empty replays eagerly.
    let mut partial = handles.clone();
    partial[0] = GraphHandle(0);
    let mixed = Log::default();
    walk(&steps, Graphs::Replay(&partial), &gpu, 0, &mixed).unwrap();
    assert_eq!(swaps_of(&mixed.0.borrow()), swaps);
    assert_eq!(
        mixed.0.borrow()[..2],
        [Step::Run(Piece::Attention(0)), Step::Run(Piece::Project(0))]
    );

    // The capture pass: one graph a run, every swap once, in order. (The
    // host backend captures nothing, so each run also launches eagerly.)
    let mut captured = Vec::new();
    let capture = Log::default();
    walk(&steps, Graphs::Capture(&mut captured), &gpu, 0, &capture).unwrap();
    assert_eq!(captured.len(), run_count(&steps));
    assert_eq!(swaps_of(&capture.0.borrow()), swaps);
}

/// A two-rank pair on the mock backend: records each exchange's size and
/// lands the queued peer payload in its `dst`.
struct Pair<'a> {
    gpu: &'a MockGpuBackend,
    rank: usize,
    sizes: RefCell<Vec<usize>>,
    peer: RefCell<VecDeque<Vec<u8>>>,
}

// SAFETY: test-only, used on one thread.
unsafe impl Send for Pair<'_> {}
unsafe impl Sync for Pair<'_> {}

impl<'a> Pair<'a> {
    fn new(gpu: &'a MockGpuBackend, rank: usize) -> Self {
        Self {
            gpu,
            rank,
            sizes: RefCell::default(),
            peer: RefCell::default(),
        }
    }
}

impl CommBackend for Pair<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn exchange_async(&self, _: u64, dst: u64, bytes: usize, add: bool, _: u64) -> Result<bool> {
        assert!(!add, "a split swap copies, never adds");
        self.sizes.borrow_mut().push(bytes);
        if let Some(payload) = self.peer.borrow_mut().pop_front() {
            assert_eq!(payload.len(), bytes, "both ranks swap the same bytes");
            self.gpu.copy_h2d(&payload, DevicePtr(dst))?;
        }
        Ok(true)
    }
    fn supports_exchange_async(&self, _: usize) -> bool {
        true
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

/// One rank of the pair on its own mock device.
struct Rank {
    gpu: MockGpuBackend,
    scratch: DflashScratch,
    split: RankSplit,
}

impl Rank {
    fn new(g: Geometry) -> Self {
        Self::with_capacity(g, g.gamma)
    }

    /// A rank whose split holds `capacity` rows.
    fn with_capacity(g: Geometry, capacity: usize) -> Self {
        let gpu = MockGpuBackend::new();
        let mut scratch = zero_scratch();
        scratch.norm_buf = gpu.alloc(g.gamma * g.hidden * 2).unwrap();
        scratch.mlp_intermediate = gpu.alloc(g.gamma * g.inter * 2).unwrap();
        scratch.stream_acc = gpu.alloc(g.gamma * g.hidden * 2).unwrap();
        scratch.logits = gpu.alloc(g.gamma * g.vocab * 2).unwrap();
        let split = RankSplit::new(g, capacity, &gpu).unwrap();
        Self {
            gpu,
            scratch,
            split,
        }
    }

    /// Distinct bytes per rank, buffer and offset.
    fn fill(&self, ptr: DevicePtr, bytes: usize, tag: u8) {
        let data: Vec<u8> = (0..bytes)
            .map(|i| tag ^ (i as u8).wrapping_mul(31))
            .collect();
        self.gpu.copy_h2d(&data, ptr).unwrap();
    }

    fn read(&self, ptr: DevicePtr, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        self.gpu.copy_d2h(ptr, &mut out).unwrap();
        out
    }
}

/// `[gamma, n]` rows from rank 0's and rank 1's shares, each `[gamma, its
/// rows]` at the start of the payload it sent.
fn joined(gamma: usize, n: usize, lo: &[u8], hi: &[u8]) -> Vec<u8> {
    let (row0, row1) = (half(n, 0).1 * 2, half(n, 1).1 * 2);
    (0..gamma)
        .flat_map(|r| {
            lo[r * row0..(r + 1) * row0]
                .iter()
                .chain(&hi[r * row1..(r + 1) * row1])
                .copied()
        })
        .collect()
}

#[test]
fn swapped_halves_land_where_the_unsplit_launch_writes_whole_rows() {
    let g = geometry(ALL);
    let (r0, r1) = (Rank::new(g), Rank::new(g));
    let (p0, p1) = (Pair::new(&r0.gpu, 0), Pair::new(&r1.gpu, 1));
    // Run `swap` on both ranks, each landing what the other sent.
    let run = |swap: Swap| {
        let bytes = g.bytes(swap);
        let sent0 = r0.read(r0.split.ends(swap, 0, &Frame::serial(&r0.scratch)).0, bytes);
        let sent1 = r1.read(r1.split.ends(swap, 1, &Frame::serial(&r1.scratch)).0, bytes);
        p0.peer.borrow_mut().push_back(sent1.clone());
        p1.peer.borrow_mut().push_back(sent0.clone());
        r0.split
            .swap(swap, 0, &r0.gpu, &p0, &Frame::serial(&r0.scratch), 0)
            .unwrap();
        r1.split
            .swap(swap, 1, &r1.gpu, &p1, &Frame::serial(&r1.scratch), 0)
            .unwrap();
        (sent0, sent1)
    };

    // The head's MLP input reaches the worker's `norm_buf`.
    let hidden = g.gamma * g.hidden * 2;
    r0.fill(r0.scratch.norm_buf, hidden, 0x11);
    let (x, _) = run(Swap::Input(0));
    assert_eq!(r1.read(r1.scratch.norm_buf, hidden), x);
    assert_eq!(
        r0.read(r0.scratch.norm_buf, hidden),
        x,
        "the head keeps its own"
    );

    // Both ranks hold the whole gated activation, rank 0's half first.
    let act = g.bytes(Swap::Activation(0));
    r0.fill(r0.split.gate, act, 0x21);
    r1.fill(r1.split.gate, act, 0x22);
    let (a0, a1) = run(Swap::Activation(0));
    let whole = joined(g.gamma, g.inter, &a0, &a1);
    assert_eq!(r0.read(r0.scratch.mlp_intermediate, whole.len()), whole);
    assert_eq!(r1.read(r1.scratch.mlp_intermediate, whole.len()), whole);

    // The head holds the whole down projection; the worker's residual stream
    // is not its to write.
    let out = g.bytes(Swap::Output(0));
    r0.fill(r0.split.down, out, 0x31);
    r1.fill(r1.split.down, out, 0x32);
    r1.fill(r1.scratch.stream_acc, hidden, 0x3f);
    let (y0, y1) = run(Swap::Output(0));
    assert_eq!(
        r0.read(r0.scratch.stream_acc, hidden),
        joined(g.gamma, g.hidden, &y0, &y1)
    );
    assert_eq!(r1.read(r1.scratch.stream_acc, hidden).len(), hidden);
    let untouched: Vec<u8> = (0..hidden)
        .map(|i| 0x3f ^ (i as u8).wrapping_mul(31))
        .collect();
    assert_eq!(r1.read(r1.scratch.stream_acc, hidden), untouched);

    // The final norm rows reach the worker; the head gets the whole logits.
    r0.fill(r0.scratch.norm_buf, hidden, 0x41);
    let (h, _) = run(Swap::Hidden);
    assert_eq!(r1.read(r1.scratch.norm_buf, hidden), h);
    let half_logits = g.bytes(Swap::Logits);
    r0.fill(r0.split.logits, half_logits, 0x51);
    r1.fill(r1.split.logits, half_logits, 0x52);
    let (l0, l1) = run(Swap::Logits);
    assert_eq!(
        r0.read(r0.scratch.logits, g.gamma * g.vocab * 2),
        joined(g.gamma, g.vocab, &l0, &l1)
    );

    // Same sizes, same order, on both ranks.
    assert_eq!(*p0.sizes.borrow(), *p1.sizes.borrow());
}

#[test]
fn a_propose_that_stops_early_still_issues_every_swap() {
    let g = geometry(ALL);
    let rank = Rank::new(g);
    let pair = Pair::new(&rank.gpu, 0);
    let sizes: Vec<usize> = g.swaps().iter().map(|&s| g.bytes(s)).collect();

    // Stopped after two swaps: the rest are drained, sized as planned.
    rank.split.begin(g.gamma, CtxRows::default()).unwrap();
    for &swap in &g.swaps()[..2] {
        rank.split
            .swap(swap, 0, &rank.gpu, &pair, &Frame::serial(&rank.scratch), 0)
            .unwrap();
    }
    rank.split.finish(&pair, 0).unwrap();
    assert_eq!(*pair.sizes.borrow(), sizes);

    // A propose that reached every swap drains nothing, and the next one
    // starts from zero.
    rank.split.finish(&pair, 0).unwrap();
    assert_eq!(pair.sizes.borrow().len(), sizes.len());
    rank.split.begin(g.gamma, CtxRows::default()).unwrap();
    rank.split.finish(&pair, 0).unwrap();
    assert_eq!(pair.sizes.borrow().len(), 2 * sizes.len());
    assert!(rank.split.max_bytes() >= *sizes.iter().max().unwrap());
}

#[path = "rank_split_batch_tests.rs"]
mod batch;

#[path = "rank_split_ctx_tests.rs"]
mod ctx;
