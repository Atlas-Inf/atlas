// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the `glm_index_split` plan, exchange and startup words,
//! against a recording pair communicator over the mock GPU (the check's are
//! in `glm_index_split_check_tests.rs`).

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use anyhow::bail;
use spark_comm::CommBackend;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;
use atlas_core::config::ModelConfig;

/// Owner row counts the production pieces produce: the 8196-row first
/// chunk's 4096 and 2052 pieces, full 4096 pieces, warm appends of any
/// length.
const ROWS: [usize; 8] = [256, 258, 512, 1024, 2050, 2052, 3001, 4096];

const ON: Settings = Settings {
    min_ctx: 4096,
    check: false,
};

pub(super) fn split(rows: usize, rank: usize, check: bool) -> IndexSplit {
    IndexSplit {
        swaps: IndexSplit::zigzag(rows, rank),
        rows,
        check,
    }
}

/// One exchange as the peer saw it: `(send, dst, add, sent bytes)`.
type Call = (u64, u64, bool, Vec<u8>);

/// A two-rank copy-engine pair: records each exchange and lands the next
/// queued peer payload in its `dst`.
pub(super) struct Pair<'a> {
    gpu: &'a MockGpuBackend,
    rank: usize,
    world: usize,
    capacity: usize,
    calls: Mutex<Vec<Call>>,
    pub(super) peer: Mutex<VecDeque<Vec<u8>>>,
}

impl<'a> Pair<'a> {
    pub(super) fn new(gpu: &'a MockGpuBackend, rank: usize) -> Self {
        Self {
            gpu,
            rank,
            world: 2,
            capacity: 1 << 20,
            calls: Mutex::default(),
            peer: Mutex::default(),
        }
    }

    pub(super) fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }
}

impl CommBackend for Pair<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world
    }
    fn exchange_async(&self, send: u64, dst: u64, bytes: usize, add: bool, _: u64) -> Result<bool> {
        let mut sent = vec![0u8; bytes];
        self.gpu.copy_d2h(DevicePtr(send), &mut sent)?;
        self.calls.lock().unwrap().push((send, dst, add, sent));
        if let Some(p) = self.peer.lock().unwrap().pop_front() {
            self.gpu.copy_h2d(&p, DevicePtr(dst))?;
        }
        Ok(true)
    }
    fn supports_exchange_async(&self, bytes: usize) -> bool {
        bytes <= self.capacity
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

/// Run `f` with a TP2 forward context over `comm` (none when `None`).
pub(super) fn with_ctx<R>(
    gpu: &MockGpuBackend,
    comm: Option<&Pair>,
    capture: bool,
    tp: usize,
    f: impl FnOnce(&ForwardContext) -> R,
) -> R {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.tp_world_size = tp;
    let buffers = BufferArena::new(&config, 1, 256, 16, 1, gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    f(&ForwardContext {
        ssm_batch: None,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: comm.map(|c| c as &dyn CommBackend),
        graph_capture: capture,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    })
}

#[test]
fn quarters_tile_the_owner_once_and_pair_symmetrically() {
    for rows in ROWS {
        let (r0, r1) = (IndexSplit::zigzag(rows, 0), IndexSplit::zigzag(rows, 1));
        let mut seen = vec![0u8; rows];
        for s in r0.iter().chain(&r1) {
            assert_eq!(s.rows, rows / 4);
            seen[s.own..s.own + s.rows].iter_mut().for_each(|c| *c += 1);
        }
        let whole = 4 * (rows / 4);
        assert!(seen[..whole].iter().all(|&c| c == 1), "rows {rows}");
        assert!(seen[whole..].iter().all(|&c| c == 0), "rows {rows}");
        // Pair k: what one rank sends lands where the other expects it,
        // and both ranks exchange the same byte count in the same order.
        for k in 0..2 {
            assert_eq!((r0[k].own, r0[k].peer), (r1[k].peer, r1[k].own));
            assert_eq!(r0[k].rows, r1[k].rows);
        }
    }
    let q = 1024;
    let swap = |own, peer| Swap { own, peer, rows: q };
    assert_eq!(
        IndexSplit::zigzag(4 * q, 0),
        [swap(0, q), swap(3 * q, 2 * q)]
    );
}

#[test]
fn every_row_is_selected_here_or_received() {
    for rows in ROWS {
        for rank in 0..2 {
            let s = split(rows, rank, false);
            let mut seen = vec![0u8; rows];
            let received = s.swaps.map(|w| w.peer..w.peer + w.rows);
            for r in s.own().into_iter().chain(received) {
                seen[r].iter_mut().for_each(|c| *c += 1);
            }
            assert!(seen.iter().all(|&c| c == 1), "rows {rows} rank {rank}");
        }
    }
}

#[test]
fn own_rows_merge_into_two_ranges_with_the_tail_on_both_ranks() {
    let own = |rows, rank| split(rows, rank, false).own();
    assert_eq!(own(4096, 0), [0..1024, 3072..4096]);
    assert_eq!(
        (own(4096, 1).len(), own(4096, 1)[0].clone()),
        (1, 1024..3072)
    );
    assert_eq!(own(2050, 0), [0..512, 1536..2050]);
    assert_eq!(own(2050, 1), [512..1536, 2048..2050]);
    assert_eq!(own(3001, 0), [0..750, 2250..3001]);
    assert_eq!(own(3001, 1), [750..2250, 3000..3001]);
}

#[test]
fn zigzag_balances_causal_work_exactly() {
    // A row's logits and top-k scale with its causal extent.
    for rows in ROWS {
        for start in [0, 2048, 6144, 8196, 100_000, 524_288] {
            let work = |rank| -> usize {
                split(rows, rank, false)
                    .own()
                    .into_iter()
                    .flatten()
                    .map(|row| start + row + 1)
                    .sum()
            };
            assert_eq!(work(0), work(1), "rows {rows} start {start}");
        }
    }
}

#[test]
fn admits_long_owners_of_any_length() {
    let min = 4096;
    // Later 8K-chunk pieces and the first chunk's 2052-row tail piece.
    assert!(admits(4096, 8196, min));
    assert!(admits(2052, 6144, min));
    // Warm appends at long context, of any length.
    assert!(admits(512, 200_000, min));
    assert!(admits(2050, 8196, min));
    assert!(admits(3001, 8196, min));
    // The first chunk's 4096-row piece has too little history.
    assert!(!admits(4096, 2048, min));
    // Verify-sized owners and 4-row tails.
    assert!(!admits(255, 100_000, min));
    assert!(!admits(64, 100_000, min));
    assert!(!admits(4, 16_388, min));
    assert!(admits(4096, 0, 0));
}

#[test]
fn settings_are_off_unless_requested_and_reject_a_junk_min_ctx() {
    let parse = |vars: &[(&str, &str)]| {
        let vars: HashMap<_, _> = vars.iter().copied().collect();
        parse_settings(|name| vars.get(name).map(|v| v.to_string()))
    };
    assert_eq!(parse(&[]), Ok(None));
    assert_eq!(parse(&[("ATLAS_GLM_INDEX_SPLIT", "0")]), Ok(None));
    // A junk MIN_CTX is harmless while the split is off.
    assert_eq!(parse(&[("ATLAS_GLM_INDEX_SPLIT_MIN_CTX", "4k")]), Ok(None));
    assert_eq!(parse(&[("ATLAS_GLM_INDEX_SPLIT", "1")]), Ok(Some(ON)));
    let on = |min_ctx: &str, check: &str| {
        parse(&[
            ("ATLAS_GLM_INDEX_SPLIT", "1"),
            ("ATLAS_GLM_INDEX_SPLIT_MIN_CTX", min_ctx),
            ("ATLAS_GLM_INDEX_SPLIT_CHECK", check),
        ])
    };
    assert_eq!(
        on("16384", "1"),
        Ok(Some(Settings {
            min_ctx: 16384,
            check: true
        }))
    );
    assert_eq!(on("0", "0"), Ok(Some(Settings { min_ctx: 0, ..ON })));
    assert!(on("4k", "0").is_err());
}

#[test]
fn plan_splits_only_mirrored_eligible_owners() {
    let gpu = MockGpuBackend::new();
    let pair = Pair::new(&gpu, 1);
    let plan = |settings, comm, capture, tp, rows, start| {
        with_ctx(&gpu, comm, capture, tp, |ctx| {
            IndexSplit::plan_with(settings, rows, start, 16, ctx).unwrap()
        })
    };
    assert_eq!(
        plan(Some(ON), Some(&pair), false, 2, 2050, 8196),
        Some(split(2050, 1, false))
    );
    // Off, single GPU, under graph capture, TP world != 2, short history.
    assert_eq!(plan(None, Some(&pair), false, 2, 2050, 8196), None);
    assert_eq!(plan(Some(ON), None, false, 2, 2050, 8196), None);
    assert_eq!(plan(Some(ON), Some(&pair), true, 2, 2050, 8196), None);
    assert_eq!(plan(Some(ON), Some(&pair), false, 1, 2050, 8196), None);
    assert_eq!(plan(Some(ON), Some(&pair), false, 2, 2050, 4095), None);
    // A communicator of another size, or a quarter past the pair capacity.
    let wide = Pair {
        world: 4,
        ..Pair::new(&gpu, 1)
    };
    assert_eq!(plan(Some(ON), Some(&wide), false, 2, 2050, 8196), None);
    let small = Pair {
        capacity: 512 * 16 - 1,
        ..Pair::new(&gpu, 1)
    };
    assert_eq!(plan(Some(ON), Some(&small), false, 2, 2050, 8196), None);
    let fits = Pair {
        capacity: 512 * 16,
        ..Pair::new(&gpu, 1)
    };
    assert!(plan(Some(ON), Some(&fits), false, 2, 2050, 8196).is_some());
}

#[test]
fn plan_checks_only_with_scratch_for_every_row() {
    let gpu = MockGpuBackend::new();
    let pair = Pair::new(&gpu, 0);
    let check = Some(Settings { check: true, ..ON });
    with_ctx(&gpu, Some(&pair), false, 2, |ctx| {
        let row_bytes = ctx.buffers.sizes().expert_down_out / (2 * 256);
        assert_eq!(
            IndexSplit::plan_with(check, 256, 8196, row_bytes, ctx).unwrap(),
            Some(split(256, 0, true))
        );
        assert!(IndexSplit::plan_with(check, 256, 8196, row_bytes + 1, ctx).is_err());
        // Without the check the scratch is not needed.
        assert!(
            IndexSplit::plan_with(Some(ON), 256, 8196, row_bytes + 1, ctx)
                .unwrap()
                .is_some()
        );
    });
}

#[test]
fn exchange_swaps_the_zigzag_pairs_in_the_same_order_on_both_ranks() {
    let (rows, rb, q) = (258, 16, 64);
    for rank in 0..2 {
        let gpu = MockGpuBackend::new();
        let pair = Pair::new(&gpu, rank);
        with_ctx(&gpu, Some(&pair), false, 2, |ctx| {
            let sel = ctx.buffers.expert_down_out();
            let at = |row: usize| sel.offset(row * rb).0;
            let owner = OwnerRows {
                selected: sel,
                row_bytes: rb,
                scratch: sel.offset(rows * rb),
                inputs: [(sel, 0); 2],
            };
            split(rows, rank, false)
                .exchange(&owner, 0, ctx, 7)
                .unwrap();
            let got: Vec<_> = pair
                .calls()
                .into_iter()
                .map(|(send, dst, add, sent)| (send, dst, add, sent.len()))
                .collect();
            let want = if rank == 0 {
                [(at(0), at(q)), (at(3 * q), at(2 * q))]
            } else {
                [(at(q), at(0)), (at(2 * q), at(3 * q))]
            };
            assert_eq!(got, want.map(|(s, d)| (s, d, false, q * rb)), "rank {rank}");
        });
    }
}

#[test]
fn startup_words_are_three_whatever_the_settings() {
    assert_eq!(Settings::words(None), [0; 3]);
    assert_eq!(Settings::words(Some(ON)), [1, 4096, 0]);
    let checked = Settings {
        min_ctx: 0,
        check: true,
    };
    assert_eq!(Settings::words(Some(checked)), [1, 0, 1]);
}

#[test]
fn tiles_bound_each_pass_like_the_replicated_loop() {
    let (a, b) = (DevicePtr(0x10), DevicePtr(0x20));
    let t: Vec<_> = tiles(&[(0..700, a)], 300).collect();
    assert_eq!(t, [(0, 300, a), (300, 300, a), (600, 100, a)]);
    let t: Vec<_> = tiles(&[(0..513, a), (1539..2052, a), (0..2052, b)], 2052).collect();
    assert_eq!(t, [(0, 513, a), (1539, 513, a), (0, 2052, b)]);
}

#[test]
fn passes_cover_own_rows_and_the_check_recompute() {
    let (sel, scratch) = (DevicePtr(0x1000), DevicePtr(0x9000));
    assert_eq!(passes(None, 4096, sel, scratch), [(0..4096, sel)]);
    assert_eq!(
        passes(Some(split(4096, 0, false)), 4096, sel, scratch),
        [(0..1024, sel), (3072..4096, sel)]
    );
    assert_eq!(
        passes(Some(split(2050, 1, true)), 2050, sel, scratch),
        [(512..1536, sel), (2048..2050, sel), (0..2050, scratch)]
    );
}
