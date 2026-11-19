// SPDX-License-Identifier: AGPL-3.0-only

//! Draft TP for ONE sequence (`ATLAS_QWEN4EXP_MTP_DRAFT_TP_ONE`): the
//! per-sequence propose the model announced walks the worker's swaps, one
//! row a position, whatever it does after the announce; a propose it did not
//! announce (no communicator, a grammar mask) issues none. Mock devices: the
//! drafts' VALUES are `qwen4exp_mtp_tp_gpu_tests.rs`'s.

use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::super::qwen4exp_mtp_batch::tests::{config, head};
use super::tests::Pair;
use super::*;
use crate::layer::ForwardContext;
use crate::speculative::DraftProposer;

/// The worker's walk of `plan` for the head `head()` builds.
fn worker_sizes(plan: Plan) -> Vec<usize> {
    let gpu = MockGpuBackend::new();
    let c = config();
    let lm_head = DenseWeight {
        weight: gpu.alloc(c.vocab_size * c.hidden_size * BF16).unwrap(),
    };
    let assist =
        Qwen4ExpDraftAssist::new(&lm_head, c.vocab_size, c.hidden_size, 1000, &gpu).unwrap();
    let comm = Pair::new(1, 0);
    assist.serve(&gpu, &comm, 0, plan.word().unwrap()).unwrap();
    comm.sizes()
}

/// `propose` of `drafts` positions for one sequence, handed `comm`.
fn sizes_of(drafts: usize, comm: Option<&Pair>, mask: Option<&[i32]>, split: bool) -> Vec<u32> {
    let gpu = MockGpuBackend::new();
    let c = config();
    let (mut head, _) = head(&gpu, &c);
    if split {
        head.tp = Some(DraftTp::new(&head.draft, c.hidden_size, 0, &gpu).unwrap());
    }
    let buffers = BufferArena::new(&c, 64, 4096, 16, 8, &gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        ssm_batch: None,
        buffers: &buffers,
        gpu: &gpu,
        config: &c,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: comm.map(|p| p as &dyn CommBackend),
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Skip,
    };
    let mut state = head.alloc_state(&gpu).unwrap();
    head.propose(
        1,
        DevicePtr::NULL,
        80,
        drafts,
        state.as_mut(),
        None,
        &ctx,
        0,
        None,
        mask,
        None,
    )
    .unwrap()
}

#[test]
fn an_announced_single_propose_walks_the_workers_swaps() {
    for drafts in [1, 3, 7] {
        let comm = Pair::new(0, 0);
        let got = sizes_of(drafts, Some(&comm), None, true);
        assert_eq!(got.len(), drafts);
        let plan = Plan { n: 1, drafts };
        assert_eq!(comm.sizes(), worker_sizes(plan), "{drafts} drafts");
        assert_eq!(comm.sizes().len(), 2 * drafts);
        assert_eq!(comm.sizes()[0], config().hidden_size * BF16, "one row");
    }
}

#[test]
fn an_unannounced_single_propose_issues_no_swap() {
    // No communicator: the propose was not announced.
    assert_eq!(sizes_of(3, None, None, true).len(), 3);
    // A grammar mask indexes the full head, never the split draft rows.
    let comm = Pair::new(0, 0);
    let mask = vec![-1i32; config().vocab_size.div_ceil(32)];
    sizes_of(3, Some(&comm), Some(&mask), true);
    assert!(comm.sizes().is_empty(), "a masked propose must not swap");
    // No split state on this head.
    let comm = Pair::new(0, 0);
    sizes_of(3, Some(&comm), None, false);
    assert!(comm.sizes().is_empty(), "no draft TP on this head");
}

#[test]
fn only_the_switch_announces_a_single_sequence() {
    let gpu = MockGpuBackend::new();
    let c = config();
    let (mut head, _) = head(&gpu, &c);
    head.tp = Some(DraftTp::new(&head.draft, c.hidden_size, 0, &gpu).unwrap());
    let comm = Pair::new(0, 0);
    assert_eq!(head.tp_one_word(&comm, 3, false), None, "switch off");
    head.tp_one = true;
    let word = Plan { n: 1, drafts: 3 }.word().unwrap();
    assert_eq!(head.tp_one_word(&comm, 3, false), Some(word));
    assert_eq!(head.tp_one_word(&comm, 3, true), None, "grammar mask");
    assert_eq!(head.tp_one_word(&comm, 9, false), None, "past the plan");
    assert_eq!(head.tp_one_word(&Pair::new(1, 0), 3, false), None, "worker");
    head.tp = None;
    assert_eq!(head.tp_one_word(&comm, 3, false), None, "no split state");
}
