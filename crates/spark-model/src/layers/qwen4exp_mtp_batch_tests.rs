// SPDX-License-Identifier: AGPL-3.0-only

//! The batched qwen4_exp propose without a GPU: the slab layout, the
//! confidence-stop rule, and the SHAPE of a call on the mock backend — one
//! drafter body pass per position for every sequence, one blocking readback
//! per call. Draft VALUES (batched == serial) need real kernels: see
//! `qwen4exp_mtp_batch_gpu_tests.rs`.

use super::*;
use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::{HcHeadWeights, HcLowRank};
use crate::weight_loader::qwen4_exp::Qwen4ExpMtpModule;
use atlas_core::config::{LayerType, ModelConfig};
use parking_lot::Mutex;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::sync::Arc;

#[test]
fn slab_holds_tokens_every_header_and_the_tables() {
    let tables = vec![vec![7u32, 9], vec![4]];
    let slab = pack_slab(&[11, 22], &[100, 40], &[15, 3], &tables, 3, 16, 4).unwrap();
    assert_eq!(slab.max_blocks, 2);
    let word = |at: usize| u32::from_le_bytes(slab.bytes[at..at + 4].try_into().unwrap());
    let slot = |at: usize| i64::from_le_bytes(slab.bytes[at..at + 8].try_into().unwrap());
    assert_eq!((word(0), word(4)), (11, 22));
    // Position 1, sequence 0: drafter row 16 is the first row of block 9.
    let hdr = HDR_OFF + HDR_BYTES;
    assert_eq!(word(hdr), 101);
    assert_eq!(slot(hdr + HDR_SLOT), 9 * 16);
    assert_eq!(word(hdr + HDR_SEQ_LEN), 17);
    // Position 2, sequence 1: drafter row 5 of block 4, RoPE position 42.
    let hdr = HDR_OFF + 2 * HDR_BYTES;
    assert_eq!(word(hdr + 4), 42);
    assert_eq!(slot(hdr + HDR_SLOT + 8), 4 * 16 + 5);
    assert_eq!(word(hdr + HDR_SEQ_LEN + 4), 6);
    let bt: Vec<u32> = (0..4).map(|b| word(BT_OFF + b * 4)).collect();
    assert_eq!(bt, vec![7, 9, 4, 0]);
    assert_eq!(slab.bytes.len(), BT_OFF + 2 * 2 * 4);
}

#[test]
fn slab_refuses_what_it_cannot_hold() {
    let one = vec![vec![1u32]];
    // Row 16 needs a second block.
    assert!(pack_slab(&[1], &[0], &[15], &one, 2, 16, 4).is_err());
    assert!(pack_slab(&[1], &[0], &[0], &[vec![1, 2, 3]], 1, 16, 2).is_err());
    let nine = vec![vec![1u32]; 9];
    assert!(pack_slab(&[1; 9], &[0; 9], &[0; 9], &nine, 1, 16, 4).is_err());
    assert!(pack_slab(&[1], &[0], &[0], &one, PROPOSE_BATCH_MAX_DRAFTS + 1, 16, 4).is_err());
}

/// The per-sequence `propose` loop's rule: draft 0 always, stop before the
/// first later draft under the threshold.
#[test]
fn confidence_stop_keeps_the_per_sequence_prefix() {
    assert_eq!(kept_drafts(&[0.1, 0.9, 0.2], 0.5), 2);
    assert_eq!(kept_drafts(&[0.9, 0.1, 0.9], 0.5), 1);
    assert_eq!(kept_drafts(&[0.1, 0.6, 0.7], 0.5), 3);
    assert_eq!(kept_drafts(&[0.1, 0.1, 0.1], 0.0), 3);
}

/// A body that records the row count of every call.
struct StubBody(Arc<Mutex<Vec<usize>>>);

impl TransformerLayer for StubBody {
    fn decode(
        &self,
        _: DevicePtr,
        _: DevicePtr,
        _: &mut dyn LayerState,
        _: &mut PagedKvCache,
        _: usize,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &mut Vec<u32>,
        _: &ForwardContext,
        _: u64,
    ) -> Result<()> {
        self.0.lock().push(1);
        Ok(())
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        _: DevicePtr,
        _: DevicePtr,
        num_seqs: usize,
        _: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _: &mut PagedKvCache,
        seq_lens: &[usize],
        _: &[Vec<u32>],
        ctx: &ForwardContext,
        _: u64,
    ) -> Result<()> {
        assert_eq!(states.len(), num_seqs);
        assert_eq!(seq_lens.len(), num_seqs);
        assert_eq!(ctx.attn_metadata.unwrap().num_seqs as usize, num_seqs);
        assert!(ctx.comm.is_none(), "the drafter must not join a collective");
        self.0.lock().push(num_seqs);
        Ok(())
    }

    fn alloc_state(&self, _: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(crate::layer::AttnLayerState::default()))
    }
}

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = 256;
    c.hc_mult = 4;
    c.hc_lowrank = 32;
    c.vocab_size = 4096;
    c.layer_types = vec![LayerType::FullAttention; 2];
    c
}

type Calls = Arc<Mutex<Vec<usize>>>;

fn head(gpu: &MockGpuBackend, c: &ModelConfig) -> (Qwen4ExpMtpHead, Calls) {
    let h = c.hidden_size;
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    let calls = Calls::default();
    let p = |bytes: usize| gpu.alloc(bytes).unwrap();
    let module = Qwen4ExpMtpModule {
        config: c.clone(),
        body: Box::new(StubBody(calls.clone())),
        pre_fc_norm_embedding: dw(h * 2),
        pre_fc_norm_hidden: dw(c.hc_mult * h * 2),
        fc_embedding: dw(h * h * 2),
        fc_hidden: dw(h * h * 2),
        hc_head: Some(HcHeadWeights {
            hc_fn: p(256),
            hc_base: p(256),
            hc_scale: p(256),
            lowrank: Some(HcLowRank {
                norm_w: p(c.hc_mult * h * 2),
                down_w: p(c.hc_mult * h * 64),
                up_w: p(c.hc_mult * h * 64),
                inject_w: DevicePtr::NULL,
                rank: c.hc_lowrank,
            }),
        }),
    };
    let embed = dw(c.vocab_size * h * 2);
    let lm_head = dw(c.vocab_size * h * 2);
    let head = Qwen4ExpMtpHead::new(module, embed, lm_head, gpu, 1000, 4096).unwrap();
    (head, calls)
}

/// One `propose_batch` of 3 drafts for 3 sequences.
fn propose3(
    head: &Qwen4ExpMtpHead,
    owned: &mut [Box<dyn ProposerState>],
    ctx: &ForwardContext,
    masks: Option<&[Option<Vec<i32>>]>,
) -> Option<Vec<Vec<u32>>> {
    let mut states: Vec<&mut dyn ProposerState> = Vec::new();
    for s in owned.iter_mut() {
        states.push(s.as_mut());
    }
    head.propose_batch(
        &[1, 2, 3],
        &[DevicePtr::NULL; 3],
        &[50, 60, 70],
        3,
        &mut states,
        None,
        ctx,
        0,
        None,
        masks,
    )
    .unwrap()
}

fn st(s: &mut Box<dyn ProposerState>) -> &mut Qwen4ExpMtpProposerState {
    s.as_any_mut().downcast_mut().unwrap()
}

#[test]
fn one_body_pass_per_position_and_one_readback_per_call() {
    let gpu = MockGpuBackend::new();
    let c = config();
    let (head, calls) = head(&gpu, &c);
    let buffers = BufferArena::new(&c, 64, 4096, 16, 8, &gpu).unwrap();
    assert_eq!(head.batch_width(&buffers, &c), PROPOSE_BATCH_MAX);
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let mut fast = crate::layers::ops::ModelLevers::defaults();
    fast.qwen4exp_batch_fast = true;
    let slow = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = |levers| ForwardContext {
        ssm_batch: None,
        buffers: &buffers,
        gpu: &gpu,
        config: &c,
        dispatch: &dispatch,
        derived: &derived,
        levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Skip,
    };
    let (fast, slow) = (ctx(&fast), ctx(&slow));
    let mut owned: Vec<Box<dyn ProposerState>> =
        (0..3).map(|_| head.alloc_state(&gpu).unwrap()).collect();
    for (i, s) in owned.iter_mut().enumerate() {
        st(s).seq_len = 14 + i;
    }

    // Outside the exact batching lane, and with a grammar mask: per-sequence.
    assert!(propose3(&head, &mut owned, &slow, None).is_none());
    let masked = [None, Some(vec![0]), None];
    assert!(propose3(&head, &mut owned, &fast, Some(&masked)).is_none());
    assert!(calls.lock().is_empty());

    let (d2h, syncs) = (gpu.d2h_blocking_count(), gpu.sync_count());
    let drafts = propose3(&head, &mut owned, &fast, Some(&[None, None, None])).expect("batched");
    assert_eq!(drafts, vec![vec![0; 3]; 3], "mock memory reads back zeros");
    assert_eq!(*calls.lock(), vec![3, 3, 3]);
    assert_eq!(gpu.d2h_blocking_count() - d2h, 1, "one readback per call");
    assert_eq!(gpu.sync_count() - syncs, 0);
    for (i, s) in owned.iter_mut().enumerate() {
        let st = st(s);
        assert_eq!((st.seq_len, st.last_num_drafted), (17 + i, 3));
        // Rows 14+i ..= 16+i: two blocks for every sequence.
        assert_eq!(st.block_table.len(), 2);
    }

    // The per-sequence path drains once per draft per sequence.
    let d2h = gpu.d2h_blocking_count();
    for (i, s) in owned.iter_mut().enumerate() {
        head.propose(
            1,
            DevicePtr::NULL,
            80 + i,
            3,
            s.as_mut(),
            None,
            &fast,
            0,
            None,
            None,
            None,
        )
        .unwrap();
    }
    assert_eq!(gpu.d2h_blocking_count() - d2h, 9);
}
