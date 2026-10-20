// SPDX-License-Identifier: AGPL-3.0-only

//! `forward_seqs` against the per-sequence loop it replaces, on the mock:
//! the same slot ids reach the gather row for row, every carry ends in the
//! same host state, and the row-independent stages launch once per batch.

use super::*;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;

const HIDDEN: usize = 1280;
const HC: usize = 4;

fn layer(gpu: &MockGpuBackend) -> PleLayer {
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    PleLayer::new(
        PleIdDims {
            ngram_size: 3,
            heads_per_ngram: 8,
            multipliers: vec![1, 3, 5],
            head_vocab_sizes: vec![7; 16],
            head_offsets: vec![0; 16],
            eos_token_id: 2,
        },
        80, // 16 n-gram heads x 80 = HIDDEN
        HIDDEN,
        HC,
        4,
        3,
        1e-6,
        PleWeights {
            key_proj: dw(2),
            value_proj: dw(2),
            norm_key: dw(2),
            norm_query: dw(2),
            norm_conv: dw(2),
            conv1d: dw(2),
        },
        NgramTable::Bf16(dw(2)),
        32,
        32,
        None,
        gpu,
    )
    .unwrap()
}

/// What a forward leaves on the host side of one carry.
fn carry(st: &PleSeqState) -> (Vec<u32>, Vec<u32>, Vec<u32>, usize) {
    (
        st.history.clone(),
        st.history_ckpt.clone(),
        st.verify_tokens.clone(),
        st.verify_snap_rows,
    )
}

fn slots(gpu: &MockGpuBackend, ple: &PleLayer, rows: usize) -> Vec<u8> {
    let mut b = vec![0u8; rows * ple.dims.ngram_heads() * 4];
    gpu.copy_d2h(ple.slots_dev, &mut b).unwrap();
    b
}

/// Launches shaped like the key projection (`ceil(hc*hidden/128)` wide, 256
/// threads, the GEMM's six parameters) — the GEMM the batched form runs once.
fn key_gemms(gpu: &MockGpuBackend) -> usize {
    let x = (HC * HIDDEN).div_ceil(128) as u32;
    gpu.launches_snapshot()
        .iter()
        .filter(|l| l.grid[0] == x && l.block[0] == 256 && l.args == 6)
        .count()
}

fn with_ctx(run: impl FnOnce(&ForwardContext, &MockGpuBackend)) {
    let gpu = MockGpuBackend::new();
    let config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    let buffers = BufferArena::new(&config, 9, 16, 16, 1, &gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        ssm_batch: None,
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
    run(&ctx, &gpu);
}

/// Two steps of ragged windows (the batched verify's 3/1/4 rows, then a
/// decode-like 1/1/1): batched and per-sequence forwards agree on every row's
/// slot ids and every carry, and the batched form runs one gather and one
/// pair of projections per step instead of one per sequence.
#[test]
fn seq_batch_matches_the_per_sequence_forwards() {
    with_ctx(|ctx, gpu| {
        let ple = layer(gpu);
        let highway = gpu.alloc(16 * HC * HIDDEN * 4).unwrap();
        let mut a: Vec<PleSeqState> = (0..3).map(|_| ple.new_seq_state(gpu).unwrap()).collect();
        let mut b: Vec<PleSeqState> = (0..3).map(|_| ple.new_seq_state(gpu).unwrap()).collect();
        let steps: [&[&[u32]]; 2] = [
            &[&[11, 12, 13], &[21], &[31, 32, 33, 34]],
            &[&[14], &[22], &[35]],
        ];
        for windows in steps {
            let rows: usize = windows.iter().map(|w| w.len()).sum();
            let (l0, g0) = (gpu.launch_count(), key_gemms(gpu));
            let mut want = Vec::new();
            for (st, &ids) in a.iter_mut().zip(windows) {
                ple.forward_rows(st, highway, ids, ctx, 0).unwrap();
                want.extend(slots(gpu, &ple, ids.len()));
            }
            let (l1, g1) = (gpu.launch_count(), key_gemms(gpu));
            let mut row0 = 0;
            let mut seqs: Vec<PleSeqRows<'_>> = b
                .iter_mut()
                .zip(windows)
                .map(|(st, &ids)| {
                    let r = PleSeqRows { st, row0, ids };
                    row0 += ids.len();
                    r
                })
                .collect();
            ple.forward_seqs(&mut seqs, highway, ctx, 0).unwrap();
            assert_eq!(slots(gpu, &ple, rows), want, "slot ids differ row for row");
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(carry(x), carry(y));
            }
            assert_eq!(g1 - g0, 3, "per-sequence: one key projection each");
            assert_eq!(key_gemms(gpu) - g1, 1, "batched: one key projection");
            // One gather and two GEMMs saved per extra sequence.
            assert_eq!((l1 - l0) - (gpu.launch_count() - l1), 2 * 3);
        }
    });
}

/// A member holding a decode prestage takes the per-sequence loop, which is
/// what decides whether that staging is consumed.
#[test]
fn an_armed_prestage_keeps_the_per_sequence_loop() {
    with_ctx(|ctx, gpu| {
        let ple = layer(gpu);
        let highway = gpu.alloc(16 * HC * HIDDEN * 4).unwrap();
        let mut b: Vec<PleSeqState> = (0..2).map(|_| ple.new_seq_state(gpu).unwrap()).collect();
        b[1].prestaged_va = Some(0x1234);
        b[1].prestaged_n = 3;
        let g0 = key_gemms(gpu);
        let ids: [&[u32]; 2] = [&[5, 6], &[7]];
        let mut seqs: Vec<PleSeqRows<'_>> = b
            .iter_mut()
            .zip(ids)
            .scan(0, |row0, (st, ids)| {
                let r = PleSeqRows {
                    st,
                    row0: *row0,
                    ids,
                };
                *row0 += ids.len();
                Some(r)
            })
            .collect();
        ple.forward_seqs(&mut seqs, highway, ctx, 0).unwrap();
        assert_eq!(key_gemms(gpu) - g0, 2);
        assert_eq!(
            b[1].prestaged_va, None,
            "the per-sequence forward consumed it"
        );
    });
}
