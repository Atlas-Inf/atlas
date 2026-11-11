// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn long_context_dflash_lane_resolves_tp_local_topology_before_reserve() {
    if isolated(
        "long_context::long_context_dflash_lane_resolves_tp_local_topology_before_reserve",
        &[
            ("ATLAS_GLM_MTP_LONG_CONTEXT", "1"),
            ("ATLAS_GLM_DFLASH", "1"),
        ],
    ) {
        return;
    }
    for capacity in [1, 4] {
        for rank in 0..2 {
            for chunk in [1024, 2048, 4096] {
                let mut a = args(capacity, rank);
                a.dflash = true;
                a.dflash_gamma = Some(8);
                a.max_seq_len = 32768;
                a.max_prefill_tokens = chunk;
                a.ssm_cache_slots = 0;
                a.enable_prefix_caching = false;
                let mut cfg = config();
                let (topology, reserve) = prepare_reserve(&a, &mut cfg, 128usize << 30)
                    .unwrap_or_else(|e| panic!("long-context preparation failed: {e:#}"));
                let topology = topology.expect("resolved topology must be retained for handoff");
                assert_eq!((topology.tp_size, topology.ep_size), (2, 2));
                assert_eq!((topology.tp_rank, topology.ep_rank), (rank, rank));
                assert_eq!(
                    (cfg.linear_num_key_heads, cfg.linear_num_value_heads),
                    (32, 32)
                );
                assert_eq!((cfg.num_attention_heads, cfg.num_key_value_heads), (32, 32));
                let rows = chunk + capacity;
                assert_eq!(reserve.max_batch_tokens_pre, rows);
                assert_eq!(
                    reserve.resolved_prefill.as_ref().unwrap().max_batch_tokens,
                    rows
                );
                let gdn = rows * (12288 * 2 + 32 * 2 * 4 + 4096 * 2 * 2);
                assert_eq!(reserve.gdn_two_phase_bytes, gdn);
                assert_eq!(reserve.inference_reserve, gdn + (512usize << 20));
                assert_eq!(
                    reserve.buffer_arena_bytes,
                    spark_runtime::buffers::BufferSizes::from_config(
                        &cfg, rows, 32768, 16, capacity
                    )
                    .total_bytes()
                );
                let total = reserve.inference_reserve + reserve.buffer_arena_bytes;
                assert!(prepare_reserve(&a, &mut config(), total).is_ok());
                assert!(prepare_reserve(&a, &mut config(), total - 1).is_err());
            }
        }
    }
}

/// `4x1m-tp4.json`'s shape: the error it used to die on before GPU init was
/// this resolution refusing anything but the pair.
#[test]
fn long_context_dflash_lane_resolves_the_tp4_chain_before_reserve() {
    if isolated(
        "long_context::long_context_dflash_lane_resolves_the_tp4_chain_before_reserve",
        &[
            ("ATLAS_GLM_MTP_LONG_CONTEXT", "1"),
            ("ATLAS_GLM_DFLASH", "1"),
            ("ATLAS_EP_PROTOCOL", "v2"),
            ("ATLAS_RDMA_ALLREDUCE", "1"),
            ("ATLAS_RDMA_ONESHOT", "1"),
            ("ATLAS_RDMA_PAIR_CHAIN", "1"),
        ],
    ) {
        return;
    }
    tp4_dflash_resolves();
}

fn tp4_dflash_resolves() {
    for rank in 0..4 {
        let mut a = args(4, rank);
        (a.world_size, a.tp_size, a.ep_size) = (4, 4, 4);
        a.dflash = true;
        a.dflash_gamma = Some(8);
        a.max_seq_len = 1 << 20;
        a.max_prefill_tokens = 8192;
        a.ssm_cache_slots = 0;
        a.enable_prefix_caching = false;
        let mut cfg = config();
        let (topology, reserve) = prepare_reserve(&a, &mut cfg, 512usize << 30)
            .unwrap_or_else(|e| panic!("TP4 long-context preparation failed: {e:#}"));
        let topology = topology.expect("resolved topology must be retained for handoff");
        assert_eq!((topology.tp_size, topology.ep_size), (4, 4));
        assert_eq!((topology.tp_rank, topology.ep_rank), (rank, rank));
        assert_eq!((cfg.num_attention_heads, cfg.num_key_value_heads), (16, 16));
        assert_eq!(
            (cfg.linear_num_key_heads, cfg.linear_num_value_heads),
            (16, 16)
        );
        assert_eq!(reserve.max_batch_tokens_pre, 8196);
    }
}

const TP4_CHAIN: [(&str, &str); 7] = [
    ("ATLAS_GLM_MTP_LONG_CONTEXT", "1"),
    ("ATLAS_GLM_DFLASH", "1"),
    ("ATLAS_EP_PROTOCOL", "v2"),
    ("ATLAS_RDMA_ALLREDUCE", "1"),
    ("ATLAS_RDMA_ONESHOT", "1"),
    ("ATLAS_RDMA_PAIR_CHAIN", "1"),
    ("ATLAS_GLM_FP4L_NATIVE", "1"),
];

const NATIVE_TARGET: [(&str, &str); 7] = [
    ("ATLAS_GLM_FP4L_MLA_FP8", "1"),
    ("ATLAS_GLM_FP4L_DENSE_MXFP8", "1"),
    ("ATLAS_GLM_FP4L_KDA_W8A8_PREFILL", "1"),
    ("ATLAS_GLM_FP4L_INPUT_SCALES", "1"),
    ("ATLAS_GLM_FP4L_FP8_PREFILL", "w8a8"),
    ("ATLAS_GLM_LM_HEAD_MXFP8", "1"),
    ("ATLAS_GLM_FP4L_KDA_TILED_TWIN", "0"),
];

/// The launcher's Spark flags (`serve.py serving_mode`) on the chain.
#[test]
fn spark_mode_resolves_the_tp4_chain() {
    let flags: Vec<_> = TP4_CHAIN
        .into_iter()
        .chain(NATIVE_TARGET)
        .chain([
            ("ATLAS_GLM_SERVING_MODE", "spark"),
            ("ATLAS_GLM_SHARED_TP_SPLIT", "0"),
            ("ATLAS_GLM_FP4L_KDA_MXFP8_DECODE", "0"),
            ("ATLAS_GLM_FP4L_DECODE_A16", "0"),
        ])
        .collect();
    if isolated("long_context::spark_mode_resolves_the_tp4_chain", &flags) {
        return;
    }
    tp4_dflash_resolves();
}

/// The launcher's Strict flags (`serve.py serving_mode`) on the chain.
#[test]
fn strict_mode_resolves_the_tp4_chain() {
    let flags: Vec<_> = TP4_CHAIN
        .into_iter()
        .chain(NATIVE_TARGET)
        .chain([
            ("ATLAS_GLM_SERVING_MODE", "strict"),
            ("ATLAS_GLM_FP4L_SHARED_FP8", "1"),
            ("ATLAS_GLM_FP4L_KDA_MXFP8_DECODE", "1"),
            ("ATLAS_GLM_FP4L_DECODE_A16", "1"),
            ("ATLAS_NO_DFLASH_DRAFTER_NVFP4", "1"),
            ("ATLAS_HC_PERSEQ_DECODE", "1"),
            ("ATLAS_GLM_CANONICAL_VERIFY", "1"),
            ("ATLAS_DFLASH_DRAFTER_FP8", "0"),
            ("ATLAS_DFLASH_NVFP4_TC", "0"),
            ("ATLAS_DFLASH_NVFP4_HEAD", "0"),
            ("ATLAS_DFLASH_MXFP8", "0"),
            ("ATLAS_DFLASH_MXFP8_HEAD", "0"),
            ("ATLAS_DFLASH_DROP_BF16", "0"),
            ("ATLAS_DFLASH_CTX_NVFP4", "0"),
            ("ATLAS_DFLASH_CONV_TWIN", "0"),
            ("ATLAS_GLM_DRAFT_TP", "0"),
            ("ATLAS_GLM_DRAFT_TP_BATCH", "0"),
            ("ATLAS_GLM_DRAFT_TP_CTX", "0"),
            ("ATLAS_GLM_DRAFT_TP_TOPK", "0"),
        ])
        .collect();
    if isolated("long_context::strict_mode_resolves_the_tp4_chain", &flags) {
        return;
    }
    tp4_dflash_resolves();
}

/// Four ranks without the chain's transport refuse at the GLM gate.
#[test]
fn tp4_without_the_rdma_chain_refuses() {
    if isolated(
        "long_context::tp4_without_the_rdma_chain_refuses",
        &[
            ("ATLAS_GLM_MTP_LONG_CONTEXT", "1"),
            ("ATLAS_GLM_DFLASH", "1"),
            ("ATLAS_EP_PROTOCOL", "v2"),
        ],
    ) {
        return;
    }
    let mut a = args(4, 0);
    (a.world_size, a.tp_size, a.ep_size) = (4, 4, 4);
    a.dflash = true;
    a.dflash_gamma = Some(8);
    a.max_seq_len = 32768;
    let err = prepare_reserve(&a, &mut config(), 512usize << 30)
        .map(|_| ())
        .unwrap_err();
    assert!(format!("{err:#}").contains("RDMA chain"), "{err:#}");
}

#[test]
fn long_context_lane_refuses_without_dflash() {
    if isolated(
        "long_context::long_context_lane_refuses_without_dflash",
        &[("ATLAS_GLM_MTP_LONG_CONTEXT", "1")],
    ) {
        return;
    }
    let mut a = args(1, 0);
    a.max_seq_len = 32768;
    let err = prepare_reserve(&a, &mut config(), 128usize << 30)
        .map(|_| ())
        .unwrap_err();
    assert!(format!("{err:#}").contains("GLM DFlash lane"), "{err:#}");
}

#[test]
fn strict_native_mtp_retains_c4_full_context_and_prefix_cache() {
    if isolated(
        "long_context::strict_native_mtp_retains_c4_full_context_and_prefix_cache",
        &[
            ("ATLAS_GLM_MTP_LONG_CONTEXT", "1"),
            ("ATLAS_GLM_STRICT_NATIVE_MTP", "1"),
            ("ATLAS_GLM_SERVING_MODE", "strict"),
            ("ATLAS_GLM_FP4L_NATIVE", "1"),
            ("ATLAS_GLM_FP4L_MLA_FP8", "1"),
            ("ATLAS_GLM_FP4L_DENSE_MXFP8", "1"),
            ("ATLAS_GLM_FP4L_KDA_W8A8_PREFILL", "1"),
            ("ATLAS_GLM_FP4L_INPUT_SCALES", "1"),
            ("ATLAS_GLM_FP4L_FP8_PREFILL", "w8a8"),
            ("ATLAS_GLM_LM_HEAD_MXFP8", "1"),
            ("ATLAS_GLM_FP4L_KDA_TILED_TWIN", "0"),
            ("ATLAS_GLM_FP4L_SHARED_FP8", "1"),
            ("ATLAS_GLM_FP4L_KDA_MXFP8_DECODE", "1"),
            ("ATLAS_GLM_FP4L_DECODE_A16", "1"),
        ],
    ) {
        return;
    }
    for rank in 0..2 {
        let mut a = args(4, rank);
        a.speculative = true;
        a.num_drafts = Some(6);
        a.max_seq_len = 1_048_576;
        a.max_prefill_tokens = 8192;
        a.enable_prefix_caching = true;
        let (topology, reserve) = prepare_reserve(&a, &mut config(), 128usize << 30).unwrap();
        assert_eq!(topology.unwrap().tp_rank, rank);
        assert_eq!(
            reserve.inference_reserve,
            reserve.gdn_two_phase_bytes + (512usize << 20)
        );
        // Mixing speculative families must not gain the native exception.
        a.speculative = false;
        a.dflash = true;
        assert!(prepare_reserve(&a, &mut config(), 128usize << 30).is_err());
        a.speculative = true;
        a.dflash = false;
        a.max_num_seqs = 5;
        assert!(prepare_reserve(&a, &mut config(), 128usize << 30).is_err());
        a.max_num_seqs = 4;
        a.max_seq_len += 1;
        assert!(prepare_reserve(&a, &mut config(), 128usize << 30).is_err());
    }
}

#[test]
fn strict_native_tp4_resolves_all_ranks_for_c1_through_c4_full_context() {
    if isolated(
        "long_context::strict_native_tp4_resolves_all_ranks_for_c1_through_c4_full_context",
        &[
            ("ATLAS_GLM_MTP_LONG_CONTEXT", "1"),
            ("ATLAS_GLM_STRICT_NATIVE_MTP", "1"),
            ("ATLAS_GLM_SERVING_MODE", "strict"),
            ("ATLAS_GLM_FP4L_NATIVE", "1"),
            ("ATLAS_GLM_FP4L_MLA_FP8", "1"),
            ("ATLAS_GLM_FP4L_DENSE_MXFP8", "1"),
            ("ATLAS_GLM_FP4L_KDA_W8A8_PREFILL", "1"),
            ("ATLAS_GLM_FP4L_INPUT_SCALES", "1"),
            ("ATLAS_GLM_FP4L_FP8_PREFILL", "w8a8"),
            ("ATLAS_GLM_LM_HEAD_MXFP8", "1"),
            ("ATLAS_GLM_FP4L_KDA_TILED_TWIN", "0"),
            ("ATLAS_GLM_FP4L_SHARED_FP8", "1"),
            ("ATLAS_GLM_FP4L_KDA_MXFP8_DECODE", "1"),
            ("ATLAS_GLM_FP4L_DECODE_A16", "1"),
            ("ATLAS_RDMA_ALLREDUCE", "1"),
            ("ATLAS_RDMA_ONESHOT", "1"),
            ("ATLAS_RDMA_PAIR_CHAIN", "1"),
        ],
    ) {
        return;
    }
    for capacity in 1..=4 {
        for rank in 0..4 {
            let mut a = args(capacity, rank);
            a.world_size = 4;
            a.tp_size = 4;
            a.ep_size = 4;
            a.speculative = true;
            a.num_drafts = Some(6);
            a.max_seq_len = 1_048_576;
            a.max_prefill_tokens = 8192;
            a.enable_prefix_caching = true;
            let mut cfg = config();
            let (topology, reserve) = prepare_reserve(&a, &mut cfg, 128usize << 30)
                .unwrap_or_else(|error| panic!("TP4 rank {rank} C{capacity}: {error:#}"));
            let topology = topology.unwrap();
            assert_eq!(
                (topology.world_size, topology.tp_size, topology.ep_size),
                (4, 4, 4)
            );
            assert_eq!((topology.tp_rank, topology.ep_rank), (rank, rank));
            assert_eq!((cfg.num_attention_heads, cfg.num_key_value_heads), (16, 16));
            assert_eq!(
                (cfg.linear_num_key_heads, cfg.linear_num_value_heads),
                (16, 16)
            );
            let rows = a.max_prefill_tokens + capacity;
            assert_eq!(reserve.max_batch_tokens_pre, rows);
            assert_eq!(
                reserve.buffer_arena_bytes,
                spark_runtime::buffers::BufferSizes::from_config(
                    &cfg,
                    rows,
                    a.max_seq_len,
                    16,
                    capacity
                )
                .total_bytes()
            );
            assert_eq!(
                reserve.inference_reserve,
                reserve.gdn_two_phase_bytes + (512usize << 20)
            );
            let total = reserve.inference_reserve + reserve.buffer_arena_bytes;
            assert!(prepare_reserve(&a, &mut config(), total).is_ok());
            assert!(prepare_reserve(&a, &mut config(), total - 1).is_err());
            a.max_num_seqs = capacity + 1;
            assert!(prepare_reserve(&a, &mut config(), 128usize << 30).is_err());
        }
    }
}
