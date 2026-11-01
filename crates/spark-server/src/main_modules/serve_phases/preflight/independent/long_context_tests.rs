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
