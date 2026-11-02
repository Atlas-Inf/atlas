// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 beyond two ranks: the parallelism gate and the two-rank lane table.

use super::*;

fn env<'a>(set: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| {
        set.iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.to_string())
    }
}

fn refusal(r: Result<()>) -> String {
    format!("{:#}", r.unwrap_err())
}

#[test]
fn the_pair_shapes_stay_and_four_ranks_need_the_chain() {
    for (world, tp, ep) in [(2, 1, 2), (2, 2, 2)] {
        assert!(parallelism_supported(world, tp, ep, false).is_ok());
    }
    assert!(parallelism_supported(4, 4, 4, true).is_ok());
    let said = refusal(parallelism_supported(4, 4, 4, false));
    assert!(said.contains("set ATLAS_RDMA_ALLREDUCE=1"), "{said}");
    for (world, tp, ep) in [
        (2, 2, 1),
        (4, 2, 2),
        (2, 4, 2),
        (4, 4, 2),
        (3, 3, 3),
        (8, 8, 8),
    ] {
        let said = refusal(parallelism_supported(world, tp, ep, true));
        assert!(
            said.contains("--world-size 4 --tp-size 4 --ep-size 4"),
            "{said}"
        );
    }
}

#[test]
fn two_ranks_take_every_lane() {
    let all: Vec<(&str, &str)> = LANES.iter().map(|(n, _)| (*n, "1")).collect();
    assert!(check(2, env(&all)).is_ok());
}

#[test]
fn four_ranks_refuse_every_failing_lane_at_once() {
    let set = [
        ("ATLAS_GLM_SPARSE_NATIVE", "1"),
        ("ATLAS_GLM_KV_SHARD_OVERLAP", "1"),
        ("ATLAS_GLM_KV_SHARD", "1"),
        // Off lanes and switches set to 0 are no refusal.
        ("ATLAS_GLM_PREFILL_SP", "1"),
        ("ATLAS_GLM_C4_DECODE", "0"),
        ("ATLAS_KDA_FLASH_PREFILL", ""),
    ];
    let said = refusal(check(4, env(&set)));
    let named = "ATLAS_GLM_KV_SHARD_OVERLAP=1";
    assert!(said.contains(named), "{named} missing: {said}");
    for unnamed in [
        "PREFILL_SP",
        "C4_DECODE",
        "KDA_FLASH",
        "KV_SHARD=1",
        "SPARSE_NATIVE",
    ] {
        assert!(!said.contains(unnamed), "{unnamed} named: {said}");
    }
}

#[test]
fn four_ranks_boot_with_only_lanes_that_stay_off() {
    let off: Vec<(&str, &str)> = LANES
        .iter()
        .filter(|(_, b)| matches!(b, Off(_)))
        .map(|(n, _)| (*n, "1"))
        .collect();
    assert!(!off.is_empty());
    assert!(check(4, env(&off)).is_ok());
    assert!(check(4, env(&[])).is_ok());
}

#[test]
fn four_ranks_take_the_sixteen_head_attention_lanes() {
    // The sparse MLA kernels mask the rows past a TP4 rank's 16 heads; the
    // long-context lane, its staged projections and the fp8_g128 latent ride
    // them.
    let set: Vec<(&str, &str)> = [
        "ATLAS_GLM_MTP_LONG_CONTEXT",
        "ATLAS_GLM_LONG_BATCH_VERIFY",
        "ATLAS_GLM_SPARSE_DECODE_TC",
        "ATLAS_GLM_SPARSE_NATIVE",
        "ATLAS_GLM_MLA_FUSE",
        "ATLAS_GLM_SPARSE_DECODE_SPLIT",
        "ATLAS_GLM_SPARSE_PREFILL_TC",
        "ATLAS_GLM_SPARSE_PREFILL_KV_REUSE",
        "ATLAS_GLM_SPARSE_PREFILL_PIPE",
        "ATLAS_GLM_K3_MLA_O_BATCHM",
        "ATLAS_GLM_K3_MLA_QUERY_BATCHM",
        "ATLAS_GLM_K3_MLA_O_COMPARE",
        "ATLAS_GLM_MULTI_SEQ_SPARSE",
        "ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS",
        "ATLAS_GLM_VERIFY_GRAPH",
    ]
    .iter()
    .map(|n| (*n, "1"))
    .collect();
    assert!(check(4, env(&set)).is_ok());
}

#[test]
fn four_ranks_take_the_moe_lanes() {
    let moe = [
        "ATLAS_GLM_EXPERT_TP",
        "ATLAS_GLM_SHARED_TP_SPLIT",
        "ATLAS_GLM_TARGET_SHARED_FP8",
        "ATLAS_GLM_C3_GROUPED_MOE",
        "ATLAS_GLM_MOE_GATE_UP_M16",
        "ATLAS_GLM_M5_ROUTER_BN4",
        "ATLAS_GLM_K5_FUSED_MOE_HC",
    ];
    assert!(moe.iter().all(|m| LANES.iter().all(|(n, _)| n != m)));
    let set: Vec<(&str, &str)> = moe.iter().map(|n| (*n, "1")).collect();
    assert!(check(4, env(&set)).is_ok());
}

#[test]
fn four_ranks_take_the_split_verify_head_and_drafter() {
    let head = [
        "ATLAS_GLM_VERIFY_VOCAB_SPLIT",
        "ATLAS_GLM_LM_HEAD_MXFP8",
        "ATLAS_GLM_STRICT_SPEC",
        "ATLAS_GLM_DRAFT_TP",
    ];
    assert!(head.iter().all(|h| LANES.iter().all(|(n, _)| n != h)));
    let set: Vec<(&str, &str)> = head.iter().map(|n| (*n, "1")).collect();
    assert!(check(4, env(&set)).is_ok());
}

#[test]
fn every_lane_is_listed_once() {
    let mut names: Vec<&str> = LANES.iter().map(|(n, _)| *n).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(names.len(), before);
}

#[test]
fn options_that_require_a_refused_lane_are_refused_with_it() {
    // Each would fail its own validation at model build once its lane is
    // off (the pipelined sparse prefill rides the sparse prefill TC, ...).
    for name in ["ATLAS_GLM_C4_GROUPED_MOE", "ATLAS_GLM_K3_MLA_QUERY_COMPARE"] {
        let said = refusal(check(4, env(&[(name, "1")])));
        assert!(said.contains(&format!("{name}=1")), "{said}");
    }
}
