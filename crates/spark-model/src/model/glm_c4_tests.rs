// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn policy() -> C4Policy<'static> {
    C4Policy {
        model_type: "glm5_next",
        enabled: true,
        world: 2,
        tp: 2,
        ep: 2,
        ep_v2: true,
        independent: true,
        kda_multi: true,
        mla_multi: true,
        sparse: false,
        sparse_graphs: false,
        c4_sparse: false,
        no_multiseq_graphs: false,
        kv_overcommit_disabled: false,
        fp32_state: true,
    }
}

#[test]
fn c4_sparse_requires_explicit_eager_bounded_policy() {
    let long = || C4Policy {
        c4_sparse: true,
        sparse: true,
        no_multiseq_graphs: true,
        kv_overcommit_disabled: true,
        ..policy()
    };
    assert!(long().validate().is_ok());
    for bad in [
        C4Policy {
            enabled: false,
            ..long()
        },
        C4Policy {
            sparse: false,
            ..long()
        },
        C4Policy {
            sparse_graphs: true,
            ..long()
        },
        C4Policy {
            no_multiseq_graphs: false,
            ..long()
        },
        C4Policy {
            kv_overcommit_disabled: false,
            ..long()
        },
        C4Policy {
            independent: false,
            ..long()
        },
        C4Policy {
            fp32_state: false,
            ..long()
        },
        C4Policy {
            mla_multi: false,
            ..long()
        },
    ] {
        assert!(bad.validate().is_err());
    }
}

#[test]
fn c4_sparse_flag_and_context_boundaries_fail_closed() {
    assert!(validate_prefill_budget(1, true).is_ok());
    assert!(validate_prefill_budget(1024, true).is_ok());
    for invalid in [0, 1025, 6144, usize::MAX] {
        assert!(validate_prefill_budget(invalid, true).is_err());
        assert!(validate_prefill_budget(invalid, false).is_ok());
    }
    assert!(!validate_sparse_flag("glm5_next", None).unwrap());
    assert!(!validate_sparse_flag("other", Some("0")).unwrap());
    assert!(validate_sparse_flag("glm5_next", Some("1")).unwrap());
    assert!(validate_sparse_flag("other", Some("1")).is_err());
    assert!(validate_sparse_flag("glm5_next", Some("true")).is_err());
    for limit in [2048, SPARSE_CONTEXT_LIMIT] {
        let sparse = limit > 2048;
        assert!(validate_launch_limits_for_mode(4, 4, limit, true, sparse).is_ok());
        for (active, admitted, context, bf16) in [
            (4, 4, 0, true),
            (4, 4, limit + 1, true),
            (3, 4, limit, true),
            (4, 5, limit, true),
            (4, 4, limit, false),
        ] {
            assert!(
                validate_launch_limits_for_mode(active, admitted, context, bf16, sparse).is_err()
            );
        }
        for rows in 2..=4 {
            assert!(validate_positions_for_mode(vec![limit - 1; rows], rows, sparse).is_ok());
            assert!(validate_positions_for_mode(vec![limit; rows], rows, sparse).is_err());
            assert!(validate_positions_for_mode(vec![usize::MAX; rows], rows, sparse).is_err());
            assert!(validate_positions_for_mode(vec![0; rows - 1], rows, sparse).is_err());
        }
    }
    assert!(validate_positions_for_mode([0], 1, true).is_err());
    assert!(validate_positions_for_mode([0; 5], 5, true).is_err());
}

#[test]
fn exact_opt_in_topology_and_modes_are_mandatory() {
    assert!(policy().validate().is_ok());
    for invalid in [
        C4Policy {
            enabled: false,
            ..policy()
        },
        C4Policy {
            model_type: "qwen3_next",
            ..policy()
        },
        C4Policy {
            world: 1,
            ..policy()
        },
        C4Policy { tp: 1, ..policy() },
        C4Policy { ep: 1, ..policy() },
        C4Policy {
            ep_v2: false,
            ..policy()
        },
        C4Policy {
            independent: false,
            ..policy()
        },
        C4Policy {
            kda_multi: false,
            ..policy()
        },
        C4Policy {
            mla_multi: false,
            ..policy()
        },
        C4Policy {
            sparse: true,
            ..policy()
        },
        C4Policy {
            sparse_graphs: true,
            ..policy()
        },
        C4Policy {
            fp32_state: false,
            ..policy()
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn initial_launch_caps_are_four_short_independent_sessions() {
    assert!(validate_launch_limits(4, 4, 1024, true).is_ok());
    assert!(validate_launch_limits(4, 4, 2048, true).is_ok());
    for (active, admitted, context, bf16) in [
        (3, 4, 1024, true),
        (4, 5, 1024, true),
        (5, 5, 1024, true),
        (4, 4, 2049, true),
        (4, 4, 0, true),
        (4, 4, 1024, false),
    ] {
        assert!(validate_launch_limits(active, admitted, context, bf16).is_err());
    }
}

#[test]
fn c4_never_falls_back_to_scalar_kda_dispatch() {
    assert!(batched_kda_rows(4, true, true).unwrap());
    assert!(batched_kda_rows(4, true, false).is_err());
    assert!(batched_kda_rows(4, false, true).is_err());
    for n in [2, 3] {
        assert!(batched_kda_rows(n, true, false).unwrap());
        assert!(!batched_kda_rows(n, false, false).unwrap());
    }
    assert!(!batched_kda_rows(1, true, true).unwrap());
}

#[test]
fn replay_positions_and_projection_handles_fail_closed() {
    assert!(validate_positions([0, 15, 1023, 2047], 4).is_ok());
    assert!(validate_positions([2048, 0, 1, 2], 4).is_err());
    assert!(validate_positions([usize::MAX, 0, 1, 2], 4).is_err());
    assert!(validate_positions([0, 1, 2], 4).is_err());
    assert!(validate_projection_handles(1, 2).is_ok());
    assert!(validate_projection_handles(0, 2).is_err());
    assert!(validate_projection_handles(1, 0).is_err());
    assert_eq!(hc_bytes(4).unwrap(), (262144, 64, 256));
    assert!(hc_bytes(usize::MAX).is_err());
    assert!(hc_bytes(0).is_err());
}

#[test]
fn every_c4_arena_is_checked_against_its_live_dtype_and_rows() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 4096;
    config.hc_mult = 4;
    let mut sizes = BufferSizes::from_config(&config, 1024, 2048, 16, 4);
    assert!(validate_scratch(&sizes, 4).is_ok());
    sizes.qkv_output = 4 * (3 * 4096 + 32 + 128 + 128) * 2;
    assert_eq!(sizes.qkv_output, 100608);
    assert!(validate_scratch(&sizes, 4).is_ok());
    sizes.qkv_output -= 1;
    assert!(validate_scratch(&sizes, 4).is_err());
    sizes.qkv_output += 1;
    sizes.hc_streams = 4 * 4 * 4096 * 2; // BF16-sized is insufficient for FP32 highway.
    assert!(validate_scratch(&sizes, 4).is_err());
    sizes.hc_streams *= 2;
    sizes.expert_up_out = 131071;
    assert!(validate_scratch(&sizes, 4).is_err());
}
