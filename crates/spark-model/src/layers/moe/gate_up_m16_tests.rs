// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::moe::prequant_fp4::CompactMoeWorklist;

fn config() -> ModelConfig {
    ModelConfig {
        model_type: "glm5_next".into(),
        hidden_size: 4096,
        moe_intermediate_size: 2048,
        shared_expert_intermediate_size: 2048,
        num_experts: 288,
        num_experts_per_tok: 8,
        tp_world_size: 2,
        ep_world_size: 2,
        scoring_func: "sigmoid".into(),
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}

fn work(rows: u32) -> CompactMoeWorklist {
    CompactMoeWorklist {
        worklist: DevicePtr(0x10000),
        total_tiles: DevicePtr(0x8000),
        max_tiles: rows * 8 * 16,
    }
}

#[test]
fn eligible_c4_k5_preserve_c3_and_larger_prefill() {
    for rows in [4, 5] {
        assert!(eligible(&config(), rows, 2048, 4096, 288, work(rows), true));
    }
    for rows in [0, 1, 2, 3, 6, 64, u32::MAX] {
        assert!(!eligible(&config(), rows, 2048, 4096, 288, work(4), true));
    }
}

#[test]
fn eligibility_rejects_each_shape_topology_and_native_resource_mismatch() {
    let changes: [fn(&mut ModelConfig); 9] = [
        |c| c.model_type = "deepseek_v4".into(),
        |c| c.hidden_size = 2048,
        |c| c.moe_intermediate_size = 4096,
        |c| c.shared_expert_intermediate_size = 1024,
        |c| c.num_experts = 256,
        |c| c.num_experts_per_tok = 4,
        |c| c.tp_world_size = 1,
        |c| c.ep_world_size = 1,
        |c| c.scoring_func = "softmax".into(),
    ];
    for change in changes {
        let mut c = config();
        change(&mut c);
        assert!(!eligible(&c, 4, 2048, 4096, 288, work(4), true));
    }
    for (n, k, experts) in [(4096, 2048, 288), (2048, 2048, 288), (2048, 4096, 144)] {
        assert!(!eligible(&config(), 4, n, k, experts, work(4), true));
    }
    assert!(!eligible(&config(), 4, 2048, 4096, 288, work(4), false));
}

#[test]
fn eligibility_requires_exact_compact_envelope_and_aligned_nonnull_metadata() {
    for change in [
        (|w: &mut CompactMoeWorklist| w.max_tiles -= 1) as fn(&mut CompactMoeWorklist),
        |w| w.max_tiles += 1,
        |w| w.worklist = DevicePtr(0),
        |w| w.total_tiles = DevicePtr(0),
        |w| w.worklist = DevicePtr(0x10001),
        |w| w.total_tiles = DevicePtr(0x8001),
    ] {
        let mut w = work(4);
        change(&mut w);
        assert!(!eligible(&config(), 4, 2048, 4096, 288, w, true));
    }
}

#[test]
fn verification_guard_uses_actual_graph_decision_not_disable_flag_assumptions() {
    assert!(reject_graphs("glm5_next", true, true).is_err());
    assert!(reject_graphs("glm5_next", true, false).is_ok());
    assert!(reject_graphs("glm5_next", false, true).is_ok());
    assert!(reject_graphs("other", true, true).is_ok());
    assert!(crate::layers::moe::validate_m16_gate_up_graphs("other", true).is_ok());
}

#[test]
fn explicit_toggle_and_verify_dependency_are_fail_closed() {
    assert!(!parse_toggle(None).unwrap());
    assert!(!parse_toggle(Some("0")).unwrap());
    assert!(parse_toggle(Some("1")).unwrap());
    for value in ["", "true", "yes", "2"] {
        assert!(parse_toggle(Some(value)).is_err());
    }
    assert!(validate_toggles(false, true).is_err());
    for (enabled, verify) in [(false, false), (true, false), (true, true)] {
        assert!(validate_toggles(enabled, verify).is_ok());
    }
}

#[test]
fn oracle_spans_are_bounded_disjoint_and_cannot_overflow() {
    let bytes = 4 * 8 * 2048 * 2;
    assert_eq!(
        checked_output_bytes(
            4,
            [DevicePtr(0x100000), DevicePtr(0x200000)],
            [bytes; 2],
            DevicePtr(0x300000)
        )
        .unwrap(),
        bytes
    );
    for pointers in [
        [DevicePtr(0), DevicePtr(0x200000)],
        [DevicePtr(0x100001), DevicePtr(0x200000)],
        [DevicePtr(0x100000), DevicePtr(0x100000 + bytes as u64 - 2)],
        [DevicePtr(u64::MAX - 1), DevicePtr(0x200000)],
        [DevicePtr(0x300000), DevicePtr(0x200000)],
    ] {
        assert!(checked_output_bytes(4, pointers, [bytes; 2], DevicePtr(0x300000)).is_err());
    }
    assert!(
        checked_output_bytes(
            4,
            [DevicePtr(0x100000), DevicePtr(0x200000)],
            [bytes - 1, bytes],
            DevicePtr(0x300000)
        )
        .is_err()
    );
    assert!(
        checked_output_bytes(
            u32::MAX,
            [DevicePtr(0x100000), DevicePtr(0x200000)],
            [usize::MAX; 2],
            DevicePtr(0x300000)
        )
        .is_err()
    );
}

#[test]
fn full_output_comparison_rejects_tail_mismatch_and_projection_swap() {
    assert!(compare_output("gate", &[1, 2, 3, 4], &[1, 2, 3, 4]).is_ok());
    assert!(compare_output("up", &[1, 2, 3, 4], &[1, 2, 3, 5]).is_err());
    assert!(compare_output("gate", &[1, 2], &[3, 4]).is_err());
    assert!(compare_output("gate", &[1, 2], &[1]).is_err());
}

#[test]
fn real_oracle_sequence_repoisons_and_detects_omitted_candidate_up_writes() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use std::cell::Cell;
    for omit_up in [false, true] {
        let gpu = MockGpuBackend::new();
        let outputs = [gpu.alloc(16).unwrap(), gpu.alloc(16).unwrap()];
        let calls = Cell::new(0);
        let result = verify_outputs(
            &gpu,
            0,
            outputs,
            16,
            KernelHandle(1),
            KernelHandle(2),
            |kernel| {
                assert_eq!(kernel.0, calls.get() + 1);
                calls.set(calls.get() + 1);
                for (i, output) in outputs.into_iter().enumerate() {
                    let mut before = [0; 16];
                    gpu.copy_d2h(output, &mut before)?;
                    assert_eq!(
                        before, [0x5a; 16],
                        "each projection must be freshly poisoned"
                    );
                    if !(omit_up && kernel.0 == 2 && i == 1) {
                        gpu.copy_h2d(&[10 + i as u8; 16], output)?;
                    }
                }
                Ok(())
            },
        );
        assert_eq!(calls.get(), 2);
        assert_eq!(result.is_ok(), !omit_up);
    }
}
