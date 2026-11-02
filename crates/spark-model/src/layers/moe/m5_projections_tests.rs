// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn config() -> ModelConfig {
    ModelConfig {
        model_type: "glm5_next".into(),
        hidden_size: 4096,
        moe_intermediate_size: 2048,
        shared_expert_intermediate_size: 2048,
        num_experts: 288,
        num_experts_per_tok: 8,
        scoring_func: "sigmoid".into(),
        tp_world_size: 2,
        ep_world_size: 2,
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}
fn router() -> RouterResources {
    RouterResources {
        comm: true,
        lora: false,
        bf16: true,
        pre_norm: false,
        hash: false,
        bias: true,
        old_m5: true,
    }
}
fn shared() -> SharedResources {
    SharedResources {
        comm: true,
        lora: false,
        nvfp4: true,
        transposed: true,
        fp8_cache: false,
        bf16_override: false,
        exact_gemv: false,
    }
}

#[test]
fn actual_m5_selectors_accept_only_existing_precision_paths() {
    assert!(router_eligible(&config(), 5, router()));
    assert!(shared_eligible(&config(), 5, shared()));
    for rows in [0, 1, 2, 3, 4, 6, 64, u32::MAX] {
        assert!(!router_eligible(&config(), rows, router()));
        assert!(!shared_eligible(&config(), rows, shared()));
    }
}

#[test]
fn each_router_resource_conflict_preserves_original() {
    let edits: [fn(&mut RouterResources); 7] = [
        |r| r.comm = false,
        |r| r.lora = true,
        |r| r.bf16 = false,
        |r| r.pre_norm = true,
        |r| r.hash = true,
        |r| r.bias = false,
        |r| r.old_m5 = false,
    ];
    for edit in edits {
        let mut r = router();
        edit(&mut r);
        assert!(!router_eligible(&config(), 5, r));
    }
}

#[test]
fn each_shared_cache_or_precision_conflict_preserves_original() {
    let edits: [fn(&mut SharedResources); 7] = [
        |r| r.comm = false,
        |r| r.lora = true,
        |r| r.nvfp4 = false,
        |r| r.transposed = false,
        |r| r.fp8_cache = true,
        |r| r.bf16_override = true,
        |r| r.exact_gemv = true,
    ];
    for edit in edits {
        let mut r = shared();
        edit(&mut r);
        assert!(!shared_eligible(&config(), 5, r));
    }
}

#[test]
fn every_target_geometry_change_and_mtp_tp1_preserves_original() {
    let edits: [fn(&mut ModelConfig); 9] = [
        |c| c.model_type = "other".into(),
        |c| c.hidden_size = 2048,
        |c| c.moe_intermediate_size = 1024,
        |c| c.shared_expert_intermediate_size = 1024,
        |c| c.num_experts = 256,
        |c| c.num_experts_per_tok = 4,
        |c| c.tp_world_size = 1,
        |c| c.ep_world_size = 1,
        |c| c.scoring_func = "softmax".into(),
    ];
    for edit in edits {
        let mut c = config();
        edit(&mut c);
        assert!(!router_eligible(&c, 5, router()));
        assert!(!shared_eligible(&c, 5, shared()));
    }
}

#[test]
fn independent_strict_flags_and_actual_width_graph_guard() {
    for (main, verify, ok) in [
        (None, None, true),
        (Some("0"), Some("0"), true),
        (Some("1"), Some("1"), true),
        (Some("0"), Some("1"), false),
        (Some("true"), None, false),
    ] {
        assert_eq!(Toggle::parse(main, verify).is_ok(), ok);
    }
    for router in [false, true] {
        for shared in [false, true] {
            for rows in [1, 2, 3, 4, 5, 6] {
                for graphs in [false, true] {
                    assert_eq!(
                        reject_graphs("glm5_next", rows, graphs, router, shared).is_err(),
                        rows == 5 && graphs && (router || shared)
                    );
                    assert!(reject_graphs("other", rows, graphs, router, shared).is_ok());
                }
            }
        }
    }
}

#[test]
fn checked_span_rejects_every_boundary_independently() {
    use spark_runtime::gpu::DevicePtr as P;
    assert!(span(P(0x1000), 64, 16).is_ok());
    assert!(span(P(0), 64, 16).is_err());
    assert!(span(P(0x1002), 64, 16).is_err());
    assert!(span(P(u64::MAX - 15), 32, 16).is_err());
    assert!(span(P(0x1000), 0, 16).is_err());
    assert!(span(P(0x1000), 64, 0).is_err());
    assert!(span(P(0x1000), 64, 3).is_err());
    assert!(disjoint(&(100..200), &(200..300)).is_ok());
    assert!(disjoint(&(100..201), &(200..300)).is_err());
}

#[test]
fn actual_feature_marks_only_success_and_each_projection_is_independent() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use std::cell::Cell;
    let gpu = MockGpuBackend::new();
    let output = gpu.alloc(16).unwrap();
    let feature = ProjectionFeature {
        kernel: KernelHandle(2),
        verify: true,
        checked: AtomicU32::new(0),
        selected: AtomicU32::new(0),
        label: "test",
    };
    let calls = Cell::new(0);
    let run = |bit, fault| {
        feature.run(
            bit,
            output,
            16,
            KernelHandle(1),
            &gpu,
            false,
            0,
            false,
            |k| {
                calls.set(calls.get() + 1);
                if fault && k.0 == 2 {
                    anyhow::bail!("candidate failure");
                }
                gpu.copy_h2d(&[0x3f; 16], output)
            },
        )
    };
    assert!(run(1, true).is_err());
    assert_eq!(feature.checked.load(Ordering::Relaxed), 0);
    calls.set(0);
    run(1, false).unwrap();
    assert_eq!(calls.get(), 2);
    calls.set(0);
    run(1, false).unwrap();
    assert_eq!(calls.get(), 1);
    calls.set(0);
    run(2, false).unwrap();
    assert_eq!(calls.get(), 2);
    assert_eq!(feature.checked.load(Ordering::Relaxed), 3);
    assert!(
        feature
            .run(
                1,
                output,
                16,
                KernelHandle(1),
                &gpu,
                true,
                0,
                false,
                |_| panic!("capture launch")
            )
            .is_err()
    );
    assert_eq!(
        gpu.alloc_count(),
        1,
        "no ordinary or diagnostic GPU allocations"
    );
}
