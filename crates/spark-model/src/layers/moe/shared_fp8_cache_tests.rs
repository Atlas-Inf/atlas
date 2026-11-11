// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn config() -> ModelConfig {
    ModelConfig {
        model_type: "glm5_next".into(),
        hidden_size: 4096,
        moe_intermediate_size: 2048,
        shared_expert_intermediate_size: 2048,
        num_hidden_layers: 45,
        mlp_only_layers: vec![0, 1, 2],
        num_experts: 288,
        num_experts_per_tok: 8,
        tp_world_size: 2,
        ep_world_size: 2,
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}

#[test]
fn target_shared_plan_counts_replicated_weights_and_excludes_mtp() {
    let c = config();
    let p = target_plan(&c, 3, true, true).unwrap().unwrap();
    assert_eq!(p.total_bytes, 1_056_964_608);
    assert_eq!(p.remaining_bytes, p.total_bytes);
    assert_eq!(
        target_plan(&c, 44, true, true)
            .unwrap()
            .unwrap()
            .remaining_bytes,
        25_165_824
    );
    assert!(target_plan(&c, 45, false, true).unwrap().is_none());
    assert!(target_plan(&c, 3, true, false).unwrap().is_none());
    assert!(target_plan(&c, 45, true, true).is_err());
    assert!(target_plan(&c, 0, true, true).is_err());
}

#[test]
fn target_shared_plan_rejects_geometry_and_invalid_dense_map() {
    for change in [
        (|c: &mut ModelConfig| c.hidden_size = 2048) as fn(&mut ModelConfig),
        |c| c.moe_intermediate_size = 1024,
        |c| c.shared_expert_intermediate_size = 1024,
        |c| c.tp_world_size = 1,
        |c| c.ep_world_size = 1,
        |c| c.num_hidden_layers = usize::MAX,
        |c| c.mlp_only_layers = vec![0, 0, 1],
        |c| c.mlp_only_layers = vec![0, 1, 45],
    ] {
        let mut c = config();
        change(&mut c);
        assert!(target_plan(&c, 3, true, true).is_err());
    }
}

#[test]
fn target_shared_reserve_preserves_four_gib_floor_without_overflow() {
    let reserve_bytes = 1_056_964_608;
    let floor = 4 * 1024 * 1024 * 1024usize;
    let reserve = SharedFp8Reserve::new(0, 0).unwrap();
    assert!(reserve.check(floor + reserve_bytes, reserve_bytes).is_ok());
    assert!(
        reserve
            .check(floor + reserve_bytes - 1, reserve_bytes)
            .is_err()
    );
    assert!(reserve.check(0, reserve_bytes).is_err());
    assert!(reserve.check(usize::MAX, usize::MAX).is_err());
}

#[test]
fn transaction_frees_all_partial_allocations_and_publishes_only_complete_triple() {
    use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
    for fail in 0..3 {
        let gpu = MockGpuBackend::new();
        let result = transaction(
            || gpu.alloc(16),
            |p| gpu.free(p),
            |i, _| {
                anyhow::ensure!(i != fail, "injected launch/sync/byte-oracle failure");
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(gpu.alloc_count(), 0);
    }
    for fail in 0..3 {
        let gpu = MockGpuBackend::new();
        let mut index = 0;
        let result = transaction(
            || {
                let i = index;
                index += 1;
                anyhow::ensure!(i != fail, "injected allocation failure");
                gpu.alloc(16)
            },
            |p| gpu.free(p),
            |_, _| Ok(()),
        );
        assert!(result.is_err());
        assert_eq!(gpu.alloc_count(), 0);
    }
    let gpu = MockGpuBackend::new();
    let outputs = transaction(
        || gpu.alloc(16),
        |p| gpu.free(p),
        |i, p| gpu.copy_h2d(&[i as u8; 16], p),
    )
    .unwrap();
    assert_eq!(gpu.alloc_count(), 3);
    for (i, p) in outputs.into_iter().enumerate() {
        let mut bytes = [0; 16];
        gpu.copy_d2h(p, &mut bytes).unwrap();
        assert_eq!(bytes, [i as u8; 16]);
        gpu.free(p).unwrap();
    }
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn transaction_attempts_remaining_cleanup_after_one_cleanup_error() {
    let mut next = 0;
    let cleaned = std::cell::RefCell::new(Vec::new());
    let result = transaction(
        || {
            next += 1;
            Ok(DevicePtr(next))
        },
        |p| {
            cleaned.borrow_mut().push(p.0);
            anyhow::ensure!(p.0 != 1, "injected free error");
            Ok(())
        },
        |i, _| {
            anyhow::ensure!(i != 2, "injected conversion error");
            Ok(())
        },
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("injected free error")
    );
    assert_eq!(*cleaned.borrow(), vec![1, 2, 3]);
}

#[test]
fn factory_reserve_preserves_arena_and_inference_without_double_cache_debit() {
    let cache = 1_056_964_608;
    let arena = 314_572_800;
    let inference = 5 * 1024 * 1024 * 1024usize;
    assert_eq!(
        factory_required(arena, inference, cache).unwrap(),
        arena + inference + cache
    );
    assert_eq!(
        factory_required(arena, inference, 0).unwrap(),
        arena + inference
    );
    for (a, i, c) in [(usize::MAX, 1, 0), (1, usize::MAX, 0), (0, 1, usize::MAX)] {
        assert!(factory_required(a, i, c).is_err());
    }
}

#[test]
fn deferred_reserve_keeps_future_arena_inference_and_physical_floor() {
    let arena = 3_121_929_980;
    let inference = 5_368_709_120;
    let cache = 1_056_964_608;
    let early_free = 8_832_421_888;
    assert!(early_free >= factory_required(arena, inference, 0).unwrap());
    let budget = SharedFp8Reserve::new(arena, inference).unwrap();
    assert!(
        budget.check(early_free, cache).is_err(),
        "cache cannot yet fit at early transient point"
    );
    assert!(budget.check(arena + inference + cache, cache).is_ok());
    assert!(budget.check(arena + inference + cache - 1, cache).is_err());
    assert!(budget.check(arena + inference, 0).is_ok());
    let floor = 4 * 1024 * 1024 * 1024;
    let small = SharedFp8Reserve::new(1, 1).unwrap();
    assert!(small.check(floor + cache, cache).is_ok());
    assert!(small.check(floor + cache - 1, cache).is_err());
    assert!(SharedFp8Reserve::new(usize::MAX, 1).is_err());
    assert!(small.check(usize::MAX, usize::MAX).is_err());
}

#[test]
fn verify_profile_requires_explicit_no_overlap_before_weight_loading() {
    assert!(validate_verify_overlap(true, Some("0")).is_ok());
    for value in [None, Some("1"), Some("true"), Some("false"), Some("")] {
        assert!(validate_verify_overlap(true, value).is_err());
        assert!(validate_verify_overlap(false, value).is_ok());
    }
}

#[test]
fn explicit_2048_profile_checks_flag_and_configured_budget_before_loading() {
    const CHILD: &str = "ATLAS_TEST_SHARED_2048_PROFILE";
    if let Ok(value) = std::env::var(CHILD) {
        let enabled = matches!(value.as_str(), "1");
        if matches!(value.as_str(), "bad") {
            assert!(prefill_2048_enabled().is_err());
            assert!(validate_shared_fp8_cache_profile(&config(), 1024, false).is_err());
            return;
        }
        assert_eq!(prefill_2048_enabled().unwrap(), enabled);
        for rows in [1, 1024, 1025, 2048, 2049, 2052, 0] {
            assert_eq!(
                validate_shared_fp8_cache_profile(&config(), rows, false).is_ok(),
                (1..=if enabled { 2048 } else { 1024 }).contains(&rows)
            );
        }
        return;
    }
    for value in ["0", "1", "bad"] {
        let full = concat!(
            module_path!(),
            "::explicit_2048_profile_checks_flag_and_configured_budget_before_loading"
        );
        let test = full.split_once("::").expect("crate-qualified test").1;
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test])
            .env(CHILD, value)
            .env("ATLAS_GLM_PREFILL_2048", value)
            .env("ATLAS_GLM_PREFILL_4096", "0")
            .env("ATLAS_GLM_TARGET_SHARED_FP8", "1")
            .env("ATLAS_GLM_TARGET_SHARED_FP8_VERIFY", "0")
            .env("ATLAS_GLM_K5_BATCHED_SHARED", "0")
            .env("ATLAS_NVFP4_MMQ_MOE", "0")
            .env("ATLAS_MOE_GROUPED_CUTLASS", "0")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("running 1 test"),
            "profile={value}: {stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn explicit_4096_profile_bounds_and_conflicting_flags_fail_before_loading() {
    const CHILD: &str = "ATLAS_TEST_SHARED_4096_PROFILE";
    if let Ok(case) = std::env::var(CHILD) {
        for rows in [0, 1, 1024, 1025, 2048, 2049, 4096, 4097, 4100, 8196] {
            assert_eq!(
                validate_shared_fp8_cache_profile(&config(), rows, false).is_ok(),
                match case.as_str() {
                    "on" => rows >= 1,
                    "off" => (1..=1024).contains(&rows),
                    "both" | "bad" => false,
                    _ => unreachable!(),
                },
                "case={case} rows={rows}"
            );
        }
        return;
    }
    let full = concat!(
        module_path!(),
        "::explicit_4096_profile_bounds_and_conflicting_flags_fail_before_loading"
    );
    let test = full.split_once("::").unwrap().1;
    for (case, two, four) in [
        ("on", "0", "1"),
        ("off", "0", "0"),
        ("both", "1", "1"),
        ("bad", "0", "true"),
    ] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test])
            .env(CHILD, case)
            .env("ATLAS_GLM_PREFILL_2048", two)
            .env("ATLAS_GLM_PREFILL_4096", four)
            .env("ATLAS_GLM_TARGET_SHARED_FP8", "1")
            .env("ATLAS_GLM_TARGET_SHARED_FP8_VERIFY", "0")
            .env("ATLAS_GLM_K5_BATCHED_SHARED", "0")
            .env("ATLAS_NVFP4_MMQ_MOE", "0")
            .env("ATLAS_MOE_GROUPED_CUTLASS", "0")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{case}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
