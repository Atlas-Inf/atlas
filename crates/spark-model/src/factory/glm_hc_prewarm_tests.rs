// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.hc_mult = 4;
    c.max_batch_tokens = 4100;
    c
}

fn storage() -> Storage {
    Storage {
        activation: DevicePtr(0x1000_0000),
        activation_capacity: 268_697_600,
        scratch: DevicePtr(0x3000_0000),
        scratch_capacity: 4_723_200,
    }
}

#[test]
fn hc_tf32_prewarm_strict_flag_and_shape_admission() {
    for value in [None, Some("0")] {
        assert!(!parse(value).unwrap());
    }
    assert!(parse(Some("1")).unwrap());
    for value in ["", "true", "yes", "2", " 1"] {
        assert!(parse(Some(value)).is_err());
    }
    for value in ["1", "true", "yes"] {
        Plan::new(&config(), 4100, Some(value), storage()).unwrap();
    }
    for value in [None, Some("0"), Some("TRUE")] {
        assert!(Plan::new(&config(), 4100, value, storage()).is_err());
    }
    for mutate in [
        (|c: &mut ModelConfig| c.model_type = "qwen4_exp".into()) as fn(&mut ModelConfig),
        |c| c.hidden_size = 2048,
        |c| c.hc_mult = 8,
        |c| c.max_batch_tokens = 4096,
    ] {
        let mut c = config();
        mutate(&mut c);
        assert!(Plan::new(&c, 4100, Some("1"), storage()).is_err());
    }
    for rows in [0, 4096, 4101, usize::MAX] {
        assert!(Plan::new(&config(), rows, Some("1"), storage()).is_err());
    }
}

#[test]
fn hc_tf32_prewarm_rejects_short_overlapping_or_wrapping_storage() {
    let good = storage();
    for bad in [
        Storage {
            activation_capacity: 268_697_599,
            ..good
        },
        Storage {
            scratch_capacity: 1_966_463,
            ..good
        },
        Storage {
            activation: DevicePtr::NULL,
            ..good
        },
        Storage {
            scratch: DevicePtr(0x3000_0001),
            ..good
        },
        Storage {
            scratch: good.activation,
            ..good
        },
        Storage {
            scratch: DevicePtr(0x1000_0100),
            ..good
        },
        Storage {
            activation: DevicePtr(u64::MAX - 255),
            ..good
        },
        Storage {
            scratch_capacity: usize::MAX,
            ..good
        },
    ] {
        assert!(Plan::new(&config(), 4100, Some("1"), bad).is_err());
    }
    let plan = Plan::new(&config(), 4100, Some("1"), good).unwrap();
    assert_eq!(plan.activation_bytes, 268_697_600);
    assert_eq!(plan.scratch_bytes, 1_966_464);
    assert_eq!(plan.output, good.scratch.offset(1_572_864));
    assert_eq!((plan.n, plan.k), (24, 16384));
    Plan::new(
        &config(),
        4100,
        Some("1"),
        Storage {
            scratch_capacity: plan.scratch_bytes,
            ..good
        },
    )
    .unwrap();
    assert!(bytes(usize::MAX, 2).is_err());
}

#[test]
fn hc_tf32_prewarm_executes_exact_calls_zeros_dead_spans_and_drains_errors() {
    let gpu = MockGpuBackend::new();
    let activation = gpu.alloc(268_697_600).unwrap();
    let scratch = gpu.alloc(4_723_200).unwrap();
    let plan = Plan::new(
        &config(),
        4100,
        Some("1"),
        Storage {
            activation,
            activation_capacity: 268_697_600,
            scratch,
            scratch_capacity: 4_723_200,
        },
    )
    .unwrap();
    gpu.memset(activation, 0x7f, plan.activation_bytes).unwrap();
    gpu.memset(scratch, 0x7f, 4_723_200).unwrap();
    let mut calls = Vec::new();
    execute(&gpu, &plan, 19, |call| {
        calls.push(call);
        Ok(())
    })
    .unwrap();
    assert_eq!(calls.len(), 3);
    for (call, m) in calls.iter().zip([4100, 4096, 3515]) {
        assert_eq!(
            *call,
            Projection {
                activation,
                weight: scratch,
                output: plan.output,
                m,
                n: 24,
                k: 16384,
                stream: 19
            }
        );
    }
    assert_eq!(gpu.alloc_count(), 2, "warmup must never allocate scratch");
    assert_eq!(gpu.sync_count(), 1);
    for (ptr, expected) in [
        (activation, 0),
        (activation.offset(plan.activation_bytes - 1), 0),
        (scratch, 0),
        (scratch.offset(plan.scratch_bytes - 1), 0),
        (scratch.offset(plan.scratch_bytes), 0x7f),
    ] {
        let mut byte = [0];
        gpu.copy_d2h(ptr, &mut byte).unwrap();
        assert_eq!(byte[0], expected);
    }
    let syncs = gpu.sync_count();
    let mut attempted = Vec::new();
    let error = execute(&gpu, &plan, 19, |call| {
        attempted.push(call.m);
        anyhow::ensure!(call.m != 4096, "injected TF32 launch failure");
        Ok(())
    })
    .unwrap_err();
    assert!(error.to_string().contains("injected TF32 launch failure"));
    assert_eq!(attempted, [4100, 4096]);
    assert_eq!(gpu.sync_count(), syncs + 1, "drain queued work on failure");
    gpu.free(activation).unwrap();
    gpu.free(scratch).unwrap();
}
