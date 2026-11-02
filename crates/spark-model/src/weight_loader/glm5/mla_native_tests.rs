// SPDX-License-Identifier: AGPL-3.0-only

//! The native MLA twins on the recording mock, at a scaled-down GLM shape
//! (hidden 256, q_lora / kv_lora 128, 4 heads of 128, TP2).

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightTensor};
use std::collections::HashMap;

const P: &str = "model.language_model.layers.3.self_attn";

fn plan(rank: usize) -> MlaTpPlan {
    MlaTpPlan {
        tp_rank: rank,
        tp_size: 2,
        hidden: 256,
        q_lora: 128,
        kv_lora: 128,
        local_q_heads: 2,
        local_kv_heads: 2,
        q_head_dim: 128,
        kv_head_dim: 256,
        v_head_dim: 128,
    }
}

fn config() -> ModelConfig {
    ModelConfig {
        hidden_size: 256,
        q_lora_rank: 128,
        kv_lora_rank: 128,
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}

fn upload(gpu: &MockGpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, p).unwrap();
    p
}

/// The four projections as FP8 block tensors (`scale` F32, 4-D grid), or
/// q_a as MXFP8 (`mx_q_a`).
fn store(gpu: &MockGpuBackend, mx_q_a: bool) -> WeightStore {
    let shapes = [(128, 256), (128, 256), (512, 128), (256, 512)];
    let mut map = HashMap::new();
    for (i, (name, (n, k))) in PROJECTIONS.iter().zip(shapes).enumerate() {
        let w: Vec<u8> = (0..n * k).map(|j| (j + i) as u8 & 0x7E).collect();
        map.insert(
            format!("{P}.{name}.weight"),
            WeightTensor {
                ptr: upload(gpu, &w),
                shape: vec![n, k],
                dtype: WeightDtype::FP8E4M3,
            },
        );
        let (scale, shape, dtype) = if mx_q_a && i == 0 {
            (vec![120u8; n * k / 32], vec![n, k / 32], WeightDtype::UInt8)
        } else {
            let s: Vec<u8> = (0..(n / 128) * (k / 128))
                .flat_map(|j| (1.0f32 + j as f32).to_le_bytes())
                .collect();
            (s, vec![n / 128, 1, k / 128, 1], WeightDtype::FP32)
        };
        map.insert(
            format!("{P}.{name}.weight_scale"),
            WeightTensor {
                ptr: upload(gpu, &scale),
                shape,
                dtype,
            },
        );
    }
    WeightStore::from_map(map)
}

const ON: NativePolicy = NativePolicy {
    mla_fp8: true,
    shared_fp8: false,
    dense_mxfp8: false,
    kda_decode: false,
    kda_w8a8: false,
};

#[test]
fn each_rank_gets_its_local_shapes() {
    for rank in 0..2 {
        let gpu = MockGpuBackend::new();
        let st = store(&gpu, false);
        let twins = mla_native_twins(ON, &st, P, &plan(rank), &config(), false, &gpu)
            .unwrap()
            .unwrap();
        let shapes: Vec<[usize; 2]> = twins.iter().map(|(_, s)| *s).collect();
        // q_a, kv_a replicated; q_b half its rows; o half its columns.
        assert_eq!(shapes, [[128, 256], [128, 256], [256, 128], [256, 256]]);
        assert!(twins.iter().all(|(t, _)| !t.tiled));
    }
}

#[test]
fn off_mtp_body_or_another_format_builds_nothing() {
    let gpu = MockGpuBackend::new();
    let st = store(&gpu, false);
    let live = gpu.alloc_count();
    let off = NativePolicy::default();
    assert!(
        mla_native_twins(off, &st, P, &plan(0), &config(), false, &gpu)
            .unwrap()
            .is_none()
    );
    assert!(
        mla_native_twins(ON, &st, P, &plan(0), &config(), true, &gpu)
            .unwrap()
            .is_none()
    );
    let mx = store(&gpu, true);
    let live_mx = gpu.alloc_count();
    assert!(
        mla_native_twins(ON, &mx, P, &plan(0), &config(), false, &gpu)
            .unwrap()
            .is_none()
    );
    assert_eq!(gpu.alloc_count(), live_mx);
    assert!(live <= live_mx);
}

#[test]
fn a_shape_the_config_does_not_expect_frees_what_was_built() {
    let gpu = MockGpuBackend::new();
    let st = store(&gpu, false);
    let live = gpu.alloc_count();
    let wrong = ModelConfig {
        kv_lora_rank: 256,
        ..config()
    };
    assert!(mla_native_twins(ON, &st, P, &plan(0), &wrong, false, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), live);
}

#[test]
fn bf16_copies_stay_for_bf16_prefill_and_are_freed_otherwise() {
    let gpu = MockGpuBackend::new();
    let mut ptrs: Vec<DevicePtr> = (0..4).map(|_| gpu.alloc(64).unwrap()).collect();
    let original = ptrs.clone();
    let [a, b, c, d] = &mut ptrs[..] else {
        unreachable!()
    };
    let keys = settle_bf16(&mut [a, b, c, d], Fp8bPrefill::Bf16, 3, &gpu).unwrap();
    assert_eq!(keys.to_vec(), original);
    assert_eq!(ptrs, original);
    let live = gpu.alloc_count();

    let [a, b, c, d] = &mut ptrs[..] else {
        unreachable!()
    };
    let keys = settle_bf16(&mut [a, b, c, d], Fp8bPrefill::W8a8, 3, &gpu).unwrap();
    assert_eq!(gpu.alloc_count(), live - 4);
    assert_eq!(ptrs, keys.to_vec());
    for (i, key) in keys.iter().enumerate() {
        assert_eq!(
            *key,
            crate::layers::qwen3_attention::mla_native::freed_key(3, i)
        );
    }
}
