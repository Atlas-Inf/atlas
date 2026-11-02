// SPDX-License-Identifier: AGPL-3.0-only

//! Native shared-expert twins on the recording mock (hidden 256, inter 256).

use super::*;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightTensor};
use std::collections::HashMap;

const P: &str = "model.language_model.layers.3.mlp.shared_experts";

fn config() -> ModelConfig {
    ModelConfig {
        hidden_size: 256,
        shared_expert_intermediate_size: 256,
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}

fn upload(gpu: &MockGpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, p).unwrap();
    p
}

fn read(gpu: &MockGpuBackend, p: DevicePtr, len: usize) -> Vec<u8> {
    let mut b = vec![0u8; len];
    gpu.copy_d2h(p, &mut b).unwrap();
    b
}

/// gate / up / down as FP8 block `[256, 256]` (or down `down_n` rows), or
/// gate as MXFP8 (`mx_gate`).
fn store(gpu: &MockGpuBackend, mx_gate: bool, down_n: usize) -> (WeightStore, Vec<Vec<u8>>) {
    let mut map = HashMap::new();
    let mut values = Vec::new();
    for (i, name) in ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .enumerate()
    {
        let (n, k) = if i == 2 { (down_n, 256) } else { (256, 256) };
        let w: Vec<u8> = (0..n * k).map(|j| (j * 3 + i) as u8 & 0x7E).collect();
        map.insert(
            format!("{P}.{name}.weight"),
            WeightTensor {
                ptr: upload(gpu, &w),
                shape: vec![n, k],
                dtype: WeightDtype::FP8E4M3,
            },
        );
        values.push(w);
        let (scale, shape, dtype) = if mx_gate && i == 0 {
            (vec![120u8; n * k / 32], vec![n, k / 32], WeightDtype::UInt8)
        } else {
            let s: Vec<u8> = (0..(n / 128) * (k / 128))
                .flat_map(|j| (0.25f32 * (1 + j) as f32).to_le_bytes())
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
    (WeightStore::from_map(map), values)
}

const ON: NativePolicy = NativePolicy {
    mla_fp8: false,
    shared_fp8: true,
    dense_mxfp8: false,
    kda_decode: false,
    kda_w8a8: false,
};

#[test]
fn twins_are_the_checkpoint_bytes() {
    let gpu = MockGpuBackend::new();
    let (st, values) = store(&gpu, false, 256);
    let native = shared_native_twins(ON, &st, P, &config(), &gpu)
        .unwrap()
        .unwrap();
    for (twin, want) in [native.gate, native.up, native.down].iter().zip(&values) {
        assert_eq!(&read(&gpu, twin.data, want.len()), want);
        assert!(!twin.tiled);
    }
    // The checkpoint bytes stay for the loader's release.
    assert!(!st.was_reclaimed(&format!("{P}.down_proj.weight")));
}

#[test]
fn off_or_another_format_builds_nothing() {
    let gpu = MockGpuBackend::new();
    let (st, _) = store(&gpu, false, 256);
    let live = gpu.alloc_count();
    assert!(
        shared_native_twins(NativePolicy::default(), &st, P, &config(), &gpu)
            .unwrap()
            .is_none()
    );
    let (mx, _) = store(&gpu, true, 256);
    let live_mx = gpu.alloc_count();
    assert!(
        shared_native_twins(ON, &mx, P, &config(), &gpu)
            .unwrap()
            .is_none()
    );
    assert_eq!(gpu.alloc_count(), live_mx);
    assert!(live < live_mx);
}

#[test]
fn a_shape_the_config_does_not_expect_frees_what_was_built() {
    let gpu = MockGpuBackend::new();
    let (st, _) = store(&gpu, false, 384);
    let live = gpu.alloc_count();
    assert!(shared_native_twins(ON, &st, P, &config(), &gpu).is_err());
    assert_eq!(gpu.alloc_count(), live);
}
