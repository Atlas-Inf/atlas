// SPDX-License-Identifier: AGPL-3.0-only

//! Native MXFP8 dense twins on the recording mock (h 128, inter 256).

use super::*;
use crate::layers::glm_native::NativePolicy;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightTensor;
use std::collections::HashMap;

const P: &str = "model.language_model.layers.0.mlp";
const SHAPE: [usize; 2] = [128, 256];

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

/// gate / up `[256, 128]`, down `[down_n, 256]`, MXFP8 (or gate BF16).
fn store(gpu: &MockGpuBackend, bf16_gate: bool, down_n: usize) -> (WeightStore, Vec<Vec<u8>>) {
    let mut map = HashMap::new();
    let mut values = Vec::new();
    for (i, (name, (n, k))) in [
        ("gate_proj", (256, 128)),
        ("up_proj", (256, 128)),
        ("down_proj", (down_n, 256)),
    ]
    .into_iter()
    .enumerate()
    {
        if bf16_gate && i == 0 {
            map.insert(
                format!("{P}.{name}.weight"),
                WeightTensor {
                    ptr: upload(gpu, &vec![0u8; n * k * 2]),
                    shape: vec![n, k],
                    dtype: WeightDtype::BF16,
                },
            );
            values.push(Vec::new());
            continue;
        }
        let w: Vec<u8> = (0..n * k).map(|j| (j * 5 + i) as u8 & 0x7E).collect();
        let s: Vec<u8> = (0..n * k / 32).map(|j| 115 + (j % 5) as u8).collect();
        map.insert(
            format!("{P}.{name}.weight"),
            WeightTensor {
                ptr: upload(gpu, &w),
                shape: vec![n, k],
                dtype: WeightDtype::FP8E4M3,
            },
        );
        map.insert(
            format!("{P}.{name}.weight_scale"),
            WeightTensor {
                ptr: upload(gpu, &s),
                shape: vec![n, k / 32],
                dtype: WeightDtype::UInt8,
            },
        );
        values.push(w);
    }
    (WeightStore::from_map(map), values)
}

const ON: NativePolicy = NativePolicy {
    mla_fp8: false,
    shared_fp8: false,
    dense_mxfp8: true,
    kda_decode: false,
    kda_w8a8: false,
};

#[test]
fn twins_are_the_checkpoint_bytes_with_w8a8_operands_over_them() {
    let gpu = MockGpuBackend::new();
    let (st, values) = store(&gpu, false, 128);
    // m_max 0: the mock has no cuBLASLt to tune buckets with.
    let native = native_dense(ON, &st, P, SHAPE, &gpu).unwrap().unwrap();
    for (twin, want) in [native.gate, native.up, native.down].iter().zip(&values) {
        assert_eq!(&read(&gpu, twin.data, want.len()), want);
        assert!(!twin.tiled);
    }
    // ATLAS_GLM_FP4L_FP8_PREFILL unset: W8A8, reading the twins' bytes.
    let ops = native.w8a8.unwrap();
    for (op, twin) in ops.iter().zip([native.gate, native.up, native.down]) {
        assert_eq!(op.data, twin.data);
    }
    assert!(!st.was_reclaimed(&format!("{P}.gate_proj.weight")));
}

#[test]
fn off_or_another_format_builds_nothing() {
    let gpu = MockGpuBackend::new();
    let (st, _) = store(&gpu, false, 128);
    assert!(
        native_dense(NativePolicy::default(), &st, P, SHAPE, &gpu)
            .unwrap()
            .is_none()
    );
    let (bf16, _) = store(&gpu, true, 128);
    let live = gpu.alloc_count();
    assert!(native_dense(ON, &bf16, P, SHAPE, &gpu).unwrap().is_none());
    assert_eq!(gpu.alloc_count(), live);
}

#[test]
fn a_shape_the_config_does_not_expect_frees_what_was_built() {
    let gpu = MockGpuBackend::new();
    let (st, _) = store(&gpu, false, 96);
    let live = gpu.alloc_count();
    assert!(native_dense(ON, &st, P, SHAPE, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), live);
}
