// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

fn upload(gpu: &MockGpuBackend, bytes: &[u8]) -> DevicePtr {
    let ptr = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, ptr).unwrap();
    ptr
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
        .collect()
}

#[test]
fn glm_mtp_hidden_copies_normalized_rows_without_another_kernel() {
    // Actual weighted final RMSNorm reference, not a uniform scale: a second
    // unweighted normalization cannot make the raw and normalized rows equal.
    let h = 4;
    let weights = [0.5, 2.0, 1.25, 3.0];
    let mut raw_values = Vec::new();
    let mut normalized_values = Vec::new();
    for row in 0..5 {
        let x = [row as f32 + 1.0, -2.0, 0.5, row as f32 + 4.0];
        let inv = (x.iter().map(|v| v * v).sum::<f32>() / h as f32 + 1e-5)
            .sqrt()
            .recip();
        raw_values.extend(x);
        normalized_values.extend(x.iter().zip(weights).map(|(v, w)| v * inv * w));
    }
    let raw_bytes = bf16_bytes(&raw_values);
    let norm_bytes = bf16_bytes(&normalized_values);
    assert_ne!(raw_bytes, norm_bytes);
    // Two independent mock devices represent rank-local source ownership.
    for _rank in 0..2 {
        let gpu = MockGpuBackend::new();
        let raw = upload(&gpu, &raw_bytes);
        let normalized = upload(&gpu, &norm_bytes);
        let destination = upload(&gpu, &[0xcd; 8]);
        let allocations = gpu.alloc_count();
        for row in [0, 1, 4] {
            copy_target_hidden_row(&gpu, "glm5_next", raw, normalized, destination, h, row, 7)
                .unwrap();
            assert_eq!(
                gpu.read_alloc(destination).unwrap(),
                norm_bytes[row * 8..(row + 1) * 8],
                "GLM row {row} must use post-final-norm target hidden"
            );
        }
        assert_eq!(gpu.d2d_count(), 3);
        assert_eq!(gpu.launch_count(), 0, "do not normalize a second time");
        assert_eq!(gpu.sync_count(), 0);
        assert_eq!(gpu.alloc_count(), allocations);
    }
}

#[test]
fn other_models_keep_raw_rows_even_when_normalized_storage_is_absent() {
    for model in ["qwen3_next", "qwen3_5", "deepseek_v4", "unknown"] {
        let gpu = MockGpuBackend::new();
        let bytes: Vec<u8> = (0..40).collect();
        let raw = upload(&gpu, &bytes);
        let destination = upload(&gpu, &[0; 8]);
        copy_target_hidden_row(&gpu, model, raw, DevicePtr::NULL, destination, 4, 4, 0).unwrap();
        assert_eq!(gpu.read_alloc(destination).unwrap(), bytes[32..40]);
        assert_eq!(gpu.d2d_count(), 1);
        assert_eq!(gpu.launch_count(), 0);
    }
}

#[test]
fn glm_mtp_stash_preserves_row_permutation_and_owned_bytes() {
    let gpu = MockGpuBackend::new();
    let bytes: Vec<u8> = (1..=40).collect();
    let raw = upload(&gpu, &[0xdd; 40]);
    let normalized = upload(&gpu, &bytes);
    let stash = upload(&gpu, &[0xcc; 24]);
    for (i, row) in [4, 0, 1].into_iter().enumerate() {
        copy_target_hidden_row(
            &gpu,
            "glm5_next",
            raw,
            normalized,
            stash.offset(i * 8),
            4,
            row,
            0,
        )
        .unwrap();
    }
    gpu.copy_h2d(&[0xee; 40], normalized).unwrap();
    let destination = upload(&gpu, &[0; 8]);
    // The stash consumer performs a plain copy: no representation transform.
    gpu.copy_d2d_async(stash.offset(16), destination, 8, 0)
        .unwrap();
    assert_eq!(gpu.read_alloc(destination).unwrap(), bytes[8..16]);
    let expected: Vec<u8> = [4, 0, 1]
        .into_iter()
        .flat_map(|row| bytes[row * 8..(row + 1) * 8].iter().copied())
        .collect();
    assert_eq!(gpu.read_alloc(stash).unwrap(), expected);
    assert_eq!(gpu.d2d_count(), 4);
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn hidden_copy_overflow_errors_before_memory_operations() {
    let gpu = MockGpuBackend::new();
    for (hidden_size, row, source) in [(usize::MAX, 0, 1), (4, usize::MAX, 1), (4, 1, u64::MAX - 3)]
    {
        assert!(
            copy_target_hidden_row(
                &gpu,
                "qwen3_next",
                DevicePtr(source),
                DevicePtr::NULL,
                DevicePtr(1),
                hidden_size,
                row,
                0
            )
            .is_err()
        );
    }
    assert_eq!(gpu.d2d_count(), 0);
}
