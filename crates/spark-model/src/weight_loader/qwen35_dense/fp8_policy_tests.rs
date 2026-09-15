// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightTensor};
use std::collections::HashMap;

fn projection(
    weight_dtype: WeightDtype,
    scale_dtype: WeightDtype,
    scale_shape: Vec<usize>,
) -> WeightStore {
    WeightStore::from_map(HashMap::from([
        (
            "p.weight".into(),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape: vec![1024, 5120],
                dtype: weight_dtype,
            },
        ),
        (
            "p.weight_scale".into(),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape: scale_shape,
                dtype: scale_dtype,
            },
        ),
    ]))
}

#[test]
fn modelopt_scalar_fp8_is_native() {
    let store = projection(WeightDtype::FP8E4M3, WeightDtype::FP32, vec![]);
    assert!(proj_is_fp8_any_scale(&store, "p"));
}

#[test]
fn exact_block_grid_fp8_is_native() {
    let store = projection(WeightDtype::FP8E4M3, WeightDtype::FP32, vec![8, 40]);
    assert!(proj_is_fp8_any_scale(&store, "p"));
}

#[test]
fn per_row_fp8_is_not_sent_to_block_scaled_kernels() {
    let store = projection(WeightDtype::FP8E4M3, WeightDtype::BF16, vec![1024, 1]);
    assert!(!proj_is_fp8_any_scale(&store, "p"));
}

#[test]
fn non_fp8_weight_is_not_native_fp8() {
    let store = projection(WeightDtype::BF16, WeightDtype::FP32, vec![]);
    assert!(!proj_is_fp8_any_scale(&store, "p"));
}
