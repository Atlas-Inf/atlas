// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `exl3_from_store`: the store-tensor preflight that turns the
//! four tensors of one EXL3 linear into a typed `Exl3Weight` handle.

use std::collections::HashMap;

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{Exl3Weight, MUL1_TAG, exl3_from_store};
use crate::weight_map::QuantWeight;

/// Uploads `bytes` to the mock GPU and returns a store tensor pointing at it.
fn tensor(
    gpu: &MockGpuBackend,
    bytes: &[u8],
    shape: Vec<usize>,
    dtype: WeightDtype,
) -> WeightTensor {
    let ptr = gpu.alloc(bytes.len().max(1)).unwrap();
    gpu.copy_h2d(bytes, ptr).unwrap();
    WeightTensor { ptr, shape, dtype }
}

/// The trellis byte count for the valid linear: `[8, 16, 64]` int16.
const TRELLIS_BYTES: usize = 8 * 16 * 64 * 2;

/// A store for linear `p` with the given pieces; tensors are already uploaded.
///
/// The valid shape is trellis `[8, 16, 64]` Int16 (in 128, out 256, bits 4),
/// `suh` FP16 with 128 elements, `svh` FP16 with 256 elements, `mul1` Int32
/// scalar holding `MUL1_TAG`.
fn store(
    trellis: Option<WeightTensor>,
    suh: Option<WeightTensor>,
    svh: Option<WeightTensor>,
    mul1: Option<WeightTensor>,
) -> WeightStore {
    let mut map: HashMap<String, WeightTensor> = HashMap::new();
    for (suffix, t) in [
        ("trellis", trellis),
        ("suh", suh),
        ("svh", svh),
        ("mul1", mul1),
    ] {
        if let Some(t) = t {
            map.insert(format!("p.{suffix}"), t);
        }
    }
    WeightStore::from_map(map)
}

/// The four valid tensors of linear `p`, except where a test overrides one.
fn valid(gpu: &MockGpuBackend) -> (WeightTensor, WeightTensor, WeightTensor, WeightTensor) {
    (
        tensor(
            gpu,
            &[0u8; TRELLIS_BYTES],
            vec![8, 16, 64],
            WeightDtype::Int16,
        ),
        tensor(gpu, &[0u8; 128 * 2], vec![128], WeightDtype::FP16),
        tensor(gpu, &[0u8; 256 * 2], vec![256], WeightDtype::FP16),
        tensor(gpu, &MUL1_TAG.to_le_bytes(), vec![], WeightDtype::Int32),
    )
}

#[test]
fn valid_linear_builds_its_handle() {
    let gpu = MockGpuBackend::new();
    let (trellis, suh, svh, mul1) = valid(&gpu);
    let store = store(Some(trellis), Some(suh), Some(svh), Some(mul1));

    let w = exl3_from_store(&store, "p", &gpu).unwrap();
    assert_eq!(w.shape.in_features, 128);
    assert_eq!(w.shape.out_features, 256);
    assert_eq!(w.shape.bits, 4);
    assert!(!w.is_null());
    assert!(QuantWeight::from(w).as_exl3().is_some());
}

#[test]
fn missing_svh_names_the_tensor() {
    let gpu = MockGpuBackend::new();
    let (trellis, suh, _svh, mul1) = valid(&gpu);
    let store = store(Some(trellis), Some(suh), None, Some(mul1));

    let err = exl3_from_store(&store, "p", &gpu).unwrap_err();
    assert!(format!("{err:#}").contains("p.svh"), "{err:#}");
}

#[test]
fn trellis_with_wrong_dtype_is_refused() {
    let gpu = MockGpuBackend::new();
    let (_trellis, suh, svh, mul1) = valid(&gpu);
    let trellis = tensor(
        &gpu,
        &[0u8; TRELLIS_BYTES],
        vec![8, 16, 64],
        WeightDtype::FP16,
    );
    let store = store(Some(trellis), Some(suh), Some(svh), Some(mul1));

    let err = exl3_from_store(&store, "p", &gpu).unwrap_err();
    let text = format!("{err:#}");
    assert!(text.contains("p.trellis"), "{text}");
    assert!(text.contains("Int16"), "{text}");
}

#[test]
fn wrong_mul1_tag_is_refused_with_the_value() {
    let gpu = MockGpuBackend::new();
    let (trellis, suh, svh, _mul1) = valid(&gpu);
    let mul1 = tensor(&gpu, &0u32.to_le_bytes(), vec![], WeightDtype::Int32);
    let store = store(Some(trellis), Some(suh), Some(svh), Some(mul1));

    let err = exl3_from_store(&store, "p", &gpu).unwrap_err();
    assert!(format!("{err:#}").contains("0x00000000"), "{err:#}");
}

#[test]
fn short_suh_is_refused() {
    let gpu = MockGpuBackend::new();
    let (trellis, _suh, svh, mul1) = valid(&gpu);
    let suh = tensor(&gpu, &[0u8; 64 * 2], vec![64], WeightDtype::FP16);
    let store = store(Some(trellis), Some(suh), Some(svh), Some(mul1));

    let err = exl3_from_store(&store, "p", &gpu).unwrap_err();
    assert!(format!("{err:#}").contains("suh"), "{err:#}");
}

#[test]
fn null_handle_is_null() {
    assert!(Exl3Weight::null().is_null());
}
