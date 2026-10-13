// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::{gpu::mock::MockGpuBackend, weights::WeightTensor};

fn bf16_store(
    gpu: &dyn GpuBackend,
    shape: Vec<usize>,
    dtype: WeightDtype,
) -> (WeightStore, DevicePtr) {
    let source = gpu.alloc(2048 * 4096 * 2).unwrap();
    (
        WeightStore::from_map(std::collections::HashMap::from([(
            "shared.weight".into(),
            WeightTensor {
                ptr: source,
                shape,
                dtype,
            },
        )])),
        source,
    )
}

#[test]
fn actual_bf16_quantizer_frees_source_and_provenance_tracks_only_returned_live_nvfp4() {
    let gpu = MockGpuBackend::new();
    let (store, source) = bf16_store(&gpu, vec![2048, 4096], WeightDtype::BF16);
    let qctx = QuantizeCtx {
        absmax_k: KernelHandle(11),
        quantize_k: KernelHandle(12),
        stream: 0,
    };
    let (weight, origin) = load_with_origin(
        &store,
        "shared",
        2048,
        4096,
        &gpu,
        Nvfp4Variant::Standard,
        qctx,
        true,
    )
    .unwrap();
    assert_eq!(gpu.launch_count(), 2);
    assert!(
        gpu.copy_d2h(source, &mut [0]).is_err(),
        "existing quantized_any must free BF16"
    );
    let origin = origin.unwrap();
    assert!(origin.validate(&weight, 2048, 4096).unwrap());
    assert!(gpu.copy_d2h(weight.weight, &mut [0]).is_ok());
    assert!(gpu.copy_d2h(weight.weight_scale, &mut [0]).is_ok());
    for changed in [
        QuantizedWeight {
            weight: DevicePtr(16),
            ..weight
        },
        QuantizedWeight {
            weight_scale: DevicePtr(16),
            ..weight
        },
        QuantizedWeight {
            weight_scale_2: 2.,
            ..weight
        },
    ] {
        assert!(origin.validate(&changed, 2048, 4096).is_err());
    }
    assert!(origin.validate(&weight, 4096, 2048).is_err());
}

#[test]
fn source_geometry_dtype_and_markers_are_checked_before_quantizer_work() {
    for (shape, dtype) in [
        (vec![2048, 4095], WeightDtype::BF16),
        (vec![4096, 2048], WeightDtype::BF16),
        (vec![2048, 4096], WeightDtype::FP8E4M3),
        (vec![usize::MAX, 2], WeightDtype::BF16),
    ] {
        let gpu = MockGpuBackend::new();
        let (store, source) = bf16_store(&gpu, shape, dtype);
        assert!(source_is_bf16(&store, "shared", 2048, 4096).is_err());
        assert_eq!(gpu.launch_count(), 0);
        assert!(gpu.copy_d2h(source, &mut [0]).is_ok());
    }
}

#[test]
fn bf16_with_any_quantization_marker_is_not_claimed_as_validated_dense_origin() {
    for marker in [
        "weight_packed",
        "weight_scale",
        "weight_scale_inv",
        "weight_scale_2",
        "weight_global_scale",
        "input_scale",
        "input_global_scale",
    ] {
        let store = WeightStore::from_map(std::collections::HashMap::from([
            (
                "shared.weight".into(),
                WeightTensor {
                    ptr: DevicePtr(16),
                    shape: vec![2048, 4096],
                    dtype: WeightDtype::BF16,
                },
            ),
            (
                format!("shared.{marker}"),
                WeightTensor {
                    ptr: DevicePtr(32),
                    shape: vec![1],
                    dtype: WeightDtype::FP32,
                },
            ),
        ]));
        assert!(source_is_bf16(&store, "shared", 2048, 4096).is_err());
    }
}

#[test]
fn native_packed_dtype_and_exact_shapes_remain_explicit() {
    let store = WeightStore::from_map(std::collections::HashMap::from([
        (
            "shared.weight".into(),
            WeightTensor {
                ptr: DevicePtr(16),
                shape: vec![2048, 2048],
                dtype: WeightDtype::UInt8,
            },
        ),
        (
            "shared.weight_scale".into(),
            WeightTensor {
                ptr: DevicePtr(0x500000),
                shape: vec![2048, 256],
                dtype: WeightDtype::FP8E4M3,
            },
        ),
    ]));
    assert!(!source_is_bf16(&store, "shared", 2048, 4096).unwrap());
}

#[test]
fn actual_source_seam_rejects_unapproved_model_variant_before_quantization() {
    let gpu = MockGpuBackend::new();
    let (store, source) = bf16_store(&gpu, vec![2048, 4096], WeightDtype::BF16);
    let qctx = QuantizeCtx {
        absmax_k: KernelHandle(11),
        quantize_k: KernelHandle(12),
        stream: 0,
    };
    for variant in [Nvfp4Variant::Bf16Raw, Nvfp4Variant::Fp8Dequanted] {
        assert!(load_with_origin(&store, "shared", 2048, 4096, &gpu, variant, qctx, true).is_err());
        assert_eq!(gpu.launch_count(), 0);
        assert!(gpu.copy_d2h(source, &mut [0]).is_ok());
    }
}
