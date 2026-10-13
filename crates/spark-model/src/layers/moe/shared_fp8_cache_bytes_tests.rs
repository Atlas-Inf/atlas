// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn e4m3_hand_values_ties_saturation_and_signed_zero() {
    for (value, code) in [
        (0., 0),
        (-0., 128),
        (0.5, 0x30),
        (1., 0x38),
        (-1., 0xb8),
        (6., 0x4c),
        (448., 0x7e),
        (1000., 0x7e),
        (-1000., 0xfe),
        (1. / 1024., 0),
        (3. / 1024., 2),
        (1.0625, 0x38),
        (1.1875, 0x3a),
    ] {
        assert_eq!(encode_e4m3(value), code, "value={value:?}");
    }
}

fn fixture() -> (MockGpuBackend, QuantizedWeight, DevicePtr, Vec<u8>) {
    let gpu = MockGpuBackend::new();
    let packed = gpu.alloc(16).unwrap();
    let scales = gpu.alloc(2).unwrap();
    let output = gpu.alloc(32).unwrap();
    // Each row contains every FP4 code. Row1 uses a negative scale to test
    // signed-zero multiplication and catch a mistaken decoded transpose.
    let p = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
    let mut packed_bytes = p.to_vec();
    packed_bytes.extend(p);
    gpu.copy_h2d(&packed_bytes, packed).unwrap();
    gpu.copy_h2d(&[0x38, 0xb8], scales).unwrap();
    let row = [
        0, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c, 0x80, 0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc,
    ];
    let mut expected = row.to_vec();
    expected.extend(row.map(|b| b ^ 128));
    gpu.copy_h2d(&expected, output).unwrap();
    (
        gpu,
        QuantizedWeight {
            weight: packed,
            weight_scale: scales,
            weight_scale_2: 1.,
            ..QuantizedWeight::null()
        },
        output,
        expected,
    )
}

#[test]
fn real_byte_oracle_checks_both_rows_nibbles_and_last_byte_without_writes() {
    let (gpu, w, output, expected) = fixture();
    let packed_before = gpu.read_alloc(w.weight).unwrap();
    let scales_before = gpu.read_alloc(w.weight_scale).unwrap();
    verify_predecoded(&gpu, &w, output, 2, 16, 0).unwrap();
    let allocations = gpu.alloc_count();
    let mut corrupt = expected.clone();
    corrupt[31] ^= 1;
    gpu.copy_h2d(&corrupt, output).unwrap();
    assert!(verify_predecoded(&gpu, &w, output, 2, 16, 0).is_err());
    assert_eq!(gpu.read_alloc(output).unwrap(), corrupt);
    assert_eq!(gpu.alloc_count(), allocations);
    assert_eq!(gpu.launch_count(), 0);
    assert_eq!(gpu.read_alloc(w.weight).unwrap(), packed_before);
    assert_eq!(gpu.read_alloc(w.weight_scale).unwrap(), scales_before);
}

#[test]
fn invalid_bounds_scale_and_alias_reject_before_any_copy() {
    let (gpu, w, output, _) = fixture();
    let copies = gpu.d2h_blocking_count();
    for (n, k) in [(0, 16), (2, 0), (2, 17), (usize::MAX, 16), (4096, 4096)] {
        assert!(verify_predecoded(&gpu, &w, output, n, k, 0).is_err());
    }
    assert!(verify_predecoded(&gpu, &w, w.weight, 2, 16, 0).is_err());
    let mut bad = w;
    bad.weight_scale_2 = f32::NAN;
    assert!(verify_predecoded(&gpu, &bad, output, 2, 16, 0).is_err());
    bad = w;
    bad.weight = DevicePtr(u64::MAX - 15);
    assert!(verify_predecoded(&gpu, &bad, output, 2, 16, 0).is_err());
    assert_eq!(gpu.d2h_blocking_count(), copies);
}

#[test]
fn multiple_chunks_and_tail_check_every_byte_with_bounded_copies() {
    let gpu = MockGpuBackend::new();
    let (n, k) = (129, 4096);
    let count = n * k;
    let packed = gpu.alloc(count / 2).unwrap();
    let scales = gpu.alloc(count / 16).unwrap();
    let output = gpu.alloc(count).unwrap();
    gpu.copy_h2d(&vec![0x21; count / 2], packed).unwrap();
    let scale_values: Vec<u8> = (0..count / 16)
        .map(|g| if g % 2 == 0 { 0x38 } else { 0xb8 })
        .collect();
    gpu.copy_h2d(&scale_values, scales).unwrap();
    // Independent hand arithmetic: (0.5,1.0)*1.0*2.0=(1.0,2.0).
    let mut expected: Vec<u8> = (0..count)
        .map(|i| (if i % 2 == 0 { 0x38 } else { 0x40 }) | if (i / 16) % 2 == 0 { 0 } else { 128 })
        .collect();
    gpu.copy_h2d(&expected, output).unwrap();
    let w = QuantizedWeight {
        weight: packed,
        weight_scale: scales,
        weight_scale_2: 2.,
        ..QuantizedWeight::null()
    };
    assert!(validate(&w, output, n, k).unwrap() * 25 / 16 + 4096 + 127 * 8 + 4096 <= HOST_BOUND);
    verify_predecoded(&gpu, &w, output, n, k, 0).unwrap();
    assert_eq!(
        gpu.d2h_blocking_count(),
        9,
        "exactly three bounded copies per row chunk"
    );
    expected[count - 1] ^= 1;
    gpu.copy_h2d(&expected, output).unwrap();
    let error = verify_predecoded(&gpu, &w, output, n, k, 0)
        .unwrap_err()
        .to_string();
    assert!(error.contains(&format!("byte {}", count - 1)));
    assert_eq!(gpu.alloc_count(), 3);
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn every_finite_code_roundtrips_and_nonfinite_source_scales_fail() {
    for code in 0..=255u8 {
        if code & 127 != 127 {
            assert_eq!(encode_e4m3(decode_e4m3(code)), code);
        }
    }
    let (gpu, w, output, _) = fixture();
    gpu.copy_h2d(&[0x38, 0x7f], w.weight_scale).unwrap();
    assert!(
        verify_predecoded(&gpu, &w, output, 2, 16, 0)
            .unwrap_err()
            .to_string()
            .contains("nonfinite block scale")
    );
}

#[test]
fn resident_tensor_scale_sign_saturation_and_subnormal_ties() {
    for (scale, positive) in [
        (-1., [0x80, 0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc]),
        (1000., [0, 0x7e, 0x7e, 0x7e, 0x7e, 0x7e, 0x7e, 0x7e]),
        (1. / 512., [0, 0, 1, 2, 2, 3, 4, 6]),
    ] {
        let (gpu, mut w, output, _) = fixture();
        w.weight_scale_2 = scale;
        let mut row = positive.to_vec();
        row.extend(positive.map(|v| v ^ 128));
        let mut expected = row.clone();
        expected.extend(row.iter().map(|v| v ^ 128));
        gpu.copy_h2d(&expected, output).unwrap();
        verify_predecoded(&gpu, &w, output, 2, 16, 0).unwrap();
    }
}
