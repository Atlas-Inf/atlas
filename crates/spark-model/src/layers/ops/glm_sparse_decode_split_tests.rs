// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
#[path = "glm_sparse_decode_split_test_gpu.rs"]
mod capture;
use capture::{Capture, config};
fn args(c: &ModelConfig) -> GlmSparsePrefillTc<'_> {
    GlmSparsePrefillTc {
        config: c,
        dtype: KvCacheDtype::Bf16,
        identical_kv_latent: true,
        query: DevicePtr(0x100000),
        k_cache: DevicePtr(0x200000),
        v_cache: DevicePtr(0x300000),
        indices: DevicePtr(0x400000),
        output: DevicePtr(0x500000),
        block_table: DevicePtr(0x600000),
        rows: 1,
        heads: 32,
        head_dim: 512,
        index_width: 2051,
        block_size: 16,
        scale: 0.0625,
    }
}
#[test]
fn split_disabled_is_no_lookup_launch_sync_or_allocation() {
    let gpu = Capture::default();
    let c = config();
    assert!(!dispatch(&gpu, &args(&c), DevicePtr::NULL, 0, 19, false, false).unwrap());
    initialize(&gpu, &c, false, false).unwrap();
    assert!(gpu.arguments.lock().unwrap().is_empty());
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(gpu.inner.alloc_count(), 0);
    assert_eq!(gpu.inner.sync_count(), 0);
}
#[test]
fn split_dispatch_exact_two_kernels_offsets_stream_and_fixed_s8() {
    let gpu = Capture::default();
    let c = config();
    let a = args(&c);
    let p = DevicePtr(0x800000);
    assert!(dispatch(&gpu, &a, p, SCRATCH_BYTES, 19, true, true).unwrap());
    let calls = gpu.arguments.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        (calls[0].0, calls[0].1, calls[0].2),
        (0x701, [1, 1, 8], 69376)
    );
    assert_eq!((calls[1].0, calls[1].1, calls[1].2), (0x702, [32, 1, 1], 0));
    let mut expected: Vec<Vec<u8>> = [
        a.query.0,
        a.k_cache.0,
        a.v_cache.0,
        a.indices.0,
        p.0,
        a.block_table.0,
    ]
    .into_iter()
    .map(|x| x.to_ne_bytes().to_vec())
    .collect();
    expected.extend(
        [1u32, 32, 512, 2051, 16]
            .into_iter()
            .map(|x| x.to_ne_bytes().to_vec()),
    );
    expected.push(0.0625f32.to_ne_bytes().to_vec());
    expected.push((p.0 + PART_O_BYTES as u64).to_ne_bytes().to_vec());
    expected.push(8u32.to_ne_bytes().to_vec());
    assert_eq!(calls[0].3, expected);
    let mut expected: Vec<Vec<u8>> = [
        p.0,
        p.0 + PART_O_BYTES as u64,
        a.output.0,
        p.0 + (PART_O_BYTES + PART_LSE_BYTES) as u64,
    ]
    .into_iter()
    .map(|x| x.to_ne_bytes().to_vec())
    .collect();
    expected.extend(
        [1u32, 32, 512, 8]
            .into_iter()
            .map(|x| x.to_ne_bytes().to_vec()),
    );
    assert_eq!(calls[1].3, expected);
    assert_eq!(gpu.inner.alloc_count(), 0);
    assert_eq!(gpu.inner.sync_count(), 0);
}
#[test]
fn split_bad_geometry_storage_capacity_and_tc_dependency_fail_before_lookup() {
    let gpu = Capture::default();
    let c = config();
    let p = DevicePtr(0x800000);
    for variant in 0..13 {
        let mut a = args(&c);
        let mut ptr = p;
        let mut bytes = SCRATCH_BYTES;
        let mut tc = true;
        match variant {
            0 => a.rows = 0,
            1 => a.rows = 3,
            2 => a.identical_kv_latent = false,
            3 => a.dtype = KvCacheDtype::Fp8,
            4 => a.index_width = 2048,
            5 => a.scale = f32::NAN,
            6 => ptr = DevicePtr::NULL,
            7 => ptr = ptr.offset(2),
            8 => bytes -= 1,
            9 => ptr = DevicePtr(u64::MAX - 15),
            10 => ptr = a.query,
            11 => a.output = a.indices,
            _ => tc = false,
        }
        assert!(
            dispatch(&gpu, &a, ptr, bytes, 19, true, tc).is_err(),
            "variant{variant}"
        );
    }
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert!(gpu.arguments.lock().unwrap().is_empty());
    assert!(validate_scratch(p, SCRATCH_BYTES, &[(p.offset(100), 1000)]).is_err());
    assert!(validate_scratch(p, SCRATCH_BYTES, &[(DevicePtr(u64::MAX - 15), 32)]).is_err());
    assert!(validate_scratch(p, SCRATCH_BYTES, &[(p.offset(SCRATCH_BYTES), 16)]).is_ok());
}
#[test]
fn split_declines_fp8_g128_before_any_lookup() {
    let gpu = Capture::default();
    let c = config();
    let mut a = args(&c);
    a.dtype = KvCacheDtype::Fp8G128;
    assert!(!dispatch(&gpu, &a, DevicePtr(0x800000), SCRATCH_BYTES, 19, true, true).unwrap());
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert!(gpu.arguments.lock().unwrap().is_empty());
}
#[test]
fn split_startup_uses_two_zero_row_null_launches_then_one_sync() {
    let gpu = Capture::default();
    let c = config();
    initialize(&gpu, &c, true, true).unwrap();
    let calls = gpu.arguments.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].1, [1, 1, 8]);
    assert_eq!(calls[1].1, [1, 1, 1]);
    for arg in &calls[0].3[..6] {
        assert_eq!(*arg, 0u64.to_ne_bytes());
    }
    assert_eq!(calls[0].3[6], 0u32.to_ne_bytes());
    assert_eq!(calls[0].3[12], 0u64.to_ne_bytes());
    for arg in &calls[1].3[..4] {
        assert_eq!(*arg, 0u64.to_ne_bytes());
    }
    assert_eq!(calls[1].3[4], 0u32.to_ne_bytes());
    assert_eq!(*gpu.streams.lock().unwrap(), vec![0]);
    assert_eq!(gpu.inner.alloc_count(), 0);
}
#[test]
fn split_flag_is_strict_and_config_validation_precedes_startup() {
    for value in ["", "true", "8", "16", " 1"] {
        assert!(super::super::parse("glm5_next", FLAG, Some(value)).is_err());
    }
    for value in [None, Some("0")] {
        assert!(!super::super::parse("glm5_next", FLAG, value).unwrap());
    }
    assert!(super::super::parse("glm5_next", FLAG, Some("1")).unwrap());
    let gpu = Capture::default();
    let mut c = config();
    c.index_topk = 1024;
    assert!(initialize(&gpu, &c, true, true).is_err());
    assert_eq!(gpu.lookups.load(std::sync::atomic::Ordering::Relaxed), 0);
}
