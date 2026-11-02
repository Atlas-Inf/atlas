// SPDX-License-Identifier: AGPL-3.0-only

//! Exact MXFP8 -> BF16 expansion (the probe is tested in `mxfp8_load_tests`).
//!
//! The golden is two real FP4-Local projections dequantized by PyTorch
//! (`scripts/mxfp8_bf16_golden.py`, which also asserts the BF16 result equals
//! the f32 product). The GPU kernel path is pinned to the host reference on
//! hardware; on CPU the host reference is pinned to PyTorch here.

use super::*;
use sha2::{Digest, Sha256};

fn fixtures() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data/mxfp8")
}

#[test]
fn host_dequant_matches_pytorch_on_real_projections() {
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("golden.json")).unwrap())
            .unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for c in cases {
        let tag = c["tag"].as_str().unwrap();
        let (n, k) = (
            c["shape"][0].as_u64().unwrap() as usize,
            c["shape"][1].as_u64().unwrap() as usize,
        );
        let data = std::fs::read(fixtures().join(format!("{tag}.weight.bin"))).unwrap();
        let scales = std::fs::read(fixtures().join(format!("{tag}.scale.bin"))).unwrap();
        let bf16 = mxfp8_to_bf16_host(&data, &scales, n, k).unwrap();
        let head: Vec<u16> = c["head"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u16)
            .collect();
        assert_eq!(&bf16[..8], &head[..], "{tag}");
        let bytes: Vec<u8> = bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let sha: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(sha, c["bf16_sha256"].as_str().unwrap(), "{tag}");
    }
}

/// The exactness claim at the edges of the accepted range: every finite E4M3
/// code times 2^(e-127) has an f32 product whose low 16 bits are zero, i.e.
/// BF16 holds it without rounding.
#[test]
fn every_e4m3_code_is_exact_across_the_accepted_exponents() {
    for e in [*EXACT_E8M0.start(), 100, 127, 150, *EXACT_E8M0.end()] {
        let m = mxfp8_scales_f32(&[e]).unwrap()[0];
        for code in 0u8..=255 {
            if code & 0x7F == 0x7F {
                continue; // NaN codes
            }
            let v = atlas_core::numeric::fp8_e4m3_to_f32(code) * m;
            assert!(v.is_finite(), "e={e} code={code:#04x}");
            assert_eq!(v.to_bits() & 0xFFFF, 0, "e={e} code={code:#04x} rounds");
            // f32 multiply exact too: dividing back recovers the code value.
            assert_eq!(v / m, atlas_core::numeric::fp8_e4m3_to_f32(code));
        }
    }
}

#[test]
fn scales_outside_the_exact_range_are_refused() {
    for bad in [0u8, 2, 247, 255] {
        assert!(mxfp8_scales_f32(&[120, bad]).is_err(), "e={bad}");
    }
    assert_eq!(mxfp8_scales_f32(&[127]).unwrap(), vec![1.0]);
    assert_eq!(mxfp8_scales_f32(&[118]).unwrap(), vec![2f32.powi(-9)]);
}

#[test]
fn output_allocation_failure_releases_widened_scales() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    let data = gpu.alloc(32).unwrap();
    let scales = gpu.alloc(1).unwrap();
    gpu.copy_h2d(&[127], scales).unwrap();
    let src = RawMxfp8 {
        data,
        scales,
        n: 1,
        k: 32,
    };
    let before = gpu.alloc_count();
    gpu.fail_alloc_after(1);
    assert!(dequant_mxfp8_to_bf16(&gpu, &src).is_err());
    assert_eq!(gpu.alloc_count(), before, "temporary scales leaked");
}
