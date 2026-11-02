// SPDX-License-Identifier: AGPL-3.0-only

//! GPU check of [`dequant_mxfp8_to_bf16`] (`#[ignore]`: a GB10 and a
//! glm-5.3-flash kernel build): the kernel path is byte-identical to the
//! host reference, on the two real FP4-Local projections and on random bytes
//! that cover every finite E4M3 code and both ends of the exact exponent range.
//!
//!   cargo test --release -p spark-model --lib mxfp8_gpu -- --ignored --nocapture
//!
//! A prebuilt binary run elsewhere: set `ATLAS_TEST_DATA=<repo>/test_data`.

use super::*;
use spark_runtime::cuda_backend::AtlasCudaBackend;

fn backend() -> &'static AtlasCudaBackend {
    crate::layers::glm_gpu_test_backend::backend().expect("GLM CUDA backend")
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn kernel_vs_host(g: &dyn GpuBackend, data: &[u8], scales: &[u8], n: usize, k: usize) -> Vec<u16> {
    let src = RawMxfp8 {
        data: up(g, data),
        scales: up(g, scales),
        n,
        k,
    };
    let out = dequant_mxfp8_to_bf16(g, &src).unwrap();
    let mut got = vec![0u8; n * k * 2];
    g.copy_d2h(out.weight, &mut got).unwrap();
    for p in [out.weight, src.data, src.scales] {
        g.free(p).unwrap();
    }
    let got: Vec<u16> = got
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let want = mxfp8_to_bf16_host(data, scales, n, k).unwrap();
    let bad = got.iter().zip(&want).position(|(a, b)| a != b);
    assert!(
        bad.is_none(),
        "[{n}, {k}] kernel != host at {bad:?}: {:#06x} vs {:#06x}",
        got[bad.unwrap()],
        want[bad.unwrap()]
    );
    got
}

#[test]
#[ignore]
fn mxfp8_gpu_kernel_matches_host_on_real_projections() {
    let g = backend();
    // `ATLAS_TEST_DATA` for a test binary run away from its source tree.
    let dir = std::env::var("ATLAS_TEST_DATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_data")
        })
        .join("mxfp8");
    for tag in ["l0_gdn_in_proj_a", "l0_gdn_in_proj_b"] {
        let data = std::fs::read(dir.join(format!("{tag}.weight.bin"))).unwrap();
        let scales = std::fs::read(dir.join(format!("{tag}.scale.bin"))).unwrap();
        kernel_vs_host(g, &data, &scales, 48, 2560);
        eprintln!("{tag}: kernel == host");
    }
}

#[test]
#[ignore]
fn mxfp8_gpu_kernel_matches_host_on_every_code_and_edge_exponent() {
    let g = backend();
    let (n, k) = (96, 1024);
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    // Every finite code, cycled; NaN codes (0x7F/0xFF) dropped to zero.
    let data: Vec<u8> = (0..n * k)
        .map(|i| {
            let b = (i % 256) as u8;
            if b & 0x7F == 0x7F { 0 } else { b }
        })
        .collect();
    let edges = [*EXACT_E8M0.start(), *EXACT_E8M0.end(), 127];
    let scales: Vec<u8> = (0..n * k / MXFP8_BLOCK)
        .map(|i| {
            if i < 3 * 8 {
                edges[i / 8]
            } else {
                3 + (next() % 244) as u8
            }
        })
        .collect();
    kernel_vs_host(g, &data, &scales, n, k);
}
