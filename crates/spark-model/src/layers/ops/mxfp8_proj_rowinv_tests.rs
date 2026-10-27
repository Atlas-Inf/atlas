// SPDX-License-Identifier: AGPL-3.0-only
//! GPU test, issue #218: under the ROWINV prefill contract every
//! `mxfp8_proj` call takes the expansion + row-invariant GEMM arm, so a row
//! is byte-identical to the same row of a tall pass at ANY m — including
//! m = 1 and a non-zero row offset. `#[ignore]` per repo convention:
//!
//! ```text
//! cargo test -p spark-model --release --lib mxfp8_proj_rowinv -- --ignored --nocapture
//! ```
#[cfg(all(test, feature = "cuda"))]
mod gpu {
    use anyhow::Result;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};

    use crate::layers::ops::mxfp8_proj::{Mxfp8ProjKernels, Mxfp8Scratch, mxfp8_proj_rowinv};
    use crate::layers::ops::qwen4exp_rowinv;
    use crate::weight_map::{Mxfp8Proj, Mxfp8Weight};

    // (n, k, label): attention q/k/v/o-class and GDN in_proj-class local
    // shapes (FP4-Local TP=2, same family as `mxfp8_proj32_tests`).
    const SHAPES: &[(u32, u32, &str)] = &[
        (8192, 2560, "gdn-qkvz"),
        (2560, 3072, "out/o_proj"),
        (1280, 2560, "shared gate|up"),
        (2560, 640, "shared down"),
    ];
    const MS: &[usize] = &[1, 2, 7, 16, 31, 32, 33, 64, 100, 512];

    struct Lcg(u64);
    impl Lcg {
        fn u8(&mut self) -> u8 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u8
        }
    }

    fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
        let p = g.alloc(bytes.len().max(1))?;
        g.copy_h2d(bytes, p)?;
        Ok(p)
    }

    #[test]
    #[ignore = "GPU"]
    fn rowinv_rows_match_the_tall_pass_bit_for_bit() {
        let gpu =
            spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
                .expect("CUDA backend");
        let g: &dyn GpuBackend = &gpu;
        let stream = g.default_stream();
        let kernels = Mxfp8ProjKernels::resolve(g).expect("mxfp8 kernels");
        let big = 1024usize;
        let mut rng = Lcg(0xB0A7);
        // The ROWINV pass scope: `mxfp8_proj` would dispatch here; the test
        // calls the arm directly since it cannot set env vars.
        let _scope = qwen4exp_rowinv::enter_test(g, 2048, false, true);
        for &(n, k, label) in SHAPES {
            let w: Vec<u8> = (0..(n as usize * k as usize)).map(|_| rng.u8()).collect();
            let s: Vec<u8> = (0..(n as usize * (k / 32) as usize))
                .map(|_| rng.u8())
                .collect();
            let proj = Mxfp8Proj {
                w: Mxfp8Weight {
                    data: upload(g, &w).unwrap(),
                    scales: upload(g, &s).unwrap(),
                },
                n,
                k,
            };
            let scratch = Mxfp8Scratch {
                ptr: g.alloc(n as usize * k as usize * 2).unwrap(),
                bytes: n as usize * k as usize * 2,
            };
            let a: Vec<half::bf16> = (0..big * k as usize)
                .map(|_| half::bf16::from_f32(rng.u8() as f32 / 128.0 - 1.0))
                .collect();
            let ab: Vec<u8> = a.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
            let ad = upload(g, &ab).unwrap();
            // Reference: one tall call covers every row.
            let cbig = g.alloc(big * n as usize * 2).unwrap();
            mxfp8_proj_rowinv(g, &kernels, &proj, ad, cbig, big, scratch, stream).unwrap();
            let want = {
                let mut b = vec![0u8; big * n as usize * 2];
                g.copy_d2h(cbig, &mut b).unwrap();
                b
            };
            for &m in MS {
                // Start at a non-zero row offset inside the activation.
                let off = (big - m).min(511);
                let cs = g.alloc(m * n as usize * 2).unwrap();
                mxfp8_proj_rowinv(
                    g,
                    &kernels,
                    &proj,
                    ad.offset(off * k as usize * 2),
                    cs,
                    m,
                    scratch,
                    stream,
                )
                .unwrap();
                let mut got = vec![0u8; m * n as usize * 2];
                g.copy_d2h(cs, &mut got).unwrap();
                let want_m = &want[off * n as usize * 2..(off + m) * n as usize * 2];
                assert_eq!(got, want_m, "{label} m={m} off={off}: rows differ");
            }
        }
    }
}
