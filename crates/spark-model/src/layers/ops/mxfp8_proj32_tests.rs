// SPDX-License-Identifier: AGPL-3.0-only
//! GPU test: every M in 1..=32 through `mxfp8_proj` is byte-identical to M
//! separate `mxfp8_proj_gemv` rows, at the real FP4-Local TP=2 shapes.
#[cfg(feature = "cuda")]
mod gpu {
    use anyhow::Result;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};
    use spark_runtime::kernel_args::KernelLaunch;

    use crate::layers::ops::mxfp8_proj::{Mxfp8ProjKernels, Mxfp8Scratch, mxfp8_proj};
    use crate::weight_map::{Mxfp8Proj, Mxfp8Weight};
    use spark_runtime::kernel_args::div_ceil;

    // (n, k, label): GDN qkvz local, o/out_proj local, shared gate|up, shared down.
    const SHAPES: &[(u32, u32, &str)] = &[
        (8192, 2560, "gdn-qkvz"),
        (2560, 3072, "out/o_proj"),
        (1280, 2560, "shared gate|up"),
        (2560, 640, "shared down"),
    ];

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
    fn down(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
        let mut b = vec![0u8; bytes];
        g.copy_d2h(p, &mut b)?;
        Ok(b)
    }

    #[test]
    #[ignore = "GPU"]
    fn batchm32_rows_match_gemv_rows_bit_for_bit() {
        let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
            .expect("CUDA backend");
        let g: &dyn GpuBackend = &gpu;
        let stream = g.default_stream();
        let kernels = Mxfp8ProjKernels::resolve(g).expect("mxfp8 kernels");
        let mut rng = Lcg(0xA71A5);
        for &(n, k, label) in SHAPES {
            // Hostile bytes: full-random E4M3 (subnormals + NaN codes) and
            // E8M0 including 0 and 0xFF, plus deliberately planted edge codes.
            let mut w = (0..(n as usize * k as usize)).map(|_| rng.u8()).collect::<Vec<u8>>();
            let mut s = (0..(n as usize * (k / 32) as usize))
                .map(|_| rng.u8())
                .collect::<Vec<u8>>();
            w[0] = 0x00;
            w[1] = 0x7F;
            w[2] = 0xFF;
            s[0] = 0x00;
            s[1] = 0xFF;
            let (wd, sd) = (upload(g, &w).unwrap(), upload(g, &s).unwrap());
            let proj = Mxfp8Proj {
                w: Mxfp8Weight { data: wd, scales: sd },
                n,
                k,
            };
            let a: Vec<half::bf16> = (0..32 * k as usize)
                .map(|_| half::bf16::from_f32(rng.u8() as f32 / 128.0 - 1.0))
                .collect();
            let ab: Vec<u8> = a.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
            let ad = upload(g, &ab).unwrap();
            // M=1 reference: row r -> C1.
            let c1 = g.alloc((n as usize) * 2).unwrap();
            let cm = g.alloc(32 * (n as usize) * 2).unwrap();
            for m in 1..=32usize {
                let cm_bytes = m * n as usize * 2;
                for r in 0..m {
                    KernelLaunch::new(g, kernels.gemv)
                        .grid([div_ceil(n, 4), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(ad.offset(r * k as usize * 2))
                        .arg_ptr(wd)
                        .arg_ptr(sd)
                        .arg_ptr(c1)
                        .arg_u32(n)
                        .arg_u32(k)
                        .launch(stream)
                        .unwrap();
                    g.copy_d2d(c1, cm.offset(r * n as usize * 2), n as usize * 2)
                        .unwrap();
                }
                // bucketed launch writes into a second buffer.
                let cb = g.alloc(cm_bytes).unwrap();
                mxfp8_proj(g, &kernels, &proj, ad, cb, m, Mxfp8Scratch { ptr: cb, bytes: 0 }, stream)
                    .unwrap();
                // ref rows are in cm[0..m]; compare cb rows to cm rows.
                let (got, want) = (
                    down(g, cb, cm_bytes).unwrap(),
                    down(g, cm, cm_bytes).unwrap(),
                );
                assert_eq!(got, want, "{label} m={m}: bucketed rows differ from gemv");
            }
        }
    }
}
