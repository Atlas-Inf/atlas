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
    #[test]
    #[ignore = "GPU"]
    fn tc_rows_row_invariant_and_near_gemv() {
        let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
            .expect("CUDA backend");
        let g: &dyn GpuBackend = &gpu;
        let stream = g.default_stream();
        let kernels = Mxfp8ProjKernels::resolve(g).expect("mxfp8 kernels");
        assert!(kernels.tc.0 != 0, "mxfp8_proj_tc must be in the target module");
        let mut rng = Lcg(0xB7E55);
        for &(n, k, label) in SHAPES {
            let w = (0..(n as usize * k as usize)).map(|_| rng.u8()).collect::<Vec<u8>>();
            let s = (0..(n as usize * (k / 32) as usize)).map(|_| rng.u8()).collect::<Vec<u8>>();
            let (wd, sd) = (upload(g, &w).unwrap(), upload(g, &s).unwrap());
            let a: Vec<half::bf16> = (0..32 * k as usize)
                .map(|_| half::bf16::from_f32(rng.u8() as f32 / 128.0 - 1.0))
                .collect();
            let ab: Vec<u8> = a.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
            let ad = upload(g, &ab).unwrap();
            // Single-row references: tc (row-invariance anchor) and gemv
            // (agreement report).
            let tc_ref = g.alloc(32 * (n as usize) * 2).unwrap();
            let gemv_ref = g.alloc(32 * (n as usize) * 2).unwrap();
            for r in 0..32usize {
                let aref = ad.offset(r * k as usize * 2);
                KernelLaunch::new(g, kernels.tc)
                    .grid([div_ceil(n, 64), 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(aref)
                    .arg_ptr(wd)
                    .arg_ptr(sd)
                    .arg_ptr(tc_ref.offset(r * n as usize * 2))
                    .arg_u32(1)
                    .arg_u32(n)
                    .arg_u32(k)
                    .launch(stream)
                    .unwrap();
                KernelLaunch::new(g, kernels.gemv)
                    .grid([div_ceil(n, 4), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(aref)
                    .arg_ptr(wd)
                    .arg_ptr(sd)
                    .arg_ptr(gemv_ref.offset(r * n as usize * 2))
                    .arg_u32(n)
                    .arg_u32(k)
                    .launch(stream)
                    .unwrap();
            }
            let (tc_rows, exact_rows) = (
                down(g, tc_ref, 32 * n as usize * 2).unwrap(),
                down(g, gemv_ref, 32 * n as usize * 2).unwrap(),
            );
            // Row invariance: row r of every M-row tc launch == tc1 row r.
            let cm = g.alloc(32 * (n as usize) * 2).unwrap();
            // Determinism probe: same launch twice must agree byte-for-byte.
            {
                let cmd = g.alloc(32 * (n as usize) * 2).unwrap();
                for _ in 0..2 {
                    KernelLaunch::new(g, kernels.tc)
                        .grid([div_ceil(n, 64), div_ceil(9u32, 16), 1])
                        .block([128, 1, 1])
                        .arg_ptr(ad).arg_ptr(wd).arg_ptr(sd).arg_ptr(cmd)
                        .arg_u32(9).arg_u32(n).arg_u32(k).launch(stream).unwrap();
                }
                let d1 = down(g, cmd, 9 * n as usize * 2).unwrap();
                for _ in 0..2 {
                    KernelLaunch::new(g, kernels.tc)
                        .grid([div_ceil(n, 64), div_ceil(9u32, 16), 1])
                        .block([128, 1, 1])
                        .arg_ptr(ad).arg_ptr(wd).arg_ptr(sd).arg_ptr(cmd)
                        .arg_u32(9).arg_u32(n).arg_u32(k).launch(stream).unwrap();
                }
                let d2 = down(g, cmd, 9 * n as usize * 2).unwrap();
                eprintln!("{label}: tc m=9 relaunch deterministic: {}", d1 == d2);
                // Isolate: same launch with A rows 8..31 zeroed.
                let mut az = ab.clone();
                for i in 8 * k as usize * 2..az.len() {
                    az[i] = 0;
                }
                let adz = upload(g, &az).unwrap();
                KernelLaunch::new(g, kernels.tc)
                    .grid([div_ceil(n, 64), div_ceil(9u32, 16), 1])
                    .block([128, 1, 1])
                    .arg_ptr(adz).arg_ptr(wd).arg_ptr(sd).arg_ptr(cmd)
                    .arg_u32(9).arg_u32(n).arg_u32(k).launch(stream).unwrap();
                let dz = down(g, cmd, 9 * n as usize * 2).unwrap();
                let row0_eq = dz[..2 * n as usize] == tc_rows[..2 * n as usize];
                eprintln!("{label}: m=9 with A rows>=8 zeroed, row0 == ref: {}", row0_eq);
                // Which side is broken: compare tc(m=9) row0 and tc(m=1) row0
                // against the exact gemv row for the same A row.
                let d9 = dz[..2 * n as usize]
                    .chunks_exact(2)
                    .zip(exact_rows[..2 * n as usize].chunks_exact(2))
                    .filter(|(x, y)| x != y)
                    .count();
                let d1 = tc_rows[..2 * n as usize]
                    .chunks_exact(2)
                    .zip(exact_rows[..2 * n as usize].chunks_exact(2))
                    .filter(|(x, y)| x != y)
                    .count();
                eprintln!("{label}: m9-vs-gemv differing elems {d9}, m1-vs-gemv {d1} (of {n})");
            }
            for m in 1..=32usize {
                KernelLaunch::new(g, kernels.tc)
                    .grid([div_ceil(n, 64), div_ceil(m as u32, 16), 1])
                    .block([128, 1, 1])
                    .arg_ptr(ad)
                    .arg_ptr(wd)
                    .arg_ptr(sd)
                    .arg_ptr(cm)
                    .arg_u32(m as u32)
                    .arg_u32(n)
                    .arg_u32(k)
                    .launch(stream)
                    .unwrap();
                let got = down(g, cm, m * n as usize * 2).unwrap();
                for r in 0..m {
                    let a_row = &got[r * n as usize * 2..(r + 1) * n as usize * 2];
                    let b_row = &tc_rows[r * n as usize * 2..(r + 1) * n as usize * 2];
                    if a_row != b_row {
                        let mut shown = 0;
                        for e in 0..n as usize {
                            let (x, y) = (
                                u16::from_le_bytes([a_row[2 * e], a_row[2 * e + 1]]),
                                u16::from_le_bytes([b_row[2 * e], b_row[2 * e + 1]]),
                            );
                            if x != y && shown < 4 {
                                eprintln!(
                                    "{label} m={m} r={r} e={e}: tc={:#06x}({}) ref={:#06x}({})",
                                    x,
                                    half::bf16::from_bits(x).to_f32(),
                                    y,
                                    half::bf16::from_bits(y).to_f32()
                                );
                                shown += 1;
                            }
                        }
                        panic!("{label} m={m} r={r}: tc row not invariant");
                    }
                }
            }
            // Strict check vs an FP64 host reference on REALISTIC data:
            // non-NaN E4M3 codes, sane E8M0 scales (2^-12..2^0), N(0,1)
            // activations. Hostile bytes exercise invariance only (inf/NaN
            // make ULP meaningless).
            let mut rng2 = Lcg(0xC0FFEE);
            let w2: Vec<u8> = (0..(n as usize * k as usize))
                .map(|_| loop { let b = rng2.u8(); if b != 0x7F && b != 0xFF { break b } })
                .collect();
            let s2: Vec<u8> = (0..(n as usize * (k / 32) as usize))
                .map(|_| 115 + rng2.u8() % 13)  // 2^-12 .. 2^0
                .collect();
            let (wd2, sd2) = (upload(g, &w2).unwrap(), upload(g, &s2).unwrap());
            let cd2 = g.alloc(32 * (n as usize) * 2).unwrap();
            KernelLaunch::new(g, kernels.tc)
                .grid([div_ceil(n, 64), 2, 1])
                .block([128, 1, 1])
                .arg_ptr(ad)
                .arg_ptr(wd2)
                .arg_ptr(sd2)
                .arg_ptr(cd2)
                .arg_u32(32)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
                .unwrap();
            let got2 = down(g, cd2, 32 * n as usize * 2).unwrap();
            let e4m3 = |b: u8| -> f64 {
                let (sg, e, m) = ((b >> 7) as i64, (b >> 3) as i64 & 0xF, (b & 7) as f64);
                let v = if e == 0 { m * 2f64.powi(-9) } else { (1.0 + m / 8.0) * 2f64.powi(e as i32 - 7) };
                if sg != 0 { -v } else { v }
            };
            let e8m0 = |b: u8| -> f64 { 2f64.powi(b as i32 - 127) };
            let af: Vec<f64> = a.iter().map(|x| x.to_f32() as f64).collect();
            let mut max_ulp = 0u32;
            for r in [0usize, 3, 9, 17, 31] {
                for c in (0..n as usize).step_by(97) {
                    let mut acc = 0f64;
                    for kk in 0..k as usize {
                        acc += af[r * k as usize + kk]
                            * e4m3(w2[c * k as usize + kk])
                            * e8m0(s2[c * (k / 32) as usize + kk / 32]);
                    }
                    let refb = half::bf16::from_f32(acc as f32).to_bits() as i32;
                    let gotb = u16::from_le_bytes([
                        got2[(r * n as usize + c) * 2],
                        got2[(r * n as usize + c) * 2 + 1],
                    ]) as i32;
                    max_ulp = max_ulp.max((refb - gotb).unsigned_abs());
                }
            }
            assert!(max_ulp <= 1, "{label}: tc vs FP64 reference {max_ulp} ulp
");
            // Report divergence vs the exact CUDA-core path (informational).
            let mut diffs = 0usize;
            let mut maxd = 0f32;
            for i in 0..(32 * n as usize) {
                let a16 = u16::from_le_bytes([tc_rows[2 * i], tc_rows[2 * i + 1]]);
                let b16 = u16::from_le_bytes([exact_rows[2 * i], exact_rows[2 * i + 1]]);
                if a16 != b16 {
                    diffs += 1;
                    maxd = maxd.max(
                        (half::bf16::from_bits(a16).to_f32()
                            - half::bf16::from_bits(b16).to_f32())
                        .abs(),
                    );
                }
            }
            eprintln!("tc-vs-gemv {label}: {diffs}/{} differ, max |diff| {maxd}", 32 * n as usize);
        }
    }
    #[test]
    #[ignore = "GPU"]
    fn tc_rows_onehot_leak_map() {
        let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())
            .expect("CUDA backend");
        let g: &dyn GpuBackend = &gpu;
        let stream = g.default_stream();
        let kernels = Mxfp8ProjKernels::resolve(g).expect("mxfp8 kernels");
        // W identity: E4M3 1.0 = 0x38 at k==n, scale 0x7F (2^0) everywhere.
        let (n, k) = (512u32, 512u32);
        let mut w = vec![0u8; n as usize * k as usize];
        for i in 0..n as usize {
            w[i * k as usize + i] = 0x38;
        }
        let s = vec![0x7Fu8; n as usize * (k / 32) as usize];
        let (wd, sd) = (upload(g, &w).unwrap(), upload(g, &s).unwrap());
        // A row r = bf16(r+1) — output row r should be A row r.
        let mut a = vec![0u8; 32 * k as usize * 2];
        for r in 0..32usize {
            let v = half::bf16::from_f32((r + 1) as f32).to_bits().to_le_bytes();
            for c in 0..k as usize {
                a[(r * k as usize + c) * 2] = v[0];
                a[(r * k as usize + c) * 2 + 1] = v[1];
            }
        }
        let ad = upload(g, &a).unwrap();
        let cd = g.alloc(32 * n as usize * 2).unwrap();
        for m in [1usize, 8, 9, 16, 32] {
            KernelLaunch::new(g, kernels.tc)
                .grid([div_ceil(n, 64), div_ceil(m as u32, 16), 1])
                .block([128, 1, 1])
                .arg_ptr(ad)
                .arg_ptr(wd)
                .arg_ptr(sd)
                .arg_ptr(cd)
                .arg_u32(m as u32)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
                .unwrap();
            let got = down(g, cd, 32 * n as usize * 2).unwrap();
            for r in 0..m {
                let mut distinct = std::collections::BTreeSet::new();
                for c in 0..n as usize {
                    distinct.insert(u16::from_le_bytes(
                        [got[(r * n as usize + c) * 2], got[(r * n as usize + c) * 2 + 1]],
                    ));
                }
                let want = half::bf16::from_f32((r + 1) as f32).to_bits();
                if !(distinct.len() == 1 && distinct.contains(&want)) {
                    eprintln!("m={m} r={r}: bits {distinct:?} (want {want:#06x})");
                }
            }
            eprintln!("m={m}: leak map done");
        }
    }
}
