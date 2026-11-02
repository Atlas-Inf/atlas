// SPDX-License-Identifier: AGPL-3.0-only

//! GPU tests (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): `ATLAS_GLM_MXFP8_TILED`'s twins, rewritten in place by
//! [`mxfp8_tile_in_place`] and served by `mxfp8_gemv_tc{8,16,32}_tiled`,
//! against the row-major tiers at the shapes they serve on a rank (MLA q_b
//! 8192x1536 and o 4096x8192, the split verify head's 77440x4096 shard,
//! staged in ten runs) and a one-run 48x128, for every row count 1..=32: the
//! same output bytes, and every row the same at every row count.
//!
//!   cargo test --release -p spark-model --lib mxfp8_tiled -- --ignored --nocapture

use super::*;
use spark_runtime::cuda_backend::AtlasCudaBackend;

const MAX_M: u32 = 32;
const K_MAX: usize = 8192;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn backend() -> Result<&'static AtlasCudaBackend> {
    crate::layers::glm_gpu_test_backend::backend()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn raw(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    crate::layers::glm_gpu_test_backend::finite_bf16("MXFP8 projection", &b)?;
    Ok(b)
}

#[test]
#[ignore = "needs a GB10 GPU and the glm-5.3-flash kernels"]
fn tiled_tiers_match_the_row_major_tiers_bit_for_bit() -> Result<()> {
    let gpu = backend()?;
    let stream = gpu.default_stream();
    let row_major = ["mxfp8_gemv_tc8", "mxfp8_gemv_tc16", "mxfp8_gemv_tc32"]
        .map(|name| crate::layers::try_kernel(gpu, "mxfp8_gemv", name));
    anyhow::ensure!(row_major.iter().all(|k| k.0 != 0), "mxfp8_gemv tiers");
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    // BF16 activations in (-2, 2) (the top 16 bits of an f32).
    let acts: Vec<u8> = (0..MAX_M as usize * K_MAX)
        .flat_map(|_| {
            let x = 2.0 * ((rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
            ((x.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    let a = upload(gpu, &acts)?;
    for (name, n, k) in [
        ("q_b", 8192usize, 1536usize),
        ("o", 4096, 8192),
        ("lm_head", 77440, 4096),
        ("one_run", 48, 128),
    ] {
        // E4M3 values below 2 in magnitude (no NaN), scales 2^-9 .. 2^-6.
        let values: Vec<u8> = (0..n * k).map(|_| rng.next() as u8 & 0xbf).collect();
        let scales: Vec<u8> = (0..n * k / MXFP8_BLOCK)
            .map(|_| 118 + (rng.next() % 4) as u8)
            .collect();
        let rm = Mxfp8Weight {
            data: upload(gpu, &values)?,
            scales: upload(gpu, &scales)?,
            tiled: false,
        };
        let mut tiled = Mxfp8Weight {
            data: upload(gpu, &values)?,
            scales: upload(gpu, &scales)?,
            tiled: false,
        };
        mxfp8_tile_in_place(gpu, &mut tiled, n, k, stream)?;
        assert!(tiled.tiled);
        let bytes = MAX_M as usize * n * 2;
        let (c_rm, c_t) = (gpu.alloc(bytes)?, gpu.alloc(bytes)?);
        let run = |w: &Mxfp8Weight, c: DevicePtr, m: u32| -> Result<Vec<u8>> {
            gpu.memset_async(c, 0xa5, bytes, stream)?;
            let kernel = mxfp8_kernel(gpu, w, &row_major, m);
            anyhow::ensure!(kernel.0 != 0, "{name}: no kernel for {m} rows");
            mxfp8_gemv(
                gpu, kernel, a, w.data, w.scales, c, m, n as u32, k as u32, n as u32, stream,
            )?;
            gpu.synchronize(stream)?;
            raw(gpu, c, m as usize * n * 2)
        };
        let full = run(&tiled, c_t, MAX_M)?;
        for m in 1..=MAX_M {
            let (want, got) = (run(&rm, c_rm, m)?, run(&tiled, c_t, m)?);
            let differ = want.iter().zip(&got).filter(|(x, y)| x != y).count();
            assert_eq!(
                differ, 0,
                "{name} {n}x{k}: {differ} bytes differ at {m} rows"
            );
            assert!(
                got == full[..got.len()],
                "{name} {n}x{k}: rows at {m} differ from the same rows at {MAX_M}"
            );
        }
        eprintln!("{name} {n}x{k}: tiled == row-major at 1..={MAX_M} rows");
        for p in [rm.data, rm.scales, tiled.data, tiled.scales, c_rm, c_t] {
            gpu.free(p)?;
        }
    }
    gpu.free(a)?;
    Ok(())
}
