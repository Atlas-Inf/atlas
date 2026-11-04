// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): the row-local steps `sp_kda_gather_quantized` runs on one
//! rank's rows -- the MXFP8 cast on the 128-row scale tile grid and the b /
//! f_a / g_a side projections on the pipelined triple and on its packed
//! CUTLASS twin, and every head's beta on the CUTLASS twin of the pipelined
//! GEMM -- leave, on each quarter of a chunk and on each of its sixteenths
//! (the per-block gather's seam blocks), those rows' bytes of the whole
//! chunk's run.
//!
//!   cargo test --release -p spark-model --lib sp_kda_quant -- --ignored --nocapture

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

const H: usize = 4096;
const DIM: usize = 128;

fn values(seed: u64, n: usize, scale: f32) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .flat_map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let unit = (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
            let u = (unit * scale).to_bits();
            (((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn poison(gpu: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    upload(gpu, &vec![0xa5; n])
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// The cast and the side planes of `m` rows: `(q, sf, [b, f_a, g_a])`.
struct Outs {
    q: DevicePtr,
    sf: DevicePtr,
    side: [DevicePtr; 3],
    beta_full: DevicePtr,
}

/// Heads of the whole `b_proj` (every rank's): four ranks.
const RANKS: usize = 4;

impl Outs {
    fn new(gpu: &dyn GpuBackend, m: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            q: poison(gpu, m * H)?,
            sf: poison(gpu, ops::mx_lt_scale_bytes(m, H))?,
            side: [
                poison(gpu, m * heads * 2)?,
                poison(gpu, m * DIM * 2)?,
                poison(gpu, m * DIM * 2)?,
            ],
            beta_full: poison(gpu, m * RANKS * heads * 2)?,
        })
    }

    fn bytes(&self, gpu: &dyn GpuBackend, m: usize, heads: usize) -> Result<Vec<Vec<u8>>> {
        Ok(vec![
            read(gpu, self.q, m * H)?,
            read(gpu, self.sf, m * ops::mx_lt_inner_pad(H))?,
            read(gpu, self.side[0], m * heads * 2)?,
            read(gpu, self.side[1], m * DIM * 2)?,
            read(gpu, self.side[2], m * DIM * 2)?,
            read(gpu, self.beta_full, m * RANKS * heads * 2)?,
        ])
    }
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn sp_kda_quant_quarters_are_the_whole_chunks_bytes() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let quant = gpu.kernel(ops::MXFP8_ACT_MODULE, ops::MX_QUANT_KERNEL)?;
    let triple = gpu.kernel("gemm", "dense_gemm_bf16_pipelined_triple_n")?;
    for heads in [16usize, 32] {
        // b / f_a / g_a packed back to back, as the loader lays them out.
        let n = heads + 2 * DIM;
        let packed = upload(&gpu, &values(0x51de + heads as u64, n * H, 0.03))?;
        let weights = [0, heads, heads + DIM].map(|r| DenseWeight {
            weight: packed.offset(r * H * 2),
        });
        let b_full = upload(
            &gpu,
            &values(0xbe7a + heads as u64, RANKS * heads * H, 0.03),
        )?;
        for (m, seed) in [(8192usize, 1u64), (16384, 2)] {
            let x = upload(&gpu, &values(seed, m * H, 2.0))?;
            let scratch = poison(&gpu, m * n * 2)?;
            for (parts, packed_gemm) in [(4, false), (4, true), (16, false), (16, true)] {
                let rows = m / parts;
                let (whole, quarters) = (Outs::new(&gpu, m, heads)?, Outs::new(&gpu, m, heads)?);
                let run = |o: &Outs, row0: usize, rows: usize| -> Result<()> {
                    let act = ops::MxAct { q: o.q, sf: o.sf };
                    ops::mx_quant_rows(&gpu, quant, x, act, row0..row0 + rows, H, stream)?;
                    let side = [heads, DIM, DIM]
                        .iter()
                        .zip(o.side)
                        .map(|(&w, p)| p.offset(row0 * w * 2))
                        .collect::<Vec<_>>();
                    let side = [side[0], side[1], side[2]];
                    let input = x.offset(row0 * H * 2);
                    ensure!(
                        ops::bf16_gemm_pipelined_twin(
                            input,
                            b_full.0,
                            o.beta_full.offset(row0 * RANKS * heads * 2),
                            [rows as u32, (RANKS * heads) as u32, H as u32],
                            stream,
                        )?,
                        "CUTLASS declined the every-head beta GEMM"
                    );
                    if packed_gemm {
                        ensure!(
                            ops::dense_gemm_packed_triple_n(
                                &gpu,
                                input,
                                packed,
                                scratch,
                                side,
                                rows as u32,
                                [heads as u32, DIM as u32],
                                H as u32,
                                stream,
                            )?,
                            "CUTLASS declined the packed side GEMM"
                        );
                    } else {
                        ops::dense_gemm_pipelined_triple_n(
                            &gpu,
                            triple,
                            input,
                            [&weights[0], &weights[1], &weights[2]],
                            side,
                            rows as u32,
                            [heads as u32, DIM as u32],
                            H as u32,
                            stream,
                        )?;
                    }
                    Ok(())
                };
                run(&whole, 0, m)?;
                for part in 0..parts {
                    run(&quarters, part * rows, rows)?;
                }
                gpu.synchronize(stream)?;
                let (want, got) = (
                    whole.bytes(&gpu, m, heads)?,
                    quarters.bytes(&gpu, m, heads)?,
                );
                for (name, (w, g)) in ["q", "sf", "b", "f_a", "g_a", "beta_full"]
                    .iter()
                    .zip(want.iter().zip(&got))
                {
                    ensure!(
                        w == g,
                        "heads {heads}, {m} rows in {parts}, packed {packed_gemm}: {name} of the parts differ"
                    );
                }
            }
        }
    }
    println!(
        "MXFP8 cast, side projections, every-head beta: quarters and sixteenths == whole (16/32 heads, 8192/16384 rows)"
    );
    Ok(())
}
