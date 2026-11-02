// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10 and a glm-5.3-flash
//! kernel build): what `ATLAS_GLM_SP_MLA_QUANT_GATHER` feeds q_a, kv_a and
//! the semantic indexer gives their outputs the whole chunk's bytes.
//!
//! - The per-token-group cast of each quarter and sixteenth of a chunk (a
//!   rank's part, a seam block), placed at its rows, is the whole chunk's
//!   cast, and the W8A8 block FP8 GEMM on those cast rows
//!   (`Fp8bRows::Cast`) writes the BF16 rows' output: on the probed path
//!   and on both of the in-tree hand paths, with an FP8 scratch that takes
//!   the chunk in one pass and one that takes it in passes with a tail; the
//!   o_proj's (`ATLAS_GLM_SP_MLA_OPROJ_AHEAD`) also a seam block a call.
//! - The indexer's BF16 GEMM of an owner's rows, run at the owner's height
//!   on an input of which only one rank's part is set (the rest NaN),
//!   writes that part's rows as on the whole input.
//!
//!   cargo test --release -p spark-model --lib sp_mla_cast -- --ignored --nocapture

use anyhow::{Result, ensure};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::{self, Fp8ActScratch, Fp8bRows};
use crate::weight_map::{FP8_BLOCK, Fp8BlockWeight};

const H: usize = 4096;
const RANKS: usize = 4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn bf16(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                let unit = (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
                let u = (unit * scale).to_bits();
                (((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect()
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn poison(gpu: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    upload(gpu, &vec![0xff; n])
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn scratch(gpu: &dyn GpuBackend, rows: usize) -> Result<Fp8ActScratch> {
    let (act_bytes, scale_bytes) = (rows * H, rows * H / FP8_BLOCK * 4);
    Ok(Fp8ActScratch {
        act: gpu.alloc(act_bytes)?,
        act_bytes,
        scale: gpu.alloc(scale_bytes)?,
        scale_bytes,
    })
}

#[test]
#[ignore = "needs a GB10 and a glm-5.3-flash kernel build"]
fn sp_mla_cast_quarters_are_the_whole_chunks_bytes() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let kernels = ops::native_kernels(g);
    let mut rng = Rng(0x0317_ca57);
    let groups = H / FP8_BLOCK;
    // q_a, kv_a and o of a TP4 rank (o's K, 16 heads of 256, is H).
    let weights = [1536usize, 512, H].map(|n| {
        let values: Vec<u8> = (0..n * H).map(|_| rng.next() as u8 & 0xF7).collect();
        let grid: Vec<u8> = (0..(n / 128) * groups)
            .flat_map(|_| (0.005 + (rng.next() % 1000) as f32 * 1e-5).to_le_bytes())
            .collect();
        (n, values, grid)
    });
    let weights = weights
        .iter()
        .map(|(n, v, s)| {
            Ok((
                *n,
                Fp8BlockWeight {
                    data: upload(g, v)?,
                    scales: upload(g, s)?,
                    tiled: false,
                },
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    // A chunk on the 16-row pass grid and one whose parts end mid-pass.
    for m in [8192usize, 8196] {
        let x = upload(g, &rng.bf16(m * H, 2.0))?;
        // The whole chunk's cast, then each part's at its rows.
        let (act, scale) = (poison(g, m * H)?, poison(g, m * groups * 4)?);
        let quant = |input: DevicePtr, rows: (usize, usize), a: DevicePtr, s: DevicePtr| {
            ops::per_token_group_quant_fp8_warp(
                g,
                kernels.act_quant,
                input.offset(rows.0 * H * 2),
                a.offset(rows.0 * H),
                s.offset(rows.0 * groups * 4),
                rows.1 as u32,
                H as u32,
                stream,
            )
        };
        quant(x, (0, m), act, scale)?;
        let want = [read(g, act, m * H)?, read(g, scale, m * groups * 4)?];
        for parts in [RANKS, 4 * RANKS]
            .into_iter()
            .filter(|p| m.is_multiple_of(*p))
        {
            let rows = m / parts;
            let (pa, ps) = (poison(g, m * H)?, poison(g, m * groups * 4)?);
            for p in 0..parts {
                quant(x, (p * rows, rows), pa, ps)?;
            }
            let got = [read(g, pa, m * H)?, read(g, ps, m * groups * 4)?];
            ensure!(got == want, "{m} rows in {parts}: the parts' cast differs");
            g.free(pa)?;
            g.free(ps)?;
        }
        // One pass of the chunk, and passes of 3000 rows (tails of 8 and 12):
        // the cast rows whole and, for the o_proj, a sixteenth of the chunk
        // (a seam block of one part) a call.
        for pass_rows in [m + 64, 3000] {
            let s = scratch(g, pass_rows + 16)?;
            for &(n, ref w) in &weights {
                let out = |cast_step: Option<usize>, how: Option<bool>| -> Result<Vec<u8>> {
                    let y = poison(g, m * n * 2)?;
                    let step = cast_step.unwrap_or(m);
                    for r0 in (0..m).step_by(step) {
                        let rows = step.min(m - r0);
                        let input = match cast_step {
                            Some(_) => Fp8bRows::Cast {
                                act: act.offset(r0 * H),
                                scale: scale.offset(r0 * groups * 4),
                            },
                            None => Fp8bRows::Bf16(x.offset(r0 * H * 2)),
                        };
                        let (y, shape) = (y.offset(r0 * n * 2), [rows, n, H]);
                        match how {
                            None => {
                                ops::fp8b_w8a8_gemm(g, &kernels, s, w, input, y, shape, stream)?
                            }
                            Some(cutlass) => ops::fp8b_w8a8_hand(
                                g, &kernels, s, w, input, y, shape, cutlass, stream,
                            )?,
                        }
                    }
                    let bytes = read(g, y, m * n * 2)?;
                    g.free(y)?;
                    Ok(bytes)
                };
                let mut steps = vec![m];
                if n == H && m.is_multiple_of(4 * RANKS) {
                    steps.push(m / (4 * RANKS));
                }
                for how in [None, Some(false), Some(true)] {
                    let want = out(None, how)?;
                    for &step in &steps {
                        ensure!(
                            out(Some(step), how)? == want,
                            "{m} rows, [{n}, {H}], passes of {pass_rows}, {how:?}: cast rows {step} a call differ"
                        );
                    }
                }
            }
            for p in [s.act, s.scale] {
                g.free(p)?;
            }
        }
        // The indexer's keys GEMM per owner (the whole chunk, or a first
        // piece and the rest) on one part's rows.
        let wk = upload(g, &rng.bf16(128 * H, 0.03))?;
        let rows = m / RANKS;
        for owners in [vec![(0, m)], vec![(0, 2048), (2048, m - 2048)]] {
            let gemm = |input: DevicePtr, out: DevicePtr| -> Result<()> {
                for &(row0, on) in &owners {
                    ops::bf16_gemm(
                        input.offset(row0 * H * 2),
                        wk.0,
                        out.offset(row0 * 256),
                        on as u32,
                        128,
                        H as u32,
                        stream,
                    )?;
                }
                Ok(())
            };
            let whole = poison(g, m * 256)?;
            gemm(x, whole)?;
            let want = read(g, whole, m * 256)?;
            for part in 0..RANKS {
                let only = poison(g, m * H * 2)?;
                g.copy_d2d_async(
                    x.offset(part * rows * H * 2),
                    only.offset(part * rows * H * 2),
                    rows * H * 2,
                    stream,
                )?;
                let out = poison(g, m * 256 + 4096)?;
                // A differently placed output, as the gathered scratch is.
                gemm(only, out.offset(4096))?;
                let got = read(g, out.offset(4096), m * 256)?;
                let span = part * rows * 256..(part + 1) * rows * 256;
                ensure!(
                    got[span.clone()] == want[span],
                    "{m} rows, owners {owners:?}: part {part}'s index keys differ"
                );
                g.free(only)?;
                g.free(out)?;
            }
            g.free(whole)?;
        }
        for p in [x, act, scale, wk] {
            g.free(p)?;
        }
    }
    println!(
        "MLA cast gather: parts' cast == whole, W8A8 GEMM on cast rows == on BF16 rows \
         (probed / hand / CUTLASS, 1 and 3 passes; o_proj per seam block too), index GEMM \
         on one part == whole"
    );
    Ok(())
}
