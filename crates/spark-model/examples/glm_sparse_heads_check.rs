// SPDX-License-Identifier: AGPL-3.0-only
//! GLM sparse MLA kernels at a TP4 rank's 16 local heads against TP2's 32:
//! every tensor-core variant (unsplit, kv_pad, fp8_g128 kv_pad and pipe, the
//! decode and verify splits with their merge) runs once over 32 heads and
//! once over the first 16 of the same queries, on a synthetic latent cache.
//! The kernels mask the rows past `heads`, so the 16-head output must equal
//! the first 16 heads of the 32-head output bit for bit. Reports both times
//! and the bandwidth floor (selected latent bytes / 273 GB/s).
//!
//! Exit: 0 pass, 1 any bit differs.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example glm_sparse_heads_check

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const DIM: usize = 512;
const WIDTH: u32 = 2051;
const TOKENS: usize = 32768;
const BLOCK: u32 = 16;
const FP8_TOKEN: usize = 528;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn f(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

fn bf16(v: f32) -> [u8; 2] {
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(16))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn time(g: &dyn GpuBackend, f: &dyn Fn() -> Result<()>) -> Result<f64> {
    f()?;
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() / 10.0)
}

/// One kernel under test: module, symbol, its 16-head twin (`""`: the same
/// kernel) with that twin's shared bytes and threads (0: the same) and
/// whether it must match bit for bit (else its error is reported), shared
/// bytes, whether it reads the fp8_g128 cache, and its split count (0:
/// writes BF16 directly).
struct Variant {
    module: &'static str,
    symbol: &'static str,
    head16: &'static str,
    head16_launch: (u32, u32),
    exact: bool,
    shared: u32,
    fp8: bool,
    splits: u32,
}

const VARIANTS: &[Variant] = &[
    Variant {
        module: "glm_sparse_prefill_tc",
        symbol: "glm_sparse_mla_prefill_bf16_head32_tc",
        head16: "",
        head16_launch: (0, 0),
        exact: true,
        shared: 101120,
        fp8: false,
        splits: 0,
    },
    Variant {
        module: "glm_sparse_prefill_kv_reuse",
        symbol: "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad",
        head16: "",
        head16_launch: (0, 0),
        exact: true,
        shared: 69376,
        fp8: false,
        splits: 0,
    },
    Variant {
        module: "glm_sparse_prefill_kv_reuse",
        symbol: "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad",
        head16: "",
        head16_launch: (0, 0),
        exact: true,
        shared: 69376,
        fp8: true,
        splits: 0,
    },
    Variant {
        module: "glm_sparse_prefill_kv_reuse",
        symbol: "glm_sparse_mla_prefill_fp8g128_head32_tc_pipe",
        head16: "glm_sparse_mla_prefill_fp8g128_head16_tc_pipe",
        head16_launch: (0, 0),
        exact: true,
        shared: 80128,
        fp8: true,
        splits: 0,
    },
    Variant {
        module: "glm_sparse_prefill_kv_reuse",
        symbol: "glm_sparse_mla_prefill_fp8g128_head32_tc_pipe",
        head16: "glm_sparse_mla_prefill_fp8g128_head16_tc_pipe_wg2",
        head16_launch: (96768, 512),
        exact: false,
        shared: 80128,
        fp8: true,
        splits: 0,
    },
    Variant {
        module: "glm_sparse_decode_split",
        symbol: "atlas_sparse_decode_split",
        head16: "",
        head16_launch: (0, 0),
        exact: true,
        shared: 69376,
        fp8: false,
        splits: 8,
    },
    Variant {
        module: "glm_sparse_prefill_kv_reuse",
        symbol: "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split",
        head16: "",
        head16_launch: (0, 0),
        exact: true,
        shared: 69376,
        fp8: false,
        splits: 6,
    },
    Variant {
        module: "glm_sparse_prefill_kv_reuse",
        symbol: "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split",
        head16: "",
        head16_launch: (0, 0),
        exact: true,
        shared: 69376,
        fp8: true,
        splits: 6,
    },
];

struct Caches {
    bf16: DevicePtr,
    fp8: DevicePtr,
    table: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
fn run(
    g: &dyn GpuBackend,
    v: &Variant,
    k: KernelHandle,
    merge: KernelHandle,
    c: &Caches,
    (q, idx, out): (DevicePtr, DevicePtr, DevicePtr),
    (part, plse, olse): (DevicePtr, DevicePtr, DevicePtr),
    rows: u32,
    heads: u32,
) -> Result<()> {
    let cache = if v.fp8 { c.fp8 } else { c.bf16 };
    let l = KernelLaunch::new(g, k)
        .grid([1, rows, v.splits.max(1)])
        .block([
            if heads == 16 && v.head16_launch.1 != 0 {
                v.head16_launch.1
            } else {
                256
            },
            1,
            1,
        ])
        .shared_mem(if heads == 16 && v.head16_launch.0 != 0 {
            v.head16_launch.0
        } else {
            v.shared
        })
        .arg_ptr(q)
        .arg_ptr(cache)
        .arg_ptr(cache)
        .arg_ptr(idx)
        .arg_ptr(if v.module == "glm_sparse_decode_split" {
            part
        } else {
            out
        })
        .arg_ptr(c.table)
        .arg_u32(rows)
        .arg_u32(heads)
        .arg_u32(DIM as u32)
        .arg_u32(WIDTH)
        .arg_u32(BLOCK)
        .arg_f32(1.0 / 16.0);
    if v.splits == 0 {
        return l.launch(0);
    }
    // The decode split writes its partials at O and takes its split count;
    // the verify split takes the partials last and reads gridDim.z.
    let l = if v.module == "glm_sparse_decode_split" {
        l.arg_ptr(plse).arg_u32(v.splits)
    } else {
        l.arg_ptr(part).arg_ptr(plse)
    };
    l.launch(0)?;
    KernelLaunch::new(g, merge)
        .grid([rows * heads, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(part)
        .arg_ptr(plse)
        .arg_ptr(out)
        .arg_ptr(olse)
        .arg_u32(rows)
        .arg_u32(heads)
        .arg_u32(DIM as u32)
        .arg_u32(v.splits)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let merge = g.kernel(
        "glm_sparse_decode_split_merge",
        "glm_sparse_decode_split_merge",
    )?;
    let mut rng = Lcg(0x5A5A);
    let cache: Vec<u8> = (0..TOKENS * DIM).flat_map(|_| bf16(rng.f())).collect();
    // fp8_g128 blocks: [16][512] E4M3 codes (no NaN codes), then [16][4] scales.
    let mut fp8 = vec![0u8; TOKENS * FP8_TOKEN];
    for block in fp8.chunks_exact_mut(BLOCK as usize * FP8_TOKEN) {
        let (codes, scales) = block.split_at_mut(BLOCK as usize * DIM);
        codes
            .iter_mut()
            .for_each(|b| *b = (rng.next() as u8) & 0xF7);
        for s in scales.chunks_exact_mut(4) {
            s.copy_from_slice(&(0.001 + (rng.next() % 1000) as f32 * 1e-5).to_le_bytes());
        }
    }
    let table: Vec<u8> = (0..(TOKENS as u32 / BLOCK))
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let c = Caches {
        bf16: up(g, &cache)?,
        fp8: up(g, &fp8)?,
        table: up(g, &table)?,
    };
    let mut fail = false;
    for rows in [1u32, 8, 2048] {
        let q32: Vec<u8> = (0..rows as usize * 32 * DIM)
            .flat_map(|_| bf16(rng.f() * 2.0))
            .collect();
        let q16: Vec<u8> = q32
            .chunks_exact(32 * DIM * 2)
            .flat_map(|row| row[..16 * DIM * 2].to_vec())
            .collect();
        // Selected IDs: random context tokens, the last three slots empty.
        let idx: Vec<u8> = (0..rows * WIDTH)
            .flat_map(|i| {
                let t = if i % WIDTH >= WIDTH - 3 {
                    -1
                } else {
                    (rng.next() % TOKENS as u64) as i32
                };
                t.to_le_bytes()
            })
            .collect();
        let (d_q32, d_q16, d_idx) = (up(g, &q32)?, up(g, &q16)?, up(g, &idx)?);
        let out_bytes = rows as usize * 32 * DIM * 2;
        let (d_o32, d_o16) = (g.alloc(out_bytes)?, g.alloc(out_bytes)?);
        let rh = rows as usize * 32;
        let part = g.alloc(8 * rh * DIM * 4)?;
        let (plse, olse) = (g.alloc(8 * rh * 4)?, g.alloc(rh * 4)?);
        for v in VARIANTS {
            // The decode split is one row; the verify split few-row.
            if (v.module == "glm_sparse_decode_split" && rows != 1) || (v.splits > 0 && rows > 8) {
                continue;
            }
            let k = g.kernel(v.module, v.symbol)?;
            anyhow::ensure!(k.0 != 0, "{} is not in {}", v.symbol, v.module);
            let k16 = match v.head16 {
                "" => k,
                twin => g.kernel(v.module, twin)?,
            };
            anyhow::ensure!(k16.0 != 0, "{} is not in {}", v.head16, v.module);
            let launch = |q, out, heads| {
                run(
                    g,
                    v,
                    if heads == 16 { k16 } else { k },
                    merge,
                    &c,
                    (q, d_idx, out),
                    (part, plse, olse),
                    rows,
                    heads,
                )
            };
            let t32 = time(g, &|| launch(d_q32, d_o32, 32))?;
            let t16 = time(g, &|| launch(d_q16, d_o16, 16))?;
            let (mut o32, mut o16) = (vec![0u8; out_bytes], vec![0u8; out_bytes / 2]);
            g.copy_d2h(d_o32, &mut o32)?;
            g.copy_d2h(d_o16, &mut o16)?;
            let want: Vec<u8> = o32
                .chunks_exact(32 * DIM * 2)
                .flat_map(|row| row[..16 * DIM * 2].to_vec())
                .collect();
            let differ = want.iter().zip(&o16).filter(|(a, b)| a != b).count();
            // A reordered sum: the largest difference against the 32-head
            // kernel's, relative to that row's largest output.
            let worst = want
                .chunks_exact(DIM * 2)
                .zip(o16.chunks_exact(DIM * 2))
                .map(|(x, y)| {
                    let f =
                        |b: &[u8]| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
                    let scale = x
                        .chunks_exact(2)
                        .map(|b| f(b).abs())
                        .fold(1e-30f32, f32::max);
                    x.chunks_exact(2)
                        .zip(y.chunks_exact(2))
                        .map(|(a, b)| (f(a) - f(b)).abs() / scale)
                        .fold(0f32, f32::max)
                })
                .fold(0f32, f32::max);
            let token = if v.fp8 { FP8_TOKEN } else { DIM * 2 };
            let floor = (rows as usize * (WIDTH as usize - 3) * token) as f64 / 273e9;
            println!(
                "{:<52} rows={rows:<5} 32h {:>9.1} us  16h {:>9.1} us  floor {:>8.1} us  \
                 16h {:>5.1}% of floor  bytes differing {differ} (max {worst:.2e} of row max)",
                v.symbol,
                t32 * 1e6,
                t16 * 1e6,
                floor * 1e6,
                100.0 * floor / t16,
            );
            fail |= v.exact && differ != 0;
        }
        for p in [d_q32, d_q16, d_idx, d_o32, d_o16, part, plse, olse] {
            g.free(p)?;
        }
    }
    println!("{}", if fail { "FAIL" } else { "PASS" });
    std::process::exit(i32::from(fail));
}
