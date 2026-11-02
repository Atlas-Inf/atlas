// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention; a GB10, a glm-5.3-flash kernel
//! build and CUTLASS): the W8A8 block FP8 prefill on CUTLASS's SM120
//! blockwise kernel (`ATLAS_GLM_FP8B_CUTLASS=1`) against the in-tree
//! `fp8_gemm_t_blockscaled` on the same quantized operands, at TP4's native
//! MLA and shared-expert shapes: how many BF16 words differ, the largest
//! relative difference, and both times.
//!
//!   cargo test --release -p spark-model --lib fp8b_cutlass -- --ignored --nocapture

use super::*;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn bf16_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// `per_token_group_quant_fp8_warp` writes `per_token_group_quant_fp8`'s
/// bytes and scales for `m` rows of `input` (`[m, k]`).
fn quant_twins_agree(
    g: &dyn GpuBackend,
    kernels: &NativeKernels,
    input: DevicePtr,
    m: usize,
    k: usize,
) -> Result<()> {
    let one = g.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?;
    let groups = k / FP8_BLOCK;
    let bufs = [
        g.alloc(m * k)?,
        g.alloc(m * groups * 4)?,
        g.alloc(m * k)?,
        g.alloc(m * groups * 4)?,
    ];
    let stream = g.default_stream();
    let (m32, k32) = (m as u32, k as u32);
    super::super::per_token_group_quant_fp8(g, one, input, bufs[0], bufs[1], m32, k32, stream)?;
    super::super::per_token_group_quant_fp8_warp(
        g,
        kernels.act_quant,
        input,
        bufs[2],
        bufs[3],
        m32,
        k32,
        stream,
    )?;
    let t = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
        f()?;
        g.synchronize(stream)?;
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            f()?;
        }
        g.synchronize(stream)?;
        Ok(t0.elapsed().as_secs_f64() / 10.0 * 1e3)
    };
    let t_one = t(&|| {
        super::super::per_token_group_quant_fp8(g, one, input, bufs[0], bufs[1], m32, k32, stream)
    })?;
    let t_warp = t(&|| {
        super::super::per_token_group_quant_fp8_warp(
            g,
            kernels.act_quant,
            input,
            bufs[2],
            bufs[3],
            m32,
            k32,
            stream,
        )
    })?;
    let read = |p: DevicePtr, n: usize| -> Result<Vec<u8>> {
        let mut v = vec![0u8; n];
        g.copy_d2h(p, &mut v)?;
        Ok(v)
    };
    anyhow::ensure!(
        read(bufs[0], m * k)? == read(bufs[2], m * k)?
            && read(bufs[1], m * groups * 4)? == read(bufs[3], m * groups * 4)?,
        "per_token_group_quant_fp8_warp differs at {m} x {k}"
    );
    println!(
        "act quant {m} x {k}: one CTA a group {t_one:.3} ms, a warp a group {t_warp:.3} ms, same bytes"
    );
    for p in bufs {
        g.free(p)?;
    }
    Ok(())
}

#[test]
#[ignore = "needs a GB10, a glm-5.3-flash kernel build and CUTLASS"]
fn fp8b_cutlass_matches_the_in_tree_w8a8_gemm() -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let kernels = native_kernels(g);
    let mut rng = Rng(0x0f8b_c0de);
    let stream = g.default_stream();
    let max_rows = 8192usize;
    let act_bytes = max_rows * 4096;
    let scratch = Fp8ActScratch {
        act: g.alloc(act_bytes)?,
        act_bytes,
        scale: g.alloc(max_rows * 4096 / 128 * 4)?,
        scale_bytes: max_rows * 4096 / 128 * 4,
    };
    let mut worst = 0f32;
    // [n, k]: MLA o, q_b, q_a, kv_a, shared gate|up and down of a TP4 rank.
    for [n, k] in [
        [4096, 4096],
        [4096, 1536],
        [1536, 4096],
        [512, 4096],
        [4096, 512],
    ] {
        let values: Vec<u8> = (0..n * k).map(|_| rng.next() as u8 & 0xF7).collect();
        let grid: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| (0.005 + (rng.next() % 1000) as f32 * 1e-5).to_le_bytes())
            .collect();
        let w = Fp8BlockWeight {
            data: upload(g, &values)?,
            scales: upload(g, &grid)?,
            tiled: false,
        };
        for m in [37usize, 2048, 8192] {
            let x: Vec<u8> = (0..m * k)
                .flat_map(|_| {
                    let v = ((rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 2.0;
                    (((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16)
                        .to_le_bytes()
                })
                .collect();
            let input = upload(g, &x)?;
            if n == 4096 {
                quant_twins_agree(g, &kernels, input, m, k)?;
            }
            let (a, b) = (g.alloc(m * n * 2)?, g.alloc(m * n * 2)?);
            let run = |out: DevicePtr, cutlass: bool| {
                fp8b_w8a8_hand(
                    g,
                    &kernels,
                    scratch,
                    &w,
                    Fp8bRows::Bf16(input),
                    out,
                    [m, n, k],
                    cutlass,
                    stream,
                )
            };
            let time = |out: DevicePtr, cutlass: bool| -> Result<f64> {
                run(out, cutlass)?;
                g.synchronize(stream)?;
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    run(out, cutlass)?;
                }
                g.synchronize(stream)?;
                Ok(t.elapsed().as_secs_f64() / 5.0 * 1e3)
            };
            let (t_hand, t_cut) = (time(a, false)?, time(b, true)?);
            let (mut ha, mut hb) = (vec![0u8; m * n * 2], vec![0u8; m * n * 2]);
            g.copy_d2h(a, &mut ha)?;
            g.copy_d2h(b, &mut hb)?;
            let (fa, fb) = (bf16_to_f32(&ha), bf16_to_f32(&hb));
            let differ = ha
                .chunks(2)
                .zip(hb.chunks(2))
                .filter(|(x, y)| x != y)
                .count();
            let scale = fa.iter().fold(0f32, |s, v| s.max(v.abs())).max(1e-6);
            let rel = fa
                .iter()
                .zip(&fb)
                .fold(0f32, |s, (x, y)| s.max((x - y).abs()))
                / scale;
            worst = worst.max(rel);
            println!(
                "[{n}, {k}] rows {m}: in-tree {t_hand:.3} ms, CUTLASS {t_cut:.3} ms \
                 ({:.0} TFLOPS); {differ} of {} words differ, max |diff| / max|y| {rel:.2e}",
                2.0 * (m * n * k) as f64 / t_cut / 1e9,
                m * n
            );
            for p in [input, a, b] {
                g.free(p)?;
            }
        }
    }
    anyhow::ensure!(worst < 1e-2, "CUTLASS block FP8 diverges: {worst:.2e}");
    Ok(())
}
