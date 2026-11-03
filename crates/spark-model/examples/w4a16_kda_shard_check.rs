// SPDX-License-Identifier: AGPL-3.0-only

//! The NVFP4 kernels a GLM KDA projection takes (decode GEMV, verify batch
//! tiers and their PDL touch twins, the fused K=5 Q/K/V, the tensor-core
//! tiers, the prefill GEMM) at the TP4 shard shapes (Q/K/V `[2048, 4096]`,
//! O `[4096, 2048]`) and the TP2 control `[4096, 4096]`, through the
//! production `layers::ops` launchers, against a CPU dequantized f64
//! reference. Q/K/V shards are row slices of one TP2 weight (the loader's
//! TP2-scale requant), so each path's sliced output must also be the TP2
//! output's rows, bit for bit.
//!
//! ```text
//! ATLAS_PDL=1 ATLAS_W4A16_TC=1 ATLAS_GLM_DECODE_GEMV_BATCH=1 \
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!   --example w4a16_kda_shard_check
//! ```

use anyhow::{Result, bail};
use half::bf16;
use spark_model::layers::ops;
use spark_model::layers::w4a16_gemv_tiers::{W4a16BatchmTiers, tc_kernel};
use spark_model::weight_map::QuantizedWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn e4m3(b: u8) -> f32 {
    let (e, m) = ((b >> 3) & 0xF, (b & 7) as f32);
    if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7)
    }
}

/// Host NVFP4 of `w` `[n, k]`: packed bytes, block scales, scale2, and the
/// dequantized weight the kernels must reproduce.
struct Host {
    packed: Vec<u8>,
    scales: Vec<u8>,
    scale2: f32,
    deq: Vec<f32>,
    n: usize,
    k: usize,
}

fn quantize(w: &[f32], n: usize, k: usize) -> Host {
    let amax = w.iter().fold(0f32, |a, v| a.max(v.abs()));
    let scale2 = amax / (6.0 * 448.0);
    let table: Vec<f32> = (0..0x7Fu8).map(e4m3).collect();
    let (mut packed, mut scales, mut deq) = (
        vec![0u8; n * k / 2],
        vec![0u8; n * k / 16],
        vec![0f32; n * k],
    );
    for r in 0..n {
        for g in 0..k / 16 {
            let blk = &w[r * k + g * 16..][..16];
            let bmax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            let want = bmax / (6.0 * scale2);
            let sb = (0..table.len())
                .min_by(|&a, &b| (table[a] - want).abs().total_cmp(&(table[b] - want).abs()))
                .unwrap() as u8;
            scales[r * k / 16 + g] = sb;
            let s = e4m3(sb) * scale2;
            for (j, &v) in blk.iter().enumerate() {
                let q = if s == 0.0 {
                    0
                } else {
                    (0..8)
                        .min_by(|&a, &b| {
                            (E2M1[a] - (v / s).abs())
                                .abs()
                                .total_cmp(&(E2M1[b] - (v / s).abs()).abs())
                        })
                        .unwrap()
                };
                let code = q as u8 | if v < 0.0 && q != 0 { 8 } else { 0 };
                let idx = r * k + g * 16 + j;
                packed[idx / 2] |= code << ((idx % 2) * 4);
                deq[idx] = (if code & 8 != 0 { -1.0 } else { 1.0 }) * E2M1[q] * e4m3(sb) * scale2;
            }
        }
    }
    Host {
        packed,
        scales,
        scale2,
        deq,
        n,
        k,
    }
}

/// Rows `r0..r0+n` of `h`, keeping its scale2 (the TP2-scale shard).
fn rows(h: &Host, r0: usize, n: usize) -> Host {
    let k = h.k;
    Host {
        packed: h.packed[r0 * k / 2..(r0 + n) * k / 2].to_vec(),
        scales: h.scales[r0 * k / 16..(r0 + n) * k / 16].to_vec(),
        scale2: h.scale2,
        deq: h.deq[r0 * k..(r0 + n) * k].to_vec(),
        n,
        k,
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

fn to_gpu(gpu: &dyn GpuBackend, h: &Host) -> Result<QuantizedWeight> {
    Ok(QuantizedWeight {
        weight: upload(gpu, &h.packed)?,
        weight_scale: upload(gpu, &h.scales)?,
        weight_scale_2: h.scale2,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    })
}

struct Kernels {
    gemv: KernelHandle,
    gemv_sw: KernelHandle,
    b2: KernelHandle,
    b3: KernelHandle,
    b5qkv: KernelHandle,
    gemm: KernelHandle,
    gemm_t_m128: KernelHandle,
    tiers: W4a16BatchmTiers,
}

/// Every KDA route for `m` rows, as `(name, launch)`.
type Launch<'a> =
    Box<dyn Fn(DevicePtr, &QuantizedWeight, DevicePtr, u32, u32, u32) -> Result<()> + 'a>;

fn routes<'a>(gpu: &'a dyn GpuBackend, kn: &'a Kernels, m: u32) -> Vec<(String, Launch<'a>)> {
    let mut out: Vec<(String, Launch<'a>)> = Vec::new();
    if m == 1 {
        out.push((
            "gemv".into(),
            Box::new(move |a, w, c, _m, n, k| {
                ops::w4a16_decode_gemv(gpu, kn.gemv, kn.gemv_sw, false, a, w, c, n, k, 0)
            }),
        ));
        if kn.gemv_sw.0 != 0 {
            out.push((
                "gemv_sw".into(),
                Box::new(move |a, w, c, _m, n, k| {
                    ops::w4a16_decode_gemv(gpu, kn.gemv, kn.gemv_sw, true, a, w, c, n, k, 0)
                }),
            ));
        }
    }
    if m == 2 {
        out.push((
            "batch2".into(),
            Box::new(move |a, w, c, _m, n, k| ops::w4a16_gemv_batch2(gpu, kn.b2, a, w, c, n, k, 0)),
        ));
    }
    if m == 3 {
        out.push((
            "batch3".into(),
            Box::new(move |a, w, c, _m, n, k| ops::w4a16_gemv_batch3(gpu, kn.b3, a, w, c, n, k, 0)),
        ));
    }
    if (4..=8).contains(&m) {
        let s = kn.tiers.scalar_kernel(m);
        out.push((
            format!("batchm_scalar{m}"),
            Box::new(move |a, w, c, m, n, k| ops::w4a16_gemv_batchm(gpu, s, a, w, c, m, n, k, 0)),
        ));
    }
    if (2..=8).contains(&m) {
        // project_hot_multi_decode: the touch twin of the tier, else the tier.
        let tier = kn.tiers.kernel(m);
        out.push((
            format!("multi_decode_touch{m}"),
            Box::new(move |a, w, c, m, n, k| {
                ops::w4a16_verify_touch(gpu, tier, a, w, c, m, n, k, 0).unwrap_or_else(|| match m {
                    2 => ops::w4a16_gemv_batch2(gpu, kn.b2, a, w, c, n, k, 0),
                    3 => ops::w4a16_gemv_batch3(gpu, kn.b3, a, w, c, n, k, 0),
                    _ => ops::w4a16_gemv_batchm(gpu, tier, a, w, c, m, n, k, 0),
                })
            }),
        ));
    }
    let tc = tc_kernel(m);
    if tc.0 != 0 && m >= 2 {
        out.push((
            format!("tc{m}"),
            Box::new(move |a, w, c, m, n, k| ops::w4a16_gemv_batchm(gpu, tc, a, w, c, m, n, k, 0)),
        ));
        if m > 8 {
            out.push((
                format!("tc_touch{m}"),
                Box::new(move |a, w, c, m, n, k| {
                    ops::w4a16_verify_touch(gpu, tc, a, w, c, m, n, k, 0)
                        .unwrap_or_else(|| ops::w4a16_gemv_batchm(gpu, tc, a, w, c, m, n, k, 0))
                }),
            ));
        }
    }
    if m >= 9 || m == 1 {
        out.push((
            format!("gemm{m}"),
            Box::new(move |a, w, c, m, n, k| ops::w4a16_gemm(gpu, kn.gemm, a, w, c, m, n, k, 0)),
        ));
    }
    if m > 128 {
        // Prefill above 128 rows: the transposed twin (its weight is swapped
        // in by the caller for routes named `gemm_t_m128`).
        out.push((
            "gemm_t_m128".into(),
            Box::new(move |a, w, c, m, n, k| {
                ops::w4a16_gemm_n128_m128(gpu, kn.gemm_t_m128, a, w, c, m, n, k, 0)
            }),
        ));
    }
    if m == 5 && kn.b5qkv.0 != 0 {
        // The fused Q/K/V with all three planes on one weight.
        out.push((
            "batch5_qkv".into(),
            Box::new(move |a, w, c, m, n, k| {
                ops::w4a16_gemv_batch5_qkv(gpu, kn.b5qkv, None, a, w, w, w, c, m, n, k, 0)
            }),
        ));
        out.push((
            "batch5_qkv_touch".into(),
            Box::new(move |a, w, c, m, n, k| {
                ops::w4a16_gemv_batch5_qkv(
                    gpu,
                    kn.b5qkv,
                    ops::gemv_touch(ops::Twin::W4Batch5Qkv),
                    a,
                    w,
                    w,
                    w,
                    c,
                    m,
                    n,
                    k,
                    0,
                )
            }),
        ));
    }
    out
}

fn run(
    gpu: &dyn GpuBackend,
    f: &Launch,
    a: DevicePtr,
    w: &QuantizedWeight,
    m: usize,
    n: usize,
    k: usize,
    planes: usize,
) -> Result<Vec<u16>> {
    let bytes = planes * m * n * 2;
    let c = gpu.alloc(bytes)?;
    gpu.copy_h2d(&vec![0xFFu8; bytes], c)?; // NaN poison: unwritten outputs show
    f(a, w, c, m as u32, n as u32, k as u32)?;
    gpu.synchronize(0)?;
    let mut raw = vec![0u8; bytes];
    gpu.copy_d2h(c, &mut raw)?;
    gpu.free(c)?;
    Ok(raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

fn compare(out: &[u16], reference: &[f64]) -> (f64, f64, usize) {
    let (mut dot, mut na, mut nb, mut worst, mut bad) = (0f64, 0f64, 0f64, 0f64, 0usize);
    let rms = (reference.iter().map(|v| v * v).sum::<f64>() / reference.len() as f64).sqrt();
    for (&o, &r) in out.iter().zip(reference) {
        let g = bf16::from_bits(o).to_f32() as f64;
        if !g.is_finite() {
            bad += 1;
            continue;
        }
        dot += g * r;
        na += g * g;
        nb += r * r;
        let e = (g - r).abs() / r.abs().max(rms);
        worst = worst.max(e);
        if e > 0.02 {
            bad += 1;
        }
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-30), worst, bad)
}

fn main() -> Result<()> {
    spark_runtime::cuda_backend::configure_pdl("glm-5.3-flash");
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    ops::gemv_touch_resolve(gpu);
    let try_k = |f: &str| gpu.kernel("w4a16_gemv", f).unwrap_or(KernelHandle(0));
    let kn = Kernels {
        gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
        gemv_sw: try_k("w4a16_gemv_sw"),
        b2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
        b3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
        b5qkv: try_k("w4a16_gemv_batch5_qkv"),
        gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
        gemm_t_m128: gpu.kernel("w4a16", "w4a16_gemm_t_m128")?,
        tiers: W4a16BatchmTiers::resolve(gpu, true),
    };
    println!(
        "PDL {} touch {} tc {}",
        spark_runtime::cuda_backend::pdl_enabled(),
        ops::gemv_touch(ops::Twin::W4Batch2).is_some(),
        tc_kernel(8).0 != 0
    );
    let mut rng = Rng(0x7A4_C4A1);
    let mut failures = 0usize;
    let ms: Vec<usize> = std::env::var("ROWS").ok().map_or(
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 32, 64, 129],
        |s| s.split(',').filter_map(|v| v.parse().ok()).collect(),
    );
    // (label, N, K): the TP2 control first; the TP4 Q/K/V slices come from it.
    for (label, n, k) in [("tp2 qkv/o", 4096usize, 4096usize), ("tp4 o", 4096, 2048)] {
        let w: Vec<f32> = (0..n * k).map(|_| rng.next() * 0.2).collect();
        let full = quantize(&w, n, k);
        let mut shards = vec![(label.to_string(), full)];
        if n == 4096 && k == 4096 {
            let lo = rows(&shards[0].1, 0, 2048);
            let hi = rows(&shards[0].1, 2048, 2048);
            shards.push(("tp4 qkv rows 0..2048".into(), lo));
            shards.push(("tp4 qkv rows 2048..4096".into(), hi));
        }
        let gpu_w: Vec<QuantizedWeight> = shards
            .iter()
            .map(|(_, h)| to_gpu(gpu, h))
            .collect::<Result<_>>()?;
        let gpu_wt: Vec<QuantizedWeight> = gpu_w
            .iter()
            .zip(&shards)
            .map(|(w, (_, h))| w.transpose_for_gemm(gpu, h.n, h.k))
            .collect::<Result<_>>()?;
        for &m in &ms {
            let a: Vec<f32> = (0..m * k)
                .map(|_| bf16::from_f32(rng.next() * 4.0).to_f32())
                .collect();
            let a_dev = upload(
                gpu,
                &a.iter()
                    .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                    .collect::<Vec<u8>>(),
            )?;
            let mut full_out: Vec<(String, Vec<u16>)> = Vec::new();
            for (si, (sl, h)) in shards.iter().enumerate() {
                let reference: Vec<f64> = (0..m * h.n)
                    .map(|i| {
                        let (r, c) = (i / h.n, i % h.n);
                        (0..h.k)
                            .map(|j| a[r * h.k + j] as f64 * h.deq[c * h.k + j] as f64)
                            .sum()
                    })
                    .collect();
                for (name, f) in routes(gpu, &kn, m as u32) {
                    let planes = if name.starts_with("batch5_qkv") { 3 } else { 1 };
                    let weight = if name == "gemm_t_m128" {
                        &gpu_wt[si]
                    } else {
                        &gpu_w[si]
                    };
                    let out = run(gpu, &f, a_dev, weight, m, h.n, h.k, planes)?;
                    // t_m128 rounds weights and activations to E4M3 by design:
                    // report it, gate only its slice equality.
                    let lossy = name == "gemm_t_m128";
                    let mut worst_line = String::new();
                    for p in 0..planes {
                        let (cos, worst, bad) =
                            compare(&out[p * m * h.n..(p + 1) * m * h.n], &reference);
                        let ok = lossy || (cos > 0.9999 && bad == 0);
                        if !ok {
                            failures += 1;
                            worst_line +=
                                &format!(" FAIL plane{p} cos {cos:.6} worst {worst:.4} bad {bad}");
                        }
                    }
                    // Slices must equal the TP2 output's rows bit for bit.
                    let mut slice_note = String::new();
                    if si == 0 {
                        full_out.push((name.clone(), out.clone()));
                    } else if let Some((_, fo)) = full_out.iter().find(|(nm, _)| *nm == name) {
                        let r0 = (si - 1) * 2048;
                        let diff = (0..m)
                            .map(|r| {
                                (0..h.n)
                                    .filter(|&c| fo[r * 4096 + r0 + c] != out[r * h.n + c])
                                    .count()
                            })
                            .sum::<usize>();
                        if diff != 0 {
                            failures += 1;
                            slice_note = format!(" SLICE-MISMATCH {diff}");
                        }
                    }
                    if lossy
                        || !worst_line.is_empty()
                        || !slice_note.is_empty()
                        || std::env::var_os("VERBOSE").is_some()
                    {
                        let (cos, worst, _) = compare(&out[..m * h.n], &reference);
                        println!(
                            "{sl:>24} m={m:<3} {name:<22} cos {cos:.7} worst {worst:.4}{worst_line}{slice_note}"
                        );
                    }
                }
            }
            gpu.free(a_dev)?;
        }
        println!("{label}: done");
    }
    if failures != 0 {
        bail!("{failures} failing route(s)");
    }
    println!(
        "PASS: every KDA NVFP4 route matches the dequantized reference; TP4 slices equal TP2 rows"
    );
    Ok(())
}
