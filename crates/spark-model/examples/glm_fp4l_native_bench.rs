// SPDX-License-Identifier: AGPL-3.0-only
//! FP4-Local native-format GEMVs against the production paths they replace,
//! at GLM-5.3-Flash's per-rank (TP2 / EP2) shapes and the decode / verify /
//! multi-sequence row counts M = 1, 2, 4, 8, 16, 32 (plus a correctness gate).
//!
//! Groups (production arm -> native arm, `ATLAS_GLM_FP4L_NATIVE=1`):
//! - KDA q/k/v [3 x 4096 x 4096]: NVFP4 `w4a16_gemv_tc8` x3 (canonical) ->
//!   MXFP8 `mxfp8_gemv_tc*` x3 and the fused `mxfp8_gemv_tc*_multi` (one launch);
//!   KDA o [4096 x 4096]: NVFP4 tc8 -> MXFP8.
//! - MLA q_a + kv_a [1536 + 512 x 4096], q_b [8192 x 1536], o [4096 x 8192]:
//!   production MXFP8 twins -> FP8 block `fp8b_gemv_tc*` (q_a + kv_a also as one
//!   `fp8b_gemv_tc*_multi` launch).
//! - shared expert, TP split (rank half): NVFP4 tc8 gate + up [1024 x 4096] and
//!   strided down [4096 x 1024 of 2048] -> `fp8b_gemv_tc*_multi` + `fp8b_gemv_tc*_ld`.
//! - dense MLP 0..2 (replicated): NVFP4 tc8 gate + up [12288 x 4096], down
//!   [4096 x 12288] -> `mxfp8_gemv_tc*_multi` + `mxfp8_gemv_tc*`.
//!
//! Weights cycle over enough copies (>= 256 MiB) that every launch streams them
//! from DRAM, like a decode step: a call reading several weights takes a
//! disjoint set of copies (copy 3c + i for the i-th of three), so nothing it
//! reads is still in L2 from the call before. Each line prints the group's us per call
//! (best of `REPS` batches), the bytes it reads and the GB/s; the floor column
//! is a device-to-device copy of the same bytes (read + write, so its GB/s is
//! 2 x bytes / time), a rough stream roofline.
//!
//! Gate (exit 1 on failure):
//! 1. every FP8 block tier row at M equals the same row at 32 bit for bit
//!    (decode == verify row), for each MLA / shared shape;
//! 2. the multi-projection launches equal their single-projection launches
//!    bit for bit (KDA q/k/v, MLA q_a + kv_a, shared halves);
//! 3. `fp8b_gemv` at M = 4 is within 1e-2 (relative to the row's |y| max) of
//!    an f64 host reference built from the same E4M3 bytes and f32 grid.
//!
//! Run (GB10, GLM kernel build):
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash ATLAS_TARGET_QUANT=nvfp4 \
//!     cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example glm_fp4l_native_bench [reps=7] [tp=2]
//!
//! `tp=4`: the MLA q_b / o at a TP4 rank's 16 heads ([4096 x 1536] /
//! [4096 x 4096]); the q_b / o lines then add the FP8 block twin in the
//! chunk-tiled layout (`ATLAS_GLM_MXFP8_TILED`).

use anyhow::{Result, ensure};
use spark_model::layers::ops::{self, MxPart, native_kernels};
use spark_model::weight_map::{Fp8BlockWeight, Mxfp8Weight};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const ROWS: [usize; 6] = [1, 2, 4, 8, 16, 32];
const COLD_BYTES: usize = 256 << 20;
const ITERS: usize = 20;

#[path = "glm_fp4l_native_bench/gate.rs"]
mod gate;
#[path = "glm_fp4l_native_bench/support.rs"]
mod support;
use support::*;

fn main() -> Result<()> {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let tp: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    ensure!(matches!(tp, 2 | 4), "tp must be 2 or 4");
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let missing = ops::missing_native_kernels(g);
    ensure!(
        missing.is_empty(),
        "this kernel build lacks {missing:?}. With ATLAS_TARGET_MODEL=glm-5.3-flash that is a \
         stale atlas-kernels build-script output (an mtime-preserving sync left the .cu files \
         older than the last build): touch kernels/gb10/glm-5.3-flash/nvfp4/mxfp8_gemv.cu or \
         `cargo clean -p atlas-kernels`, then rebuild"
    );
    let nk = native_kernels(g);
    let tc8 = g.kernel("w4a16_gemv", "w4a16_gemv_tc8")?;
    let tc8_ld = g.kernel("w4a16_gemv", "w4a16_gemv_tc8_ld")?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    // Activations: 32 rows of the widest K (12288), and outputs.
    let x_host: Vec<f32> = (0..32 * 12288).map(|_| rng.unit()).collect();
    let x = up(g, &bf16_bytes(&x_host))?;
    let outs: Vec<DevicePtr> = (0..4)
        .map(|_| g.alloc(32 * 12288 * 2))
        .collect::<Result<_>>()?;
    let floor_dst = g.alloc(COLD_BYTES)?;
    let mut fail = false;

    let kda = Weight::new(g, &mut rng, 4096, 4096)?;
    let q_a = Weight::new(g, &mut rng, 1536, 4096)?;
    let kv_a = Weight::new(g, &mut rng, 512, 4096)?;
    let heads_dim = 16384 / tp;
    let q_b = Weight::new(g, &mut rng, heads_dim, 1536)?;
    let mla_o = Weight::new(g, &mut rng, 4096, heads_dim)?;
    // The FP8 block twins of q_b / o again, value-tiled (TP4 only).
    let tiled_fp8b = |w: &Weight| -> Result<Vec<Fp8BlockWeight>> {
        if tp != 4 {
            return Ok(Vec::new());
        }
        let mut set = Vec::new();
        for t in &w.fp8b {
            let data = g.alloc(w.n * w.k)?;
            g.copy_d2d(t.data, data, w.n * w.k)?;
            ops::mx_tile_values_in_place(g, data, w.n, w.k, 0)?;
            set.push(Fp8BlockWeight {
                data,
                tiled: true,
                ..*t
            });
        }
        Ok(set)
    };
    let (q_b_tiled, mla_o_tiled) = (tiled_fp8b(&q_b)?, tiled_fp8b(&mla_o)?);
    let sh_gu = Weight::new(g, &mut rng, 2048, 4096)?;
    let sh_down = Weight::new(g, &mut rng, 4096, 2048)?;
    let dense_gu = Weight::new(g, &mut rng, 12288, 4096)?;
    let dense_down = Weight::new(g, &mut rng, 4096, 12288)?;

    // A chunk-tiled copy set of the KDA weight (values + scales relaid).
    let kda_tiled: Vec<Mxfp8Weight> = {
        let fresh = Weight::new(g, &mut rng, 4096, 4096)?;
        let mut set = fresh.mx;
        for w in set.iter_mut() {
            ops::mxfp8_tile_in_place(g, w, 4096, 4096, 0)?;
        }
        set
    };

    let floor = |w: &Weight, bytes: usize| -> Result<f64> {
        let copies = w.mx.len();
        time(g, reps, copies, &|c| {
            g.copy_d2d_async(w.mx[c].data, floor_dst, bytes.min(w.n * w.k), 0)
        })
        .map(|us| us / 2.0 * (bytes as f64 / bytes.min(w.n * w.k) as f64))
    };

    println!("== KDA q/k/v [3 x 4096 x 4096] and o [4096 x 4096] (per rank, 34 layers)");
    for m in ROWS {
        let n = 4096;
        let copies = kda.mx.len();
        let nv = time(g, reps, copies, &|c| {
            for i in 0..3 {
                nv_tc(g, tc8, kda.nv[(3 * c + i) % copies], x, outs[i], m, n, 4096)?;
            }
            Ok(())
        })?;
        line("qkv NVFP4 tc8 x3 (production)", m, nv, 3 * kda.nv_bytes());
        let mx3 = time(g, reps, copies, &|c| {
            for i in 0..3 {
                ops::mxfp8_gemv_rows(
                    g,
                    &nk.mx.row_major,
                    &kda.mx[(3 * c + i) % copies],
                    x,
                    outs[i],
                    m,
                    n,
                    4096,
                    n,
                    0,
                )?;
            }
            Ok(())
        })?;
        line("qkv MXFP8 x3", m, mx3, 3 * kda.mx_bytes());
        let fused = time(g, reps, copies, &|c| {
            let parts: Vec<MxPart> = (0..3)
                .map(|i| MxPart::mx(&kda.mx[(3 * c + i) % copies], outs[i], n))
                .collect();
            ops::mx_multi_rows(g, &nk.mx_multi, false, x, &parts, m, 4096, 0)
        })?;
        line("qkv MXFP8 fused (native)", m, fused, 3 * kda.mx_bytes());
        // Attribution arms on the same shape and the same tier body:
        // FP8 block scales (one f32 per 128 x 128, broadcast) instead of
        // UE8M0 per 32 (two scattered byte loads per lane per chunk) ...
        let fb = time(g, reps, copies, &|c| {
            let parts: Vec<MxPart> = (0..3)
                .map(|i| MxPart::fp8b(&kda.fp8b[(3 * c + i) % copies], outs[i], n))
                .collect();
            ops::mx_multi_rows(g, &nk.fp8b_multi, false, x, &parts, m, 4096, 0)
        })?;
        line(
            "qkv FP8 block fused (scales only)",
            m,
            fb,
            3 * kda.fp8b_bytes(),
        );
        // ... and the chunk-tiled MXFP8 layout (ATLAS_GLM_FP4L_KDA_TILED_TWIN).
        let tiled = time(g, reps, kda_tiled.len(), &|c| {
            let copies = kda_tiled.len();
            let parts: Vec<MxPart> = (0..3)
                .map(|i| MxPart::mx(&kda_tiled[(3 * c + i) % copies], outs[i], n))
                .collect();
            ops::mx_multi_rows(g, &nk.mx_multi, true, x, &parts, m, 4096, 0)
        })?;
        line("qkv MXFP8 fused tiled", m, tiled, 3 * kda.mx_bytes());
        let o_nv = time(g, reps, copies, &|c| {
            nv_tc(g, tc8, kda.nv[c], x, outs[0], m, n, 4096)
        })?;
        line("o NVFP4 tc8 (production)", m, o_nv, kda.nv_bytes());
        let o_mx = time(g, reps, copies, &|c| {
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &kda.mx[c],
                x,
                outs[0],
                m,
                n,
                4096,
                n,
                0,
            )
        })?;
        line("o MXFP8 (native)", m, o_mx, kda.mx_bytes());
    }
    let f = floor(&kda, 3 * kda.mx_bytes())?;
    line(
        "floor: d2d of the qkv MXFP8 bytes",
        0,
        f,
        3 * kda.mx_bytes(),
    );

    // Informational: the cuBLASLt MXFP8 W8A8 GEMM (row-invariant, NVIDIA's
    // MXFP8 recipe: activations to E4M3 per 32) at decode / verify heights.
    match ops::MxW8a8::from_twin(g, &kda.mx[0], [4096, 4096], 32, 0) {
        Ok(w8a8) => {
            let q = g.alloc(32 * 4096)?;
            let sf = g.alloc(ops::mx_lt_scale_bytes(32, 4096))?;
            let act = ops::MxAct { q, sf };
            for m in ROWS {
                let us = time(g, reps, 1, &|_| {
                    ops::mx_quant_rows(g, nk.mx_quant, x, act, 0..m, 4096, 0)?;
                    w8a8.gemm(act, outs[0], m, 0)
                })?;
                line(
                    "q MXFP8 W8A8 cuBLASLt (1 copy, warm)",
                    m,
                    us,
                    kda.mx_bytes(),
                );
            }
        }
        Err(e) => println!("  (W8A8 arm skipped: {e:#})"),
    }

    println!("== MLA q_a + kv_a, q_b, o (per rank, 11 layers)");
    for m in ROWS {
        let cq = q_a.mx.len().min(kv_a.mx.len());
        let mx = time(g, reps, cq, &|c| {
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &q_a.mx[c],
                x,
                outs[0],
                m,
                1536,
                4096,
                1536,
                0,
            )?;
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &kv_a.mx[c],
                x,
                outs[1],
                m,
                512,
                4096,
                512,
                0,
            )
        })?;
        line(
            "q_a + kv_a MXFP8 twins (production)",
            m,
            mx,
            q_a.mx_bytes() + kv_a.mx_bytes(),
        );
        let fb = time(g, reps, cq, &|c| {
            let parts = [
                MxPart::fp8b(&q_a.fp8b[c], outs[0], 1536),
                MxPart::fp8b(&kv_a.fp8b[c], outs[1], 512),
            ];
            ops::mx_multi_rows(g, &nk.fp8b_multi, false, x, &parts, m, 4096, 0)
        })?;
        line(
            "q_a + kv_a FP8 block fused (native)",
            m,
            fb,
            q_a.fp8b_bytes() + kv_a.fp8b_bytes(),
        );
        for (label, w, tiled) in [("q_b", &q_b, &q_b_tiled), ("o", &mla_o, &mla_o_tiled)] {
            let copies = w.mx.len();
            let mx = time(g, reps, copies, &|c| {
                ops::mxfp8_gemv_rows(
                    g,
                    &nk.mx.row_major,
                    &w.mx[c],
                    x,
                    outs[0],
                    m,
                    w.n,
                    w.k,
                    w.n,
                    0,
                )
            })?;
            line(
                &format!("{label} MXFP8 twin (production)"),
                m,
                mx,
                w.mx_bytes(),
            );
            let fb = time(g, reps, copies, &|c| {
                ops::fp8b_gemv_rows(g, &nk.fp8b, &w.fp8b[c], x, outs[0], m, w.n, w.k, w.n, 0)
            })?;
            line(
                &format!("{label} FP8 block (native)"),
                m,
                fb,
                w.fp8b_bytes(),
            );
            if !tiled.is_empty() {
                let ft = time(g, reps, tiled.len(), &|c| {
                    ops::fp8b_gemv_rows(g, &nk.fp8b, &tiled[c], x, outs[0], m, w.n, w.k, w.n, 0)
                })?;
                line(&format!("{label} FP8 block tiled"), m, ft, w.fp8b_bytes());
                line(
                    &format!("{label} floor"),
                    m,
                    floor(w, w.fp8b_bytes())?,
                    w.fp8b_bytes(),
                );
            }
        }
    }

    println!(
        "== shared expert, TP split rank half (42 layers): gate + up [1024 x 4096], down [4096 x 1024/2048]"
    );
    for m in ROWS {
        let copies = sh_gu.nv.len().min(sh_down.nv.len());
        let nv = time(g, reps, copies, &|c| {
            let (gw, gs) = sh_gu.nv[2 * c % copies];
            let half = (gw.offset(1024 * 4096 / 2), gs.offset(1024 * 4096 / 16));
            nv_tc(g, tc8, half, x, outs[0], m, 1024, 4096)?;
            nv_tc(
                g,
                tc8,
                sh_gu.nv[(2 * c + 1) % copies],
                x,
                outs[1],
                m,
                1024,
                4096,
            )?;
            nv_tc_ld(
                g,
                tc8_ld,
                sh_down.nv[c],
                outs[0],
                outs[2],
                m,
                4096,
                1024,
                2048,
                1024,
            )
        })?;
        let nv_bytes = 2 * sh_gu.nv_bytes() / 2 + sh_down.nv_bytes() / 2;
        line("NVFP4 tc8 split (production)", m, nv, nv_bytes);
        let fb = time(g, reps, copies, &|c| {
            let rows = |w: &Fp8BlockWeight| Fp8BlockWeight {
                data: w.data.offset(1024 * 4096),
                scales: w.scales.offset(8 * 32 * 4),
                tiled: false,
            };
            let parts = [
                MxPart::fp8b(&rows(&sh_gu.fp8b[2 * c % copies]), outs[0], 1024),
                MxPart::fp8b(&rows(&sh_gu.fp8b[(2 * c + 1) % copies]), outs[1], 1024),
            ];
            ops::mx_multi_rows(g, &nk.fp8b_multi, false, x, &parts, m, 4096, 0)?;
            let d = &sh_down.fp8b[c];
            ops::fp8b_gemv_ld_rows(
                g,
                &nk.fp8b_ld,
                d.data.offset(1024),
                d.scales.offset(8 * 4),
                outs[0],
                outs[2],
                m,
                4096,
                1024,
                2048,
                0,
            )
        })?;
        let fb_bytes = 2 * sh_gu.fp8b_bytes() / 2 + sh_down.fp8b_bytes() / 2;
        line("FP8 block split (native)", m, fb, fb_bytes);
    }

    println!(
        "== dense MLP 0..2, replicated (3 layers): gate + up [12288 x 4096], down [4096 x 12288]"
    );
    for m in ROWS {
        let copies = dense_gu.nv.len().min(dense_down.nv.len());
        let nv = time(g, reps, copies, &|c| {
            nv_tc(
                g,
                tc8,
                dense_gu.nv[2 * c % copies],
                x,
                outs[0],
                m,
                12288,
                4096,
            )?;
            nv_tc(
                g,
                tc8,
                dense_gu.nv[(2 * c + 1) % copies],
                x,
                outs[1],
                m,
                12288,
                4096,
            )?;
            nv_tc(g, tc8, dense_down.nv[c], outs[0], outs[2], m, 4096, 12288)
        })?;
        line(
            "NVFP4 tc8 (production)",
            m,
            nv,
            2 * dense_gu.nv_bytes() + dense_down.nv_bytes(),
        );
        let mx = time(g, reps, copies, &|c| {
            let parts = [
                MxPart::mx(&dense_gu.mx[2 * c % copies], outs[0], 12288),
                MxPart::mx(&dense_gu.mx[(2 * c + 1) % copies], outs[1], 12288),
            ];
            ops::mx_multi_rows(g, &nk.mx_multi, false, x, &parts, m, 4096, 0)?;
            ops::mxfp8_gemv_rows(
                g,
                &nk.mx.row_major,
                &dense_down.mx[c],
                outs[0],
                outs[2],
                m,
                4096,
                12288,
                4096,
                0,
            )
        })?;
        line(
            "MXFP8 fused gate+up, down (native)",
            m,
            mx,
            2 * dense_gu.mx_bytes() + dense_down.mx_bytes(),
        );
    }

    let weights = [&kda, &q_a, &kv_a, &q_b, &mla_o, &sh_down];
    fail |= gate::run(g, nk, x, &x_host, &outs, weights)?;
    println!("{}", if fail { "gate: FAIL" } else { "gate: pass" });
    std::process::exit(i32::from(fail));
}
