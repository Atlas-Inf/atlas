// SPDX-License-Identifier: AGPL-3.0-only

//! GPU: the determinism contracts of FP4-Local's BF16 router
//! (`row_invariant_bf16_router`), at its shape (512 experts, hidden 2560,
//! top 10):
//!
//! * decode/verify rows from the `dense_rows` tiers are the single-row
//!   `dense_gemv_bf16` bytes at every row count, so a verify row routes like
//!   the serial row;
//! * the prefill router GEMM (`q38_router_gemm`) may round differently, but
//!   picks the same top-k experts wherever the k-th and (k+1)-th logits are
//!   more than twice the largest logit difference apart;
//! * the softmax top-k kernels are deterministic launch after launch; the
//!   serial-decode (single row) and verify (rows) kernels pick the same
//!   experts on every exact tie, and the prefill (batched) kernel agrees with
//!   them on ties at the k boundary.
//!
//! `#[ignore]` per repo convention. On a GB10:
//! ```text
//! ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo test -p spark-model --release --lib router_determinism_gpu -- --ignored --nocapture --test-threads=1
//! ```

use std::collections::BTreeSet;

use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::ops;
use crate::weight_map::DenseWeight;

const E: usize = 512;
const H: usize = 2560;
const TOP_K: usize = 10;

fn backend() -> AtlasCudaBackend {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend")
}

/// SplitMix64 BF16 values in about [-scale, scale), with -0 and subnormals
/// planted (no overflow bait: router logits stay finite).
fn bf16_bytes(n: usize, seed: u64, scale: f32) -> Vec<u8> {
    let mut z = seed;
    (0..n)
        .flat_map(|_| {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            let v = ((x >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale;
            let bits = match x % 509 {
                0 => 0x8000,
                1 => (x >> 20) as u16 & 0x807F,
                _ => (v.to_bits() >> 16) as u16,
            };
            bits.to_le_bytes()
        })
        .collect()
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn read(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Vec<u8> {
    g.synchronize(g.default_stream()).unwrap();
    let mut out = vec![0u8; bytes];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

fn to_f32(bf16: &[u8]) -> Vec<f32> {
    bf16.chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
        .collect()
}

/// Per-row single-row GEMV logits: the serial decode router's bytes.
fn serial_logits(g: &dyn GpuBackend, x: DevicePtr, w: &DenseWeight, rows: usize) -> Vec<u8> {
    let k_gemv = g.kernel("gemv", "dense_gemv_bf16").unwrap();
    let out = g.alloc(rows * E * 2).unwrap();
    for r in 0..rows {
        ops::dense_gemv(
            g,
            k_gemv,
            x.offset(r * H * 2),
            w,
            out.offset(r * E * 2),
            E as u32,
            H as u32,
            g.default_stream(),
        )
        .unwrap();
    }
    let bytes = read(g, out, rows * E * 2);
    g.free(out).unwrap();
    bytes
}

/// Expert ids by descending logit (ties by ascending id).
fn ranked(row: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&i, &j| row[j].total_cmp(&row[i]).then(i.cmp(&j)));
    idx
}

#[test]
#[ignore]
fn row_invariant_router_rows_equal_single_row_gemv() {
    // The served profile's switches: the wide BF16 tiers resolve only under
    // them (each is read once per process; run this module alone).
    unsafe {
        std::env::set_var("ATLAS_QWEN4EXP_BATCH_FAST", "1");
        std::env::set_var("ATLAS_QWEN4EXP_ROWS32_TILE", "1");
    }
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let wide = ops::Qwen4ExpWideRows::resolve(g, "qwen4_exp");
    let k_batchm = crate::layers::try_kernel(g, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
    assert_ne!(
        k_batchm.0, 0,
        "dense_gemv_bf16_batchm must be in the target"
    );
    println!("wide tiers: {wide:?}");
    let w = DenseWeight {
        weight: up(g, &bf16_bytes(E * H, 1, 0.05)),
    };
    let max_m = 64;
    let x = up(g, &bf16_bytes(max_m * H, 2, 2.0));
    let want = serial_logits(g, x, &w, max_m);
    let out = g.alloc(max_m * E * 2).unwrap();
    for m in (1..=32).chain([33, 64]) {
        g.memset(out, 0xAA, m * E * 2).unwrap();
        wide.dense_rows(
            g,
            k_batchm,
            x,
            &w,
            out,
            (m as u32, E as u32, H as u32, E as u32),
            stream,
        )
        .unwrap();
        let got = read(g, out, m * E * 2);
        let diff = got
            .chunks(2)
            .zip(want.chunks(2))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            diff, 0,
            "m={m}: {diff} router logits differ from the single-row GEMV"
        );
    }
    println!("ok: dense_rows m=1..=32,33,64 byte-identical to dense_gemv_bf16 per row");
}

#[test]
#[ignore]
fn prefill_router_gemm_topk_matches_decode_outside_near_ties() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let rows = 64usize;
    let k_gemm = g.kernel("gemm", "dense_gemm_bf16_pipelined").unwrap();
    let w = DenseWeight {
        weight: up(g, &bf16_bytes(E * H, 3, 0.05)),
    };
    let x = up(g, &bf16_bytes(rows * H, 4, 2.0));
    let decode = to_f32(&serial_logits(g, x, &w, rows));
    let logits = g.alloc(rows * E * 2).unwrap();
    super::forward_prefill_route_sp::q38_router_gemm(
        g,
        k_gemm,
        [x, w.weight, logits],
        [rows as u32, E as u32, H as u32],
        g.default_stream(),
    )
    .unwrap();
    let prefill = to_f32(&read(g, logits, rows * E * 2));
    let d = decode
        .iter()
        .zip(&prefill)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mut excluded = 0;
    for r in 0..rows {
        let (a, b) = (&decode[r * E..(r + 1) * E], &prefill[r * E..(r + 1) * E]);
        let ra = ranked(a);
        if a[ra[TOP_K - 1]] - a[ra[TOP_K]] <= 2.0 * d {
            excluded += 1;
            continue;
        }
        let top_a: BTreeSet<usize> = ra[..TOP_K].iter().copied().collect();
        let top_b: BTreeSet<usize> = ranked(b)[..TOP_K].iter().copied().collect();
        assert_eq!(
            top_a, top_b,
            "row {r}: prefill and decode route differently"
        );
    }
    println!(
        "prefill vs decode router logits: max |diff| d = {d:e} (bit-identical: {}); \
         near-tie rows excluded: {excluded}/{rows}",
        d == 0.0
    );
    assert!(
        excluded < rows,
        "every row was a near tie: the check proved nothing"
    );
}

#[test]
#[ignore]
fn routing_topk_is_deterministic_on_ties() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k_single = g.kernel("moe_topk", "moe_topk_softmax").unwrap();
    let k_batched = g.kernel("moe_topk", "moe_topk_softmax_batched").unwrap();
    let k_rows = crate::layers::try_kernel(g, "moe_topk", "moe_topk_softmax_rows");
    // Row 0: all 512 logits equal. Rows 1..: nine clear winners, then an
    // exact tie for the tenth place — {300, 7}, {450, 20, 333}, {511, 256}.
    let winners = [500usize, 3, 250, 77, 12, 401, 99, 180, 64];
    let ties: [&[usize]; 3] = [&[300, 7], &[450, 20, 333], &[511, 256]];
    let rows = 1 + ties.len();
    let mut host = vec![0f32; rows * E];
    for (t, tied) in ties.iter().enumerate() {
        let l = &mut host[(t + 1) * E..(t + 2) * E];
        l.fill(-1.0);
        for (i, &id) in winners.iter().enumerate() {
            l[id] = 4.0 - 0.25 * i as f32;
        }
        for &id in tied.iter() {
            l[id] = 1.0;
        }
    }
    let bytes: Vec<u8> = host
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let logits = up(g, &bytes);
    let ids = g.alloc(rows * TOP_K * 4).unwrap();
    let wts = g.alloc(rows * TOP_K * 4).unwrap();
    let run = |which: usize| -> Vec<u8> {
        g.memset(ids, 0xFF, rows * TOP_K * 4).unwrap();
        g.memset(wts, 0xFF, rows * TOP_K * 4).unwrap();
        match which {
            0 => {
                for r in 0..rows {
                    ops::moe_topk_softmax(
                        g,
                        k_single,
                        logits.offset(r * E * 2),
                        ids.offset(r * TOP_K * 4),
                        wts.offset(r * TOP_K * 4),
                        E as u32,
                        TOP_K as u32,
                        true,
                        stream,
                    )
                    .unwrap();
                }
            }
            1 => ops::moe_topk_softmax_batched(
                g,
                k_batched,
                logits,
                ids,
                wts,
                E as u32,
                TOP_K as u32,
                true,
                rows as u32,
                stream,
            )
            .unwrap(),
            _ => KernelLaunch::new(g, k_rows)
                .grid([rows as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(logits)
                .arg_ptr(ids)
                .arg_ptr(wts)
                .arg_u32(E as u32)
                .arg_u32(TOP_K as u32)
                .arg_u32(1)
                .arg_u32(E as u32)
                .launch(stream)
                .unwrap(),
        }
        let mut out = read(g, ids, rows * TOP_K * 4);
        out.extend(read(g, wts, rows * TOP_K * 4));
        out
    };
    let id_sets = |out: &[u8]| -> Vec<BTreeSet<u32>> {
        (0..rows)
            .map(|r| {
                out[r * TOP_K * 4..(r + 1) * TOP_K * 4]
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            })
            .collect()
    };
    // Each kernel is deterministic launch over launch, ids and weights.
    let mut picks = Vec::new();
    for (which, name) in [(0, "single"), (1, "batched"), (2, "rows")] {
        if which == 2 && k_rows.0 == 0 {
            continue;
        }
        let first = run(which);
        for launch in 1..3 {
            assert_eq!(
                run(which),
                first,
                "{name}: launch {launch} differs from launch 0"
            );
        }
        let sets = id_sets(&first);
        println!("{name}: {sets:?}");
        picks.push((name, sets));
    }
    let single = &picks[0].1;
    for (name, sets) in &picks[1..] {
        match *name {
            // Serial decode (single) and multi-row verify (rows) must agree on
            // every row, the fully degenerate one included.
            "rows" => assert_eq!(sets, single, "rows picks other experts than single"),
            // Prefill (batched) must agree on boundary ties; a row of 512
            // equal logits is reported, not required.
            _ => {
                assert_eq!(
                    &sets[1..],
                    &single[1..],
                    "{name} breaks a boundary tie differently"
                );
                println!(
                    "{name} on 512 equal logits matches single: {}",
                    sets[0] == single[0]
                );
            }
        }
    }
    for (t, tied) in ties.iter().enumerate() {
        let won: Vec<usize> = tied
            .iter()
            .copied()
            .filter(|&id| single[t + 1].contains(&(id as u32)))
            .collect();
        println!("tie {tied:?} -> {won:?}");
        assert_eq!(
            won.len(),
            1,
            "exactly one tied expert takes the tenth place"
        );
    }
}
