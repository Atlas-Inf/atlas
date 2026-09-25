// SPDX-License-Identifier: AGPL-3.0-only

//! QSA prefill-side GPU tests, second half: device top-k, scorer drift and
//! bitwise checks. Split from `qsa_tests_prefill.rs` (≤500 LoC cap); same
//! module position, so `super` is `qsa_tests` and `super::super` is `qsa`.

#![allow(unused_imports)]

use super::*;

/// Stage 1B: `qsa_topk_rows` must produce exactly the list the host selection
/// produced — same blocks, SAME ORDER.
///
/// Order is not cosmetic here: `qsa_prefill_attn` walks the list warp-striped
/// and its online softmax accumulates in list order, so a permutation
/// reassociates the sum. The reference below is the original comparator
/// (`partial_cmp(b, a).then(a.cmp(&b))`) rather than the packed key, so this
/// tests the SPEC and would catch a bug in the packing as well as in the
/// kernel.
///
/// Geometry is chosen to exercise what a single-chunk test would not:
/// `complete` ~= 1250 spans three of the kernel's 512-wide chunks, so the
/// running-best merge runs twice; exact ties force the index tie-break; and
/// both +0.0 and -0.0 appear, which is the one value where the bit map and
/// IEEE comparison disagree unless zero is canonicalised.
#[test]
#[ignore]
fn qsa_topk_rows_matches_host_selection() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g.kernel("qsa_indexer", "qsa_topk_rows").unwrap();

    let (rows, ratio, topk, stride) = (7usize, 4usize, 512usize, 1300usize);
    let first_pos = 5000usize;
    assert!(ops::qsa_topk_rows_ok(topk as u32));

    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut sc = vec![-1e30f32; rows * stride];
    for r in 0..rows {
        let complete = (first_pos + r + 1) / ratio;
        assert!(complete > 2 * 512 && complete <= stride, "want a multi-chunk row");
        for b in 0..complete {
            let v = ((next() >> 40) as f32 / 512.0) - 16.0;
            sc[r * stride + b] = match b % 37 {
                0 => 0.0,               // exact ties on zero, across many blocks
                1 => -0.0,              // the one IEEE-vs-bitmap disagreement
                2 => 3.5,               // exact ties on a normal value
                _ => v,
            };
        }
    }

    let sc_bytes: Vec<u8> = sc.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sc_dev = upload(g, &sc_bytes);
    let lists_dev = g.alloc(rows * topk * 4).unwrap();

    ops::qsa_topk_rows(
        g,
        k,
        sc_dev,
        lists_dev,
        rows as u32,
        first_pos as u32,
        stride as u32,
        ratio as u32,
        topk as u32,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let mut raw = vec![0u8; rows * topk * 4];
    g.copy_d2h(lists_dev, &mut raw).unwrap();
    let got: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for r in 0..rows {
        let complete = (first_pos + r + 1) / ratio;
        let row = &sc[r * stride..r * stride + complete];
        let mut order: Vec<u32> = (0..complete as u32).collect();
        order.sort_by(|&a, &b| {
            row[b as usize]
                .partial_cmp(&row[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let want: Vec<i32> = order[..topk].iter().map(|&v| v as i32).collect();
        let mine = &got[r * topk..(r + 1) * topk];
        assert_eq!(
            want, mine,
            "row {r} (complete={complete}) list differs from the host selection"
        );
    }
    println!("qsa_topk_rows == host selection on {rows} rows x {topk}");
}

/// How far `qsa_score_rows_gemm` moves the scores away from `qsa_score_rows`.
///
/// This is a DISCRIMINATOR, not a gate. The GEMM scorer contracts `d` serially
/// per thread instead of through a block-wide tree, so the scores must differ —
/// the question is by how much. A relative difference at the 1e-6 level is
/// ordinary FP32 reassociation over 128 terms, and any downstream behaviour
/// change is the top-k amplifying it (near-tied blocks either side of the 512th
/// place). Anything larger is a bug in the kernel, and the two cases call for
/// completely different responses.
#[test]
#[ignore]
fn qsa_score_rows_gemm_vs_reference_drift() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k_ref = g.kernel("qsa_indexer", "qsa_score_rows").unwrap();
    let k_gemm = g.kernel("qsa_indexer", "qsa_score_rows_gemm").unwrap();

    let (rows, n_heads, hd, ratio) = (37usize, 4usize, 128usize, 4usize);
    let first_pos = 5000usize;
    let n_blocks_max = (first_pos + rows) / ratio;
    let stride = n_blocks_max + 8;

    let mut s = 0x9E3779B97F4A7C15u64;
    let mut nextf = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / 1024.0) - 12.0
    };
    let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };

    let q_host: Vec<f32> = (0..rows * n_heads * hd).map(|_| nextf()).collect();
    let k_host: Vec<u16> = (0..n_blocks_max * hd).map(|_| bf(nextf())).collect();

    let q_dev = upload(g, &q_host.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let k_dev = upload(g, &k_host.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let a_dev = g.alloc(rows * stride * 4).unwrap();
    let b_dev = g.alloc(rows * stride * 4).unwrap();

    ops::qsa_score_rows(
        g, k_ref, q_dev, k_dev, a_dev, rows as u32, n_blocks_max as u32,
        first_pos as u32, stride as u32, ratio as u32, n_heads as u32, hd as u32, stream,
    )
    .unwrap();
    ops::qsa_score_rows_gemm(
        g, k_gemm, q_dev, k_dev, b_dev, rows as u32, n_blocks_max as u32,
        first_pos as u32, stride as u32, ratio as u32, n_heads as u32, hd as u32, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let a = dl_f32(g, a_dev, rows * stride);
    let b = dl_f32(g, b_dev, rows * stride);

    let mut worst_rel = 0.0f64;
    let mut worst_abs = 0.0f64;
    let mut n_cmp = 0usize;
    let mut exact = 0usize;
    for r in 0..rows {
        let complete = (first_pos + r + 1) / ratio;
        for bb in 0..complete {
            let (x, y) = (a[r * stride + bb] as f64, b[r * stride + bb] as f64);
            n_cmp += 1;
            if x.to_bits() == y.to_bits() {
                exact += 1;
            }
            let d = (x - y).abs();
            worst_abs = worst_abs.max(d);
            if x.abs() > 1e-6 {
                worst_rel = worst_rel.max(d / x.abs());
            }
        }
    }
    println!(
        "qsa_score_rows_gemm vs reference: {n_cmp} scores, {exact} bit-exact, \
         worst abs {worst_abs:.3e}, worst rel {worst_rel:.3e}"
    );
    // 1e-4 is far above FP32 reassociation over 128 terms and far below a
    // wrong-layout bug, so it separates the two cases cleanly.
    assert!(
        worst_rel < 1e-4,
        "score drift {worst_rel:.3e} is too large to be reassociation — kernel bug"
    );
}

/// `qsa_score_rows_exact` must be BIT-IDENTICAL to `qsa_score_rows` — every
/// score, every bit.
///
/// The point of that kernel is that bit-identity does not require a 128-thread
/// block-wide reduction, only the same FP addition DAG: it replays the
/// reference's `__shfl_down_sync` tree inside one thread. If nvcc contracts a
/// `q*k + t` into an FMA anywhere in that tree, or the warp-partial fold order
/// drifts, this fails — which is exactly what it is for. A cosine or
/// small-epsilon check would not catch either.
#[test]
#[ignore]
fn qsa_score_rows_exact_is_bitwise_identical() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k_ref = g.kernel("qsa_indexer", "qsa_score_rows").unwrap();
    let k_exact = g.kernel("qsa_indexer", "qsa_score_rows_exact").unwrap();

    let (rows, n_heads, hd, ratio) = (37usize, 4usize, 128usize, 4usize);
    let first_pos = 5000usize;
    let n_blocks_max = (first_pos + rows) / ratio;
    let stride = n_blocks_max + 8;
    assert!(ops::qsa_score_rows_exact_ok(n_heads as u32, hd as u32));

    let mut s = 0x9E3779B97F4A7C15u64;
    let mut nextf = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / 1024.0) - 12.0
    };
    let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };

    let q_host: Vec<f32> = (0..rows * n_heads * hd).map(|_| nextf()).collect();
    let k_host: Vec<u16> = (0..n_blocks_max * hd).map(|_| bf(nextf())).collect();

    let q_dev = upload(g, &q_host.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let k_dev = upload(g, &k_host.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let a_dev = g.alloc(rows * stride * 4).unwrap();
    let b_dev = g.alloc(rows * stride * 4).unwrap();

    ops::qsa_score_rows(
        g, k_ref, q_dev, k_dev, a_dev, rows as u32, n_blocks_max as u32,
        first_pos as u32, stride as u32, ratio as u32, n_heads as u32, hd as u32, stream,
    )
    .unwrap();
    ops::qsa_score_rows_exact(
        g, k_exact, q_dev, k_dev, b_dev, rows as u32, n_blocks_max as u32,
        first_pos as u32, stride as u32, ratio as u32, n_heads as u32, hd as u32, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let a = dl_f32(g, a_dev, rows * stride);
    let b = dl_f32(g, b_dev, rows * stride);

    let mut diffs = 0usize;
    let mut n_cmp = 0usize;
    let mut first: Option<(usize, usize, f32, f32)> = None;
    for r in 0..rows {
        let complete = (first_pos + r + 1) / ratio;
        for bb in 0..complete {
            let (x, y) = (a[r * stride + bb], b[r * stride + bb]);
            n_cmp += 1;
            if x.to_bits() != y.to_bits() {
                diffs += 1;
                if first.is_none() {
                    first = Some((r, bb, x, y));
                }
            }
        }
    }
    assert_eq!(
        diffs, 0,
        "exact-tree scorer is not bit-identical: {diffs}/{n_cmp} differ, first {first:?}"
    );
    println!("qsa_score_rows_exact == qsa_score_rows on {n_cmp} scores, bit for bit");
}

/// Minimal repro for the dense chunk-0 flash zeroing rows past ~1280 at
/// qwen4_exp geometry (nq=24, nkv=2, hd=256, causal, seq 2809). Synthetic
/// q/k/v, CPU reference at probe rows. If this passes, the corruption is in
/// the K/V staging upstream of the kernel, not the kernel.
#[test]
#[ignore]
fn flash64_long_seq_rows_repro() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g
        .kernel("inferspark_prefill", "inferspark_prefill_64")
        .unwrap();

    let (n, nq, nkv, hd) = (2809usize, 24usize, 2usize, 256usize);
    let mut seed = 0xBEEFu32;
    let mut nextf = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
    };
    let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
    let unbf = |u: u16| -> f32 { f32::from_bits((u as u32) << 16) };
    let q_host: Vec<u16> = (0..n * nq * hd).map(|_| bf(nextf())).collect();
    let k_host: Vec<u16> = (0..n * nkv * hd).map(|_| bf(nextf())).collect();
    let v_host: Vec<u16> = (0..n * nkv * hd).map(|_| bf(nextf())).collect();
    let as_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let q_dev = upload(g, &as_bytes(&q_host));
    let k_dev = upload(g, &as_bytes(&k_host));
    let v_dev = upload(g, &as_bytes(&v_host));
    let out_dev = g.alloc(n * nq * hd * 2).unwrap();
    // Poison the output so unwritten rows are detectable.
    let poison = vec![0x3Fu8; n * nq * hd * 2];
    g.copy_h2d_async(&poison, out_dev, stream).unwrap();
    let scale = 1.0 / (hd as f32).sqrt();

    ops::prefill_attention_64(
        g, k, q_dev, k_dev, v_dev, out_dev, n as u32, 1, nq as u32, nkv as u32, hd as u32, scale,
        true, 0, stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let got = dl_bf16(g, out_dev, n * nq * hd);

    let group = nq / nkv;
    for &row in &[100usize, 1024, 1200, 1279, 1280, 1290, 1500, 2051, 2808] {
        // CPU reference for head 0 only (cheap).
        let h = 0usize;
        let kvh = h / group;
        let qv: Vec<f32> = (0..hd)
            .map(|d| unbf(q_host[(row * nq + h) * hd + d]))
            .collect();
        let mut m = f32::MIN;
        let scores: Vec<f32> = (0..=row)
            .map(|t| {
                let base = (t * nkv + kvh) * hd;
                let s: f32 = (0..hd).map(|d| qv[d] * unbf(k_host[base + d])).sum::<f32>() * scale;
                m = m.max(s);
                s
            })
            .collect();
        let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let l: f32 = exps.iter().sum();
        let mut refv = vec![0.0f32; hd];
        for (t, e) in exps.iter().enumerate() {
            let base = (t * nkv + kvh) * hd;
            let w = e / l;
            for d in 0..hd {
                refv[d] += w * unbf(v_host[base + d]);
            }
        }
        let gv = &got[(row * nq + h) * hd..(row * nq + h) * hd + hd];
        let dot: f64 = gv
            .iter()
            .zip(&refv)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let ng: f64 = gv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let nr: f64 = refv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (ng * nr).max(1e-30);
        println!("  flash64 row {row:>4}: cos={cos:.6} |got|={ng:.4} |ref|={nr:.4}");
    }
}
