// SPDX-License-Identifier: AGPL-3.0-only

//! GPU parity for the QSA indexer against the reference module.
//!
//! Fixtures come from `bench/qwen4_exp/qsa_golden.py`, which runs the REAL
//! `Qwen4ExpTextQSAIndexer` on REAL layer-3 checkpoint weights over T=2200
//! tokens (550 blocks > block_topk=512 — selection actively prunes).
//!
//! The golden ran in FP32; the engine stores raw/block keys in BF16 (exactly
//! what real BF16 inference does), so scores near the 512-block cutoff can
//! legitimately flip. The selection-set compare therefore allows a small
//! number of near-tie block swaps and asserts each is within tolerance of
//! the cutoff score — everything else is exact.
//!
//! GPU test: `#[ignore]` per repo convention. Run with
//! ```text
//! ATLAS_QSA_TEST_DATA=/tank/atlas-testdata/qwen4exp_qsa/qsa_golden_bins \
//!   cargo test -p spark-model --release qsa_matches -- --ignored --nocapture
//! ```

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;

fn bins_dir() -> String {
    std::env::var("ATLAS_QSA_TEST_DATA").expect(
        "set ATLAS_QSA_TEST_DATA — generate with \
         `bench/qwen4_exp/qsa_golden.py --out .../qsa_golden.npz`",
    )
}

fn bin(name: &str) -> Vec<u8> {
    let p = format!("{}/{name}.bin", bins_dir());
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn f32s(name: &str) -> Vec<f32> {
    bin(name)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i32s(name: &str) -> Vec<i32> {
    bin(name)
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d_async(bytes, p, g.default_stream()).unwrap();
    p
}

fn dl_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn dl_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn compare(label: &str, got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let rms =
        (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt() as f32;
    let tol = (rms * rel).max(1e-3);
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let (mut ng, mut nw) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(want) {
        max_abs = max_abs.max((a - b).abs());
        dot += a as f64 * b as f64;
        ng += a as f64 * a as f64;
        nw += b as f64 * b as f64;
    }
    let cos = dot / (ng.sqrt() * nw.sqrt()).max(1e-30);
    println!("  {label:<12} max|diff|={max_abs:.4e} cos={cos:.9} ref_rms={rms:.4e}");
    assert!(
        max_abs <= tol,
        "{label}: max|diff| {max_abs:.4e} > {tol:.4e}"
    );
    assert!(cos > 0.9999, "{label}: cosine {cos:.9}");
}

#[test]
#[ignore]
fn qsa_matches_reference() {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/meta.json", bins_dir())).unwrap(),
    )
    .unwrap();
    assert_eq!(
        meta["norm_convention"].as_str().unwrap(),
        "normed * (1.0 + weight)"
    );
    let t_all = meta["num_tokens"].as_u64().unwrap() as usize;
    let n_heads = meta["index_n_heads"].as_u64().unwrap() as usize;
    let hd = meta["index_head_dim"].as_u64().unwrap() as usize;
    let ratio = meta["compress_ratio"].as_u64().unwrap() as usize;
    let budget = meta["token_budget"].as_u64().unwrap() as usize;
    let topk = meta["block_topk"].as_u64().unwrap() as usize;
    let hidden = meta["hidden_size"].as_u64().unwrap() as usize;
    let rot = meta["rotary_dim"].as_u64().unwrap() as usize;
    let eps = meta["rms_norm_eps"].as_f64().unwrap() as f32;
    // rope_theta comes from the model config (1e7 on this checkpoint); the
    // golden's cos/sin were generated under it — a mismatch fails the
    // q_post/block_keys compares immediately, so it is self-checking.
    let theta = 1.0e7f32;

    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let qsa = QsaIndexer::new(
        upload(g, &bin("w_qk_proj")),
        upload(g, &bin("w_q_norm")),
        upload(g, &bin("w_k_norm")),
        n_heads,
        hd,
        ratio,
        budget,
        rot,
        theta,
        eps,
        hidden,
        /* nkv_attn */ 2,
        /* hd_attn */ 256,
        g,
    )
    .unwrap();

    let hidden_dev = upload(g, &bin("hidden"));

    // ── Prefill ingest of T-1 tokens (the last token arrives at decode). ──
    let t_pre = t_all - 1;
    qsa.prefill_ingest(hidden_dev, t_pre, 0, g, stream).unwrap();
    g.synchronize(stream).unwrap();

    let want_raw = f32s("raw_keys");
    compare(
        "raw_keys",
        &dl_bf16(g, qsa.raw_keys, t_pre * hd),
        &want_raw[..t_pre * hd],
        0.05,
    );
    let n_blocks_pre = t_pre / ratio;
    let want_bk = f32s("block_keys");
    compare(
        "block_keys",
        &dl_bf16(g, qsa.block_keys, n_blocks_pre * hd),
        &want_bk[..n_blocks_pre * hd],
        0.05,
    );

    // ── Decode step for the LAST token: ingest + select + gather. ──
    // Dummy paged pools sized for T positions (gather source; contents
    // uninspected — the layout math is exercised, values are checked e2e).
    let bs = 16usize;
    let pages = t_all.div_ceil(bs);
    let row = 2 * 256;
    let kpool = g.alloc(pages * bs * row * 2).unwrap();
    let vpool = g.alloc(pages * bs * row * 2).unwrap();
    let ident: Vec<u8> = (0..pages as i32).flat_map(|v| v.to_le_bytes()).collect();
    let table = upload(g, &ident);

    let last_row = hidden_dev.offset((t_all - 1) * hidden * 2);
    let sel = qsa
        .decode_select(
            last_row,
            t_all - 1,
            kpool,
            vpool,
            table,
            bs as u32,
            g,
            stream,
        )
        .unwrap()
        .expect("selection must be ACTIVE at T=2200");
    g.synchronize(stream).unwrap();

    // q_post for the last token.
    let want_q: Vec<f32> = f32s("q_post")[(t_all - 1) * n_heads * hd..].to_vec();
    compare(
        "q_post",
        &dl_f32(g, qsa.q_post, n_heads * hd),
        &want_q,
        0.05,
    );

    // Block scores for the last query.
    let n_blocks = t_all / ratio;
    let want_scores = f32s("scores_last");
    let got_scores = dl_f32(g, qsa.scores_dev, n_blocks);
    compare("scores", &got_scores, &want_scores, 0.05);

    // Selection SET: exact up to near-tie flips at the 512th-block cutoff
    // (BF16 key storage vs the FP32 golden). Each mismatched block's score
    // must sit within tolerance of the cutoff.
    let want_sel = i32s("selected_last");
    assert_eq!(sel.n_sel as usize, want_sel.len(), "selected count");
    let got_sel = {
        let mut b = vec![0u8; sel.n_sel as usize * 4];
        g.copy_d2h(qsa.sel_dev, &mut b).unwrap();
        b.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<i32>>()
    };
    let got_set: std::collections::BTreeSet<i32> = got_sel.iter().copied().collect();
    let want_set: std::collections::BTreeSet<i32> = want_sel.iter().copied().collect();
    let missing: Vec<i32> = want_set.difference(&got_set).copied().collect();
    let extra: Vec<i32> = got_set.difference(&want_set).copied().collect();
    let flipped_blocks = missing.len().div_ceil(ratio);
    println!(
        "  selection    {} tokens, {} missing / {} extra (<= {} near-tie \
         block flips allowed)",
        want_sel.len(),
        missing.len(),
        extra.len(),
        3
    );
    assert_eq!(missing.len(), extra.len(), "selection: count drift");
    assert!(
        flipped_blocks <= 3,
        "selection: {flipped_blocks} block flips — more than near-tie noise \
         (missing {missing:?})"
    );
    if !missing.is_empty() {
        // Every flip must be a near-tie: |score - cutoff| small.
        let mut sorted: Vec<f32> = want_scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let cutoff = sorted[topk - 1];
        for m in missing.iter().chain(extra.iter()) {
            let b = (*m as usize) / ratio;
            let d = (want_scores[b] - cutoff).abs();
            assert!(
                d < 0.02 * cutoff.abs().max(1.0),
                "selection: block {b} flipped with score {:.5} vs cutoff \
                 {cutoff:.5} — not a near-tie",
                want_scores[b]
            );
        }
    }
    println!("qsa parity OK: n_sel={} (budget {})", sel.n_sel, budget);
}

/// Stage 2A: prefill selection SETS vs the golden mask rows. Ingest all T
/// tokens, run `prefill_select_chunk0` (attention numerics unchecked here —
/// q is zeros, pools are dummies; test B covers them), read back the
/// uploaded block lists and compare each selective row's set against the
/// reference `selected_mask` row.
#[test]
#[ignore]
fn qsa_prefill_select_sets_match_reference() {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/meta.json", bins_dir())).unwrap(),
    )
    .unwrap();
    let t_all = meta["num_tokens"].as_u64().unwrap() as usize;
    let n_heads = meta["index_n_heads"].as_u64().unwrap() as usize;
    let hd = meta["index_head_dim"].as_u64().unwrap() as usize;
    let ratio = meta["compress_ratio"].as_u64().unwrap() as usize;
    let budget = meta["token_budget"].as_u64().unwrap() as usize;
    let topk = meta["block_topk"].as_u64().unwrap() as usize;
    let hidden = meta["hidden_size"].as_u64().unwrap() as usize;
    let rot = meta["rotary_dim"].as_u64().unwrap() as usize;
    let eps = meta["rms_norm_eps"].as_f64().unwrap() as f32;

    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let (nq, nkv, hd_attn, bs) = (24u32, 2usize, 256usize, 16usize);
    let qsa = QsaIndexer::new(
        upload(g, &bin("w_qk_proj")),
        upload(g, &bin("w_q_norm")),
        upload(g, &bin("w_k_norm")),
        n_heads,
        hd,
        ratio,
        budget,
        rot,
        1.0e7,
        eps,
        hidden,
        nkv,
        hd_attn,
        g,
    )
    .unwrap();
    let hidden_dev = upload(g, &bin("hidden"));
    qsa.prefill_ingest(hidden_dev, t_all, 0, g, stream).unwrap();

    let q_row = nq as usize * hd_attn;
    let q_roped = g.alloc(t_all * q_row * 2).unwrap();
    let attn_ctx = g.alloc(t_all * q_row * 2).unwrap();
    let pages = t_all.div_ceil(bs);
    let kpool = g.alloc(pages * bs * nkv * hd_attn * 2).unwrap();
    let vpool = g.alloc(pages * bs * nkv * hd_attn * 2).unwrap();
    let host_table: Vec<u32> = (0..pages as u32).collect();

    const ROWS: usize = 2048;
    let qkw = (n_heads + 1) * hd;
    let stride = t_all.div_ceil(ratio);
    let scratch = g
        .alloc(ROWS * qkw * 2 + ROWS * n_heads * hd * 4 + ROWS * stride * 4 + ROWS * topk * 4)
        .unwrap();

    qsa.prefill_select_chunk0(
        hidden_dev,
        q_roped,
        attn_ctx,
        kpool,
        vpool,
        &host_table,
        t_all,
        nq,
        bs as u32,
        1.0 / (hd_attn as f32).sqrt(),
        scratch,
        g,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();

    let bound = qsa.inert_bound();
    let n_sel = t_all - bound;
    assert!(n_sel <= ROWS, "test assumes one slab");
    let lists_off = ROWS * qkw * 2 + ROWS * n_heads * hd * 4 + ROWS * stride * 4;
    let mut raw = vec![0u8; n_sel * topk * 4];
    g.copy_d2h(scratch.offset(lists_off), &mut raw).unwrap();
    let lists: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // The engine's own scores for the (single) slab, for flip adjudication.
    let scores_off = ROWS * qkw * 2 + ROWS * n_heads * hd * 4;
    let mut sraw = vec![0u8; n_sel * stride * 4];
    g.copy_d2h(scratch.offset(scores_off), &mut sraw).unwrap();
    let eng_scores: Vec<f32> = sraw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Golden: [n_sel, T] u8 mask rows for pos in [bound, T).
    let mask = bin("mask_sel_rows");
    assert_eq!(mask.len(), n_sel * t_all, "mask_sel_rows shape");
    let mut rows_exact = 0usize;
    let mut total_flips = 0usize;
    for r in 0..n_sel {
        let pos = bound + r;
        let complete = (pos + 1) / ratio;
        let want: std::collections::BTreeSet<i32> = (0..complete * ratio)
            .filter(|&t| mask[r * t_all + t] != 0)
            .map(|t| (t / ratio) as i32)
            .collect();
        for t in complete * ratio..=pos {
            assert_eq!(
                mask[r * t_all + t],
                1,
                "row {pos}: tail token {t} not visible"
            );
        }
        let got: std::collections::BTreeSet<i32> =
            lists[r * topk..(r + 1) * topk].iter().copied().collect();
        assert_eq!(want.len(), topk, "row {pos}: golden block count");
        assert_eq!(got.len(), topk, "row {pos}: duplicate blocks in list");
        let flips = want.difference(&got).count();
        total_flips += flips;
        if flips == 0 {
            rows_exact += 1;
        }
        // BF16 key storage jitters scores ~1.3e-2 vs the FP32 golden
        // (measured cos 0.9999976 on the decode path), so blocks NEAR the
        // 512th-place cutoff can legitimately swap. Adjudicate every flip
        // against the ENGINE's own score ordering: a golden-selected block
        // the engine dropped must sit within jitter distance BELOW the
        // engine's cutoff — a far-from-cutoff flip is a real defect no
        // matter how rare.
        if flips > 0 {
            let row_sc = &eng_scores[r * stride..r * stride + complete];
            let mut sorted: Vec<f32> = row_sc.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let cutoff = sorted[topk - 1];
            for b in want.difference(&got) {
                let d = cutoff - row_sc[*b as usize];
                assert!(
                    (0.0..0.05).contains(&d),
                    "row {pos}: golden block {b} dropped with engine score \
                     {:.5} vs cutoff {cutoff:.5} (gap {d:.5}) — not a \
                     near-tie",
                    row_sc[*b as usize]
                );
            }
        }
    }
    println!(
        "qsa prefill sets: {rows_exact}/{n_sel} rows EXACT, {total_flips} total near-tie flips"
    );
}

/// Stage 2B: the qsa_prefill_attn kernel vs a CPU reference on synthetic
/// data — validates the online softmax, GQA head mapping, paged addressing
/// and tail handling at small topk.
#[test]
#[ignore]
fn qsa_prefill_attn_matches_cpu() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g.kernel("qsa_indexer", "qsa_prefill_attn").unwrap();

    let (rows, nq, nkv, hd, ratio, topk, bs) =
        (3usize, 24usize, 2usize, 256usize, 4usize, 8usize, 16usize);
    let first_pos = 41usize; // complete = 10 > topk = 8; tail = 2 at row 0
    let n_pos = first_pos + rows;
    let pages = n_pos.div_ceil(bs);

    let mut seed = 0x12345u32;
    let mut nextf = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 8) as f32 / (1 << 24) as f32) - 0.5
    };
    let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
    let unbf = |u: u16| -> f32 { f32::from_bits((u as u32) << 16) };

    let q_host: Vec<u16> = (0..rows * nq * hd).map(|_| bf(nextf())).collect();
    let kv_elems = pages * bs * nkv * hd;
    let k_host: Vec<u16> = (0..kv_elems).map(|_| bf(nextf())).collect();
    let v_host: Vec<u16> = (0..kv_elems).map(|_| bf(nextf())).collect();
    let lists_host: Vec<i32> = (0..rows)
        .flat_map(|r| (0..topk as i32).map(move |i| (i * 5 + r as i32) % 10))
        .collect();

    let as_bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let q_dev = upload(g, &as_bytes(&q_host));
    let k_dev = upload(g, &as_bytes(&k_host));
    let v_dev = upload(g, &as_bytes(&v_host));
    let lists_dev = upload(
        g,
        &lists_host
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let ident: Vec<u8> = (0..pages as i32).flat_map(|v| v.to_le_bytes()).collect();
    let table = upload(g, &ident);
    let out_dev = g.alloc(rows * nq * hd * 2).unwrap();
    let scale = 1.0 / (hd as f32).sqrt();

    ops::qsa_prefill_attn(
        g,
        k,
        q_dev,
        k_dev,
        v_dev,
        table,
        lists_dev,
        out_dev,
        rows as u32,
        first_pos as u32,
        topk as u32,
        ratio as u32,
        bs as u32,
        nq as u32,
        nkv as u32,
        hd as u32,
        scale,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let got = dl_bf16(g, out_dev, rows * nq * hd);

    let group = nq / nkv;
    let mut worst_cos = 1.0f64;
    for r in 0..rows {
        let pos = first_pos + r;
        let complete = (pos + 1) / ratio;
        let tail = (pos + 1) - complete * ratio;
        let mut toks: Vec<usize> = lists_host[r * topk..(r + 1) * topk]
            .iter()
            .flat_map(|&b| (0..ratio).map(move |i| b as usize * ratio + i))
            .collect();
        toks.extend(complete * ratio..complete * ratio + tail);
        for h in 0..nq {
            let kvh = h / group;
            let qv: Vec<f32> = (0..hd)
                .map(|d| unbf(q_host[(r * nq + h) * hd + d]))
                .collect();
            let scores: Vec<f32> = toks
                .iter()
                .map(|&t| {
                    let base = (t * nkv + kvh) * hd;
                    (0..hd).map(|d| qv[d] * unbf(k_host[base + d])).sum::<f32>() * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let l: f32 = exps.iter().sum();
            let mut refv = vec![0.0f32; hd];
            for (i, &t) in toks.iter().enumerate() {
                let base = (t * nkv + kvh) * hd;
                let w = exps[i] / l;
                for d in 0..hd {
                    refv[d] += w * unbf(v_host[base + d]);
                }
            }
            let gv = &got[(r * nq + h) * hd..(r * nq + h + 1) * hd];
            let dot: f64 = gv
                .iter()
                .zip(&refv)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let ng: f64 = gv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let nr: f64 = refv.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (ng * nr).max(1e-30);
            worst_cos = worst_cos.min(cos);
        }
    }
    println!("qsa_prefill_attn vs CPU: worst cos = {worst_cos:.9}");
    assert!(worst_cos > 0.999, "attention kernel diverges: {worst_cos}");
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
