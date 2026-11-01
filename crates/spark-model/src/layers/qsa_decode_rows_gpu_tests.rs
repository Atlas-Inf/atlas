// SPDX-License-Identifier: AGPL-3.0-only

//! GPU test (`#[ignore]` per repo convention): the long-context decode
//! switches change no byte, wiring included.
//!
//! Three indexers over the same weights and the same prefilled carry decode
//! the same R rows of one sequence:
//!   A  switches off — the production per-row path (`decode_select`, host
//!      top-k past 16384 blocks), gather + bs=1 BF16 paged decode attention;
//!   B  `QSA_DECODE_ROWS` — `decode_select_rows` + `attend_rows`;
//!   C  `QSA_TOPK_WIDE` + `QSA_DECODE_ROWS` single-row — `decode_select`
//!      (device radix at every width, exact tiled scorer) + the same attention.
//! Attention outputs, carries (counters + pooled keys) must be identical.
//!
//!   cargo test -p spark-model --release --lib decode_rows_equal -- --ignored --nocapture

use super::*;

const HIDDEN: usize = 2560;
const NQ: u32 = 12; // one TP2 rank
const NKV: u32 = 1;
const HD: u32 = 256;
const BS: u32 = 16;
const QSTRIDE: usize = (NQ as usize * 2 + 2 * NKV as usize) * HD as usize; // [Q|K|V|gate]

/// Reproducible BF16 bytes in about [-scale, scale].
fn bf16_bytes(n: usize, seed: u64, scale: f32) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .flat_map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let u = ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32;
            (((u * 2.0 - 1.0) * scale).to_bits() >> 16)
                .to_le_bytes()
                .to_vec()
                .into_iter()
                .take(2)
        })
        .collect()
}

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = g.alloc(bytes.len()).unwrap();
    g.copy_h2d(bytes, p).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut raw = vec![0u8; n];
    g.copy_d2h(p, &mut raw).unwrap();
    raw
}

#[test]
#[ignore]
fn decode_rows_equal_the_serial_rows() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*' or qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k_pd = g.kernel("paged_decode", "paged_decode_attn").unwrap();
    let (n_heads, ihd, ratio, budget) = (4usize, 128usize, 4usize, 2048usize);
    let qk_w = upload(g, &bf16_bytes((n_heads + 1) * ihd * HIDDEN, 1, 0.03));
    let qn_w = upload(g, &bf16_bytes(ihd, 2, 0.2));
    let kn_w = upload(g, &bf16_bytes(ihd, 3, 0.2));
    let make = |rows: bool, wide: bool| {
        let mut q = QsaIndexer::new(
            qk_w,
            qn_w,
            kn_w,
            n_heads,
            ihd,
            ratio,
            budget,
            131072,
            64,
            1.0e7,
            1e-6,
            HIDDEN,
            NKV as usize,
            HD as usize,
            g,
        )
        .unwrap();
        let sel_cap = budget + ratio + SHARE_MARGIN;
        q.rows =
            RowsPath::with_switches(rows, wide, [n_heads, ihd, sel_cap, HD as usize], g).unwrap();
        q
    };
    let (qa, qb, qc) = (make(false, false), make(true, false), make(true, true));
    // Prefill source: 16K random rows, read through shifting windows so the
    // keys vary (and repeat often enough to make exact score ties).
    let src = upload(g, &bf16_bytes(16384 * HIDDEN, 4, 1.0));
    let inv_sqrt_d = 1.0 / (HD as f32).sqrt();

    // (prefix, rows): first active at the bound; mid; past 16384 blocks.
    for (prefix, rows) in [(2051usize, 5usize), (20003, 8), (70001, 4), (70002, 16)] {
        let total = prefix + rows;
        let pages = total.div_ceil(BS as usize) + 1;
        let mut table: Vec<i32> = (0..pages as i32).collect();
        for i in (1..pages).rev() {
            table.swap(i, (i * 7919 + 13) % (i + 1));
        }
        let table_b: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tab = upload(g, &table_b);
        let kv = pages * BS as usize * (NKV * HD) as usize;
        let (kp, vp) = (
            upload(g, &bf16_bytes(kv, 5, 2.0)),
            upload(g, &bf16_bytes(kv, 6, 1.0)),
        );
        let normed = upload(g, &bf16_bytes(rows * HIDDEN, 7, 1.0));
        let q = upload(g, &bf16_bytes(rows * QSTRIDE, 8, 1.0));
        let out_bytes = rows * (NQ * HD) as usize * 2;
        let outs: Vec<DevicePtr> = (0..3).map(|_| g.alloc(out_bytes).unwrap()).collect();
        let mut sts: Vec<QsaSeqState> = (0..3).map(|_| qa.new_seq_state(g).unwrap()).collect();
        for (qsa, st) in [&qa, &qb, &qc].into_iter().zip(sts.iter_mut()) {
            let mut at = 0;
            while at < prefix {
                let n = (prefix - at).min(8192);
                let rows_off = (at / 8192 * 977) % 8192;
                qsa.prefill_ingest(st, src.offset(rows_off * HIDDEN * 2), n, at, g, stream)
                    .unwrap();
                at += n;
            }
        }
        // A and C: the serial per-row decode, gather + bs=1 attention.
        for (k, qsa) in [(0usize, &qa), (2, &qc)] {
            for i in 0..rows {
                let sel = qsa
                    .decode_select(
                        &mut sts[k],
                        normed.offset(i * HIDDEN * 2),
                        prefix + i,
                        kp,
                        vp,
                        tab,
                        BS,
                        g,
                        stream,
                    )
                    .unwrap()
                    .expect("active row");
                ops::paged_decode_attn_bf16(
                    g,
                    k_pd,
                    q.offset(i * QSTRIDE * 2),
                    sel.k_scratch,
                    sel.v_scratch,
                    outs[k].offset(i * (NQ * HD) as usize * 2),
                    sel.table_dev,
                    sel.seq_len_dev,
                    sel.max_blocks,
                    1,
                    NQ,
                    NKV,
                    HD,
                    BS,
                    inv_sqrt_d,
                    NQ * HD,
                    0,
                    stream,
                )
                .unwrap();
            }
        }
        // B: the batched rows path.
        let sel = qb
            .decode_select_rows(&mut sts[1], normed, HIDDEN * 2, prefix, rows, g, stream)
            .unwrap();
        qb.attend_rows(
            &sel,
            q,
            QSTRIDE as u32,
            kp,
            vp,
            tab,
            outs[1],
            NQ,
            NKV,
            BS,
            inv_sqrt_d,
            g,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();

        let want = download(g, outs[0], out_bytes);
        for (k, name) in [(1usize, "DECODE_ROWS"), (2, "TOPK_WIDE single-row")] {
            let got = download(g, outs[k], out_bytes);
            let diff = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            println!("prefix {prefix} rows {rows}: {name} vs serial: {diff} differing bytes");
            assert_eq!(
                diff, 0,
                "prefix {prefix} rows {rows}: {name} attention differs"
            );
            assert_eq!(
                (sts[k].ingested, sts[k].pooled),
                (sts[0].ingested, sts[0].pooled),
                "{name}: carry counters"
            );
            let keys = sts[0].pooled * ihd * 2;
            assert_eq!(
                download(g, sts[k].block_keys, keys),
                download(g, sts[0].block_keys, keys),
                "{name}: pooled keys"
            );
        }
        assert!(want.iter().any(|b| *b != 0), "attention wrote nothing");
        for (qsa, st) in [&qa, &qb, &qc].into_iter().zip(sts.iter_mut()) {
            qsa.free_seq_state(st, g).unwrap();
        }
        for p in [tab, kp, vp, normed, q].into_iter().chain(outs) {
            g.free(p).unwrap();
        }
    }
}
