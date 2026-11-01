// SPDX-License-Identifier: AGPL-3.0-only

//! `qsa_prefill_attn_gp` (ATLAS_QWEN4EXP_PREFILL_QSA_GP) against
//! `qsa_prefill_attn_g`, through the production launchers. Same module
//! position as `qsa_tests_prefill.rs`: `super` is `qsa_tests`.

#![allow(unused_imports)]

use super::*;

/// `qsa_prefill_attn_gp` must be BYTE-IDENTICAL to `qsa_prefill_attn_g` --
/// it reschedules `_g`'s loads, shuffles and softmax bookkeeping, and claims
/// every operation and its order unchanged. Real selection geometry (topk 512
/// blocks of 4 tokens, so the slot table and ring are sized as in serving),
/// a shuffled block table, every tail length (rows 0..3 of four positions),
/// the TP2 rank shape (12 q / 1 kv) and TP1's (24 q / 2 kv, kv head 1 read
/// through its offset).
#[test]
#[ignore]
fn qsa_prefill_attn_gp_matches_g_bitwise() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL='*'");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let kg = g.kernel("qsa_indexer", "qsa_prefill_attn_g").unwrap();

    let mut seed = 0x2545f491u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed >> 8
    };
    let (hd, ratio, topk, bs) = (256usize, 4usize, 512usize, 16usize);
    for (nq, nkv, first_pos, rows) in [(12usize, 1usize, 9001usize, 67usize), (24, 2, 2050, 33)] {
        let pages = (first_pos + rows).div_ceil(bs);
        let mut table: Vec<i32> = (0..pages as i32).collect();
        for i in (1..pages).rev() {
            table.swap(i, next() as usize % (i + 1));
        }
        let mut f = || ((next() as f32 / (1 << 24) as f32) - 0.5) * 2.0;
        let bf = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
        let q: Vec<u16> = (0..rows * nq * hd).map(|_| bf(f())).collect();
        let kv = pages * bs * nkv * hd;
        let k: Vec<u16> = (0..kv).map(|_| bf(f())).collect();
        let v: Vec<u16> = (0..kv).map(|_| bf(f())).collect();
        let lists: Vec<i32> = (0..rows)
            .flat_map(|r| {
                let complete = (first_pos + r + 1) / ratio;
                (0..topk).map(move |i| ((i * 7 + r * 13) % complete) as i32)
            })
            .collect();
        let b16 = |x: &[u16]| -> Vec<u8> { x.iter().flat_map(|e| e.to_le_bytes()).collect() };
        let b32 = |x: &[i32]| -> Vec<u8> { x.iter().flat_map(|e| e.to_le_bytes()).collect() };
        let (q_dev, k_dev, v_dev) = (
            upload(g, &b16(&q)),
            upload(g, &b16(&k)),
            upload(g, &b16(&v)),
        );
        let (table_dev, lists_dev) = (upload(g, &b32(&table)), upload(g, &b32(&lists)));
        let out_g = g.alloc(rows * nq * hd * 2).unwrap();
        let out_gp = g.alloc(rows * nq * hd * 2).unwrap();
        let inv_sqrt_d = 1.0 / (hd as f32).sqrt();
        ops::qsa_prefill_attn_g(
            g,
            kg,
            q_dev,
            k_dev,
            v_dev,
            table_dev,
            lists_dev,
            out_g,
            rows as u32,
            first_pos as u32,
            topk as u32,
            ratio as u32,
            bs as u32,
            nq as u32,
            nkv as u32,
            hd as u32,
            inv_sqrt_d,
            stream,
        )
        .unwrap();
        let slab = ops::qwen4exp_prefill::QsaAttnSlab {
            q: q_dev,
            k_cache: k_dev,
            v_cache: v_dev,
            block_table: table_dev,
            lists: lists_dev,
            attn_out: out_gp,
            rows: rows as u32,
            first_pos: first_pos as u32,
            topk: topk as u32,
            ratio: ratio as u32,
            block_size: bs as u32,
            nq: nq as u32,
            nkv: nkv as u32,
            hd: hd as u32,
            inv_sqrt_d,
        };
        assert!(
            ops::qwen4exp_prefill::launch_qsa_prefill_attn_gp(g, &slab, stream).unwrap(),
            "qsa_prefill_attn_gp must serve nq={nq} nkv={nkv} or the test proves nothing"
        );
        g.synchronize(stream).unwrap();
        let a = dl_bf16(g, out_g, rows * nq * hd);
        let b = dl_bf16(g, out_gp, rows * nq * hd);
        let diffs = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            diffs,
            0,
            "nq={nq} nkv={nkv}: {diffs}/{} elements differ",
            a.len()
        );
        println!(
            "qsa_prefill_attn_gp == qsa_prefill_attn_g: nq={nq} nkv={nkv} rows={rows}, {} elements",
            a.len()
        );
    }
}
