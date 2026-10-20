// SPDX-License-Identifier: AGPL-3.0-only

//! GPU: the two ranks' shards of the draft head, assembled, are the bytes
//! the unsplit draft head writes, for the BF16 rows and the NVFP4 copy, at
//! every batched propose width. Synthetic weights and inputs (+-0, BF16
//! subnormals and overflow bait planted); a ragged row count, so the shards
//! differ in width and rank 0 projects columns it does not keep.
//! `#[ignore]` per repo convention. On a GB10:
//! ```text
//! ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo test -p spark-model --release --lib qwen4exp_mtp_tp_gpu -- --ignored --nocapture
//! ```

use super::*;

const VOCAB: usize = 20_011;
const HIDDEN: usize = 2560;

/// SplitMix64 BF16 values in about [-scale, scale), with specials planted.
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
                0 => 0x8000,                    // -0
                1 => (x >> 20) as u16 & 0x807F, // subnormal
                2 => 0x7149,                    // ~1e30
                _ => (v.to_bits() >> 16) as u16,
            };
            bits.to_le_bytes()
        })
        .collect()
}

#[test]
#[ignore]
fn qwen4exp_mtp_tp_gpu_shards_assemble_to_the_unsplit_head() {
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let lm_head = DenseWeight {
        weight: g.alloc(VOCAB * HIDDEN * BF16).unwrap(),
    };
    g.copy_h2d(&bf16_bytes(VOCAB * HIDDEN, 7, 0.05), lm_head.weight)
        .unwrap();
    let input = g.alloc(PROPOSE_BATCH_MAX * HIDDEN * BF16).unwrap();
    g.copy_h2d(&bf16_bytes(PROPOSE_BATCH_MAX * HIDDEN, 11, 2.0), input)
        .unwrap();
    let k = crate::layers::try_kernel(g, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
    let full = g.alloc(PROPOSE_BATCH_MAX * VOCAB * BF16).unwrap();
    let joined = g.alloc(PROPOSE_BATCH_MAX * VOCAB * BF16).unwrap();
    for nvfp4 in [false, true] {
        let draft = DraftHead::build_with(&lm_head, VOCAB, HIDDEN, 0, nvfp4, g).unwrap();
        let tp = [0, 1].map(|rank| DraftTp::new(&draft, HIDDEN, rank, g).unwrap());
        for n in 2..=PROPOSE_BATCH_MAX {
            let bytes = n * VOCAB * BF16;
            g.memset(full, 0x55, bytes).unwrap();
            g.memset(joined, 0xAA, bytes).unwrap();
            draft
                .project_rows(g, k, input, full, n as u32, HIDDEN as u32, stream)
                .unwrap();
            for t in &tp {
                t.project(&draft, g, k, input, n, stream).unwrap();
            }
            // The peer's shard is rank 1's own `send`.
            assemble(
                g,
                tp[0].geom,
                0,
                (tp[0].send, tp[1].send),
                joined,
                n,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            let (mut a, mut b) = (vec![0u8; bytes], vec![0u8; bytes]);
            g.copy_d2h(full, &mut a).unwrap();
            g.copy_d2h(joined, &mut b).unwrap();
            let diff = a.chunks(2).zip(b.chunks(2)).filter(|(x, y)| x != y).count();
            assert_eq!(diff, 0, "nvfp4={nvfp4} n={n}: {diff} logits differ");
            println!(
                "ok nvfp4={nvfp4} n={n}: {} logits byte-identical",
                n * VOCAB
            );
        }
    }
}

/// What a rank's shard saves: the 100k-row NVFP4 draft head (the profiled
/// configuration) at 8 rows, whole against one shard, wall time over a
/// stream sync (the projections are ~1 ms).
#[test]
#[ignore]
fn qwen4exp_mtp_tp_gpu_shard_time() {
    const ROWS: usize = 100_000;
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("build with ATLAS_TARGET_MODEL=qwen3.8-flash-next");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let lm_head = DenseWeight {
        weight: g.alloc(ROWS * HIDDEN * BF16).unwrap(),
    };
    g.copy_h2d(&bf16_bytes(ROWS * HIDDEN, 3, 0.05), lm_head.weight)
        .unwrap();
    let input = g.alloc(PROPOSE_BATCH_MAX * HIDDEN * BF16).unwrap();
    let out = g.alloc(PROPOSE_BATCH_MAX * ROWS * BF16).unwrap();
    let k = crate::layers::try_kernel(g, "dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm");
    let draft = DraftHead::build_with(&lm_head, ROWS, HIDDEN, 0, true, g).unwrap();
    let tp = DraftTp::new(&draft, HIDDEN, 1, g).unwrap();
    let time = |f: &dyn Fn()| {
        f();
        g.synchronize(stream).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..20 {
            f();
        }
        g.synchronize(stream).unwrap();
        t.elapsed().as_secs_f64() * 1e6 / 20.0
    };
    let n = PROPOSE_BATCH_MAX;
    let whole = time(&|| {
        draft
            .project_rows(g, k, input, out, n as u32, HIDDEN as u32, stream)
            .unwrap()
    });
    let shard = time(&|| tp.project(&draft, g, k, input, n, stream).unwrap());
    println!(
        "NVFP4 draft head {ROWS} rows, {n} inputs: whole {whole:.0} us, one shard {shard:.0} us"
    );
}
