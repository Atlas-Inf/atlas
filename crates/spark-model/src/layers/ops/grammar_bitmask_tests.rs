// SPDX-License-Identifier: AGPL-3.0-only

//! `atlas_apply_grammar_bitmask` — the #102 draft-0 mask pre-pass. The
//! kernel writes bf16(-inf) over disallowed ids; these tests pin the bit
//! convention (word `id>>5`, bit `id&31`, set = allowed) and the row-local
//! effect. GPU test is `#[ignore]` per repo convention — run with:
//! ```text
//! cargo test -p spark-model --release grammar_bitmask -- --ignored --nocapture
//! ```

use crate::layers::ops;

fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Build a bitmask allowing exactly `allowed` out of `vocab` ids.
fn mask_allowing(vocab: usize, allowed: &[u32]) -> Vec<i32> {
    let mut words = vec![0i32; ops::grammar_bitmask_words(vocab as u32)];
    for &t in allowed {
        words[(t >> 5) as usize] |= 1 << (t & 31);
    }
    words
}

#[test]
fn grammar_bitmask_words_rounds_up() {
    assert_eq!(ops::grammar_bitmask_words(1), 1);
    assert_eq!(ops::grammar_bitmask_words(32), 1);
    assert_eq!(ops::grammar_bitmask_words(33), 2);
    assert_eq!(ops::grammar_bitmask_words(151_936), 4_748);
}

#[test]
fn bitmask_word_layout_matches_xgrammar() {
    // The kernel reads `bitmask[i >> 5] >> (i & 31) & 1`; xgrammar's
    // `bitmask_data` uses the identical layout (grammar/state.rs).
    let m = mask_allowing(64, &[0, 31, 33, 63]);
    for t in 0..64u32 {
        let allowed = (m[(t >> 5) as usize] >> (t & 31)) & 1 == 1;
        assert_eq!(allowed, [0u32, 31, 33, 63].contains(&t), "token {t}");
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn grammar_bitmask_masks_row_01_only() {
    use spark_runtime::gpu::GpuBackend;
    let set = atlas_kernels::ptx_for_exact_target("qwen3.6-27b", "nvfp4")
        .expect("target with common grammar_bitmask module");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let stream = gpu.default_stream();
    let mask_k = gpu
        .kernel("grammar_bitmask", "atlas_apply_grammar_bitmask")
        .expect("atlas_apply_grammar_bitmask");
    let argmax_k = gpu.kernel("argmax", "argmax_bf16").expect("argmax_bf16");

    // vocab=96: row logits = index (id 95 is argmax). γ=4 rows; mask forbids
    // 95 and 94 → legal argmax = 93.
    let vocab = 96usize;
    let gamma = 4usize;
    let mut logits_host: Vec<u16> = Vec::with_capacity(gamma * vocab);
    for _r in 0..gamma {
        for i in 0..vocab {
            logits_host.push(bf16(i as f32));
        }
    }
    let logits = gpu.alloc(gamma * vocab * 2).unwrap();
    let bytes: Vec<u8> = logits_host.iter().flat_map(|b| b.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, logits).unwrap();

    let words = ops::grammar_bitmask_words(vocab as u32);
    let mask = mask_allowing(vocab, &(0..94u32).collect::<Vec<_>>());
    let mask_bytes: Vec<u8> = mask.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mask_dev = gpu.alloc(words * 4).unwrap();
    gpu.copy_h2d(&mask_bytes, mask_dev).unwrap();

    // Mask rows 0 and 1 only (the pos+1 predictors); rows 2,3 untouched.
    for row in 0..2usize {
        ops::apply_grammar_bitmask(
            &gpu,
            mask_k,
            logits.offset(row * vocab * 2),
            mask_dev,
            vocab as u32,
            stream,
        )
        .unwrap();
    }
    gpu.synchronize(stream).unwrap();

    let out = gpu.alloc(4).unwrap();
    let mut got = vec![0u8; 4];
    let argmax_of = |row: usize, got: &mut Vec<u8>| {
        ops::argmax_bf16(
            &gpu,
            argmax_k,
            logits.offset(row * vocab * 2),
            out,
            vocab as u32,
            stream,
        )
        .unwrap();
        gpu.synchronize(stream).unwrap();
        gpu.copy_d2h(out, got).unwrap();
        u32::from_le_bytes([got[0], got[1], got[2], got[3]])
    };
    assert_eq!(argmax_of(0, &mut got), 93, "row 0 masked argmax");
    assert_eq!(argmax_of(1, &mut got), 93, "row 1 masked argmax");
    assert_eq!(argmax_of(2, &mut got), 95, "row 2 must stay unmasked");
    assert_eq!(argmax_of(3, &mut got), 95, "row 3 must stay unmasked");

    // -inf is really written: row 0 elements 94/95 read back as -inf.
    let mut back = vec![0u8; vocab * 2];
    gpu.copy_d2h(logits, &mut back).unwrap();
    for &i in &[94usize, 95] {
        let v = bf16_to_f32(u16::from_le_bytes([back[2 * i], back[2 * i + 1]]));
        assert!(
            v.is_infinite() && v < 0.0,
            "masked id {i} must be -inf, got {v}"
        );
    }
    gpu.free(logits).ok();
    gpu.free(mask_dev).ok();
    gpu.free(out).ok();
}
