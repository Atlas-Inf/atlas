// SPDX-License-Identifier: AGPL-3.0-only
//! Attention-path kernel parity (prefill, decode, softmax-topp, kv_cache_append).

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// `attention_prefill` parity. Multi-token causal attention.
/// FP32 reference matches the kernel's algorithm exactly: causal
/// mask everything past `m`, max-subtract softmax, FP32 sum of
/// V-weighted scores. Verifies the (m, h) flat-grid decoding
/// inside the kernel.
#[test]
fn metal_attention_prefill_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let num_tokens: u32 = 5;
    let seq_len: u32 = 5; // K/V align with Q in this test
    let num_heads: u32 = 4;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 8;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<half::bf16> = (0..(num_tokens * num_heads * head_dim))
        .map(|i| half::bf16::from_f32(0.05 + 0.005 * i as f32))
        .collect();
    let k: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
        .map(|i| half::bf16::from_f32(0.04 + 0.003 * i as f32))
        .collect();
    let v: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
        .map(|i| half::bf16::from_f32(0.5 + 0.001 * i as f32))
        .collect();

    let mut expected: Vec<half::bf16> =
        vec![half::bf16::ZERO; (num_tokens * num_heads * head_dim) as usize];
    let group = num_heads / num_kv_heads;
    for m in 0..num_tokens as usize {
        let cutoff = m + 1;
        for h in 0..num_heads as usize {
            let kv_h = h / group as usize;
            let mut scores: Vec<f32> = (0..seq_len as usize)
                .map(|s| {
                    if s >= cutoff {
                        f32::NEG_INFINITY
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim as usize {
                            let qv =
                                q[(m * num_heads as usize + h) * head_dim as usize + d].to_f32();
                            let kvv = k[(s * num_kv_heads as usize + kv_h) * head_dim as usize + d]
                                .to_f32();
                            dot += qv * kvv;
                        }
                        dot * scale
                    }
                })
                .collect();
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - mx).exp();
                sum += *s;
            }
            let inv = 1.0 / sum;
            for d in 0..head_dim as usize {
                let mut acc = 0.0f32;
                for s in 0..seq_len as usize {
                    let vv = v[(s * num_kv_heads as usize + kv_h) * head_dim as usize + d].to_f32();
                    acc += scores[s] * inv * vv;
                }
                expected[(m * num_heads as usize + h) * head_dim as usize + d] =
                    half::bf16::from_f32(acc);
            }
        }
    }

    let q_bytes = bf16_slice_to_bytes(&q);
    let k_bytes = bf16_slice_to_bytes(&k);
    let v_bytes = bf16_slice_to_bytes(&v);
    let q_ptr = backend.alloc(q_bytes.len()).unwrap();
    let k_ptr = backend.alloc(k_bytes.len()).unwrap();
    let v_ptr = backend.alloc(v_bytes.len()).unwrap();
    let out_ptr = backend.alloc(q_bytes.len()).unwrap();
    backend.copy_h2d(&q_bytes, q_ptr).unwrap();
    backend.copy_h2d(&k_bytes, k_ptr).unwrap();
    backend.copy_h2d(&v_bytes, v_ptr).unwrap();

    let kernel = backend
        .kernel("attention_prefill", "attention_prefill")
        .unwrap();
    // Flat 1-D grid: num_heads * num_tokens threadgroups.
    let total_groups = num_heads * num_tokens;
    backend
        .launch_typed(
            kernel,
            [total_groups, 1, 1],
            [32, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&num_tokens.to_le_bytes()),
                KernelArg::Bytes(&seq_len.to_le_bytes()),
                KernelArg::Bytes(&num_heads.to_le_bytes()),
                KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&scale.to_le_bytes()),
                KernelArg::Buffer(q_ptr),
                KernelArg::Buffer(k_ptr),
                KernelArg::Buffer(v_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch attention_prefill");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; q_bytes.len()];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..(num_tokens * num_heads * head_dim) as usize {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.02,
        "attention_prefill: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(q_ptr).unwrap();
    backend.free(k_ptr).unwrap();
    backend.free(v_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}

/// `softmax_topp` correctness. Plant a winner-takes-all logit
/// distribution (one token wildly larger than the rest) so any
/// sane top-p sampler must pick that token regardless of the
/// uniform sample. Independently of the `p` and `uniform`
/// parameters, the result has to be the planted index.
#[test]
fn metal_softmax_topp_dominant_logit() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let vocab: u32 = 256;
    let mut logits: Vec<half::bf16> = (0..vocab)
        .map(|i| half::bf16::from_f32(-2.0 + 0.001 * i as f32))
        .collect();
    let target_idx = 137usize;
    // 30 logit units at temperature 1.0 → softmax mass essentially 1.
    logits[target_idx] = half::bf16::from_f32(30.0);

    let bytes = bf16_slice_to_bytes(&logits);
    let logits_ptr = backend.alloc(bytes.len()).unwrap();
    let result_ptr = backend.alloc(4).unwrap();
    backend.copy_h2d(&bytes, logits_ptr).unwrap();

    let kernel = backend.kernel("softmax_topp", "softmax_topp").unwrap();
    // Try a few (p, uniform) combinations — none should change
    // the result given the dominant logit.
    for &(p, uniform) in &[(0.9f32, 0.1f32), (0.95, 0.5), (1.0, 0.99)] {
        let temp: f32 = 1.0;
        backend
            .launch_typed(
                kernel,
                [1, 1, 1],
                [128, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&vocab.to_le_bytes()),
                    KernelArg::Bytes(&temp.to_le_bytes()),
                    KernelArg::Bytes(&p.to_le_bytes()),
                    KernelArg::Bytes(&uniform.to_le_bytes()),
                    KernelArg::Buffer(logits_ptr),
                    KernelArg::Buffer(result_ptr),
                ],
            )
            .expect("launch softmax_topp");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut result_raw = [0u8; 4];
        backend.copy_d2h(result_ptr, &mut result_raw).unwrap();
        let actual = u32::from_le_bytes(result_raw) as usize;
        assert_eq!(
            actual, target_idx,
            "softmax_topp: p={p}, uniform={uniform}: expected {target_idx}, got {actual}"
        );
    }

    backend.free(logits_ptr).unwrap();
    backend.free(result_ptr).unwrap();
}

/// `kv_cache_append` parity. Writes a single token's K and V
/// projections at slot `cache_pos` and verifies the cache
/// updates exactly there, with neighbouring slots untouched.
#[test]
fn metal_kv_cache_append_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let max_seq: u32 = 8;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 4;
    let cache_pos: u32 = 3;

    // Pre-fill cache with a sentinel (-1) so untouched slots are
    // visibly distinct from the new K/V data.
    let cache_slots = (max_seq * num_kv_heads * head_dim) as usize;
    let init: Vec<half::bf16> = vec![half::bf16::from_f32(-1.0); cache_slots];
    let new_k: Vec<half::bf16> = (0..(num_kv_heads * head_dim))
        .map(|i| half::bf16::from_f32(0.5 + 0.1 * i as f32))
        .collect();
    let new_v: Vec<half::bf16> = (0..(num_kv_heads * head_dim))
        .map(|i| half::bf16::from_f32(2.0 + 0.05 * i as f32))
        .collect();

    let init_bytes = bf16_slice_to_bytes(&init);
    let nk_bytes = bf16_slice_to_bytes(&new_k);
    let nv_bytes = bf16_slice_to_bytes(&new_v);

    let k_cache_ptr = backend.alloc(init_bytes.len()).unwrap();
    let v_cache_ptr = backend.alloc(init_bytes.len()).unwrap();
    let new_k_ptr = backend.alloc(nk_bytes.len()).unwrap();
    let new_v_ptr = backend.alloc(nv_bytes.len()).unwrap();
    backend.copy_h2d(&init_bytes, k_cache_ptr).unwrap();
    backend.copy_h2d(&init_bytes, v_cache_ptr).unwrap();
    backend.copy_h2d(&nk_bytes, new_k_ptr).unwrap();
    backend.copy_h2d(&nv_bytes, new_v_ptr).unwrap();

    let kernel = backend
        .kernel("kv_cache_append", "kv_cache_append")
        .unwrap();
    backend
        .launch_typed(
            kernel,
            [head_dim, num_kv_heads, 1],
            [1, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&cache_pos.to_le_bytes()),
                KernelArg::Buffer(new_k_ptr),
                KernelArg::Buffer(new_v_ptr),
                KernelArg::Buffer(k_cache_ptr),
                KernelArg::Buffer(v_cache_ptr),
            ],
        )
        .expect("launch kv_cache_append");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut k_after = vec![0u8; init_bytes.len()];
    let mut v_after = vec![0u8; init_bytes.len()];
    backend.copy_d2h(k_cache_ptr, &mut k_after).unwrap();
    backend.copy_d2h(v_cache_ptr, &mut v_after).unwrap();
    let k_actual = bytes_to_bf16_vec(&k_after);
    let v_actual = bytes_to_bf16_vec(&v_after);

    let slot_size = (num_kv_heads * head_dim) as usize;
    let slot_start = cache_pos as usize * slot_size;
    for i in 0..cache_slots {
        let in_slot = i >= slot_start && i < slot_start + slot_size;
        let expect_k = if in_slot {
            new_k[i - slot_start]
        } else {
            half::bf16::from_f32(-1.0)
        };
        let expect_v = if in_slot {
            new_v[i - slot_start]
        } else {
            half::bf16::from_f32(-1.0)
        };
        assert_eq!(
            k_actual[i], expect_k,
            "kv_cache_append: K[{i}] mismatch (in_slot={in_slot})"
        );
        assert_eq!(
            v_actual[i], expect_v,
            "kv_cache_append: V[{i}] mismatch (in_slot={in_slot})"
        );
    }

    backend.free(k_cache_ptr).unwrap();
    backend.free(v_cache_ptr).unwrap();
    backend.free(new_k_ptr).unwrap();
    backend.free(new_v_ptr).unwrap();
}

/// `attention_decode` parity. Single-token query against a 16-
/// element KV cache with GQA (4 query heads, 2 KV heads, group=2).
/// Independent FP32 reference computes scaled-dot-product
/// attention exactly the way the kernel does (max-subtraction
/// softmax, FP32 accumulation throughout) so any deviation
/// surfaces only as BF16 round error.
#[test]
fn metal_attention_decode_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let seq_len: u32 = 16;
    let num_heads: u32 = 4;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 8;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<half::bf16> = (0..(num_heads * head_dim))
        .map(|i| half::bf16::from_f32(0.05 + 0.01 * i as f32))
        .collect();
    let k: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
        .map(|i| half::bf16::from_f32(0.02 + 0.003 * i as f32))
        .collect();
    let v: Vec<half::bf16> = (0..(seq_len * num_kv_heads * head_dim))
        .map(|i| half::bf16::from_f32(0.5 + 0.001 * i as f32))
        .collect();

    // FP32 reference: same algorithm as the kernel.
    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; (num_heads * head_dim) as usize];
    let group = num_heads / num_kv_heads;
    for h in 0..num_heads as usize {
        let kv_h = h / group as usize;

        // Scores
        let mut scores: Vec<f32> = (0..seq_len as usize)
            .map(|s| {
                let mut dot = 0.0f32;
                for d in 0..head_dim as usize {
                    let qv = q[h * head_dim as usize + d].to_f32();
                    let kvv =
                        k[(s * num_kv_heads as usize + kv_h) * head_dim as usize + d].to_f32();
                    dot += qv * kvv;
                }
                dot * scale
            })
            .collect();
        // Softmax with max subtraction.
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - mx).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        // Output
        for d in 0..head_dim as usize {
            let mut acc = 0.0f32;
            for s in 0..seq_len as usize {
                let vv = v[(s * num_kv_heads as usize + kv_h) * head_dim as usize + d].to_f32();
                acc += scores[s] * inv * vv;
            }
            expected[h * head_dim as usize + d] = half::bf16::from_f32(acc);
        }
    }

    let q_bytes = bf16_slice_to_bytes(&q);
    let k_bytes = bf16_slice_to_bytes(&k);
    let v_bytes = bf16_slice_to_bytes(&v);
    let q_ptr = backend.alloc(q_bytes.len()).unwrap();
    let k_ptr = backend.alloc(k_bytes.len()).unwrap();
    let v_ptr = backend.alloc(v_bytes.len()).unwrap();
    let out_ptr = backend.alloc(q_bytes.len()).unwrap();
    backend.copy_h2d(&q_bytes, q_ptr).unwrap();
    backend.copy_h2d(&k_bytes, k_ptr).unwrap();
    backend.copy_h2d(&v_bytes, v_ptr).unwrap();

    let kernel = backend
        .kernel("attention_decode", "attention_decode")
        .unwrap();
    backend
        .launch_typed(
            kernel,
            [num_heads, 1, 1],
            [32, 1, 1], // one simdgroup per head
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&seq_len.to_le_bytes()),
                KernelArg::Bytes(&num_heads.to_le_bytes()),
                KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&scale.to_le_bytes()),
                KernelArg::Buffer(q_ptr),
                KernelArg::Buffer(k_ptr),
                KernelArg::Buffer(v_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch attention_decode");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; q_bytes.len()];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..(num_heads * head_dim) as usize {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    // Output magnitudes ≈ 0.5 (V scaled by softmax weights).
    // BF16 ULP at 0.5 is ≈ 0.004; tolerate one ULP plus the
    // accumulator-order drift between kernel and reference.
    assert!(
        max_abs_diff < 0.02,
        "attention_decode: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(q_ptr).unwrap();
    backend.free(k_ptr).unwrap();
    backend.free(v_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}
