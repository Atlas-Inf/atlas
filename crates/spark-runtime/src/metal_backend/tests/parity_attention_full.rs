// SPDX-License-Identifier: AGPL-3.0-only
//! Non-causal full-sequence attention kernel parity.

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;

/// `attention_full` parity. Same as `attention_prefill` minus
/// the causal mask; the FP32 reference simply drops the
/// `s >= cutoff` branch.
#[test]
fn metal_attention_full_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let num_tokens: u32 = 4;
    let seq_len: u32 = 6; // K/V wider than Q to exercise the non-causal path
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

    let mut expected = vec![half::bf16::ZERO; (num_tokens * num_heads * head_dim) as usize];
    let group = num_heads / num_kv_heads;
    for m in 0..num_tokens as usize {
        for h in 0..num_heads as usize {
            let kv_h = h / group as usize;
            let mut scores: Vec<f32> = (0..seq_len as usize)
                .map(|s| {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim as usize {
                        let qv = q[(m * num_heads as usize + h) * head_dim as usize + d].to_f32();
                        let kvv =
                            k[(s * num_kv_heads as usize + kv_h) * head_dim as usize + d].to_f32();
                        dot += qv * kvv;
                    }
                    dot * scale
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

    let kernel = backend.kernel("attention_full", "attention_full").unwrap();
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
        .expect("launch attention_full");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; q_bytes.len()];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..(num_tokens * num_heads * head_dim) as usize {
        let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.02,
        "attention_full: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(q_ptr).unwrap();
    backend.free(k_ptr).unwrap();
    backend.free(v_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}
