// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel checks at native/512K/1M boundaries, including distinct MRoPE axes.
//! These require actual Qwen PTX. They are not product quality qualification.

use super::*;
use atlas_core::{config::ModelConfig, qwen_yarn};

fn policy(factor: u8) -> (Vec<f32>, f32) {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen4_exp".into();
    c.head_dim = 256;
    c.partial_rotary_factor = 0.25;
    c.rope_theta = 1e7;
    c.max_position_embeddings = 262144;
    qwen_yarn::configure_context(&mut c, Some(factor), 524288).unwrap();
    (
        qwen_yarn::inverse_frequencies(&c).unwrap(),
        c.yarn_attention_factor,
    )
}

fn rotate(input: &[f32], freq: &[f32], scale: f32, axes: [u32; 3]) -> Vec<f32> {
    let mut out = input.to_vec();
    for j in 0..32 {
        let angle = axes[j % 3] as f32 * freq[j];
        let (s, c) = (angle.sin() * scale, angle.cos() * scale);
        out[j] = input[j] * c - input[j + 32] * s;
        out[j + 32] = input[j + 32] * c + input[j] * s;
    }
    out
}

fn norm(input: &[f32], weight: &[f32]) -> Vec<f32> {
    let mean = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let inv = 1.0 / (mean + EPS).sqrt();
    input
        .iter()
        .zip(weight)
        .map(|(v, w)| v * inv * (1.0 + w))
        .collect()
}

#[test]
#[ignore = "requires GB10 and real Qwen3.8-Flash-Next PTX"]
fn qwen_yarn_mrope_gpu_matches_reference_at_extended_positions() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let mut rng = Rng(0x18e325d);
    let axes: [[u32; 3]; 5] = [
        [0, 0, 0],
        [262143, 262143, 262143],
        [524287, 524287, 524287],
        [999999, 999999, 999999],
        [900000, 37, 901],
    ];
    let q = bf16_round(&rng.vec(axes.len() * 2 * 256, 1.0));
    let k = bf16_round(&rng.vec(axes.len() * 256, 1.0));
    let positions: Vec<_> = (0..3)
        .map(|axis| {
            let bytes: Vec<_> = axes.iter().flat_map(|p| p[axis].to_le_bytes()).collect();
            upload(g, &bytes)
        })
        .collect();
    let kernel = g
        .kernel(
            "rope_mrope_interleaved",
            "rope_forward_mrope_interleaved_yarn",
        )
        .unwrap();
    for factor in [2, 4] {
        let (freq, scale) = policy(factor);
        let df = upload(g, &f32_bytes(&freq));
        let dq = upload(g, &bf16_bytes(&q));
        let dk = upload(g, &bf16_bytes(&k));
        ops::rope_yarn_mrope(
            g,
            kernel,
            dq,
            dk,
            positions[0],
            positions[1],
            positions[2],
            axes.len() as u32,
            2,
            1,
            256,
            64,
            df,
            scale,
            g.default_stream(),
        )
        .unwrap();
        g.synchronize(g.default_stream()).unwrap();
        for (input, ptr, heads) in [(&q, dq, 2), (&k, dk, 1)] {
            let mut expected = Vec::new();
            for (row, positions) in axes.iter().enumerate() {
                for head in 0..heads {
                    let start = (row * heads + head) * 256;
                    expected.extend(bf16_round(&rotate(
                        &input[start..start + 256],
                        &freq,
                        scale,
                        *positions,
                    )));
                }
            }
            let got = download_bf16(g, ptr, input.len());
            check("YaRN MRoPE", &got, &expected);
            for (i, (&a, &b)) in got.iter().zip(input).enumerate() {
                if i % 256 >= 64 {
                    assert_eq!(a, b, "non-rotary dimension changed");
                }
            }
            // Unscaled passthrough must fail, including at position zero (amplitude).
            assert!(measure(&got, input).0 > 0.1);
        }
        for ptr in [df, dq, dk] {
            g.free(ptr).unwrap();
        }
    }
    for ptr in positions {
        g.free(ptr).unwrap();
    }
}

#[test]
#[ignore = "requires GB10 and real Qwen3.8-Flash-Next PTX"]
fn qwen_yarn_qsa_gpu_matches_reference_in_all_rotation_kernels() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let mut rng = Rng(0x7a113d);
    let raw = bf16_round(&rng.vec(16 * 128, 1.0));
    let weights = bf16_round(&rng.vec(128, 0.1));
    let draw = upload(g, &bf16_bytes(&raw));
    let dw = upload(g, &bf16_bytes(&weights));
    let out = g.alloc(16 * 128 * 4).unwrap();
    for factor in [2, 4] {
        let (freq, scale) = policy(factor);
        let df = upload(g, &f32_bytes(&freq));
        let kernel = g.kernel("qsa_indexer", "qsa_block_pool_yarn").unwrap();
        ops::qsa_block_pool_scaled(
            g, kernel, draw, dw, out, 0, 4, 4, 128, 64, 1e7, EPS, df, scale, stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        let mut expected = Vec::new();
        for block in 0..4 {
            let mean: Vec<_> = (0..128)
                .map(|d| (0..4).map(|r| raw[(block * 4 + r) * 128 + d]).sum::<f32>() / 4.0)
                .collect();
            expected.extend(bf16_round(&rotate(
                &norm(&mean, &weights),
                &freq,
                scale,
                [block as u32 * 4; 3],
            )));
        }
        check("YaRN QSA pool", &download_bf16(g, out, 4 * 128), &expected);
        for pos in [262143, 524287, 999998] {
            let kernel = g.kernel("qsa_indexer", "qsa_qprep_yarn").unwrap();
            ops::qsa_qprep_scaled(
                g, kernel, draw, dw, out, 4, 128, 64, pos, 1e7, EPS, df, scale, stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            let expected: Vec<_> = (0..4)
                .flat_map(|h| {
                    rotate(
                        &norm(&raw[h * 128..(h + 1) * 128], &weights),
                        &freq,
                        scale,
                        [pos; 3],
                    )
                })
                .collect();
            check("YaRN QSA query", &download_f32(g, out, 4 * 128), &expected);
            let kernel = g.kernel("qsa_indexer", "qsa_qprep_rows_yarn").unwrap();
            ops::qsa_qprep_rows_scaled(
                g,
                kernel,
                draw,
                dw,
                out,
                2,
                pos,
                5 * 128,
                4,
                128,
                64,
                1e7,
                EPS,
                df,
                scale,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            let mut expected = Vec::new();
            for row in 0..2 {
                for h in 0..4 {
                    let at = (row * 5 + h) * 128;
                    expected.extend(rotate(
                        &norm(&raw[at..at + 128], &weights),
                        &freq,
                        scale,
                        [pos + row as u32; 3],
                    ));
                }
            }
            check("YaRN QSA rows", &download_f32(g, out, 8 * 128), &expected);
        }
        g.free(df).unwrap();
    }
    for ptr in [draw, dw, out] {
        g.free(ptr).unwrap();
    }
}
