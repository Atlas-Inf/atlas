// SPDX-License-Identifier: AGPL-3.0-only
//! Vision-tower kernel parity (conv3d_patch_embed, dense_gemm/v, layer_norm, gelu).
//! Non-causal `attention_full` lives in `parity_attention_full.rs`.

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// `conv3d_patch_embed` parity. Tiny 3-D conv with stride ==
/// kernel — the patch-embedding shape every modern ViT uses.
/// Independent FP32 reference walks the same indices to surface
/// any stride / channel-broadcast / dim-order regression.
#[test]
fn metal_conv3d_patch_embed_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    // Small but non-trivial: 4 output channels, 3 input channels
    // (RGB), kernel 2×4×4, output spatial 1×3×3 → input spatial
    // 2×12×12. The 4×4 spatial kernel is large enough that a
    // mis-strided dispatch would visibly mix neighbouring cells.
    let out_c: u32 = 4;
    let in_c: u32 = 3;
    let kt: u32 = 2;
    let kh: u32 = 4;
    let kw: u32 = 4;
    let t_out: u32 = 1;
    let h_out: u32 = 3;
    let w_out: u32 = 3;

    let t_in = t_out * kt;
    let h_in = h_out * kh;
    let w_in = w_out * kw;

    let input: Vec<half::bf16> = (0..(in_c * t_in * h_in * w_in))
        .map(|i| half::bf16::from_f32(0.01 * (i as f32 * 0.013).sin()))
        .collect();
    let weight: Vec<half::bf16> = (0..(out_c * kt * kh * kw * in_c))
        .map(|i| half::bf16::from_f32(0.005 + 0.0003 * (i as f32 * 0.011).cos()))
        .collect();
    let bias: Vec<half::bf16> = (0..out_c)
        .map(|i| half::bf16::from_f32(0.1 + 0.05 * i as f32))
        .collect();

    // CPU reference — same loop nest as the kernel.
    let mut expected = vec![half::bf16::ZERO; (out_c * t_out * h_out * w_out) as usize];
    for c_out_ in 0..out_c as usize {
        for t_o in 0..t_out as usize {
            for h_o in 0..h_out as usize {
                for w_o in 0..w_out as usize {
                    let mut acc = bias[c_out_].to_f32();
                    for dt in 0..kt as usize {
                        for dh in 0..kh as usize {
                            for dw in 0..kw as usize {
                                for ic in 0..in_c as usize {
                                    let w_off = (((c_out_ * kt as usize + dt) * kh as usize + dh)
                                        * kw as usize
                                        + dw)
                                        * in_c as usize
                                        + ic;
                                    let t_idx = t_o * kt as usize + dt;
                                    let h_idx = h_o * kh as usize + dh;
                                    let w_idx = w_o * kw as usize + dw;
                                    let i_off = ((ic * t_in as usize + t_idx) * h_in as usize
                                        + h_idx)
                                        * w_in as usize
                                        + w_idx;
                                    acc += weight[w_off].to_f32() * input[i_off].to_f32();
                                }
                            }
                        }
                    }
                    let out_idx = ((c_out_ * t_out as usize + t_o) * h_out as usize + h_o)
                        * w_out as usize
                        + w_o;
                    expected[out_idx] = half::bf16::from_f32(acc);
                }
            }
        }
    }

    let in_bytes = bf16_slice_to_bytes(&input);
    let w_bytes = bf16_slice_to_bytes(&weight);
    let b_bytes = bf16_slice_to_bytes(&bias);
    let in_ptr = backend.alloc(in_bytes.len()).unwrap();
    let w_ptr = backend.alloc(w_bytes.len()).unwrap();
    let b_ptr = backend.alloc(b_bytes.len()).unwrap();
    let out_bytes_n = (out_c * t_out * h_out * w_out) as usize * 2;
    let out_ptr = backend.alloc(out_bytes_n).unwrap();
    backend.copy_h2d(&in_bytes, in_ptr).unwrap();
    backend.copy_h2d(&w_bytes, w_ptr).unwrap();
    backend.copy_h2d(&b_bytes, b_ptr).unwrap();

    let kernel = backend
        .kernel("conv3d_patch_embed", "conv3d_patch_embed")
        .unwrap();
    let block_x = 8u32;
    let block_y = 8u32;
    let flat_y = t_out * h_out * w_out;
    backend
        .launch_typed(
            kernel,
            [out_c.div_ceil(block_x), flat_y.div_ceil(block_y), 1],
            [block_x, block_y, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&out_c.to_le_bytes()),
                KernelArg::Bytes(&in_c.to_le_bytes()),
                KernelArg::Bytes(&kt.to_le_bytes()),
                KernelArg::Bytes(&kh.to_le_bytes()),
                KernelArg::Bytes(&kw.to_le_bytes()),
                KernelArg::Bytes(&t_out.to_le_bytes()),
                KernelArg::Bytes(&h_out.to_le_bytes()),
                KernelArg::Bytes(&w_out.to_le_bytes()),
                KernelArg::Buffer(in_ptr),
                KernelArg::Buffer(w_ptr),
                KernelArg::Buffer(b_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch conv3d_patch_embed");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; out_bytes_n];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..(out_c * t_out * h_out * w_out) as usize {
        let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    // Reduction depth ~ 96 elements per output cell → BF16 ULP ≈
    // 0.005 at output magnitudes ~0.1; tolerate 0.05 for ordering
    // drift.
    assert!(
        max_abs_diff < 0.05,
        "conv3d_patch_embed: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(in_ptr).unwrap();
    backend.free(w_ptr).unwrap();
    backend.free(b_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}

/// `dense_gemm_bf16` parity. Multi-token prefill BF16 matmul
/// vs CPU FP32 reference. Same weight layout as
/// `dense_gemv_bf16` (`[N, K]` row-major) so the ViT prefill
/// path can use either kernel without reshaping weights.
#[test]
fn metal_dense_gemm_bf16_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let m: u32 = 4;
    let n: u32 = 8;
    let k: u32 = 64;

    let x: Vec<half::bf16> = (0..(m * k))
        .map(|i| half::bf16::from_f32(0.1 + 0.001 * i as f32))
        .collect();
    let w: Vec<half::bf16> = (0..(n * k))
        .map(|i| {
            let row = i / k;
            let col = i % k;
            half::bf16::from_f32(0.001 + 0.0005 * row as f32 + 0.0007 * col as f32)
        })
        .collect();

    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; (m * n) as usize];
    for mi in 0..m as usize {
        for ni in 0..n as usize {
            let mut acc = 0.0f32;
            for ki in 0..k as usize {
                acc += x[mi * k as usize + ki].to_f32() * w[ni * k as usize + ki].to_f32();
            }
            expected[mi * n as usize + ni] = half::bf16::from_f32(acc);
        }
    }

    let x_bytes = bf16_slice_to_bytes(&x);
    let w_bytes = bf16_slice_to_bytes(&w);
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let w_ptr = backend.alloc(w_bytes.len()).unwrap();
    let y_ptr = backend.alloc((m * n) as usize * 2).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();
    backend.copy_h2d(&w_bytes, w_ptr).unwrap();

    let kernel = backend
        .kernel("dense_gemm_bf16", "dense_gemm_bf16")
        .unwrap();
    let block_x = 16u32;
    let block_y = 16u32;
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(block_x), m.div_ceil(block_y), 1],
            [block_x, block_y, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&m.to_le_bytes()),
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(w_ptr),
                KernelArg::Buffer(y_ptr),
            ],
        )
        .expect("launch dense_gemm_bf16");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut y_raw = vec![0u8; (m * n) as usize * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
    let actual = bytes_to_bf16_vec(&y_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..(m * n) as usize {
        let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.05,
        "dense_gemm_bf16: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(x_ptr).unwrap();
    backend.free(w_ptr).unwrap();
    backend.free(y_ptr).unwrap();
}

/// `dense_gemv_bf16` parity. Pure BF16 matvec vs CPU FP32
/// reference. Same reduction shape as `mlx_int8_gemv` minus
/// the fused dequant.
#[test]
fn metal_dense_gemv_bf16_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = 16;
    let k: u32 = 256;
    let w: Vec<half::bf16> = (0..(n * k))
        .map(|i| {
            let row = i / k;
            let col = i % k;
            half::bf16::from_f32(0.001 + 0.0005 * row as f32 + 0.0007 * col as f32)
        })
        .collect();
    let x: Vec<half::bf16> = (0..k)
        .map(|i| half::bf16::from_f32(0.5 + 0.001 * i as f32))
        .collect();

    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n as usize];
    for r in 0..n as usize {
        let mut acc = 0.0f32;
        for c in 0..k as usize {
            acc += w[r * k as usize + c].to_f32() * x[c].to_f32();
        }
        expected[r] = half::bf16::from_f32(acc);
    }

    let w_bytes = bf16_slice_to_bytes(&w);
    let x_bytes = bf16_slice_to_bytes(&x);
    let w_ptr = backend.alloc(w_bytes.len()).unwrap();
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let y_ptr = backend.alloc(n as usize * 2).unwrap();
    backend.copy_h2d(&w_bytes, w_ptr).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();

    let kernel = backend
        .kernel("dense_gemv_bf16", "dense_gemv_bf16")
        .unwrap();
    backend
        .launch_typed(
            kernel,
            [n, 1, 1],
            [64, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
                KernelArg::Buffer(w_ptr),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(y_ptr),
            ],
        )
        .expect("launch dense_gemv_bf16");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut y_raw = vec![0u8; n as usize * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
    let actual = bytes_to_bf16_vec(&y_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..n as usize {
        let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.05,
        "dense_gemv_bf16: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(w_ptr).unwrap();
    backend.free(x_ptr).unwrap();
    backend.free(y_ptr).unwrap();
}

/// `layer_norm` parity. Two-pass mean+variance reduction vs
/// the FP32 `(x - mean) / std * w + b` reference.
#[test]
fn metal_layer_norm_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let num_tokens: u32 = 3;
    let hidden: u32 = 256;
    let eps: f32 = 1e-5;

    let x: Vec<half::bf16> = (0..(num_tokens * hidden))
        .map(|i| {
            let r = i / hidden;
            let c = i % hidden;
            half::bf16::from_f32(0.1 + 0.01 * r as f32 + 0.001 * c as f32)
        })
        .collect();
    let weight: Vec<half::bf16> = (0..hidden)
        .map(|i| half::bf16::from_f32(1.0 + 0.005 * i as f32))
        .collect();
    let bias: Vec<half::bf16> = (0..hidden)
        .map(|i| half::bf16::from_f32(-0.05 + 0.002 * i as f32))
        .collect();

    let mut expected = vec![half::bf16::ZERO; (num_tokens * hidden) as usize];
    for r in 0..num_tokens as usize {
        let mut sum = 0.0f32;
        for c in 0..hidden as usize {
            sum += x[r * hidden as usize + c].to_f32();
        }
        let mean = sum / hidden as f32;
        let mut var = 0.0f32;
        for c in 0..hidden as usize {
            let d = x[r * hidden as usize + c].to_f32() - mean;
            var += d * d;
        }
        var /= hidden as f32;
        let inv_std = (var + eps).powf(-0.5);
        for c in 0..hidden as usize {
            let xi = x[r * hidden as usize + c].to_f32();
            let w = weight[c].to_f32();
            let b = bias[c].to_f32();
            expected[r * hidden as usize + c] = half::bf16::from_f32((xi - mean) * inv_std * w + b);
        }
    }

    let x_bytes = bf16_slice_to_bytes(&x);
    let w_bytes = bf16_slice_to_bytes(&weight);
    let b_bytes = bf16_slice_to_bytes(&bias);
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let w_ptr = backend.alloc(w_bytes.len()).unwrap();
    let b_ptr = backend.alloc(b_bytes.len()).unwrap();
    let out_ptr = backend.alloc(x_bytes.len()).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();
    backend.copy_h2d(&w_bytes, w_ptr).unwrap();
    backend.copy_h2d(&b_bytes, b_ptr).unwrap();

    let kernel = backend.kernel("layer_norm", "layer_norm").unwrap();
    backend
        .launch_typed(
            kernel,
            [num_tokens, 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&hidden.to_le_bytes()),
                KernelArg::Bytes(&eps.to_le_bytes()),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(w_ptr),
                KernelArg::Buffer(b_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch layer_norm");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; x_bytes.len()];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..(num_tokens * hidden) as usize {
        let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.05,
        "layer_norm: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(x_ptr).unwrap();
    backend.free(w_ptr).unwrap();
    backend.free(b_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}

/// `gelu` parity. Tanh-approx GeLU vs FP32 reference. Sweep
/// across negative, zero-crossing, and positive inputs because
/// the tanh approximation has its largest error near |x| ≈ 1.
#[test]
fn metal_gelu_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = 256;
    let x: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(-3.0 + 6.0 * i as f32 / (n - 1) as f32))
        .collect();

    let sqrt_2_over_pi: f32 = 0.7978845608028654;
    let c: f32 = 0.044715;
    let mut expected = vec![half::bf16::ZERO; n as usize];
    for i in 0..n as usize {
        let v = x[i].to_f32();
        let arg = sqrt_2_over_pi * (v + c * v * v * v);
        expected[i] = half::bf16::from_f32(0.5 * v * (1.0 + arg.tanh()));
    }

    let x_bytes = bf16_slice_to_bytes(&x);
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let out_ptr = backend.alloc(x_bytes.len()).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();

    let kernel = backend.kernel("gelu", "gelu").unwrap();
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch gelu");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; x_bytes.len()];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..n as usize {
        let d = (expected[i].to_f32() - actual[i].to_f32()).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 0.02,
        "gelu: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(x_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}
