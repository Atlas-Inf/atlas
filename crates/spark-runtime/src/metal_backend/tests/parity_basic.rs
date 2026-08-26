// SPDX-License-Identifier: AGPL-3.0-only
//! Smoke + simple element-wise kernel parity (alloc, bf16_add, sigmoid_gate).

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// End-to-end check: alloc → memcpy → kernel launch → memcpy back.
/// The kernel is `noop_smoke` from `kernels/metal/common/`. It
/// writes 0.0 to the first `n` floats of `out`, so after launching
/// with `n=4` the first 4 floats should be exactly zero regardless
/// of what we initialised the buffer with.
#[test]
fn metal_alloc_copy_launch_roundtrip() {
    // Pull the metallib bytes the build script embedded; skip the
    // test gracefully when no Metal device is available (CI runner).
    let Some(backend) = maybe_backend() else {
        return;
    };

    // Round-trip a known byte pattern through alloc/copy_h2d/copy_d2h.
    let bytes = 64;
    let ptr = backend.alloc(bytes).expect("alloc");
    let pattern: Vec<u8> = (0..bytes as u8).collect();
    backend.copy_h2d(&pattern, ptr).expect("copy_h2d");
    let mut readback = vec![0u8; bytes];
    backend.copy_d2h(ptr, &mut readback).expect("copy_d2h");
    assert_eq!(pattern, readback, "h2d/d2h round-trip mismatch");

    // Zero the first 4 floats via the noop_smoke kernel.
    let n: u32 = 4;
    let kernel = backend
        .kernel("noop_smoke", "noop_smoke")
        .expect("kernel lookup");
    backend
        .launch_typed(
            kernel,
            [1, 1, 1],
            [n, 1, 1],
            0,
            backend.default_stream(),
            &[KernelArg::Buffer(ptr), KernelArg::Bytes(&n.to_le_bytes())],
        )
        .expect("launch_typed");
    backend
        .synchronize(backend.default_stream())
        .expect("synchronize");

    // First 16 bytes should now be all-zero floats; the rest of
    // the buffer should retain the original pattern.
    let mut after = vec![0u8; bytes];
    backend
        .copy_d2h(ptr, &mut after)
        .expect("copy_d2h post-launch");
    assert_eq!(&after[..16], &[0u8; 16], "kernel did not zero out[0..4]");
    assert_eq!(
        &after[16..],
        &pattern[16..],
        "kernel touched out-of-range bytes"
    );

    backend.free(ptr).expect("free");
}

/// `bf16_add` parity. Trivial element-wise check — the kernel
/// is one line of math but it's the residual primitive every
/// transformer block uses, so a regression here would silently
/// blow up every layer's output.
#[test]
fn metal_bf16_add_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = 257; // odd to verify bounds-check on tail thread
    let a: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(0.1 + 0.001 * i as f32))
        .collect();
    let b: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(-0.05 + 0.0007 * i as f32))
        .collect();

    let mut expected = vec![half::bf16::ZERO; n as usize];
    for i in 0..n as usize {
        expected[i] = half::bf16::from_f32(a[i].to_f32() + b[i].to_f32());
    }

    let a_bytes = bf16_slice_to_bytes(&a);
    let b_bytes = bf16_slice_to_bytes(&b);
    let a_ptr = backend.alloc(a_bytes.len()).unwrap();
    let b_ptr = backend.alloc(b_bytes.len()).unwrap();
    let out_ptr = backend.alloc(a_bytes.len()).unwrap();
    backend.copy_h2d(&a_bytes, a_ptr).unwrap();
    backend.copy_h2d(&b_bytes, b_ptr).unwrap();

    let kernel = backend.kernel("bf16_add", "bf16_add").unwrap();
    let block: u32 = 64;
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(block), 1, 1],
            [block, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(a_ptr),
                KernelArg::Buffer(b_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch bf16_add");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; a_bytes.len()];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    for i in 0..n as usize {
        assert!(
            (expected[i].to_f32() - actual[i].to_f32()).abs() < 1e-4,
            "bf16_add mismatch at idx {i}"
        );
    }

    backend.free(a_ptr).unwrap();
    backend.free(b_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}

/// `sigmoid_gate` parity. `out = sigmoid(gate) * x`. Distinct
/// from `silu_gate` (which is `gate * sigmoid(gate) * up`) —
/// Qwen3.5 uses this for `attn_output_gate`.
#[test]
fn metal_sigmoid_gate_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = 128;
    let gate: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(-3.0 + 6.0 * i as f32 / (n - 1) as f32))
        .collect();
    let x: Vec<half::bf16> = (0..n)
        .map(|i| half::bf16::from_f32(0.5 + 0.01 * i as f32))
        .collect();

    let mut expected = vec![half::bf16::ZERO; n as usize];
    for i in 0..n as usize {
        let g = gate[i].to_f32();
        let v = x[i].to_f32();
        let sig = 1.0 / (1.0 + (-g).exp());
        expected[i] = half::bf16::from_f32(sig * v);
    }

    let g_bytes = bf16_slice_to_bytes(&gate);
    let x_bytes = bf16_slice_to_bytes(&x);
    let g_ptr = backend.alloc(g_bytes.len()).unwrap();
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let out_ptr = backend.alloc(g_bytes.len()).unwrap();
    backend.copy_h2d(&g_bytes, g_ptr).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();

    let kernel = backend.kernel("sigmoid_gate", "sigmoid_gate").unwrap();
    backend
        .launch_typed(
            kernel,
            [n.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Buffer(g_ptr),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch sigmoid_gate");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; g_bytes.len()];
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
        "sigmoid_gate: max |expected - actual| = {max_abs_diff}"
    );

    backend.free(g_ptr).unwrap();
    backend.free(x_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}
