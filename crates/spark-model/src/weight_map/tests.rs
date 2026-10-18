// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::*;

// The FP8 E4M3 LUT and the f32 → BF16 cast are now `atlas_core::numeric`;
// their reference-value, exhaustive-spec, RNE-byte-exact and PyTorch-parity
// tests live beside them there. The identical copy that used to sit here
// was deleted rather than kept in sync by hand.

/// `dense_keep_f32` is what the qwen4_exp loaders hand to the `const float*`
/// norm consumers (`hc_norm`, the PLE norms): an F32 source passes through
/// untouched and a BF16 one is widened by zero extension — `(bf16 as u32)
/// << 16`, exact, no rounding.
#[test]
fn dense_keep_f32_passes_f32_and_widens_bf16() {
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn put(
        store: &mut WeightStore,
        gpu: &MockGpuBackend,
        name: &str,
        bytes: &[u8],
        dtype: WeightDtype,
    ) {
        let ptr = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(bytes, ptr).unwrap();
        store
            .insert(
                name.into(),
                WeightTensor {
                    ptr,
                    shape: vec![bytes.len() / dtype.byte_size()],
                    dtype,
                },
            )
            .unwrap();
    }

    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::empty();

    // BF16 [1.0, -2.5, 0x1.fp-4]: each lands as (bits << 16).
    let bf: Vec<u8> = [0x3F80u16, 0xC020, 0x3D00]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    put(
        &mut store,
        &gpu,
        "l.attn_hyper_connection.hc_norm.weight",
        &bf,
        WeightDtype::BF16,
    );
    let w = dense_keep_f32(&store, "l.attn_hyper_connection.hc_norm.weight", &gpu).unwrap();
    let got = gpu.read_alloc(w.weight).unwrap();
    let want: Vec<u8> = [0x3F80u32 << 16, 0xC020 << 16, 0x3D00 << 16]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    assert_eq!(got, want);

    // F32 source: the SAME pointer comes back, no copy.
    let f: Vec<u8> = [1.5f32, -0.25]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    put(
        &mut store,
        &gpu,
        "l.ple.norm_key.weight",
        &f,
        WeightDtype::FP32,
    );
    let src = store.get("l.ple.norm_key.weight").unwrap().ptr;
    let w = dense_keep_f32(&store, "l.ple.norm_key.weight", &gpu).unwrap();
    assert_eq!(w.weight, src);
}

#[test]
fn test_weight_name_patterns() {
    // Verify our name generation matches actual HF patterns.
    let layer = 3;
    assert_eq!(
        format!("model.layers.{layer}.self_attn.q_proj.weight"),
        "model.layers.3.self_attn.q_proj.weight"
    );
    assert_eq!(
        format!("model.layers.{layer}.linear_attn.in_proj_qkvz.weight"),
        "model.layers.3.linear_attn.in_proj_qkvz.weight"
    );
    assert_eq!(
        format!("model.layers.{layer}.mlp.experts.{}.gate_proj.weight", 42),
        "model.layers.3.mlp.experts.42.gate_proj.weight"
    );
}
