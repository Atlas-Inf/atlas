// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence / verify Q, K, V of a native MXFP8 attention layer
//! (`qkv_mx8.rs`) on the mock backend.

use super::super::super::Qwen3AttentionLayer;
use super::super::super::mx8::Mx8AttnRt;
use super::tests::qkv_phase_d2d;
use crate::layers::FfnComponent;
use crate::layers::ops::{Mxfp8ProjKernels, Mxfp8Scratch};
use crate::weight_map::{AttentionWeights, DenseWeight, Mxfp8Proj, Mxfp8Weight, QuantizedWeight};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;

/// An attention layer as `attn_mx8::build` wires it: NULL BF16 slots, no
/// NVFP4 twin; `bf16_slots` instead fills the BF16 slots.
fn layer(gpu: &MockGpuBackend, config: &ModelConfig, bf16_slots: bool) -> Qwen3AttentionLayer {
    let h = config.hidden_size;
    let dw = |bytes: usize| DenseWeight {
        weight: if bytes == 0 {
            DevicePtr::NULL
        } else {
            gpu.alloc(bytes).unwrap()
        },
    };
    let slot = |bytes: usize| dw(if bf16_slots { bytes } else { 0 });
    let attn = AttentionWeights {
        q_proj: slot(h * 2),
        k_proj: slot(h * 2),
        v_proj: slot(h * 2),
        o_proj: QuantizedWeight::null(),
        q_norm: dw(config.head_dim * 2),
        k_norm: dw(config.head_dim * 2),
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    Qwen3AttentionLayer::new(
        dw(h * 2),
        attn,
        dw(h * 2),
        FfnComponent::None,
        0,
        None,
        None,
        None,
        gpu,
        KvCacheDtype::Bf16,
        0,
        config,
    )
    .unwrap()
}

fn projections(gpu: &MockGpuBackend, config: &ModelConfig) -> ([Mxfp8Proj; 4], Mx8AttnRt) {
    let h = config.hidden_size;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let proj = |n: usize, k: usize| Mxfp8Proj {
        w: Mxfp8Weight {
            data: gpu.alloc(n * k).unwrap(),
            scales: gpu.alloc(n * k / 32).unwrap(),
        },
        n: n as u32,
        k: k as u32,
    };
    let projs = [
        proj(2 * q_dim, h),
        proj(kv_dim, h),
        proj(kv_dim, h),
        proj(h, q_dim),
    ];
    let bytes = 2 * q_dim * h * 2;
    let rt = Mx8AttnRt {
        w8a8: [None, None, None, None],
        kernels: Mxfp8ProjKernels::resolve(gpu).unwrap(),
        scratch: Mxfp8Scratch {
            ptr: gpu.alloc(bytes).unwrap(),
            bytes,
        },
    };
    (projs, rt)
}

/// Four verify rows: one weight-streaming launch per projection (grid
/// ceil(N/4)), then three scatter copies a row into `qkv_buf`.
#[test]
fn verify_rows_read_each_mxfp8_projection_once() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut l = layer(&gpu, &config, false);
    let (projs, rt) = projections(&gpu, &config);
    l.set_mx8_projections(projs, rt).unwrap();
    let (copies, launches) = qkv_phase_d2d(&gpu, &config, &l, 4);
    assert_eq!(copies, 3 * 4, "three scatter copies a row");
    let q_rows = 2 * config.num_attention_heads * config.head_dim;
    let kv_rows = config.num_key_value_heads * config.head_dim;
    let grids: Vec<[u32; 3]> = launches.iter().map(|l| l.grid).collect();
    for (what, n) in [("q", q_rows), ("k/v", kv_rows)] {
        assert!(
            grids.contains(&[(n / 4) as u32, 1, 1]),
            "{what} MXFP8 GEMV missing; grids: {grids:?}"
        );
    }
}

/// BF16 slots still holding weights means a second copy: refused.
#[test]
fn a_layer_with_bf16_slots_is_refused() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut l = layer(&gpu, &config, true);
    let (projs, rt) = projections(&gpu, &config);
    let err = l.set_mx8_projections(projs, rt).unwrap_err();
    assert!(err.to_string().contains("another copy"), "{err:#}");
}
