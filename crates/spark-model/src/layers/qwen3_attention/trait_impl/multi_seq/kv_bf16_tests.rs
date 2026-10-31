// SPDX-License-Identifier: AGPL-3.0-only

use super::super::kv_bf16::project_rows;
use super::*;

#[test]
fn kv_bf16_batches_skip_q_and_cover_tail_rows() {
    let gpu = MockGpuBackend::new();
    let k = DenseWeight {
        weight: DevicePtr(0x1000),
    };
    let v = DenseWeight {
        weight: DevicePtr(0x2000),
    };
    for n in [1usize, 2, 7, 8, 9, 16, 64, 257, 16385] {
        let before = gpu.launch_count();
        project_rows(
            &gpu,
            KernelHandle(42),
            DevicePtr(0x3000),
            [&k, &v],
            DevicePtr(0x4000),
            n,
            2560,
            256,
            6144,
            6656,
            0,
        )
        .unwrap();
        let launches = gpu.launches_snapshot();
        let added = &launches[before..];
        assert_eq!(added.len(), 2 * n.div_ceil(8), "n={n}");
        assert!(
            added
                .iter()
                .all(|l| l.func == 42 && l.grid == [64, 1, 1] && l.args == 7)
        );
    }
}

#[test]
fn kv_bf16_requires_plain_qwen_weights_and_kernel() {
    let gpu = MockGpuBackend::new();
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let mut layer = attn_layer(&gpu, &config);
    assert!(!layer.kv_bf16_batchm_supported("qwen4_exp"));
    layer.q_weight = None;
    layer.k_weight = None;
    layer.v_weight = None;
    layer.gated = false;
    assert!(layer.kv_bf16_batchm_supported("qwen4_exp"));
    assert!(!layer.kv_bf16_batchm_supported("glm5_next"));
    assert!(!layer.kv_bf16_batchm_supported("qwen3_next"));
    layer.gated = true;
    assert!(layer.kv_bf16_batchm_supported("qwen4_exp"));
    layer.gated = false;
    layer.q_weight = Some(crate::weight_map::QuantWeight::Dense(layer.attn.q_proj));
    assert!(!layer.kv_bf16_batchm_supported("qwen4_exp"));
    layer.q_weight = None;
    layer.dense_gemv_batchm_k = KernelHandle(0);
    assert!(!layer.kv_bf16_batchm_supported("qwen4_exp"));
    layer.dense_gemv_batchm_k = KernelHandle(42);
    layer.attn.k_proj.weight = DevicePtr::NULL;
    assert!(!layer.kv_bf16_batchm_supported("qwen4_exp"));
    layer.attn.k_proj.weight = DevicePtr(0x1000);
    // A native MXFP8 projection is not a plain BF16 one.
    layer.k_weight = Some(crate::weight_map::QuantWeight::Mxfp8(
        crate::weight_map::Mxfp8Proj {
            w: crate::weight_map::Mxfp8Weight {
                data: DevicePtr(0x2000),
                scales: DevicePtr(0x3000),
                tiled: false,
            },
            n: 256,
            k: 2560,
        },
    ));
    assert!(!layer.kv_bf16_batchm_supported("qwen4_exp"));
}
