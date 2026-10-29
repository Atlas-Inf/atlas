// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch of native MXFP8 GDN projections (`mx8.rs`) on the mock backend.

use super::mx8::Mx8Gdn;
use super::tests::{gpu_launch_grids, run_batched_verify};
use super::*;
use crate::layers::ops::{Mxfp8Proj, Mxfp8ProjKernels, Mxfp8Scratch};
use crate::weight_map::{Mxfp8Weight, WeightQuantFormat};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::mock::MockGpuBackend;

/// A GDN layer wired like `gdn_mx8::build`: every other projection slot
/// empty, the MXFP8 pair installed.
fn mx8_layer(gpu: &MockGpuBackend, config: &ModelConfig) -> Qwen3SsmLayer {
    let mut layer = bare_layer(gpu, config);
    layer.set_mx8_projections(mx8(gpu, config)).unwrap();
    layer
}

fn bare_layer(gpu: &MockGpuBackend, config: &ModelConfig) -> Qwen3SsmLayer {
    let h = config.hidden_size;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    let ssm = SsmWeights {
        in_proj_qkvz: DenseWeight {
            weight: DevicePtr::NULL,
        },
        in_proj_ba: dw(config.ssm_ba_size() * h * 2),
        conv1d: dw(
            (2 * config.linear_num_key_heads * config.linear_key_head_dim + value_dim)
                * config.linear_conv_kernel_dim
                * 2,
        ),
        a_log: dw(config.linear_num_value_heads * 4),
        dt_bias: dw(config.linear_num_value_heads * 4),
        norm: dw(config.linear_value_head_dim * 2),
        out_proj: QuantizedWeight::null(),
    };
    Qwen3SsmLayer::new_sequential(
        dw(h * 2),
        ssm,
        dw(h * 2),
        FfnComponent::None,
        None,
        None,
        None,
        config,
        gpu,
    )
    .unwrap()
}

fn mx8(gpu: &MockGpuBackend, config: &ModelConfig) -> Mx8Gdn {
    let h = config.hidden_size;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let proj = |n: usize, k: usize| Mxfp8Proj {
        w: Mxfp8Weight {
            data: gpu.alloc(n * k).unwrap(),
            scales: gpu.alloc(n * k / 32).unwrap(),
        },
        n: n as u32,
        k: k as u32,
    };
    let qkvz = proj(config.ssm_qkvz_size(), h);
    let out = proj(h, value_dim);
    let bytes = qkvz.bf16_bytes().max(out.bf16_bytes());
    Mx8Gdn {
        qkvz,
        out,
        kernels: Mxfp8ProjKernels::resolve(gpu).unwrap(),
        scratch: Mxfp8Scratch {
            ptr: gpu.alloc(bytes).unwrap(),
            bytes,
        },
        w8a8: None,
    }
}

/// A 7-row verify (two sequences, K=4 and K=3) reads each MXFP8 projection
/// with the weight-streaming GEMV — one launch of grid ceil(N/4) per
/// projection — and never expands a weight.
#[test]
fn verify_rows_stream_the_mxfp8_weights() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = mx8_layer(&gpu, &config);
    run_batched_verify(&gpu, &config, &layer, &[4, 3]).unwrap();
    let seen = gpu_launch_grids(&gpu);
    let qkvz = config.ssm_qkvz_size() as u32;
    let h = config.hidden_size as u32;
    for (what, n) in [("QKVZ", qkvz), ("out_proj", h)] {
        assert!(
            seen.contains(&([n / 4, 1, 1], [256, 1, 1])),
            "{what} MXFP8 GEMV launch missing; grids seen: {seen:?}"
        );
    }
    let dequant_grid = (qkvz * h / 16).div_ceil(256);
    assert!(
        !seen.iter().any(|(g, _)| g[0] == dequant_grid),
        "a verify window must not expand the weight: {seen:?}"
    );
}

/// The MXFP8 pair must be the layer's only copy of its projections.
#[test]
fn a_second_projection_copy_is_refused() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let h = config.hidden_size;
    let mut layer = bare_layer(&gpu, &config);
    layer.set_fp8_decode_weights(
        Some(Fp8Weight {
            weight: gpu.alloc(config.ssm_qkvz_size() * h).unwrap(),
            row_scale: gpu.alloc(4).unwrap(),
            n: config.ssm_qkvz_size() as u32,
            k: h as u32,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        }),
        None,
    );
    let err = layer.set_mx8_projections(mx8(&gpu, &config)).unwrap_err();
    assert!(err.to_string().contains("another copy"), "{err:#}");
}

/// A scratch too small for the prefill expansion is refused at build.
#[test]
fn an_undersized_scratch_is_refused() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut layer = bare_layer(&gpu, &config);
    let mut m = mx8(&gpu, &config);
    m.scratch.bytes = m.qkvz.bf16_bytes() - 2;
    let err = layer.set_mx8_projections(m).unwrap_err();
    assert!(err.to_string().contains("prefill scratch"), "{err:#}");
}

/// `ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE` pipes out_proj by row ranges only
/// where a range's rows are the whole pass's bytes: the dense BF16 arm, or
/// MXFP8 on W8A8 (per-row quant, in-order kernels). MXFP8 on the GEMV tiers
/// and the expansion is not: a short range would take the other family.
#[test]
fn only_w8a8_mxfp8_out_proj_pipes_by_row_ranges() {
    use crate::layers::ops::MxW8a8;
    use crate::layers::ops::qwen4exp_w8a8::MxPrefill;
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut layer = bare_layer(&gpu, &config);
    assert!(!layer.out_proj_rows_independent(), "no out_proj at all");
    let mut m = mx8(&gpu, &config);
    let op = |p: &Mxfp8Proj| MxW8a8 {
        data: p.w.data,
        sf: DevicePtr(0x1),
        n: p.n as usize,
        k: p.k as usize,
        pinned: false,
    };
    let w8a8 = MxPrefill {
        ops: [op(&m.qkvz), op(&m.out)],
        quant: spark_runtime::gpu::KernelHandle(7),
    };
    layer.set_mx8_projections(m).unwrap();
    assert!(
        !layer.out_proj_rows_independent(),
        "MXFP8 on the GEMV tiers"
    );
    let mut layer = bare_layer(&gpu, &config);
    m = mx8(&gpu, &config);
    m.w8a8 = Some(w8a8);
    layer.set_mx8_projections(m).unwrap();
    assert!(layer.out_proj_rows_independent(), "MXFP8 on W8A8");
}
