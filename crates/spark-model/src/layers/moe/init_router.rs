// SPDX-License-Identifier: AGPL-3.0-only

//! Router kernel-name selection for `MoeLayer` initialization.

use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Validates the FP32 router mode and returns the selected mode.
pub(super) fn resolve(router_f32: bool, gate_is_bf16: bool) -> anyhow::Result<bool> {
    anyhow::ensure!(
        !router_f32 || gate_is_bf16,
        "FP32 MoE router logits need a BF16 router weight (the NVFP4 router GEMVs store BF16)"
    );
    Ok(router_f32)
}

pub(super) fn kernel_name(router_f32: bool, name: &str) -> String {
    if router_f32 {
        format!("{name}_f32")
    } else {
        name.to_string()
    }
}

/// The router's single-row GEMV: `dense` unless logits are FP32.
pub(super) fn gemv(
    gpu: &dyn GpuBackend,
    router_f32: bool,
    dense: KernelHandle,
) -> anyhow::Result<KernelHandle> {
    if router_f32 {
        gpu.kernel("gemv", "dense_gemv_bf16_fp32out")
    } else {
        Ok(dense)
    }
}

/// The router's GEMM: `dense` unless logits are FP32.
pub(super) fn gemm(
    gpu: &dyn GpuBackend,
    router_f32: bool,
    dense: KernelHandle,
) -> anyhow::Result<KernelHandle> {
    if router_f32 {
        gpu.kernel("gemm", "dense_gemm_bf16_f32out")
    } else {
        Ok(dense)
    }
}

/// The router-specialised GEMM, where this build has one.
pub(super) fn gemm_batched(gpu: &dyn GpuBackend, router_f32: bool) -> KernelHandle {
    let name = if router_f32 {
        "dense_gemm_bf16_router_f32out"
    } else {
        "dense_gemm_bf16_router"
    };
    super::super::try_kernel(gpu, "gemm", name)
}
