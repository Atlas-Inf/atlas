// SPDX-License-Identifier: AGPL-3.0-only

//! Ling kernel handles, resolved once per model and shared by every layer
//! (module names are file stems of the `ling-3.2-air` target plus the
//! common modules — the table in `model/bailing_c1.rs`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Every kernel the Ling decode path launches. Resolved eagerly at load so
/// a target that missed a module fails loudly there, never mid-forward.
pub(crate) struct LingKernels {
    // common
    pub rms_norm_k: KernelHandle,
    pub residual_add_k: KernelHandle,
    pub gemv_k: KernelHandle,
    pub gemm_k: KernelHandle,
    pub conv_update_k: KernelHandle,
    pub conv_prefill_k: KernelHandle,
    pub moe_transpose_batched_k: KernelHandle,
    /// Common `bf16_to_f32` (module `residual_add`): the primitive-ai build
    /// stores the FP32-routing gate/expert-bias tensors in BF16, promoted at
    /// load time (512 x hidden each — a few MB per model).
    pub bf16_to_f32_k: KernelHandle,
    // kda.cu
    pub kda_recurrent_k: KernelHandle,
    pub kda_gated_norm_k: KernelHandle,
    // gated_mla.cu
    pub mla_kv_a_cache_k: KernelHandle,
    pub mla_q_absorb_k: KernelHandle,
    pub mla_paged_attn_k: KernelHandle,
    pub mla_v_gated_k: KernelHandle,
    pub mla_prefill_attn_k: KernelHandle,
    // moe_decode.cu
    pub moe_route_k: KernelHandle,
    pub moe_gate_up_k: KernelHandle,
    pub moe_down_k: KernelHandle,
    pub moe_unpermute_k: KernelHandle,
    pub shared_gate_up_k: [KernelHandle; 5],
    pub shared_down_k: [KernelHandle; 5],
}

/// Resolve kernels over the five row buckets {1, 4, 8, 16, 32}.
fn five(f: impl FnMut(u32) -> Result<KernelHandle>) -> Result<[KernelHandle; 5]> {
    let handles: Vec<KernelHandle> = [1u32, 4, 8, 16, 32]
        .map(f)
        .into_iter()
        .collect::<Result<_>>()?;
    handles
        .try_into()
        .map_err(|_| anyhow::anyhow!("ling: five buckets"))
}

fn required(gpu: &dyn GpuBackend, module: &str, name: &str) -> Result<KernelHandle> {
    let k = gpu.kernel(module, name)?;
    ensure!(
        k.0 != 0,
        "ling: kernel {module}:{name} missing from the target"
    );
    Ok(k)
}

impl LingKernels {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            rms_norm_k: required(gpu, "rms_norm_vanilla", "rms_norm_vanilla")?,
            residual_add_k: required(gpu, "residual_add", "bf16_residual_add")?,
            gemv_k: required(gpu, "gemv", "dense_gemv_bf16")?,
            gemm_k: required(gpu, "gemm", "dense_gemm_bf16")?,
            conv_update_k: required(gpu, "causal_conv1d", "causal_conv1d_update")?,
            conv_prefill_k: required(gpu, "causal_conv1d", "causal_conv1d_update_prefill")?,
            moe_transpose_batched_k: required(
                gpu,
                "moe_transpose_batched",
                "moe_transpose_u8_batched",
            )?,
            bf16_to_f32_k: required(gpu, "residual_add", "bf16_to_f32")?,
            kda_recurrent_k: required(gpu, "kda", "kda_recurrent_bf16")?,
            kda_gated_norm_k: required(gpu, "kda", "kda_sigmoid_gated_rms_norm")?,
            mla_kv_a_cache_k: required(gpu, "gated_mla", "ling_mla_kv_a_norm_rope_cache")?,
            mla_q_absorb_k: required(gpu, "gated_mla", "ling_mla_q_absorb_rope")?,
            mla_paged_attn_k: required(gpu, "gated_mla", "ling_mla_paged_attn")?,
            mla_v_gated_k: required(gpu, "gated_mla", "ling_mla_v_extract_gated")?,
            mla_prefill_attn_k: required(gpu, "gated_mla", "ling_mla_prefill_attn")?,
            moe_route_k: required(gpu, "moe_decode", "ling_moe_route")?,
            moe_gate_up_k: required(gpu, "moe_decode", "ling_moe_gate_up_silu")?,
            moe_down_k: required(gpu, "moe_decode", "ling_moe_down")?,
            moe_unpermute_k: required(gpu, "moe_decode", "ling_moe_unpermute")?,
            shared_gate_up_k: five(|mr| {
                required(gpu, "moe_decode", &format!("ling_moe_shared_gate_up_r{mr}"))
            })?,
            shared_down_k: five(|mr| {
                required(gpu, "moe_decode", &format!("ling_moe_shared_down_r{mr}"))
            })?,
        })
    }
}
