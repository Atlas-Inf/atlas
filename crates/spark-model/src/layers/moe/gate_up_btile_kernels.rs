// SPDX-License-Identifier: AGPL-3.0-only
//! Private staged family; no serving lookup or raw handle publication.
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

pub(super) struct KernelFamily<'g> {
    pub(super) gpu: &'g dyn GpuBackend,
    pub(super) handles: [KernelHandle; 16],
}
impl<'g> KernelFamily<'g> {
    pub(super) fn resolve(
        gpu: &'g dyn GpuBackend,
        config: &ModelConfig,
        stream: u64,
    ) -> Result<Self> {
        ensure!(
            !gpu.stream_is_capturing(stream),
            "B-tile resolution during capture"
        );
        validate_profile(config)?;
        let mut handles = [KernelHandle(0); 16];
        for (slot, &(module, name)) in handles.iter_mut().zip(EXPORTS.iter()) {
            *slot = gpu.kernel(module, name)?;
            ensure!(slot.0 != 0, "missing B-tile export {module}::{name}");
        }
        Ok(Self { gpu, handles })
    }
}
pub(super) fn validate_profile(config: &ModelConfig) -> Result<()> {
    ensure!(
        config.model_type == "glm5_next"
            && config.hidden_size == 4096
            && config.moe_intermediate_size == 2048
            && config.shared_expert_intermediate_size == 2048
            && config.num_experts == 288
            && config.num_experts_per_tok == 8
            && config.tp_world_size == 2
            && config.ep_world_size == 2
            && config.ep_rank < 2
            && config.tp_rank == config.ep_rank
            && config.adapter_max_rank == 0
            && (config.weight_prefix.is_empty() || config.weight_prefix == "model"),
        "unsupported B-tile family profile"
    );
    Ok(())
}
const EXPORTS: [(&str, &str); 16] = [
    ("glm_moe_btile_native_repack", "glm_native_to_btile_u8"),
    ("transpose_u8", "transpose_u8"),
    ("glm_moe_btile_decode", "glm_btile_decode_word1"),
    ("glm_moe_btile_decode", "glm_btile_decode_word2"),
    ("glm_moe_btile_decode", "glm_btile_decode_word3"),
    ("glm_moe_btile_decode", "glm_btile_decode_vec1"),
    ("glm_moe_btile_decode", "glm_btile_decode_vec2"),
    ("glm_moe_btile_decode", "glm_btile_decode_vec3"),
    ("moe_w4a16", "glm_moe_gate_up_btile"),
    ("moe_w4a16", "glm_moe_gate_up_btile_vecscale"),
    ("moe_w4a16", "glm_moe_gate_up_btile_m64"),
    ("moe_w4a16", "glm_moe_gate_up_btile_m64_vecscale"),
    ("moe_w4a16", "glm_moe_btile_m64_dense"),
    ("moe_w4a16", "glm_moe_btile_m64_vecscale_dense"),
    ("moe_w4a16", "glm_moe_btile_m64_compact"),
    ("moe_w4a16", "glm_moe_btile_m64_vecscale_compact"),
];

#[cfg(test)]
#[path = "gate_up_btile_kernels_tests.rs"]
mod tests;
