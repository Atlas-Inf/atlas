// SPDX-License-Identifier: AGPL-3.0-only

//! Router kernel-name selection for `MoeLayer` initialization.

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
