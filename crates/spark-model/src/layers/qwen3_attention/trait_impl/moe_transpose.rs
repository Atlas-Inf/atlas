// SPDX-License-Identifier: AGPL-3.0-only

//! `TransformerLayer::transpose_moe_for_prefill_hybrid` and
//! `set_moe_down_transpose_scratch` for `Qwen3AttentionLayer`, split from `trait_impl.rs` (500-LoC cap) per the
//! `*_inner` delegation idiom: a trait impl cannot span files, so the body
//! lives here and the method in `trait_impl.rs` delegates one call.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Qwen3AttentionLayer;
use crate::layers::FfnComponent;

pub(super) fn transpose_moe_for_prefill_hybrid_inner(
    layer: &mut Qwen3AttentionLayer,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> Result<()> {
    if let FfnComponent::Moe(moe) = &mut layer.ffn {
        moe.transpose_for_prefill_hybrid(gpu, config)?;
    }
    if let Some(FfnComponent::Moe(moe)) = layer.moe_ffn.as_mut() {
        moe.transpose_for_prefill_hybrid(gpu, config)?;
    }
    Ok(())
}

pub(super) fn set_moe_down_transpose_scratch_inner(
    layer: &mut Qwen3AttentionLayer,
    [scratch_packed, scratch_scale, packed_ptrs_t, scale_ptrs_t]: [DevicePtr; 4],
) {
    if let FfnComponent::Moe(moe) = &mut layer.ffn {
        moe.set_down_transpose_scratch(scratch_packed, scratch_scale, packed_ptrs_t, scale_ptrs_t);
    }
    if let Some(FfnComponent::Moe(moe)) = layer.moe_ffn.as_mut() {
        moe.set_down_transpose_scratch(scratch_packed, scratch_scale, packed_ptrs_t, scale_ptrs_t);
    }
}
