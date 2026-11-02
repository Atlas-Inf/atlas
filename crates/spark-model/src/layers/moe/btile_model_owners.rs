// SPDX-License-Identifier: AGPL-3.0-only
//! Actual GLM FFN traversal at arena adoption and before model owner release.
use crate::{
    layer::TransformerLayer,
    layers::{FfnComponent, Glm5KdaLayer, qwen3_attention::Qwen3AttentionLayer},
};
use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::{buffers::BufferArena, gpu::GpuBackend, weights::WeightStore};

fn visit(
    config: &ModelConfig,
    layers: &mut [Box<dyn TransformerLayer>],
    mut apply: impl FnMut(&mut super::MoeLayer) -> Result<()>,
) -> Result<()> {
    if config.model_type != "glm5_next" {
        return Ok(());
    }
    for layer in layers {
        let Some(any) = layer.as_any_mut() else {
            continue;
        };
        let ffn = if any.is::<Glm5KdaLayer>() {
            any.downcast_mut::<Glm5KdaLayer>()
                .expect("checked KDA")
                .glm_shared_cache_ffn()
                .1
        } else if any.is::<Qwen3AttentionLayer>() {
            any.downcast_mut::<Qwen3AttentionLayer>()
                .expect("checked attention")
                .glm_shared_cache_ffn()
                .1
        } else {
            continue;
        };
        if let FfnComponent::Moe(moe) = ffn {
            apply(moe)?;
        }
    }
    Ok(())
}

pub(crate) fn bind(
    config: &ModelConfig,
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    layers: &mut [Box<dyn TransformerLayer>],
    arena: &BufferArena,
) -> Result<()> {
    visit(config, layers, |moe| {
        moe.bind_btile_arena_if_resident(store, config, gpu, arena, gpu.default_stream())
    })
}

pub(crate) fn invalidate(config: &ModelConfig, layers: &mut [Box<dyn TransformerLayer>]) {
    // This walk performs no fallible GPU work and never changes Legacy FFNs.
    let _ = visit(config, layers, |moe| {
        moe.invalidate_btile_before_release();
        Ok(())
    });
}
