// SPDX-License-Identifier: AGPL-3.0-only
//! Deferred target-only shared cache, after weight replacement and MTP load.
use crate::layer::TransformerLayer;
use crate::layers::moe::SharedFp8Reserve;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::layers::{FfnComponent, Glm5KdaLayer};
use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

#[derive(Clone)]
struct TargetInfo {
    ordinal: usize,
    kind: LayerType,
    moe: bool,
}
fn cache_ordinals(config: &ModelConfig, infos: &[TargetInfo]) -> Result<Vec<usize>> {
    anyhow::ensure!(
        config.num_hidden_layers == 45 && infos.len() == 45 && config.mlp_only_layers == [0, 1, 2],
        "deferred shared FP8 requires exactly 45 target layers with dense 0..2"
    );
    for (i, info) in infos.iter().enumerate() {
        anyhow::ensure!(
            info.ordinal == i
                && info.kind == config.layer_type(i)
                && matches!(
                    info.kind,
                    LayerType::LinearAttention | LayerType::FullAttention
                )
                && info.moe == (i >= 3),
            "deferred shared FP8 layer identity/type/FFN mismatch at {i}"
        );
    }
    Ok(infos
        .iter()
        .filter(|info| info.moe)
        .map(|info| info.ordinal)
        .collect())
}

fn inspect(layer: &mut dyn TransformerLayer) -> Result<(TargetInfo, &mut FfnComponent)> {
    let any = layer
        .as_any_mut()
        .ok_or_else(|| anyhow::anyhow!("deferred shared FP8 unsupported layer wrapper"))?;
    let (ordinal, kind, ffn) = if any.is::<Glm5KdaLayer>() {
        let (ordinal, ffn) = any
            .downcast_mut::<Glm5KdaLayer>()
            .expect("checked KDA type")
            .glm_shared_cache_ffn();
        (ordinal, LayerType::LinearAttention, ffn)
    } else if any.is::<Qwen3AttentionLayer>() {
        let (ordinal, ffn) = any
            .downcast_mut::<Qwen3AttentionLayer>()
            .expect("checked MLA type")
            .glm_shared_cache_ffn();
        (ordinal, LayerType::FullAttention, ffn)
    } else {
        anyhow::bail!("deferred shared FP8 unexpected concrete layer");
    };
    let moe = match ffn {
        FfnComponent::Moe(_) => true,
        FfnComponent::Dense(_) => false,
        FfnComponent::None => anyhow::bail!("deferred shared FP8 missing FFN at {ordinal}"),
    };
    Ok((TargetInfo { ordinal, kind, moe }, ffn))
}

/// No cache work occurs before this post-target/MTP/head boundary. The first
/// pass validates every target identity before any new GPU allocation.
pub(super) fn initialize(
    config: &ModelConfig,
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    layers: &mut [Box<dyn TransformerLayer>],
    reserve: Option<SharedFp8Reserve>,
) -> Result<()> {
    let Some(reserve) = reserve else {
        return Ok(());
    };
    anyhow::ensure!(
        config.model_type == "glm5_next",
        "deferred shared FP8 target model type"
    );
    anyhow::ensure!(
        layers.len() == 45,
        "deferred shared FP8 requires exactly 45 target layers"
    );
    let mut infos = Vec::with_capacity(layers.len());
    for layer in layers.iter_mut() {
        infos.push(inspect(layer.as_mut())?.0);
    }
    let ordinals = cache_ordinals(config, &infos)?;
    let total = reserve.remaining_bytes(ordinals.len())?;
    let free = gpu.free_memory()?;
    reserve.check(free, total)?;
    tracing::info!(
        free,
        total,
        "GLM deferred shared FP8 pass admitted after target/MTP/head loading"
    );
    let variant = crate::weight_map::detect_nvfp4_variant(store, config);
    for ordinal in ordinals {
        let (_, ffn) = inspect(layers[ordinal].as_mut())?;
        let FfnComponent::Moe(moe) = ffn else {
            anyhow::bail!("deferred shared FP8 FFN changed at {ordinal}");
        };
        let prefix = format!("{}.mlp.shared_experts", config.layer_prefix(ordinal));
        moe.maybe_cache_glm_target_shared_fp8(
            store,
            &prefix,
            config,
            ordinal,
            true,
            variant,
            gpu,
            gpu.default_stream(),
            reserve,
        )?;
    }
    reserve.check(gpu.free_memory()?, 0)?;
    Ok(())
}

#[cfg(test)]
#[path = "glm_shared_cache_tests.rs"]
mod tests;
