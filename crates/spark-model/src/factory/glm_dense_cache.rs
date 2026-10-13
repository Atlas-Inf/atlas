// SPDX-License-Identifier: AGPL-3.0-only
//! Deferred prefill-only dense FFN cache, accounted before arena and KV pools.
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;

pub(crate) const CACHE_BYTES: usize = 9 * 12288 * 4096 * 2;
fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1" | "2") => Ok(true),
        _ => anyhow::bail!("ATLAS_GLM_DENSE_PREFILL_BF16 must be 0, 1 or 2"),
    }
}

/// `ATLAS_GLM_DENSE_PREFILL_BF16=2`: the same BF16 prefill GEMMs, with each
/// weight dequantized into idle arena scratch just before its GEMM instead of
/// a persistent 906 MB cache (bit-identical weights; ~0.5 ms per projection).
pub(crate) fn transient() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_DENSE_PREFILL_BF16").as_deref() == Ok("2"))
}
fn required(arena: usize, inference: usize) -> Result<usize> {
    arena
        .checked_add(inference)
        .map(|future| future.max(4 * 1024 * 1024 * 1024))
        .and_then(|future| future.checked_add(CACHE_BYTES))
        .ok_or_else(|| anyhow::anyhow!("dense BF16 prefill reserve overflow"))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn initialize(
    config: &ModelConfig,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    layers: &mut [Box<dyn crate::layer::TransformerLayer>],
    max_batch_tokens: usize,
    max_seq_len: usize,
    block_size: usize,
    max_batch_size: usize,
    inference: usize,
) -> Result<()> {
    if config.model_type != "glm5_next" {
        return Ok(());
    }
    let enabled = match std::env::var("ATLAS_GLM_DENSE_PREFILL_BF16") {
        Ok(value) => parse(Some(&value))?,
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => return Err(error.into()),
    };
    if !enabled {
        return Ok(());
    }
    validate(config)?;
    if transient() {
        tracing::info!("GLM dense BF16 prefill: transient per-call dequant (no persistent cache)");
        return Ok(());
    }
    ensure!(
        crate::layers::ops::GemmDispatch::from_env().cublas_gemm,
        "dense BF16 prefill cache requires ATLAS_CUBLAS_GEMM=1"
    );
    ensure!(
        config.adapter_max_rank == 0,
        "dense BF16 cache excludes adapters"
    );
    ensure!(
        layers.len() == 45,
        "dense BF16 cache requires exactly45 target layers"
    );
    // Validate all identities before allocating even the first projection.
    for (ordinal, layer) in layers.iter_mut().take(3).enumerate() {
        dense_target(layer.as_mut(), ordinal)?;
    }
    let arena = spark_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens,
        max_seq_len,
        block_size,
        max_batch_size,
    )
    .total_bytes();
    let required = required(arena, inference)?;
    let protected = required - CACHE_BYTES;
    ensure!(
        gpu.free_memory()? >= required,
        "dense BF16 cache needs {required} bytes including future arena/inference"
    );
    for (ordinal, layer) in layers.iter_mut().take(3).enumerate() {
        let remaining = CACHE_BYTES / 3 * (3 - ordinal);
        ensure!(
            gpu.free_memory()? >= protected + remaining,
            "dense BF16 cache memory changed during deferred allocation"
        );
        dense_target(layer.as_mut(), ordinal)?.cache_glm_prefill_bf16(config, gpu)?;
    }
    ensure!(
        gpu.free_memory()? >= protected,
        "dense BF16 cache violated future memory reserve"
    );
    tracing::info!(
        bytes = CACHE_BYTES,
        arena,
        inference,
        protected,
        "GLM dense BF16 prefill cache installed before KV accounting"
    );
    Ok(())
}
fn dense_target(
    layer: &mut dyn crate::layer::TransformerLayer,
    ordinal: usize,
) -> Result<&mut crate::layers::DenseFfnLayer> {
    let kda = layer
        .as_any_mut()
        .and_then(|any| any.downcast_mut::<crate::layers::Glm5KdaLayer>())
        .ok_or_else(|| anyhow::anyhow!("dense BF16 target layer{ordinal} is not GLM KDA"))?;
    let (actual, ffn) = kda.glm_shared_cache_ffn();
    ensure!(
        actual == ordinal && ordinal < 3,
        "dense BF16 target ordinal mismatch"
    );
    match ffn {
        crate::layers::FfnComponent::Dense(dense) => Ok(dense),
        _ => anyhow::bail!("dense BF16 target layer{ordinal} missing dense FFN"),
    }
}
pub(crate) fn validate(config: &ModelConfig) -> Result<()> {
    ensure!(
        config.model_type == "glm5_next"
            && config.num_hidden_layers == 45
            && config.hidden_size == 4096
            && config.intermediate_size == 12288
            && config.mlp_only_layers == [0, 1, 2],
        "dense BF16 prefill cache requires GLM-5.3 target geometry"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_prefill_bf16_reserve_counts_nine_full_weights_plus_future_once() {
        assert_eq!(CACHE_BYTES, 864 * 1024 * 1024);
        assert_eq!(
            required(1259374672, 7625510912).unwrap(),
            8884885584 + CACHE_BYTES
        );
        assert_eq!(
            required(0, 0).unwrap(),
            4 * 1024 * 1024 * 1024 + CACHE_BYTES
        );
        assert!(required(usize::MAX, 1).is_err());
    }
    #[test]
    fn dense_prefill_bf16_target_geometry() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(validate(&c).is_err());
        c.model_type = "glm5_next".into();
        c.num_hidden_layers = 45;
        c.hidden_size = 4096;
        c.intermediate_size = 12288;
        c.mlp_only_layers = vec![0, 1, 2];
        assert!(validate(&c).is_ok());
        c.mlp_only_layers.push(45);
        assert!(validate(&c).is_err());
    }
    #[test]
    fn dense_prefill_bf16_explicit_flag() {
        assert!(!parse(None).unwrap());
        assert!(!parse(Some("0")).unwrap());
        assert!(parse(Some("1")).unwrap());
        for invalid in ["", "true", "3"] {
            assert!(parse(Some(invalid)).is_err());
        }
    }
}
