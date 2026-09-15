// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::rowwise_fp8;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::{DenseFfnLayer, FfnComponent, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, QuantizedWeight, dense, dequant_fp8_blockscaled_to_bf16,
    load_kv_scales,
};

const ATTN_ENV: &str = "ATLAS_FP8_DEQUANT_ATTN_TO_BF16";
const FFN_ENV: &str = "ATLAS_FP8_DEQUANT_FFN_TO_BF16";

fn env_enabled(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

fn component_policy(enabled: bool, tp_size: usize, all_per_row: bool) -> bool {
    enabled && tp_size == 1 && all_per_row
}

fn all_per_row(store: &WeightStore, base: &str, names: &[&str]) -> bool {
    names
        .iter()
        .all(|name| rowwise_fp8::proj_is_fp8_per_row(store, &format!("{base}.{name}")))
}

pub(super) fn preserve_attention(store: &WeightStore, base: &str, tp_size: usize) -> bool {
    let enabled = env_enabled(ATTN_ENV);
    if !enabled || tp_size != 1 {
        return false;
    }
    component_policy(
        enabled,
        tp_size,
        all_per_row(store, base, &["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
}

pub(super) fn preserve_ffn(store: &WeightStore, base: &str, tp_size: usize) -> bool {
    let enabled = env_enabled(FFN_ENV);
    if !enabled || tp_size != 1 {
        return false;
    }
    component_policy(
        enabled,
        tp_size,
        all_per_row(store, base, &["gate_proj", "up_proj", "down_proj"]),
    )
}

pub(super) fn dequant_per_row(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    anyhow::ensure!(
        rowwise_fp8::proj_is_fp8_per_row(store, prefix),
        "{prefix} is not FP8 E4M3 with a [N] or [N,1] scale"
    );
    let source = store.get(&format!("{prefix}.weight"))?.ptr;
    let bf16 = dequant_fp8_blockscaled_to_bf16(store, prefix, gpu)?;
    debug_assert_ne!(
        source, bf16.weight,
        "per-row FP8 dequant must return an independently owned BF16 buffer"
    );
    Ok(bf16)
}

pub(super) fn dequant_ffn(
    store: &WeightStore,
    base: &str,
    gpu: &dyn GpuBackend,
) -> Result<(DenseWeight, DenseWeight, DenseWeight)> {
    Ok((
        dequant_per_row(store, &format!("{base}.gate_proj"), gpu)?,
        dequant_per_row(store, &format!("{base}.up_proj"), gpu)?,
        dequant_per_row(store, &format!("{base}.down_proj"), gpu)?,
    ))
}

pub(super) fn null_ffn_weights() -> DenseFfnWeights {
    DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    }
}

pub(super) fn synchronize_and_reclaim(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    stream: u64,
    prefixes: &[String],
) -> Result<()> {
    gpu.synchronize(stream)?;
    for prefix in prefixes {
        store.reclaim(gpu, &format!("{prefix}.weight"))?;
        store.reclaim(gpu, &format!("{prefix}.weight_scale"))?;
    }
    Ok(())
}

pub(super) fn install_ffn(
    layer: &mut DenseFfnLayer,
    weights: (DenseWeight, DenseWeight, DenseWeight),
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    stream: u64,
    base: &str,
) -> Result<()> {
    layer.set_bf16_weights(weights.0, weights.1, weights.2);
    let sources = [
        format!("{base}.gate_proj"),
        format!("{base}.up_proj"),
        format!("{base}.down_proj"),
    ];
    synchronize_and_reclaim(store, gpu, stream, &sources)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_attention(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    base: &str,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    attn_idx: usize,
    kv_dtype: KvCacheDtype,
    stream: u64,
) -> Result<Qwen3AttentionLayer> {
    let sources = [
        format!("{base}.q_proj"),
        format!("{base}.k_proj"),
        format!("{base}.v_proj"),
        format!("{base}.o_proj"),
    ];
    let q_bf16 = dequant_per_row(store, &sources[0], gpu)?;
    let k_bf16 = dequant_per_row(store, &sources[1], gpu)?;
    let v_bf16 = dequant_per_row(store, &sources[2], gpu)?;
    let o_bf16 = dequant_per_row(store, &sources[3], gpu)?;
    let (k_scale, v_scale) = load_kv_scales(store, base, gpu);
    let attn = AttentionWeights {
        q_proj: q_bf16,
        k_proj: k_bf16,
        v_proj: v_bf16,
        o_proj: QuantizedWeight::null(),
        q_norm: dense(store, &format!("{base}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{base}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    let mut layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        None,
        None,
        None,
        gpu,
        kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    layer.set_o_dense_bf16(o_bf16);
    synchronize_and_reclaim(store, gpu, stream, &sources)?;
    Ok(layer)
}

fn gdn_reclaim_policy(is_hip: bool, tp_size: usize, all_fp8_with_scale: bool) -> bool {
    is_hip && tp_size == 1 && all_fp8_with_scale
}

pub(super) fn reclaim_gdn_bf16_sources(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    stream: u64,
    base: &str,
    tp_size: usize,
) -> Result<()> {
    let names = ["in_proj_qkv", "in_proj_z", "out_proj"];
    let all_fp8_with_scale = names.iter().all(|name| {
        let prefix = format!("{base}.{name}");
        matches!(
            store.get(&format!("{prefix}.weight")).map(|w| w.dtype),
            Ok(WeightDtype::FP8E4M3)
        ) && store.contains(&format!("{prefix}.weight_scale"))
    });
    if gdn_reclaim_policy(cfg!(atlas_hip), tp_size, all_fp8_with_scale) {
        let sources = names.map(|name| format!("{base}.{name}"));
        synchronize_and_reclaim(store, gpu, stream, &sources)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{component_policy, gdn_reclaim_policy};

    #[test]
    fn preservation_requires_opt_in_tp1_and_all_projections() {
        assert!(component_policy(true, 1, true));
        assert!(!component_policy(false, 1, true));
        assert!(!component_policy(true, 2, true));
        assert!(!component_policy(true, 1, false));
    }

    #[test]
    fn gdn_reclaim_requires_hip_tp1_and_complete_fp8_sources() {
        assert!(gdn_reclaim_policy(true, 1, true));
        assert!(!gdn_reclaim_policy(false, 1, true));
        assert!(!gdn_reclaim_policy(true, 2, true));
        assert!(!gdn_reclaim_policy(true, 1, false));
    }
}
