// SPDX-License-Identifier: AGPL-3.0-only

//! Repro/regression for the long-prompt prefill crash: a 3052-token
//! prompt (serve chunked prefill, 512-row scratch segments) faults with
//! CUDA_ERROR_ILLEGAL_ADDRESS in `ling_moe_gate_up_silu`
//! (grid=[6,768,1], items = max_items(512) = 512 + 512*8/16 = 768) at
//! model layer 5 — the FIRST gated-MLA layer — while the same 512-row
//! segment passes through the KDA-attended MoE layers 2..4. `#[ignore]`d:
//!
//! ```sh
//! ATLAS_LING_CKPT=/home/spark/ling/Ling-3.2-air-NVFP4 \
//! cargo test --release -p spark-model --lib \
//!     bailing::layers::mla_moe_prefill_repro_gpu -- --ignored --nocapture
//! ```
//!
//! Drives layer 5 exactly as the 512-row prefill segment does:
//! `LingMlaLayer::attention_half_prefill` (the whole-prompt contiguous
//! path) then the layer's own `ffn_half` (post-norm -> MoE rows=512 ->
//! residual), staging snapshots between halves and reporting a
//! non-finite scan of every intermediate so the first bad side of the
//! attention/FFN seam is visible directly.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, ensure};
use atlas_core::config::parse_config;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::weights::{WeightStore, WeightTensor};

use super::layers_moe_parity_gpu::{Ckpt, gpu_backend, widen, x_rows};
use crate::model::bailing_c1::{FfnKind, LingKernels, LingMlaLayer, LingScratch, LingSpec};
use crate::weight_map::dense;

const HIDDEN: usize = 2560;
const ROWS: usize = 512; // the serve scratch segment size this crash rides on
const EXPS: usize = 512;
const TOP_K: usize = 8;
const MLB: usize = 576; // MLA K row: 512 latent + 64 rope
const BLOCKS: usize = ROWS / 8;

#[test]
#[ignore = "needs a GB10, a ling-3.2-air kernel build and ATLAS_LING_CKPT"]
fn ling_mla_moe_prefill_repro_gpu() -> Result<()> {
    let gpu = gpu_backend()?;
    let dir = std::env::var("ATLAS_LING_CKPT")
        .map(Into::into)
        .unwrap_or_else(|_| "/home/spark/ling/Ling-3.2-air-NVFP4".to_owned());
    let config = parse_config(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
    let spec = LingSpec::from_config(&config)?;
    ensure!(
        spec.layers[5].ffn == FfnKind::Moe,
        "layer 5 is not a MoE layer"
    );
    let kernels = Arc::new(LingKernels::resolve(gpu)?);
    let scratch = LingScratch::arc(gpu, ROWS, HIDDEN, EXPS, TOP_K, 6144, BLOCKS)?;

    let mut ckpt = Ckpt::open()?;
    let layer = build_layer(gpu, &mut ckpt, &spec, &kernels, &scratch, 5)?;

    let kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: 8,
            num_kv_heads: 1,
            head_dim: MLB,
            num_layers: 7,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        BLOCKS,
        gpu,
    )?;
    let block_table: Vec<u32> = (0..BLOCKS as u32).collect();

    let stream = gpu.default_stream();
    let x = x_rows(ROWS, 5);
    let bits: Vec<u16> = x
        .iter()
        .map(|v| atlas_core::numeric::f32_to_bf16(*v))
        .collect();
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    let hidden = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, hidden)?;
    report_hidden(gpu, hidden, ROWS, HIDDEN, "input hidden")?;

    // ── attention half: the whole-prompt contiguous prefill path.
    layer.attention_half_prefill(gpu, hidden, &mut kv, ROWS, 0, true, &block_table, stream)?;
    gpu.synchronize(stream)?;
    report_hidden(gpu, hidden, ROWS, HIDDEN, "post-attention hidden")?;

    // ── FFN half: post-norm -> MoE rows=512 -> residual.
    layer.ffn_half(gpu, hidden, ROWS as u32, stream)?;
    gpu.synchronize(stream)?;
    report_hidden(gpu, hidden, ROWS, HIDDEN, "final hidden (post-FFN)")?;
    eprintln!("layer 5 attention+MoE prefill at rows={ROWS}: CLEAN");
    Ok(())
}

fn build_layer(
    gpu: &dyn GpuBackend,
    ckpt: &mut Ckpt,
    spec: &LingSpec,
    kernels: &Arc<LingKernels>,
    scratch: &Arc<LingScratch>,
    layer: usize,
) -> Result<LingMlaLayer> {
    let lp = format!("model.layers.{layer}");
    let mut names: Vec<String> = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        .iter()
        .map(|s| format!("model.layers.{layer}.{s}"))
        .collect();
    for stem in [
        "q_proj",
        "kv_a_proj_with_mqa",
        "kv_a_layernorm",
        "g_proj",
        "dense",
        "kv_b_proj",
    ] {
        names.push(format!("{lp}.attention.{stem}.weight"));
    }
    names.push(format!("{lp}.mlp.gate.weight"));
    names.push(format!("{lp}.mlp.gate.expert_bias"));
    for p in ["gate_proj", "up_proj", "down_proj"] {
        names.push(format!("{lp}.mlp.shared_experts.{p}.weight"));
        for e in 0..spec.experts {
            for part in [
                "weight_packed",
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            ] {
                names.push(format!("{lp}.mlp.experts.{e}.{p}.{part}"));
            }
        }
    }
    let store = upload(gpu, ckpt, &names)?;
    let weights = super::load_mla_weights(&store, &lp, gpu)?;
    let ffn = super::load_ffn(&store, &lp, spec.layers[layer], spec, gpu, kernels, scratch)?;
    let input_norm = dense(
        &store,
        &format!("model.layers.{layer}.input_layernorm.weight"),
    )?;
    let post_norm = dense(
        &store,
        &format!("model.layers.{layer}.post_attention_layernorm.weight"),
    )?;
    LingMlaLayer::new(
        input_norm,
        post_norm,
        weights,
        ffn,
        Arc::clone(kernels),
        Arc::clone(scratch),
        0,
        &spec_model_config()?,
    )
}

fn spec_model_config() -> Result<atlas_core::config::ModelConfig> {
    let dir = std::env::var("ATLAS_LING_CKPT")
        .map(Into::into)
        .unwrap_or_else(|_| "/home/spark/ling/Ling-3.2-air-NVFP4".to_owned());
    Ok(parse_config(&std::fs::read_to_string(format!(
        "{dir}/config.json"
    ))?)?)
}

fn upload(gpu: &dyn GpuBackend, ckpt: &mut Ckpt, names: &[String]) -> Result<WeightStore> {
    let mut map = HashMap::new();
    for name in names {
        let (dtype, shape, data) = ckpt.tensor(name)?;
        let ptr = gpu.alloc(data.len().max(1))?;
        gpu.copy_h2d(&data, ptr)?;
        map.insert(name.clone(), WeightTensor { ptr, dtype, shape });
    }
    Ok(WeightStore::from_map(map))
}

/// Non-finite scan of one `[rows, cols]` BF16 buffer.
fn report_hidden(
    gpu: &dyn GpuBackend,
    p: DevicePtr,
    rows: usize,
    cols: usize,
    label: &str,
) -> Result<()> {
    let mut b = vec![0u8; rows * cols * 2];
    gpu.copy_d2h(p, &mut b)?;
    let vals: Vec<f32> = b
        .chunks_exact(2)
        .map(|c| widen(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let nonfinite = vals.iter().filter(|v| !v.is_finite()).count();
    let max = vals.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "{label}: {} values, {} non-finite, max |v| = {max:.3}",
        vals.len(),
        nonfinite
    );
    if nonfinite > 0 {
        for (i, v) in vals.iter().enumerate() {
            if !v.is_finite() {
                let (r, c) = (i / cols, i % cols);
                eprintln!("  first non-finite at row {r} col {c}: {v}");
                break;
            }
        }
    }
    Ok(())
}
