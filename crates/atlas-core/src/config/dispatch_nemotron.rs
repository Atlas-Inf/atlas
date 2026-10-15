// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H model-type parsers for [`super::parse_config`]. Split out of
//! `dispatch.rs` for file-size budget — handles `nemotron_h`,
//! `nemotron_h_puzzle`, and the nested-`llm_config` `nemotron_h_omni`
//! wrapper (Nemotron 3.5 Super VL; the vision tower stays unloaded —
//! text-only serve).

use anyhow::{Context, Result};

use super::{LayerType, ModelConfig, default_conv_kernel, default_rms_eps, finalize_config};

pub fn parse_nemotron_h(raw: &serde_json::Value, top_model_type: &str) -> Result<ModelConfig> {
    // Puzzle: num_hidden_layers is JSON null and the hybrid schedule lives
    // in layers_block_type / block_configs (per-block MoE channel pruning).
    // Rewrite the JSON so serde can deserialize, then map to Atlas fields.
    let mut raw_mut = raw.clone();
    if top_model_type == "nemotron_h_puzzle" {
        apply_nemotron_puzzle_json(&mut raw_mut)?;
    }
    let mut config: ModelConfig = serde_json::from_value(raw_mut.clone())
        .context("Failed to parse nemotron_h config.json")?;
    // Map Nemotron-H field names → Atlas canonical names
    if config.num_experts == 0 && config.n_routed_experts > 0 {
        config.num_experts = config.n_routed_experts;
    }
    if config.rms_norm_eps == default_rms_eps() && config.norm_eps > 0.0 {
        config.rms_norm_eps = config.norm_eps;
    }
    if config.linear_conv_kernel_dim == default_conv_kernel() && config.conv_kernel > 0 {
        config.linear_conv_kernel_dim = config.conv_kernel;
    }
    if config.shared_expert_intermediate_size == 0 && config.moe_shared_expert_intermediate_size > 0
    {
        config.shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size;
    }
    // Architecture flags
    config.attn_gated = false;
    config.weight_prefix = "backbone".to_string();
    // A config shipping BOTH fields with DIFFERENT schedule lengths
    // is contradictory: fail closed here rather than letting one
    // silently win the precedence and the other be ignored.
    if !config.hybrid_override_pattern.is_empty()
        && let Some(block_types) = raw.get("layers_block_type").and_then(|v| v.as_array())
        && block_types.len() != config.hybrid_override_pattern.chars().count()
    {
        anyhow::bail!(
            "nemotron_h config has contradictory schedules: \
             hybrid_override_pattern has {} layers but layers_block_type has {}",
            config.hybrid_override_pattern.chars().count(),
            block_types.len()
        );
    }
    // Parse hybrid_override_pattern → layer_types (Nano / Super)
    if !config.hybrid_override_pattern.is_empty() && config.layer_types.is_empty() {
        config.layer_types = config
            .hybrid_override_pattern
            .chars()
            .map(|c| match c {
                'M' => LayerType::LinearAttention,
                'E' => LayerType::Moe,
                '*' => LayerType::FullAttention,
                other => panic!("Unknown hybrid_override_pattern char: '{other}'"),
            })
            .collect();
    }
    // Nemotron-H variants without the pattern (e.g. Nemotron 3.5
    // Lightning's public config) ship the hybrid schedule as
    // `layers_block_type`. Without this, layer_types stays empty and
    // the model silently degrades to all-attention with RoPE —
    // coherent-looking garbage (observed 2026-08-17: 52 attention /
    // 0 SSM layers, degenerate repeated-token output at 602 tok/s).
    //
    if config.layer_types.is_empty()
        && let Some(block_types) = raw.get("layers_block_type").and_then(|v| v.as_array())
    {
        config.layer_types = block_types
            .iter()
            .map(|v| {
                let s = v.as_str().unwrap_or("");
                Ok(match s {
                    "mamba" => LayerType::LinearAttention,
                    "moe" => LayerType::Moe,
                    "attention" => LayerType::FullAttention,
                    other => {
                        anyhow::bail!("unknown layers_block_type entry: '{other}'")
                    }
                })
            })
            .collect::<Result<Vec<_>>>()?;
    }
    // Puzzle: layers_block_type + block_configs → layer_types + per-layer MoE dims
    if top_model_type == "nemotron_h_puzzle" {
        apply_nemotron_puzzle_config(&mut config, &raw_mut)?;
    }
    finalize_config(&mut config, &raw_mut)?;
    Ok(config)
}

/// Nemotron 3.5 Super VL — the language model is a stock `nemotron_h`
/// backbone nested under `llm_config`, inside an omni wrapper that also
/// declares a RADIO vision tower (unused here — text-only serve).
///
/// The nested block omits `num_hidden_layers` (schedule lives only in
/// `layers_block_type`) and ships `layer_norm_epsilon` instead of
/// `norm_eps`; both are rewritten before the flat nemotron_h parse. The
/// wrapper's top-level `quantization_config` is carried down so
/// `finalize_config`'s quant detection sees it.
pub fn parse_nemotron_h_omni(raw: &serde_json::Value) -> Result<ModelConfig> {
    let llm = raw
        .get("llm_config")
        .context("nemotron_h_omni config missing llm_config")?;
    if !llm.is_object() {
        anyhow::bail!("nemotron_h_omni llm_config is not a JSON object");
    }
    if let Some(inner_type) = llm.get("model_type").and_then(|v| v.as_str())
        && inner_type != "nemotron_h"
    {
        anyhow::bail!(
            "nemotron_h_omni llm_config.model_type must be \"nemotron_h\" (or absent), \
             found \"{inner_type}\""
        );
    }
    let mut llm_mut = llm.clone();
    let obj = llm_mut
        .as_object_mut()
        .context("nemotron_h_omni llm_config is not a JSON object")?;
    if obj
        .get("num_hidden_layers")
        .map(|v| v.is_null() || v.as_u64() == Some(0))
        .unwrap_or(true)
    {
        let n = obj
            .get("layers_block_type")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .context(
                "nemotron_h_omni llm_config has no num_hidden_layers and no \
                 layers_block_type to derive it from",
            )?;
        obj.insert("num_hidden_layers".into(), serde_json::json!(n));
    }
    if obj.get("norm_eps").is_none()
        && let Some(eps) = obj.get("layer_norm_epsilon").cloned()
    {
        obj.insert("norm_eps".into(), eps);
    }
    // Keys the downstream parse (`finalize_config` / quant detection) reads
    // from the JSON root — the omni wrapper declares them at top level.
    for key in ["quantization_config"] {
        if !obj.contains_key(key)
            && let Some(v) = raw.get(key)
        {
            obj.insert(key.into(), v.clone());
        }
    }
    let mut config = parse_nemotron_h(&llm_mut, "nemotron_h")?;
    // A distinct identity (like `nemotron_h_puzzle`): drives kernel-target
    // resolution and the tool/reasoning defaults.
    config.model_type = "nemotron_h_omni".to_string();
    config.weight_prefix = "language_model.backbone".to_string();
    // The omni latent-MoE MTP head isn't loaded; DFlash is the
    // speculative path for this checkpoint.
    config.mtp_num_hidden_layers = 0;
    // RADIO vision tower: map the omni wrapper + `vision_config` keys onto
    // `VisionConfig`. The image placeholder is `<image>` id 18 (the
    // processor expands it to `<img>` + N x id-18 + `</img>`); spatial
    // merge comes from `downsample_ratio` (0.5 -> 2x2 merge), the output
    // dim is the LLM hidden size, and `max_pixels` resolves later from
    // `max_img_size^2` if the operator does not pin `--vision-max-pixels`.
    if let Some(vc) = raw.get("vision_config") {
        let g = |k: &str| -> usize {
            vc.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize
        };
        let g3 = |k: &str| -> [f32; 3] {
            let a = vc.get(k).and_then(serde_json::Value::as_array);
            [0, 1, 2].map(|i| {
                a.and_then(|a| a.get(i))
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0) as f32
            })
        };
        let top = |k: &str| -> f64 {
            raw.get(k)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
        };
        let merge = (1.0 / top("downsample_ratio").max(1e-9)).round() as usize;
        config.vision = Some(crate::config::VisionConfig {
            depth: g("num_hidden_layers"),
            hidden_size: g("hidden_size"),
            num_heads: g("num_attention_heads"),
            patch_size: g("patch_size"),
            temporal_patch_size: g("video_temporal_patch_size"),
            spatial_merge_size: merge,
            intermediate_size: (g("hidden_size") as f64
                * vc.get("mlp_ratio")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(4.0)) as usize,
            out_hidden_size: config.hidden_size,
            deepstack_visual_indexes: vec![],
            image_pad_token_id: raw
                .get("img_context_token_id")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(18) as u32,
            video_pad_token_id: 0,
            max_pixels: None,
            omni: Some(crate::config::OmniVisionConfig {
                norm_mean: g3("norm_mean"),
                norm_std: g3("norm_std"),
                max_num_patches: g("max_num_patches"),
                min_num_patches: g("min_num_patches"),
                downsample_ratio: top("downsample_ratio"),
                image_token_id: 18,
                image_start_token_id: 19,
                image_end_token_id: 20,
                force_image_size: top("force_image_size") as usize,
                prefix_tokens: g("num_cls_tokens") + g("num_registers"),
                num_summary_tokens: g("num_cls_tokens") + g("num_registers"),
                pos_grid_side: g("max_img_size") / g("patch_size").max(1),
                projector_norm_eps: 1e-5,
                layer_norm_eps: vc
                    .get("layer_norm_eps")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(1e-6),
            }),
        });
    }
    Ok(config)
}

/// Rewrite Puzzle HF JSON so serde can load it as `ModelConfig`.
///
/// - `num_hidden_layers` is JSON null → derive from `layers_block_type` length
/// - scalar `moe_intermediate_size` / `num_experts_per_tok` may be absent → fill
///   with max-over-blocks so uniform Super-style code paths still have defaults
fn apply_nemotron_puzzle_json(raw: &mut serde_json::Value) -> Result<()> {
    let obj = raw
        .as_object_mut()
        .context("nemotron_h_puzzle config.json is not an object")?;
    let n_layers = obj
        .get("layers_block_type")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .or_else(|| {
            obj.get("block_configs")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
        })
        .context("nemotron_h_puzzle missing layers_block_type / block_configs")?;
    if obj
        .get("num_hidden_layers")
        .map(|v| v.is_null() || v.as_u64() == Some(0))
        .unwrap_or(true)
    {
        obj.insert("num_hidden_layers".into(), serde_json::json!(n_layers));
    }
    // Collect max MoE dims from block_configs for scalar fallbacks
    let mut max_inter = 0usize;
    let mut max_topk = 0usize;
    if let Some(blocks) = obj.get("block_configs").and_then(|v| v.as_array()) {
        for b in blocks {
            if let Some(mi) = b.get("moe_intermediate_size").and_then(|v| v.as_u64()) {
                max_inter = max_inter.max(mi as usize);
            }
            if let Some(tk) = b.get("num_experts_per_tok").and_then(|v| v.as_u64()) {
                max_topk = max_topk.max(tk as usize);
            }
        }
    }
    if obj
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        == 0
        && max_inter > 0
    {
        obj.insert("moe_intermediate_size".into(), serde_json::json!(max_inter));
    }
    if obj
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        == 0
        && max_topk > 0
    {
        obj.insert("num_experts_per_tok".into(), serde_json::json!(max_topk));
    }
    Ok(())
}

/// Map Puzzle `layers_block_type` / `block_configs` onto Atlas layer schedule.
fn apply_nemotron_puzzle_config(config: &mut ModelConfig, raw: &serde_json::Value) -> Result<()> {
    let block_types = raw
        .get("layers_block_type")
        .and_then(|v| v.as_array())
        .context("nemotron_h_puzzle missing layers_block_type")?;
    config.layer_types = block_types
        .iter()
        .map(|v| {
            let s = v.as_str().unwrap_or("");
            Ok(match s {
                "mamba" => LayerType::LinearAttention,
                "moe" => LayerType::Moe,
                "attention" => LayerType::FullAttention,
                other => anyhow::bail!("unknown layers_block_type entry: '{other}'"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if config.num_hidden_layers == 0 {
        config.num_hidden_layers = config.layer_types.len();
    }
    // Per-layer MoE schedule from block_configs
    let n = config.num_hidden_layers;
    let mut inters = vec![0usize; n];
    let mut topks = vec![0usize; n];
    if let Some(blocks) = raw.get("block_configs").and_then(|v| v.as_array()) {
        for (i, b) in blocks.iter().enumerate().take(n) {
            if b.get("block_type").and_then(|v| v.as_str()) != Some("moe") {
                continue;
            }
            inters[i] = b
                .get("moe_intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            topks[i] = b
                .get("num_experts_per_tok")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
        }
    }
    config.moe_intermediate_sizes = inters;
    config.num_experts_per_toks = topks;
    // Keep scalar fields as max for buffer defaults / logging
    if let Some(m) = config
        .moe_intermediate_sizes
        .iter()
        .copied()
        .filter(|&s| s > 0)
        .max()
    {
        config.moe_intermediate_size = m;
    }
    if let Some(m) = config
        .num_experts_per_toks
        .iter()
        .copied()
        .filter(|&k| k > 0)
        .max()
    {
        config.num_experts_per_tok = m;
    }
    Ok(())
}
