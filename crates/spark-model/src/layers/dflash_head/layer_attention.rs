// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer attention variants of the drafter: NoPE layers (HF
//! `no_rope_layers`) and per-layer sliding windows (HF `layer_types` mixing
//! sliding and full attention), plus the context window such a drafter needs.
//!
//! Ripple-65B's DFlash2 drafter runs layers 0-3 with a 4096-token sliding
//! window and RoPE, and layer 4 as global attention without RoPE over the
//! drafter's whole context window, which must cover its 16384-token training
//! rows (Ricursive serves SGLang with `--speculative-draft-window-size
//! 16384`). Uniform drafters (Qwen3.8, GLM-5.3, Lightning) are unchanged:
//! every layer applies RoPE and the context buffer itself is the window.

use std::ops::Range;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::BlockDiffusionDraftHead;
use crate::layers::ops;
use crate::weight_loader::DflashConfig;

/// Context-buffer capacity of a uniform drafter when `ATLAS_DFLASH_CTX_WINDOW`
/// is unset (the scratch the paged-attention path is sized for).
const UNIFORM_CTX_WINDOW: usize = 4096;

/// Each layer's `(rope, attn_window)`: `attn_window` is the layer's sliding
/// window for the attention kernel when the checkpoint mixes sliding and full
/// layers (0 on full layers), else 0 for every layer.
pub(super) fn layer_attention(config: &DflashConfig) -> Result<Vec<(bool, u32)>> {
    let rope = config.rope_layers()?;
    let windows = config.mixed_layer_windows()?;
    ensure!(
        windows.is_none() || !config.query_causal(),
        "DFlash drafter: per-layer sliding windows on a causal drafter are not supported"
    );
    let windows = windows.unwrap_or_else(|| vec![None; rope.len()]);
    rope.into_iter()
        .zip(windows)
        .map(|(rope, window)| Ok((rope, u32::try_from(window.unwrap_or(0))?)))
        .collect()
}

/// `(window_size, ctx_window)`: the context-buffer window and the scratch
/// capacity. `requested` is `--dflash-window-size`, `env_ctx_window`
/// `ATLAS_DFLASH_CTX_WINDOW`. A uniform drafter keeps its historical
/// resolution. A drafter with per-layer windows takes `requested` as its
/// global context window (SGLang's `--speculative-draft-window-size`): it must
/// exceed every layer's sliding window, or the full layers would see no more
/// than the sliding ones, and the scratch must hold it.
pub(super) fn resolve_context_window(
    requested: Option<usize>,
    config: &DflashConfig,
    env_ctx_window: Option<usize>,
) -> Result<(Option<usize>, usize)> {
    let Some(windows) = config.mixed_layer_windows()? else {
        let window = requested.or_else(|| config.sliding_window());
        return Ok((window, env_ctx_window.unwrap_or(UNIFORM_CTX_WINDOW)));
    };
    let widest = windows.iter().flatten().copied().max().unwrap_or(0);
    let window = requested.filter(|&w| w > widest).ok_or_else(|| {
        anyhow::anyhow!(
            "DFlash drafter mixes sliding ({widest}-token) and full attention layers: \
             --dflash-window-size {requested:?} must exceed {widest} and cover the full \
             layers' training context (16384 for Ripple-65B's DFlash2)"
        )
    })?;
    let ctx_window = env_ctx_window.unwrap_or(window);
    ensure!(
        ctx_window >= window,
        "ATLAS_DFLASH_CTX_WINDOW {ctx_window} is below the drafter window {window}"
    );
    Ok((Some(window), ctx_window))
}

/// Rows the head's context accumulator will have (`ctx_window` capped at
/// `max_seq_len`), for sizing it before the head exists. The factory's early
/// pool prime used the uniform 4096 and primed half of Ripple's 8192 rows:
/// the first request took that buffer and its zeroing memset ran past the end
/// (HIP InvalidValue, HTTP 500 on request 1 only).
pub(crate) fn ctx_capacity(
    requested: Option<usize>,
    config: &DflashConfig,
    max_seq_len: usize,
) -> Result<usize> {
    let env_ctx_window = std::env::var("ATLAS_DFLASH_CTX_WINDOW")
        .ok()
        .and_then(|s| s.parse().ok());
    let (_, ctx_window) = resolve_context_window(requested, config, env_ctx_window)?;
    Ok(ctx_window.min(max_seq_len))
}

impl BlockDiffusionDraftHead {
    /// The attention kernel's sliding window for layer `layer_idx`: the
    /// layer's own window when the checkpoint declares per-layer windows,
    /// else the head-wide [`Self::attn_sliding_window`].
    pub(super) fn layer_attn_window(&self, layer_idx: usize) -> u32 {
        match self.layers[layer_idx].attn_window {
            0 => self.attn_sliding_window(),
            window => window,
        }
    }

    /// RoPE on `rows` rows of Q (`num_q_heads` heads; 0 = K only) and K at
    /// `positions`, for layer `layer_idx`. A NoPE layer leaves both as they are.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn layer_rope(
        &self,
        gpu: &dyn GpuBackend,
        layer_idx: usize,
        q: DevicePtr,
        k: DevicePtr,
        positions: DevicePtr,
        rows: u32,
        num_q_heads: u32,
        stream: u64,
    ) -> Result<()> {
        if !self.layers[layer_idx].rope {
            return Ok(());
        }
        ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            q,
            k,
            positions,
            rows,
            num_q_heads,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )
    }

    /// Maximal runs of consecutive RoPE layers, so a fused all-layer RoPE
    /// stays one launch on a drafter without NoPE layers.
    pub(super) fn rope_layer_runs(&self) -> Vec<Range<usize>> {
        rope_runs(&self.layers.iter().map(|l| l.rope).collect::<Vec<_>>())
    }
}

fn rope_runs(rope: &[bool]) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::new();
    for (layer, _) in rope.iter().enumerate().filter(|(_, r)| **r) {
        match runs.last_mut() {
            Some(run) if run.end == layer => run.end = layer + 1,
            _ => runs.push(layer..layer + 1),
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loader::dflash_loader::parse_dflash_config;

    fn ripple() -> DflashConfig {
        parse_dflash_config(
            &serde_json::json!({
                "architectures": ["DFlash2DraftModel"], "hidden_size": 5120,
                "num_hidden_layers": 5, "intermediate_size": 40960,
                "num_attention_heads": 80, "num_key_value_heads": 8, "head_dim": 128,
                "vocab_size": 100352, "is_causal": false,
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                                "sliding_attention", "full_attention"],
                "use_sliding_window": true, "sliding_window": 4096, "no_rope_layers": [4],
                "dflash_config": {"mask_token_id": 100278, "target_layer_ids": [1, 20, 39, 58, 77]}
            })
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn ripple_layers_and_window() {
        let c = ripple();
        let layers = layer_attention(&c).unwrap();
        assert_eq!(
            layers,
            vec![(true, 4096); 4]
                .into_iter()
                .chain([(false, 0)])
                .collect::<Vec<_>>()
        );
        assert_eq!(
            resolve_context_window(Some(16384), &c, None).unwrap(),
            (Some(16384), 16384)
        );
        assert_eq!(
            resolve_context_window(Some(16384), &c, Some(20000)).unwrap(),
            (Some(16384), 20000)
        );
        // The CLI default (4096) and no window both fail fast.
        assert!(resolve_context_window(Some(4096), &c, None).is_err());
        assert!(resolve_context_window(None, &c, None).is_err());
        assert!(resolve_context_window(Some(16384), &c, Some(8192)).is_err());
        // What the factory primes: the head's rows, not the uniform 4096.
        assert_eq!(ctx_capacity(Some(16384), &c, 8192).unwrap(), 8192);
    }

    #[test]
    fn uniform_drafters_keep_historical_resolution() {
        let mut c = ripple();
        c.layer_types = None;
        c.no_rope_layers = None;
        c.use_sliding_window = None;
        assert_eq!(layer_attention(&c).unwrap(), vec![(true, 0); 5]);
        assert_eq!(
            resolve_context_window(Some(4096), &c, None).unwrap(),
            (Some(4096), 4096)
        );
        assert_eq!(
            resolve_context_window(None, &c, Some(16384)).unwrap(),
            (None, 16384)
        );
        c.use_sliding_window = Some(true);
        c.sliding_window = Some(2048);
        assert_eq!(
            resolve_context_window(None, &c, None).unwrap(),
            (Some(2048), 4096)
        );
    }

    #[test]
    fn rope_runs_split_around_nope_layers() {
        assert_eq!(rope_runs(&[true; 5]), vec![0..5]);
        assert_eq!(rope_runs(&[true, true, true, true, false]), vec![0..4]);
        assert_eq!(
            rope_runs(&[false, true, false, true, true]),
            vec![1..2, 3..5]
        );
        assert!(rope_runs(&[]).is_empty());
    }
}
