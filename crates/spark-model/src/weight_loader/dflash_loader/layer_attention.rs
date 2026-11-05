// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer attention variants of a DFlash drafter: HF `layer_types` that
//! mix sliding and full attention, and HF `no_rope_layers` (NoPE layers).
//! `ricursiveAI/Ripple-65B-Preview-DFlash2` ships four `sliding_attention`
//! layers (window 4096) and one global `full_attention` layer without RoPE;
//! SGLang's DFlash draft attention reads both the same way
//! (`_get_dflash_layer_attention_params`, Ricursive's `_patch_draft_attention`).

use anyhow::{Result, ensure};

use super::DflashConfig;

const SLIDING: &str = "sliding_attention";
const FULL: &str = "full_attention";

impl DflashConfig {
    /// `layer_types` declares both sliding and full layers.
    pub fn layer_types_mixed(&self) -> bool {
        self.layer_types.as_ref().is_some_and(|types| {
            types.iter().any(|t| t == SLIDING) && types.iter().any(|t| t != SLIDING)
        })
    }

    /// Each layer's sliding window (HF semantics: a key is visible when
    /// `q - k < window`) when `layer_types` mixes sliding and full layers:
    /// `Some(sliding_window)` on a sliding layer, `None` on a full one.
    /// `Ok(None)` for a uniform checkpoint, whose one window is
    /// [`Self::sliding_window`]. Fails on an unknown layer type, a list that
    /// does not cover every layer, or sliding layers without a window.
    pub fn mixed_layer_windows(&self) -> Result<Option<Vec<Option<usize>>>> {
        if !self.layer_types_mixed() {
            return Ok(None);
        }
        let types = self.layer_types.as_deref().unwrap_or_default();
        ensure!(
            types.len() == self.num_hidden_layers,
            "DFlash drafter layer_types has {} entries for {} layers",
            types.len(),
            self.num_hidden_layers
        );
        let window = self.sliding_window.filter(|&w| w > 0).ok_or_else(|| {
            anyhow::anyhow!("DFlash drafter sliding_attention layers need a sliding_window")
        })?;
        types
            .iter()
            .enumerate()
            .map(|(i, t)| match t.as_str() {
                SLIDING => Ok(Some(window)),
                FULL => Ok(None),
                other => {
                    anyhow::bail!("DFlash drafter layer_types[{i}] = {other:?} is not supported")
                }
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// Whether each layer applies RoPE: every layer but the indices in
    /// `no_rope_layers`. Fails on an index past the last layer or a repeated
    /// one (a 0/1 flag list, as some HF configs use, is not this field).
    pub fn rope_layers(&self) -> Result<Vec<bool>> {
        let mut rope = vec![true; self.num_hidden_layers];
        for &layer in self.no_rope_layers.iter().flatten() {
            ensure!(
                layer < self.num_hidden_layers && rope[layer],
                "DFlash drafter no_rope_layers entry {layer} is out of range or repeated \
                 ({} layers)",
                self.num_hidden_layers
            );
            rope[layer] = false;
        }
        Ok(rope)
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse_dflash_config;

    fn config(extra: serde_json::Value) -> super::DflashConfig {
        let mut v = serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "hidden_size": 5120, "num_hidden_layers": 5, "intermediate_size": 40960,
            "num_attention_heads": 80, "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 100352,
            "dflash_config": {"mask_token_id": 100278, "target_layer_ids": [1, 20, 39, 58, 77]}
        });
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        parse_dflash_config(&v.to_string()).unwrap()
    }

    /// Ripple-65B's DFlash2 `config.json` keys: `use_sliding_window: true`
    /// must not turn the global layer into a 4096-token one.
    #[test]
    fn ripple_mixed_layers_and_nope() {
        let c = config(serde_json::json!({
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                            "sliding_attention", "full_attention"],
            "use_sliding_window": true, "sliding_window": 4096, "no_rope_layers": [4]
        }));
        assert!(c.layer_types_mixed());
        assert_eq!(c.sliding_window(), None);
        let w = c.mixed_layer_windows().unwrap().unwrap();
        assert_eq!(
            w,
            vec![Some(4096), Some(4096), Some(4096), Some(4096), None]
        );
        assert_eq!(
            c.rope_layers().unwrap(),
            vec![true, true, true, true, false]
        );
    }

    /// GLM-5.3 (all sliding) and Qwen3.8 (no keys) keep their one window and RoPE everywhere.
    #[test]
    fn uniform_layers_unchanged() {
        let glm = config(serde_json::json!({
            "layer_types": vec!["sliding_attention"; 5], "use_sliding_window": true,
            "sliding_window": 2048
        }));
        assert!(!glm.layer_types_mixed());
        assert_eq!(glm.sliding_window(), Some(2048));
        assert_eq!(glm.mixed_layer_windows().unwrap(), None);
        let qwen = config(serde_json::json!({}));
        assert_eq!(qwen.sliding_window(), None);
        assert_eq!(qwen.mixed_layer_windows().unwrap(), None);
        assert_eq!(qwen.rope_layers().unwrap(), vec![true; 5]);
    }

    #[test]
    fn malformed_layer_declarations_fail() {
        let short = config(serde_json::json!({
            "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 4096
        }));
        assert!(short.mixed_layer_windows().is_err());
        let no_window = config(serde_json::json!({
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                            "sliding_attention", "full_attention"]
        }));
        assert!(no_window.mixed_layer_windows().is_err());
        let unknown = config(serde_json::json!({
            "layer_types": ["sliding_attention", "chunked_attention", "full_attention",
                            "full_attention", "full_attention"], "sliding_window": 4096
        }));
        assert!(unknown.mixed_layer_windows().is_err());
        assert!(
            config(serde_json::json!({"no_rope_layers": [5]}))
                .rope_layers()
                .is_err()
        );
        assert!(
            config(serde_json::json!({"no_rope_layers": [1, 1]}))
                .rope_layers()
                .is_err()
        );
    }
}
