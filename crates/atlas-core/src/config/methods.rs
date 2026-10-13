// SPDX-License-Identifier: AGPL-3.0-only

//! [`ModelConfig`] inherent helper methods. Split out of `config.rs` for
//! file-size budget. Pure derived getters + small predicates over the
//! struct fields.

#![allow(unused_imports)]

use super::{LayerType, ModelConfig};

impl ModelConfig {
    /// GQA ratio: number of Q heads per KV head.
    pub fn gqa_ratio(&self) -> usize {
        self.num_attention_heads
            .checked_div(self.num_key_value_heads)
            .unwrap_or(1)
    }

    /// Layer type for a given layer index.
    /// Falls back to full_attention_interval if layer_types is empty.
    pub fn layer_type(&self, layer_idx: usize) -> LayerType {
        if !self.layer_types.is_empty() {
            self.layer_types
                .get(layer_idx)
                .cloned()
                .unwrap_or(LayerType::FullAttention)
        } else if self.full_attention_interval > 0
            && (layer_idx + 1).is_multiple_of(self.full_attention_interval)
        {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        }
    }

    /// Number of attention (KV-cache-consuming) layers: full attention plus
    /// sliding attention. Sliding-attention layers write to the paged KV cache
    /// exactly like full-attention ones (only their attention window differs),
    /// so every consumer sized from this count — KV pool `num_layers`,
    /// `attn_layer_dtypes`, loader `layer_kv_dtypes` indexing — must see them
    /// all. Step 3.7 is the only model emitting `SlidingAttention` layer types
    /// (12 full + 33 sliding); counting full-only there undersized the dtype
    /// vec and panicked the loader at layer 13.
    pub fn num_attention_layers(&self) -> usize {
        if !self.layer_types.is_empty() {
            self.layer_types
                .iter()
                .filter(|t| matches!(t, LayerType::FullAttention | LayerType::SlidingAttention))
                .count()
        } else {
            self.num_hidden_layers
                .checked_div(self.full_attention_interval)
                .unwrap_or(self.num_hidden_layers)
        }
    }

    /// Number of SSM (linear attention) layers.
    pub fn num_ssm_layers(&self) -> usize {
        if !self.layer_types.is_empty() {
            self.layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count()
        } else {
            self.num_hidden_layers - self.num_attention_layers()
        }
    }

    /// Whether this model carries recurrent (SSM / linear-attention) state —
    /// the honest capability signal for the SSM snapshot tiers. Derived from
    /// [`Self::num_ssm_layers`] so the config-level predicate and the runtime
    /// pool predicate (`ssm_pool.num_ssm_layers > 0`) agree by construction
    /// (SSOT). A pure-attention model (dense or MoE) returns `false`:
    /// requesting an SSM tier for it must fail fast, never silently no-op.
    pub fn has_recurrent_state(&self) -> bool {
        self.num_ssm_layers() > 0
    }

    /// Whether this model has MoE routed experts — the capability signal for
    /// the expert-streaming tier. Keyed on config, never on observed expert
    /// tensors (EP ranks legitimately own zero local expert tensors).
    pub fn has_experts(&self) -> bool {
        self.num_experts > 0
    }

    /// Rotary embedding dimension.
    ///
    /// Priority:
    /// 1. Explicit `rotary_dim` field (MiniMax M2 — integer in config.json).
    /// 2. `partial_rotary_factor * head_dim` (Qwen3/Gemma-4 convention — float).
    pub fn rotary_dim(&self) -> usize {
        if self.rotary_dim > 0 {
            self.rotary_dim
        } else {
            (self.partial_rotary_factor * self.head_dim as f64) as usize
        }
    }

    /// SSM projection output size: Q + K + V + Z concatenated.
    pub fn ssm_qkvz_size(&self) -> usize {
        let q = self.linear_num_key_heads * self.linear_key_head_dim;
        let k = self.linear_num_key_heads * self.linear_key_head_dim;
        let v = self.linear_num_value_heads * self.linear_value_head_dim;
        let z = self.linear_num_value_heads * self.linear_value_head_dim;
        q + k + v + z
    }

    /// SSM QKV projection output size (without Z): Q + K + V.
    pub fn ssm_qkv_size(&self) -> usize {
        let q = self.linear_num_key_heads * self.linear_key_head_dim;
        let k = self.linear_num_key_heads * self.linear_key_head_dim;
        let v = self.linear_num_value_heads * self.linear_value_head_dim;
        q + k + v
    }

    /// SSM Z gate projection output size.
    pub fn ssm_z_size(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// SSM beta+alpha projection output size.
    pub fn ssm_ba_size(&self) -> usize {
        // beta: num_value_heads, alpha: num_value_heads
        self.linear_num_value_heads * 2
    }

    /// Range of expert indices local to this EP rank.
    /// Returns (start, end) where start is inclusive and end is exclusive.
    pub fn local_expert_range(&self) -> (usize, usize) {
        if self.ep_world_size <= 1 || self.expert_tp {
            return (0, self.num_experts);
        }
        let per_rank = self.num_experts / self.ep_world_size;
        let start = self.ep_rank * per_rank;
        let end = if self.ep_rank == self.ep_world_size - 1 {
            self.num_experts // last rank gets remainder
        } else {
            start + per_rank
        };
        (start, end)
    }

    /// Routed-expert intermediate width held by this rank: the full
    /// `moe_intermediate_size`, or its EP-rank slice under expert TP.
    pub fn routed_inter_local(&self) -> usize {
        if self.expert_tp {
            self.moe_intermediate_size / self.ep_world_size
        } else {
            self.moe_intermediate_size
        }
    }

    /// Whether the given expert ID is local to this EP rank.
    pub fn is_local_expert(&self, expert_id: usize) -> bool {
        let (start, end) = self.local_expert_range();
        expert_id >= start && expert_id < end
    }

    /// Range `[start, end)` of a `total`-sized dimension owned by this TP rank.
    /// `total` must be divisible by `tp_world_size`. Returns `(0, total)` when
    /// TP is disabled.
    pub fn tp_shard_range(&self, total: usize) -> (usize, usize) {
        if self.tp_world_size <= 1 {
            return (0, total);
        }
        debug_assert!(
            total.is_multiple_of(self.tp_world_size),
            "tp_shard_range: total={} not divisible by tp_world_size={}",
            total,
            self.tp_world_size,
        );
        let per_rank = total / self.tp_world_size;
        let start = self.tp_rank * per_rank;
        (start, start + per_rank)
    }

    /// Per-rank shard size for a `total`-sized dimension under TP.
    pub fn tp_shard_dim(&self, total: usize) -> usize {
        if self.tp_world_size <= 1 {
            return total;
        }
        total / self.tp_world_size
    }

    /// This config as one rank running the whole model alone: the head counts
    /// the TP split divided (`resolve_topology` in spark-server) restored to
    /// full width, and no TP or EP split. For a module one rank holds in full
    /// and runs without collectives (the qwen4_exp MTP head on rank 0), so its
    /// loaders take every weight unsliced and its kernels run every head.
    /// A single-rank config comes back unchanged.
    pub fn tp1_view(&self) -> Self {
        let tp = self.tp_world_size.max(1);
        Self {
            num_attention_heads: self.num_attention_heads * tp,
            num_key_value_heads: self.num_key_value_heads * tp,
            linear_num_key_heads: self.linear_num_key_heads * tp,
            linear_num_value_heads: self.linear_num_value_heads * tp,
            tp_rank: 0,
            tp_world_size: self.tp_world_size.min(1),
            ep_rank: 0,
            ep_world_size: self.ep_world_size.min(1),
            expert_tp: false,
            ..self.clone()
        }
    }

    /// Weight key prefix for layer-level weights.
    /// Returns `"model.layers"` for flat models (qwen3_next),
    /// or `"model.language_model.layers"` for conditional generation models (qwen3_5_moe).
    pub fn layer_prefix(&self, layer_idx: usize) -> String {
        if self.weight_prefix.is_empty() {
            format!("model.layers.{layer_idx}")
        } else {
            format!("{}.layers.{layer_idx}", self.weight_prefix)
        }
    }

    /// Derive model-agnostic capabilities from this config.
    pub fn capabilities(&self) -> crate::capabilities::ModelCapabilities {
        crate::capabilities::ModelCapabilities::from_config(self)
    }

    // ── Factory sub-dispatch predicates ──
    // Used only by loader_for_config() to select the right weight loader
    // within the qwen3_5_moe model_type family. Not for general use —
    // prefer config fields (attn_gated, nested_config) or capabilities.

    /// Factory use only. Prefer `config.attn_gated` or `config.capabilities()`.
    pub fn is_qwen35(&self) -> bool {
        self.model_type == "qwen3_5_moe"
    }

    /// Factory use only.
    pub fn is_qwen35_dense(&self) -> bool {
        self.model_type == "qwen3_5" && self.num_experts == 0
    }

    /// Factory use only.
    ///
    /// Recognises the upstream `qwen3_vl_moe` model_type (Qwen3-VL MoE)
    /// and Qwen3.5-VL — which ships with `model_type = "qwen3_5"` plus
    /// `architectures = ["Qwen3_5ForConditionalGeneration"]` and a
    /// populated `vision_config` block. The vision_config presence is
    /// the durable signal: the trunk model_type stays `qwen3_5` whether
    /// the checkpoint is text-only or VL, but VL ships an extra
    /// vision encoder which the parser exposes as `config.vision`.
    pub fn is_qwen3_vl(&self) -> bool {
        if self.model_type == "qwen3_vl_moe" {
            return true;
        }
        // Qwen3.5-VL: trunk model_type is `qwen3_5`; the vision tower
        // is detected by the parsed `vision_config` block.
        if self.model_type == "qwen3_5" && self.vision.is_some() {
            return true;
        }
        false
    }

    /// Whether to skip NVFP4 quantization of the LM head.
    /// MLA models (kv_lora_rank > 0) lose logit precision under NVFP4.
    /// Gemma-4 dense (31B): the LM head ties to BF16 embed_tokens whose
    /// rows have heavy outliers (final_norm.weight max=510, several
    /// embedding rows in similar range). The runtime BF16→NVFP4 path
    /// uses a single per-tensor absmax for `scale2`, which forces a
    /// coarse scale that loses ~7 bits in normal-magnitude rows. For a
    /// 262 144-row vocab matrix that compounds into the 0.14-margin
    /// argmax flip on creative prompts (verified 2026-05-01 via FP32
    /// lm_head bisection: NVFP4 output had top1=` a` 21.85 vs FP32 BF16
    /// view top1=` a` 21.85 — quantization noise was visible in the
    /// SAME logit channel that flipped the tiebreak). Skipping the
    /// runtime quantization keeps the LM head as plain BF16 dense; the
    /// FP32 lm_head path (gated by `ATLAS_GEMMA4_FP32_LMHEAD=1`) can
    /// then act on full-precision weights without the NVFP4 floor.
    pub fn skip_lm_head_quantization(&self) -> bool {
        // CLI override (`--lm-head-dtype`, set into `lm_head_bf16_override` at serve
        // time) wins. `bf16` keeps the LM head in BF16 instead of runtime-quantizing it
        // to NVFP4 — the 4-bit floor on the final vocab projection is a prime suspect for
        // argmax flips in long structured generation; vLLM keeps lm_head at checkpoint
        // precision. (Replaces the former ATLAS_LMHEAD_BF16 env var; PCND: explicit arg.)
        if let Some(force_bf16) = self.lm_head_bf16_override {
            return force_bf16;
        }
        if self.kv_lora_rank > 0 {
            return true;
        }
        if self.model_type == "laguna" {
            return true;
        }
        if self.model_type == "gemma4" && self.num_experts == 0 {
            // Allow rollback via env for A/B testing.
            return std::env::var("ATLAS_GEMMA4_LMHEAD_NVFP4").ok().as_deref() != Some("1");
        }
        // Qwen3.8-Flash-Next: the checkpoint ships lm_head in BF16 and the
        // long-form (20K-token single-file) benchmark is exactly the
        // argmax-flip regime the comment above describes; the 27B recipe
        // already pins `--lm-head-dtype bf16` for the same reason. Keep the
        // head at checkpoint precision by default; `--lm-head-dtype nvfp4`
        // still forces the packed head for the decode-speed trade.
        if self.model_type == "qwen4_exp" {
            return true;
        }
        false
    }

    /// The model's final normalization happens inside its last layer, so the
    /// model-level final RMSNorm must be skipped. qwen4_exp: the last layer's
    /// hyper-connection mixer collapses the `hc_mult` streams and its `hc_norm`
    /// IS the final norm; the reference feeds the mixer output straight to
    /// `lm_head` (no `model.norm.weight` exists).
    pub fn final_norm_is_identity(&self) -> bool {
        self.model_type == "qwen4_exp"
    }

    /// Mamba-2 d_inner = mamba_num_heads * mamba_head_dim.
    pub fn mamba2_d_inner(&self) -> usize {
        self.mamba_num_heads * self.mamba_head_dim
    }

    /// Mamba-2 d_xBC = d_inner + 2 * n_groups * ssm_state_size.
    /// This is the dimension that goes through conv1d (x + B + C concatenated).
    pub fn mamba2_d_xbc(&self) -> usize {
        self.mamba2_d_inner() + 2 * self.n_groups * self.ssm_state_size
    }

    /// Mamba-2 in_proj output size = z + xBC + dt.
    pub fn mamba2_in_proj_size(&self) -> usize {
        self.mamba2_d_inner() + self.mamba2_d_xbc() + self.mamba_num_heads
    }

    /// Per-layer SSM hidden state size in bytes (FP32).
    /// Dispatches on SSM architecture: Mamba-2 vs GDN, using config fields.
    pub fn ssm_h_state_bytes(&self) -> usize {
        if self.mamba_num_heads > 0 && self.mamba_head_dim > 0 {
            // Mamba-2: h[num_heads, head_dim, state_size] FP32
            self.mamba_num_heads * self.mamba_head_dim * self.ssm_state_size * 4
        } else {
            // GDN: h[nv, vd, kd] FP32
            self.linear_num_value_heads * self.linear_value_head_dim * self.linear_key_head_dim * 4
        }
    }

    /// Per-layer SSM conv state size in bytes (FP32).
    pub fn ssm_conv_state_bytes(&self) -> usize {
        let d_conv = self.linear_conv_kernel_dim;
        if self.mamba_num_heads > 0 && self.mamba_head_dim > 0 {
            // Mamba-2: conv: [d_xBC, d_conv] FP32
            self.mamba2_d_xbc() * d_conv * 4
        } else {
            // GDN: conv: [conv_dim, d_conv] FP32
            let conv_dim = self.linear_num_key_heads * self.linear_key_head_dim * 2
                + self.linear_num_value_heads * self.linear_value_head_dim;
            conv_dim * d_conv * 4
        }
    }

    /// SSM state normalization dimensions: (num_heads, k_dim, v_dim).
    /// Used by the state normalization kernel to prevent drift.
    pub fn ssm_state_norm_dims(&self) -> (usize, usize, usize) {
        if self.mamba_num_heads > 0 && self.mamba_head_dim > 0 {
            (
                self.mamba_num_heads,
                self.mamba_head_dim,
                self.ssm_state_size,
            )
        } else {
            (
                self.linear_num_value_heads,
                self.linear_key_head_dim,
                self.linear_value_head_dim,
            )
        }
    }

    /// MoE expert input dimension: latent size if LatentMoE, else hidden_size.
    pub fn moe_input_size(&self) -> usize {
        if self.moe_latent_size > 0 {
            self.moe_latent_size
        } else {
            self.hidden_size
        }
    }

    /// Routed expert intermediate size for layer `i`.
    ///
    /// Puzzle checkpoints prune channels non-uniformly across MoE layers;
    /// look up `moe_intermediate_sizes[i]` when populated, else the scalar.
    pub fn moe_intermediate_size_for(&self, layer: usize) -> usize {
        self.moe_intermediate_sizes
            .get(layer)
            .copied()
            .filter(|&s| s > 0)
            .unwrap_or(self.moe_intermediate_size)
    }

    /// Top-K experts per token for layer `i` (Puzzle per-block schedule).
    pub fn num_experts_per_tok_for(&self, layer: usize) -> usize {
        self.num_experts_per_toks
            .get(layer)
            .copied()
            .filter(|&k| k > 0)
            .unwrap_or(self.num_experts_per_tok)
    }

    /// Max routed intermediate across all layers (buffer / scratch sizing).
    pub fn max_moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_sizes
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(self.moe_intermediate_size)
    }

    /// Number of MoE-only layers (Nemotron-H).
    pub fn num_moe_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::Moe)
            .count()
    }

    /// Whether the radix prefix cache captures every state needed to resume
    /// this model exactly. DeepSeek V4 compression also carries a prompt-built
    /// pool and ring that are not represented by KV blocks today.
    pub fn kv_only_prefix_cache_is_safe(&self) -> bool {
        self.model_type != "deepseek_v4" || self.compress_ratios.iter().all(|&ratio| ratio == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::ModelConfig;

    #[test]
    fn any_compressed_deepseek_v4_layer_is_not_kv_cache_complete() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "deepseek_v4".to_string();

        for ratios in [vec![4, 0, 0], vec![0, 4, 0], vec![0, 0, 128]] {
            config.compress_ratios = ratios;
            assert!(!config.kv_only_prefix_cache_is_safe());
        }
    }

    #[test]
    fn kv_complete_models_can_use_the_prefix_cache() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(config.kv_only_prefix_cache_is_safe());

        config.model_type = "deepseek_v4".to_string();
        config.compress_ratios = vec![0; 3];
        assert!(config.kv_only_prefix_cache_is_safe());
    }

    #[test]
    fn final_norm_identity_only_for_qwen4_exp() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(!config.final_norm_is_identity());

        config.model_type = "qwen4_exp".to_string();
        assert!(config.final_norm_is_identity());

        config.model_type = "deepseek_v4".to_string();
        assert!(!config.final_norm_is_identity());
    }

    /// A TP=EP=2 rank's view restores the full head counts and drops the
    /// split; a single-rank config is its own view.
    #[test]
    fn tp1_view_undoes_the_tp_head_split() {
        let full = ModelConfig::qwen3_next_80b_nvfp4();
        let mut rank1 = full.clone();
        rank1.num_attention_heads /= 2;
        rank1.num_key_value_heads /= 2;
        rank1.linear_num_key_heads /= 2;
        rank1.linear_num_value_heads /= 2;
        (rank1.tp_rank, rank1.tp_world_size) = (1, 2);
        (rank1.ep_rank, rank1.ep_world_size) = (1, 2);
        let view = rank1.tp1_view();
        let heads = |c: &ModelConfig| {
            (
                c.num_attention_heads,
                c.num_key_value_heads,
                c.linear_num_key_heads,
                c.linear_num_value_heads,
            )
        };
        assert_eq!(heads(&view), heads(&full));
        assert_eq!((view.tp_rank, view.tp_world_size), (0, 1));
        assert_eq!((view.ep_rank, view.ep_world_size), (0, 1));
        assert_eq!(view.local_expert_range(), (0, full.num_experts));

        for world in [0, 1] {
            let mut single = full.clone();
            (single.tp_world_size, single.ep_world_size) = (world, world);
            let view = single.tp1_view();
            assert_eq!(heads(&view), heads(&single));
            assert_eq!((view.tp_world_size, view.ep_world_size), (world, world));
        }
    }

    #[test]
    fn nemotron_mamba2_ssm_bytes_are_not_gdn_zero() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.mamba_num_heads = 64;
        c.mamba_head_dim = 64;
        c.ssm_state_size = 128;
        c.n_groups = 8;
        c.linear_conv_kernel_dim = 4;
        c.linear_num_value_heads = 0;
        c.linear_value_head_dim = 0;
        c.linear_num_key_heads = 0;
        c.linear_key_head_dim = 0;
        assert_eq!(c.ssm_h_state_bytes(), 64 * 64 * 128 * 4);
        assert_eq!(c.ssm_conv_state_bytes(), (64 * 64 + 2 * 8 * 128) * 4 * 4);
        assert_eq!(
            c.linear_num_value_heads * c.linear_value_head_dim * c.linear_key_head_dim * 4,
            0
        );
    }
}
