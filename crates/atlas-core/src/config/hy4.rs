// SPDX-License-Identifier: AGPL-3.0-only

//! Hy4-specific metadata, kept separate from executable attention policies.

use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hy4Attention {
    DeepseekSparseAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hy4Mlp {
    Dense,
    Sparse,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Hy4Rope {
    pub rope_type: String,
    pub rope_theta: f64,
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct Hy4Ropes {
    pub deepseek_sparse_attention: Hy4Rope,
    pub sliding_attention: Hy4Rope,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Hy4Config {
    pub layer_types: Vec<Hy4Attention>,
    pub swa_pattern: Vec<u8>,
    pub mlp_layer_types: Vec<Hy4Mlp>,
    pub first_k_dense_replace: usize,
    pub swa_num_attention_heads: usize,
    pub swa_qk_head_dim: usize,
    pub swa_qk_nope_head_dim: usize,
    pub swa_qk_rope_head_dim: usize,
    pub swa_v_head_dim: usize,
    pub enable_ihc: bool,
    pub hc_magnitude: f64,
    pub routed_expert_hidden_size: usize,
    pub swiglu_limit: f64,
    pub sandwich_norm: bool,
    pub gated_mla: bool,
    pub learnable_sink: bool,
    pub dsa_skip_attn_post_norm: bool,
    pub engram_layer_ids: Vec<usize>,
    pub engram_layer_head_vocab_sizes: BTreeMap<usize, Vec<Vec<usize>>>,
    pub engram_max_ngram_size: usize,
    pub engram_n_embed_per_ngram: usize,
    pub engram_n_head_per_ngram: usize,
    pub engram_fp8_embedding: bool,
    pub engram_seed: u64,
    pub engram_pad_id: u32,
    pub use_dsa: bool,
    pub use_mla: bool,
    pub use_compressed_indexer: bool,
    pub compressed_indexer_compress_ratio: usize,
    pub compressed_indexer_group_topk: usize,
    pub compressed_indexer_candidate_topk: usize,
    pub compressed_indexer_num_attention_heads: usize,
    pub compressed_indexer_head_dim: usize,
    pub compressed_indexer_key_pooling: String,
    pub compressed_indexer_skip_fine_select: bool,
    pub rope_parameters: Hy4Ropes,
}

#[cfg(test)]
mod tests;
