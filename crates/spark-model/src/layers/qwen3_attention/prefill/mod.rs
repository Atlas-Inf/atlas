// SPDX-License-Identifier: AGPL-3.0-only

//! Q/K/V projection + Flash Attention prefill paths.
//!
//! Wave-3 refactor split this 2619-line file into two methods-per-file
//! sub-modules. Both `paged.rs` and `cache_skip.rs` exceed the 500-LoC
//! cap because each contains a single monolithic 1000-1400 LoC method
//! (`prefill_attention_paged` / `prefill_attention_with_cache_skip`)
//! whose body interleaves 10+ sections with deep cross-section state
//! coupling. Splitting further requires extracting each section as a
//! helper method with 10-20 args — multi-day kernel-level surgery
//! beyond this wave's scope.

mod cache_skip;
mod cache_skip_mla;
mod cache_skip_mla_kv_only;
mod cache_skip_qkv;
mod cache_skip_v4;
mod glm_index;
mod glm_index_split;
pub(crate) use glm_index_split::index_split_words;
mod paged;
mod paged_attn;
mod paged_attn_batched;
mod paged_attn_fp8k;
mod paged_attn_turbok;
mod paged_glm;
pub(in crate::layers::qwen3_attention) use paged_glm::shard::ShardRows;
pub(crate) use paged_glm::write_floor_legacy;
pub(in crate::layers::qwen3_attention) use paged_glm::{GlmChunkOwner, glm_chunk_pieces};
mod paged_mla;
mod paged_mla_args;
mod paged_oproj;
mod paged_qkv;
mod paged_v4;

/// `ATLAS_ATTN_W4A4` (PRESENCE, read once): the opt-in native FP4 Q/K/V and
/// o_proj prefill GEMMs. Both ranks must agree (`model::startup_parity`):
/// the o_proj arm decides whether the SP reduce-scatter is piped.
pub(crate) fn attn_w4a4_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_ATTN_W4A4").is_some())
}
