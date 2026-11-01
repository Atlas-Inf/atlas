// SPDX-License-Identifier: AGPL-3.0-only

//! Generic transformer model.
//!
//! The model loop (embed -> layers -> norm -> lm_head) is architecture-
//! agnostic. Layer-specific logic lives in `TransformerLayer` implementations.
//!
//! Wave 4b1 split: this module was originally a single 8,690 LoC `model.rs`.
//! Sub-modules now hold:
//!   - `types`             struct TransformerModel + PinnedMetaStaging
//!   - `ssm_pool`          SsmStatePool
//!   - `ssm_snapshot`      SsmSnapshotPool
//!   - `block_mgmt`        free-fn helpers (apply_evicted_blocks etc.)
//!   - `impl_a1/2/3`       first inherent `impl TransformerModel` block
//!   - `impl_b1/2/3`       second inherent `impl TransformerModel` block
//!   - `trait_impl`        single `impl Model for TransformerModel` block
//!     **FLAGGED ≤500 LoC cap** — Rust does NOT allow
//!     splitting one trait impl across files (E0119),
//!     and breaking it into inherent-helper delegation
//!     is a semantic refactor outside this wave's scope.
//!   - `drop`              `impl Drop for TransformerModel`
//!   - `tests`             extracted unit tests

#![allow(unused_imports, dead_code)]

pub(crate) mod block_mgmt;
mod block_table_upload;
pub(crate) mod decode_pieces;
<<<<<<< HEAD
=======
mod dflash_capture_batch;
pub(crate) mod dflash_ctx_window;
mod dflash_warm_ctx;
>>>>>>> 5c15afb91 (dflash: warm-ctx drafting — real per-frame windows + device-resident restore)
mod draft_assist;
pub(crate) mod qsa_commit_table;
pub use draft_assist::DraftAssist;
pub(crate) mod drafter_context;
pub(crate) mod drop;
pub mod glm_c4;
pub(crate) mod glm_cache_plan;
mod glm_fused_chunk;
pub mod glm_independent;
pub(crate) mod glm_k3_head;
mod glm_long_verify;
mod glm_prefill_sp;
mod glm_verify_masks;
mod glm_vocab_split;
mod qwen4exp_prefill_sp;
pub use glm_verify_masks::MASKED_VERIFY;
pub(crate) use glm_verify_masks::prepare as prepare_glm_verify_masks;
pub(crate) mod graph_flags;
pub(crate) use glm_vocab_split::prepare_shard_mxfp8 as prepare_glm_head_mxfp8;
pub(crate) mod dspark_generation;
#[cfg(test)]
mod dspark_generation_tests;
pub(crate) mod dspark_pool;
#[cfg(test)]
mod dspark_pool_tests;
mod final_norm;
pub(crate) mod impl_a1;
pub(crate) mod impl_a1_init;
mod impl_a1_spec_init;
pub(crate) mod impl_a2;
mod impl_a2_cmd_words;
mod impl_a2_ep_vision;
mod impl_a2_ep_worker;
mod impl_a2_seq_lens;
pub(crate) mod impl_a3;
mod impl_a3_embed;
pub(crate) mod impl_b1;
pub(crate) mod impl_b2;
pub(crate) mod impl_b3;
pub(crate) mod impl_b3_accessors;
pub(crate) mod impl_b3_dflash;
pub(crate) mod impl_lora;
pub(crate) mod impl_lora_swap;
mod impl_ngram;
pub mod kv_admission;
pub(crate) mod kv_lookahead;
mod kv_nvme;
pub(crate) mod mtp_carry;
pub(crate) mod pinned_pack;
pub(crate) mod pinned_upload;
pub(crate) mod prefix_share;
pub(crate) mod qwen4exp_batch_fast;
pub(crate) mod qwen4exp_exact_verify;
pub(crate) mod qwen4exp_lmhead_split;
pub(crate) mod qwen4exp_mtp_depth;
pub(crate) mod qwen4exp_step_copies;
pub(crate) mod ssm_batched_copy;
pub(crate) mod ssm_indexed_decode;
pub(crate) mod ssm_pool;
mod ssm_pool_teardown;
pub(crate) mod ssm_pools;
pub(crate) mod ssm_snapshot;
pub mod ssm_snapshot_budget;
pub(crate) mod ssm_snapshot_faultin;
pub(crate) mod ssm_snapshot_spill;
mod ssm_snapshot_teardown;
pub(crate) mod ssm_spill_gate;
pub(crate) mod ssm_spill_staging;
pub(crate) mod ssm_tier;
mod ssm_verify_attach;
pub mod startup_parity;
pub(crate) mod token_overlay;
pub(crate) mod trait_impl;
pub(crate) mod types;
pub(crate) mod verify_pieces;
mod vision_cache;
pub(crate) mod vision_transport;
pub use vision_transport::VisionCapacityError;
mod warm_turn;

// Served NLLB-200 / M2M-100 encoder-decoder model (CUDA/GB10 serving path).
#[cfg(feature = "cuda")]
pub mod nllb;

pub use types::TransformerModel;

/// `ATLAS_EP_PROTOCOL=v2`: the head sends a slot id ahead of every worker
/// command (`impl_a2`). Every rank must run the same value
/// ([`startup_parity`]): the worker would read the slot id as the command.
pub fn ep_protocol_v2_requested() -> bool {
    matches!(std::env::var("ATLAS_EP_PROTOCOL").as_deref(), Ok("v2"))
}
