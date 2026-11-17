// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1's attention cache: a 128-token window of keys on every
//! layer, plus compressed keys and index keys that a few source layers
//! produce and the layers after them read.

pub mod block;
mod block_attention;
mod block_compress;
mod block_draft;
mod block_index;
mod block_moe;
mod block_out;
pub mod cache_recipe;
pub mod draft;
pub mod draft_head;
pub mod draft_ops;
mod draft_propose;
pub mod draft_seq;
pub mod engram;
pub mod engram_table;
pub mod gemm;
pub mod hc;
pub mod index;
pub mod index_recipe;
pub mod layer;
pub mod moe;
pub mod moe_gemm;
pub mod moe_grouped;
pub mod ops;
pub mod parts;
mod profile;
mod seen;
mod shape;
pub mod side;
mod state;
pub mod tail;
pub mod weights;
mod weights_columns;
mod weights_experts;

#[cfg(test)]
mod test_support;
