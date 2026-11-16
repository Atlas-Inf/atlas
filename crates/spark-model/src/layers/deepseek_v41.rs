// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1's attention cache: a 128-token window of keys on every
//! layer, plus compressed keys and index keys that a few source layers
//! produce and the layers after them read.

pub mod block;
mod block_attention;
mod block_index;
mod block_moe;
pub mod cache_recipe;
pub mod hc;
pub mod index;
pub mod index_recipe;
pub mod moe;
pub mod ops;
mod shape;
pub mod side;
mod state;
pub mod weights;

#[cfg(test)]
mod test_support;
