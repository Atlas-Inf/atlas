// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone Hy4 blocks. Full-model assembly remains guarded until MLA and
//! Engram are implemented. All activations/highway buffers are BF16 except
//! FP32 routing scores and iHC gates. No implicit GLM environment policies.

pub mod attention;
pub mod batch;
pub mod engram;
pub mod ffn;
mod ffn_batch;
mod ffn_forward;
mod ffn_scratch;
pub mod hc;
pub mod timing;
mod weights;
pub use ffn_scratch::Hy4FfnScratch;

#[cfg(test)]
mod tests;
