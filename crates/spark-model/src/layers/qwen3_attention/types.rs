// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3 attention struct definitions: `MlaWeights` (latent attention
//! 2-step decode) and `Qwen3AttentionLayer` (full attention layer).

pub(crate) use super::types_weights::HeadGateActivation;
pub use super::types_weights::{HcWeights, MlaWeights};

pub use super::types_layer::Qwen3AttentionLayer;
