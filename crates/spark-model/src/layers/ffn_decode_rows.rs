// SPDX-License-Identifier: AGPL-3.0-only
//! `Ffn::forward_decode_rows` — the decode/verify multi-row sibling of
//! `forward_prefill`. Split into its own file for the 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;

use super::FfnComponent;

impl FfnComponent {
    /// Decode/verify multi-row variant of `forward_prefill` — dense FFNs
    /// tag the call [`crate::layers::dense_ffn::FfnMmPhase::DecodeRows`] so
    /// the NVFP4 MMQ arm may serve bounded-M verify rows while prefill
    /// stays W4A16. MoE and empty layers take the identical body either
    /// way (the flag is dense-only).
    pub fn forward_decode_rows(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.forward_prefill(input, num_tokens, ctx, stream),
            Self::Dense(d) => d.forward_decode_rows(input, num_tokens, ctx, stream),
            Self::None => {
                let _ = (input, num_tokens);
                Ok(())
            }
        }
    }
}
