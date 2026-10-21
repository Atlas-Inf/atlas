// SPDX-License-Identifier: AGPL-3.0-only

//! `MoeLayer::batched_prefill_guards` — the early-bail checks that precede
//! `forward_batched`'s batched decode path.

use super::*;

impl MoeLayer {
    /// Early bail-out guards for `forward_batched`: `Ok(true)` means the call
    /// was already served (routed to `forward_prefill`); `Ok(false)` falls
    /// through to the batched path.
    pub(super) fn batched_prefill_guards(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        self.btile_input_guard(input, num_tokens, ctx, stream)?;
        // The equal-memory NVFP4 MMQ repack replaces the checkpoint-native
        // expert payloads. Only the grouped dispatcher understands that block
        // layout; the per-token fused kernels below expect the original packed
        // weights. Small speculative-verification batches reach this entry
        // through the K2/K3/K5 fallbacks, so route them through the same MMQ
        // pipeline as ordinary prefill rather than interpreting repacked bytes
        // with the old-layout kernels. `forward_prefill` does not bounce MMQ
        // layers back here, so this cannot recurse.
        if self.nvfp4_mmq_layout || self.routed_scales_released {
            self.forward_prefill(input, num_tokens, ctx, stream)?;
            return Ok(true);
        }
        Ok(false)
    }
}
