// SPDX-License-Identifier: AGPL-3.0-only
//! Legacy prefill entry with the optional deferred shared blend.

use super::*;

impl MoeLayer {
    /// Every previous entry retains the exact legacy policy, including K5's
    /// optional deferred shared blend. Pair mode is not selected by an env var.
    pub(super) fn forward_prefill_impl(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
        defer_shared_hc: bool,
    ) -> Result<()> {
        self.forward_prefill_mode(input, num_tokens, ctx, stream, defer_shared_hc)
    }
}
