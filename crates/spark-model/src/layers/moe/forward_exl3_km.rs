// SPDX-License-Identifier: AGPL-3.0-only

//! K=m (m = 4..=16) verify rows on a packed-EXL3 MoE.
//!
//! The batched verify FFN arm (`FfnComponent::try_forward_km`) was dense-only,
//! so an MTP verify wider than K=3 bailed on this model's mHC layers ("no
//! batched MoE arm for K=4"). A packed-EXL3 MoE already runs any row count
//! row by row on its decode kernels and leaves `[m, hidden]` in
//! `moe_output`, which is what that arm promises. Kept in its own file so the
//! packed forward (`forward_exl3.rs`) stays untouched.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::*;

impl MoeLayer {
    /// Whether this MoE block is registered as packed EXL3.
    pub(crate) fn is_exl3_packed(&self) -> bool {
        self.exl3_packed()
    }

    /// Run `m` verify rows on the packed-EXL3 path. `Ok(false)` when this
    /// block is not packed EXL3 (caller falls back).
    pub(crate) fn exl3_forward_km(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        self.try_forward_exl3(input, m, ctx, stream)
    }
}
