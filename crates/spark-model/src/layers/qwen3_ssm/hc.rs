// SPDX-License-Identifier: AGPL-3.0-only

//! mHC on the GDN layer.
//!
//! Qwen3.8-Flash-Next carries a `hc_mult`-wide residual highway on ALL 48
//! layers, 36 of which are GDN. DeepSeek-V4 — the model Atlas built mHC for —
//! is all-attention, so `Qwen3SsmLayer` never needed to know about it.
//!
//! Holding the weights is not the same as running them: the forward
//! integration is unbuilt, and the guard below refuses rather than let a GDN
//! layer run on a stream it never mixed.

use super::Qwen3SsmLayer;
impl Qwen3SsmLayer {
    /// Attach mHC weights. See the `hc` field: holding them is not the same
    /// as running them, and the forward paths refuse while the integration
    /// is unbuilt.
    pub fn set_hc_weights(&mut self, hc: crate::layers::qwen3_attention::HcWeights) {
        self.hc = Some(hc);
    }

    /// Refuse rather than run the SSM block on a stream that was never mixed.
    pub(crate) fn ensure_no_unwired_hc(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.hc.is_none(),
            "qwen3_ssm: mHC weights are attached but the GDN forward does not \
             run them yet. Wiring needs hc_pre before the SSM block and \
             hc_post after it, in this layer's prefill AND decode paths, and \
             `prefill_inner`'s own residual add has to be reconciled with the \
             highway first so the block output is not counted twice. \
             Refusing: a GDN layer running on an unmixed stream produces \
             plausible, wrong activations. Avarok #753 item B."
        );
        Ok(())
    }
}
