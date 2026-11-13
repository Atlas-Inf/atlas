// SPDX-License-Identifier: AGPL-3.0-only

//! The deferred-commit accept of a GDN verify: the accepted rows replayed from
//! the live H0 into one layer's `h_state` (`gated_delta_rule_commit`).

use super::*;

impl TransformerModel {
    /// Deferred-commit accept: replay tokens `0..num_accepted` of this
    /// layer's staged verify inputs from the live H0 in `h_state`
    /// (`gated_delta_rule_commit` — same `gated_delta_rule_wyn_impl`
    /// arithmetic as the storing verify, so the result is bit-identical to
    /// `h_state_intermediates[num_accepted - 1]` or the final H). Covers
    /// partial AND full accepts — under defer a full accept is not a
    /// no-op because `h_state` was never written.
    ///
    /// Conv state is NOT touched here — conv intermediates are written and
    /// restored exactly as on the snapshot path.
    pub(in crate::model::trait_impl) fn commit_gdn_deferred(
        &self,
        layer_idx: usize,
        ssm: &crate::layer::SsmLayerState,
        num_accepted: usize,
        stream: u64,
    ) -> Result<()> {
        let nk = self.config.linear_num_key_heads;
        let nv = self.config.linear_num_value_heads;
        let kd = self.config.linear_key_head_dim;
        let vd = self.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let conv_dim = key_dim * 2 + nv * vd;
        let commit_k = self.layers[layer_idx].gdn_commit_kernel();
        anyhow::ensure!(
            commit_k.0 != 0,
            "deferred commit pending but gated_delta_rule_commit unresolved (layer {layer_idx})"
        );
        ops::gdn_commit_accepted(
            self.gpu.as_ref(),
            commit_k,
            ssm.h_state,
            ssm.gdn_commit_qkv,
            ssm.gdn_commit_qkv.offset(key_dim * 2),
            ssm.gdn_commit_qkv.offset(key_dim * 2 * 2),
            ssm.gdn_commit_gb,
            ssm.gdn_commit_gb.offset(nv * 4),
            num_accepted as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32, // qk_stride
            conv_dim as u32, // v_stride
            (nv * 2) as u32, // gb_stride
            stream,
        )
    }
}
