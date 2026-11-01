// SPDX-License-Identifier: AGPL-3.0-only

//! The single-sequence entry to the attention rows: one recurrent state owns
//! every row. Split out of `forward_attention.rs` (500-LoC cap).

use super::*;

impl Glm5KdaLayer {
    pub(super) fn forward_attention(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        tokens: usize,
        decode: bool,
        capture_verify_intermediates: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<FfnPhase> {
        let state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("GLM-5 KDA expected SsmLayerState"))?;
        ensure!(!state.h_is_f16, "GLM-5 KDA requires FP32 recurrent state");
        self.forward_attention_rows(
            hidden,
            tokens,
            decode,
            capture_verify_intermediates,
            ctx,
            stream,
            &mut |projected, g1, beta| {
                self.forward_recurrent(
                    projected,
                    g1,
                    beta,
                    state,
                    tokens,
                    decode,
                    capture_verify_intermediates,
                    ctx,
                    stream,
                )
            },
        )
    }
}
