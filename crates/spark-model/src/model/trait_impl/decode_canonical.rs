// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_CANONICAL_VERIFY`: a plain one-row GLM decode is a one-row
//! verify. Adaptive speculation, a failed propose, the thinking and resume
//! guards and every other serial step decode one row through
//! `Model::decode`, whose own pipeline (multi-seq serial MLA with BF16
//! projections and dense attention, the decode KDA recurrence, the BF16 head)
//! gave that row other bits than the same row verified. Here the decode runs
//! the verify forward of that one token (`decode_verify_graphed_kgamma_dispatch`:
//! the MLA prefill lane, the verify KDA recurrence and the canonical kernels),
//! commits it (`commit_accepted_prefix(1, 1)`, a no-op unless KDA records),
//! and completes the split head's half-vocabulary logits row with the peer's
//! half, so the sampler's first-index argmax picks the verify's token. Both
//! ranks take it: the EP worker's decode runs the same function.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Whether this model's plain decode is the canonical one-row verify:
    /// GLM on the DFlash prefill-verify lane, the canonical kernels on, BF16
    /// logits and the MXFP8 split verify head serving.
    pub(super) fn decode_is_canonical_verify(&self) -> bool {
        self.config.model_type == "glm5_next"
            && crate::layers::canonical_verify::enabled()
            && crate::speculative::glm_repair_policy::dflash_enabled()
            && crate::speculative::glm_repair_policy::dflash_prefill_verify()
            && !self.use_fp32_logits
            && self.glm_split_head_mxfp8()
    }

    /// The decode of `token` as a one-row verify; `None` when the plain decode
    /// pipeline serves (see [`Self::decode_is_canonical_verify`]).
    pub(super) fn decode_canonical_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        if !self.decode_is_canonical_verify() {
            return Ok(None);
        }
        self.decode_verify_graphed_kgamma_dispatch(&[token], seq, stream, None)?;
        self.commit_accepted_prefix_dispatch(seq, 1, 1)?;
        self.sync_secondary_dispatch()?;
        self.glm_split_head_full_row(self.gpu.default_stream())?;
        Ok(Some(self.buffers.logits()))
    }
}
