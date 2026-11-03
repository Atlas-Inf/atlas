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
    /// Native GLM MTP is admitted from rank-symmetric factory validation and
    /// uses the same target arithmetic as DFlash without inheriting DFlash's
    /// environment switches.
    pub(super) fn native_glm_mtp_enabled(&self) -> bool {
        self.native_glm_mtp
    }

    pub(super) fn canonical_glm_target_enabled(&self) -> bool {
        if self.native_glm_mtp {
            return self.native_canonical_target_ready();
        }
        self.config.model_type == "glm5_next"
            && crate::speculative::glm_repair_policy::dflash_enabled()
    }

    /// Native MTP is a target lane, not a license to use the generic GLM
    /// decode path. The target arithmetic requires the partitioned MXFP8
    /// vocabulary projection and BF16 mHC highway on matching TP2/EP2 or
    /// TP4/EP4 ranks. Factory startup validates the environment before NCCL; this runtime predicate keeps
    /// every rank's dispatch fail-closed if a model is constructed directly.
    pub(super) fn native_canonical_target_ready(&self) -> bool {
        self.native_glm_mtp
            && self.config.model_type == "glm5_next"
            && self.comm.as_ref().is_some_and(|comm| {
                crate::weight_loader::glm5_mode::supported_parallelism(
                    comm.world_size(),
                    self.config.tp_world_size,
                    self.config.ep_world_size,
                ) && comm.rank() < comm.world_size()
                    && self.config.tp_rank == comm.rank()
                    && self.config.ep_rank == comm.rank()
            })
            && crate::layers::canonical_verify::enabled()
            && crate::layers::ops::hc_bf16_for("glm5_next")
            && self.glm_split_head_mxfp8()
    }

    /// Whether this model's plain decode is the canonical one-row verify:
    /// native GLM or GLM on the DFlash prefill-verify lane, the canonical
    /// kernels on, BF16 logits and the MXFP8 split verify head serving.
    pub(super) fn decode_is_canonical_verify(&self) -> bool {
        self.config.model_type == "glm5_next"
            && crate::layers::canonical_verify::enabled()
            && self.canonical_glm_target_enabled()
            && (self.native_glm_mtp
                || crate::speculative::glm_repair_policy::dflash_prefill_verify())
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
        anyhow::ensure!(
            !self.native_glm_mtp || self.native_canonical_target_ready(),
            "native GLM MTP plain decode requires the canonical target path"
        );
        if !self.decode_is_canonical_verify() {
            return Ok(None);
        }
        self.decode_verify_graphed_kgamma_dispatch(&[token], seq, stream, None)?;
        self.commit_accepted_prefix_dispatch(seq, 1, 1)?;
        self.capture_native_mtp_serial(
            seq,
            token,
            self.buffers.norm_output(),
            self.gpu.default_stream(),
        )?;
        self.sync_secondary_dispatch()?;
        // Native verifies complete their rows before returning (`verify_d`).
        if !self.native_glm_mtp {
            self.glm_split_head_full_rows(1, self.gpu.default_stream())?;
        }
        Ok(Some(self.buffers.logits()))
    }
}
