// SPDX-License-Identifier: AGPL-3.0-only
//! Ordinary GLM EP prefill computes on the default stream on both ranks.
//! Its head must mirror the worker's one normalization after every chunk.
use anyhow::Result;
use spark_model::traits::{Model, SequenceState};

fn ordinary_glm_ep(model: &dyn Model) -> bool {
    model.is_ep() && model.supports_chunked_mla()
}

pub(super) fn initial(model: &dyn Model, seq: &SequenceState) -> Result<()> {
    if ordinary_glm_ep(model) {
        model.normalize_ssm_states(seq, model.default_stream())?;
    }
    Ok(())
}

pub(super) fn continuation(
    model: &dyn Model,
    seq: &SequenceState,
    prefill_stream: u64,
) -> Result<()> {
    let stream = if ordinary_glm_ep(model) {
        model.default_stream()
    } else {
        prefill_stream
    };
    model.normalize_ssm_states(seq, stream)
}
