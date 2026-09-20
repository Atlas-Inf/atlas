// SPDX-License-Identifier: AGPL-3.0-only

//! The pre-mutation QSA-inert-bound guard for the batched multi-seq path.
//!
//! Split out of `mod.rs` for the file-size cap, and because the guard is the
//! one thing in that entry point that reads a property of the LAYER rather than
//! of the step: it exists so the refusal happens before any ingest mutation,
//! where the late `sel.is_none()` check inside the ingest path fires after it.

use anyhow::Result;

use super::super::Qwen3AttentionLayer;

/// Refuse a batched row that would run with an ACTIVE QSA selection.
///
/// Past `inert_bound()` the selection is provably not all-visible, and the
/// batched ms path cannot serve that — the verify would return
/// `VerifyUnsupportedWithActiveQsa` after the mutation, which finishes the
/// request. Checking here turns it into a clean decline instead.
pub(super) fn ensure_rows_below_inert_bound(
    layer: &Qwen3AttentionLayer,
    seq_lens: &[usize],
    num_seqs: usize,
) -> Result<()> {
    let Some(qsa) = layer.qsa.as_ref() else {
        return Ok(());
    };
    let bound = qsa.inert_bound();
    for (i, &len) in seq_lens.iter().take(num_seqs).enumerate() {
        anyhow::ensure!(
            len < bound,
            "VerifyUnsupportedWithActiveQsa: row {i} visible {len} >= inert bound {bound}"
        );
    }
    Ok(())
}
