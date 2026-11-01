// SPDX-License-Identifier: AGPL-3.0-only

//! Per-draft confidence from the DFlash2 candidate selector
//! (`ATLAS_DFLASH_CONF_WIDTH=1` or `ATLAS_DFLASH_CONF_LOG=1`).
//!
//! The selector's confidence twin makes the same picks as the production
//! kernel and writes, after a block's `gamma` tokens, `gamma` f32 words: the
//! log of each pick's softmax probability over the row's scored candidates.
//! They ride the draft readback the propose already does, and the scheduler
//! sizes the verify from them (`dflash_conf_width`). Off, the head launches
//! the production selector and reads `gamma` words, as before.
//!
//! Prior art: the signal is the per-draft "log max softmax" of the selector's
//! scores that knapcio's `GLM_DRAFT_TRUNC` truncates on
//! (<https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4>,
//! `overlay/glm_draft_trunc.py`; ideas, no code; docs/glm-prior-art.md).

use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::{BlockDiffusionDraftHead, DflashScratch};

/// The selector to launch and whether it reports confidences: the confidence
/// twin when wanted and shipped by the target, else the production kernel.
pub(super) fn selector_kernel(gpu: &dyn GpuBackend, want: bool) -> (Option<KernelHandle>, bool) {
    let twin = want
        .then(|| {
            gpu.kernel(
                "glm_dflash2_selector_conf",
                "dflash2_candidate_selector_conf",
            )
        })
        .and_then(Result::ok)
        .filter(|kernel| kernel.0 != 0);
    if want && twin.is_none() {
        tracing::warn!(
            "DFlash draft confidence requested, but this target ships no confidence selector: off"
        );
    }
    match twin {
        Some(kernel) => (Some(kernel), true),
        None => (
            gpu.kernel("dflash2_candidate_selector", "dflash2_candidate_selector")
                .ok(),
            false,
        ),
    }
}

/// Bytes of one block's selector output: `gamma` tokens, then with
/// confidences `gamma` f32.
pub(super) fn record_bytes(gamma: usize, conf: bool) -> usize {
    gamma * 4 * (1 + conf as usize)
}

/// The confidences of a block's first `drafts` drafts, in verify order, from
/// its selector output: row `j + 1` is draft `j` (row 0 is the anchor, not a
/// draft). Empty for an output without confidences.
pub(super) fn draft_conf(record: &[u8], gamma: usize, drafts: usize) -> Vec<f32> {
    record
        .chunks_exact(4)
        .skip(gamma + 1)
        .take(drafts.min(gamma.saturating_sub(1)))
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

impl BlockDiffusionDraftHead {
    /// Bytes of one block's selector output on this head.
    pub(super) fn draft_record_bytes(&self) -> usize {
        record_bytes(self.gamma, self.startup.diagnostics.draft_conf)
    }

    /// Confidences of the `drafts` drafts `forward_block` just read back into
    /// the lane's pinned buffer.
    pub(super) fn host_draft_conf(&self, scratch: &DflashScratch, drafts: usize) -> Vec<f32> {
        let pinned = scratch
            .draft_tokens_host_pinned
            .load(std::sync::atomic::Ordering::Relaxed);
        if !self.startup.diagnostics.draft_conf || pinned.is_null() {
            return Vec::new();
        }
        // SAFETY: the buffer is `draft_record_bytes()` long (`from_weights`
        // allocates it with the same expression) and the readback into it
        // was synchronized before `forward_block` returned the drafts.
        let record = unsafe { std::slice::from_raw_parts(pinned, self.draft_record_bytes()) };
        draft_conf(record, self.gamma, drafts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(gamma: usize, conf: &[f32]) -> Vec<u8> {
        let mut bytes: Vec<u8> = (0..gamma as u32).flat_map(u32::to_le_bytes).collect();
        bytes.extend(conf.iter().flat_map(|c| c.to_le_bytes()));
        bytes
    }

    #[test]
    fn confidences_follow_the_drafts_in_verify_order() {
        // Row 0 is the anchor: its word is skipped, rows 1.. are drafts 0..
        let conf = [0.0, -0.1, -0.2, -0.3];
        let record = record(4, &conf);
        assert_eq!(record.len(), record_bytes(4, true));
        assert_eq!(draft_conf(&record, 4, 3), vec![-0.1, -0.2, -0.3]);
        // The draft cap truncates the confidences with the drafts.
        assert_eq!(draft_conf(&record, 4, 2), vec![-0.1, -0.2]);
        assert_eq!(draft_conf(&record, 4, 9), vec![-0.1, -0.2, -0.3]);
    }

    #[test]
    fn an_output_without_confidences_reads_as_not_measured() {
        let record = record(4, &[]);
        assert_eq!(record.len(), record_bytes(4, false));
        assert!(draft_conf(&record, 4, 3).is_empty());
    }
}
