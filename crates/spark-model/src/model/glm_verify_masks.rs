// SPDX-License-Identifier: AGPL-3.0-only
//! Per-row grammar bitmasks for a strict structured-output GLM verify
//! (`ATLAS_GLM_STRICT_SPEC=1`).
//!
//! The head walks the grammar matcher through the drafted tokens and builds
//! one bitmask per verify row (vLLM #14702, "Enable Speculative Decoding with
//! Structured Outputs"; #44297 for the reasoning boundary). On the TP2 vocab
//! split each rank takes the argmax of its own half of the vocabulary, so rank
//! 1, which holds no grammar state, needs the masks too. They live in a buffer
//! both ranks attach at load (before KV sizing), and the next split-head
//! verify applies them before each rank's partial argmax (`glm_vocab_split`).
//! Unmasked verifies never touch any of this.
//!
//! Wire contract, in this order: the head validates and uploads the masks
//! (`upload_row_masks`) before it sends anything, then sends the generic
//! verify command, the width word with [`MASKED_VERIFY`] set and the tokens;
//! then both ranks broadcast `rows * words` 32-bit words (`send_row_masks`).
//! Only that broadcast follows the flag, so nothing that can fail locally
//! separates the ranks once the worker expects masks.
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::types::TransformerModel;

/// Flag on the generic-verify width word: per-row masks follow the tokens.
pub const MASKED_VERIFY: u32 = 1 << 16;
/// Widest masked verify (the worker's generic-verify width bound).
const MAX_ROWS: usize = 32;

/// The worker's reading of a generic-verify width word: `(rows, masked)`.
pub(super) fn split_verify_width(word: u32) -> (usize, bool) {
    ((word & !MASKED_VERIFY) as usize, word & MASKED_VERIFY != 0)
}

/// 32-bit words per row mask of a `vocab`-token vocabulary.
fn mask_words(vocab: usize) -> usize {
    vocab.div_ceil(32)
}

/// Attach the mask buffer (`MAX_ROWS` rows, ~0.6 MiB for GLM-5.3) on every
/// rank whose verify head may be the TP2 vocab split. Called before KV sizing.
pub(crate) fn prepare(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    buffers: &mut BufferArena,
) -> Result<()> {
    if config.model_type != "glm5_next"
        || config.ep_world_size != 2
        || !super::glm_vocab_split::enabled()
    {
        return Ok(());
    }
    buffers.attach_verify_masks(MAX_ROWS * mask_words(config.vocab_size) * 4, gpu)
}

/// How many rows are staged for the next verify (0: none). Pure state, so the
/// stage/take protocol is testable without a GPU.
#[derive(Default)]
pub(crate) struct VerifyRowMasks {
    armed: AtomicUsize,
}

impl VerifyRowMasks {
    /// Masks for `rows` rows are in the buffer for the next verify.
    fn arm(&self, rows: usize) {
        self.armed.store(rows, Ordering::Release);
    }

    /// Nothing is staged (the worker's state before a verify command).
    pub(super) fn reset(&self) {
        self.armed.store(0, Ordering::Release);
    }

    /// Consume the staging for a verify of `rows` rows: `Ok(true)` when masks
    /// were staged for it, `Ok(false)` when none were. Always leaves nothing
    /// staged, so a later unmasked verify never sees stale masks.
    fn take(&self, rows: usize) -> Result<bool> {
        let armed = self.armed.swap(0, Ordering::AcqRel);
        ensure!(
            armed == 0 || armed == rows,
            "masked verify staged {armed} rows for a {rows}-row verify"
        );
        Ok(armed != 0)
    }
}

impl TransformerModel {
    fn verify_mask_buffer(&self, rows: usize) -> Result<(DevicePtr, usize)> {
        ensure!(
            (1..=MAX_ROWS).contains(&rows),
            "masked verify rows {rows} out of range"
        );
        let bytes = rows * mask_words(self.config.vocab_size) * 4;
        let (buf, cap) = self
            .buffers
            .verify_masks()
            .ok_or_else(|| anyhow::anyhow!("masked verify: no mask buffer attached"))?;
        ensure!(
            bytes <= cap,
            "masked verify: {bytes} B over the {cap} B buffer"
        );
        Ok((buf, bytes))
    }

    /// Head only, before any verify command: validate the masks and upload
    /// them. A failure here leaves both ranks untouched.
    pub(crate) fn upload_row_masks(&self, rows: usize, masks: &[u32]) -> Result<()> {
        let (buf, bytes) = self.verify_mask_buffer(rows)?;
        ensure!(
            masks.len() * 4 == bytes,
            "masked verify: {} mask words for {rows} rows",
            masks.len()
        );
        // SAFETY: a live &[u32] reinterpreted as its own bytes (align 1).
        let view: &[u8] = unsafe { std::slice::from_raw_parts(masks.as_ptr().cast::<u8>(), bytes) };
        self.gpu.copy_h2d(view, buf)
    }

    /// Every rank, after the verify tokens: broadcast the head's masks
    /// (synchronous) and stage them for the next verify.
    pub(crate) fn send_row_masks(&self, rows: usize) -> Result<()> {
        let (buf, bytes) = self.verify_mask_buffer(rows)?;
        if let Some(comm) = self.comm.as_ref().filter(|c| c.world_size() > 1) {
            comm.broadcast(buf.0, bytes, 0)?;
        }
        self.verify_row_masks.arm(rows);
        Ok(())
    }

    /// Take the staged masks for a verify of `rows` rows, if any were staged.
    pub(crate) fn take_verify_row_masks(&self, rows: usize) -> Result<Option<DevicePtr>> {
        if !self.verify_row_masks.take(rows)? {
            return Ok(None);
        }
        Ok(Some(self.verify_mask_buffer(rows)?.0))
    }
}

#[cfg(test)]
mod tests {
    use super::{MASKED_VERIFY, VerifyRowMasks, mask_words, split_verify_width};

    #[test]
    fn the_width_word_round_trips_rows_and_the_mask_flag() {
        // Unmasked verifies send the bare width, as before this flag existed.
        for rows in [2usize, 9, 32] {
            assert_eq!(split_verify_width(rows as u32), (rows, false));
            assert_eq!(
                split_verify_width(rows as u32 | MASKED_VERIFY),
                (rows, true)
            );
        }
        // Both ranks size the broadcast from the same rows and vocabulary.
        assert_eq!(mask_words(154_856), 4_840);
        assert_eq!(mask_words(154_880), 4_840);
    }

    #[test]
    fn staged_masks_serve_exactly_one_verify_of_their_width() {
        let m = VerifyRowMasks::default();
        assert!(!m.take(9).unwrap(), "nothing staged: an unmasked verify");
        m.arm(9);
        assert!(m.take(9).unwrap());
        assert!(!m.take(9).unwrap(), "the next verify is unmasked again");
        m.arm(9);
        assert!(m.take(5).is_err(), "a width mismatch is a protocol error");
        assert!(!m.take(5).unwrap(), "and leaves nothing staged");
        m.arm(3);
        m.reset();
        assert!(!m.take(3).unwrap(), "the worker resets before each command");
    }
}
