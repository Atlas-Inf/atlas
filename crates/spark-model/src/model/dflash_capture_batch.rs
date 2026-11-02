// SPDX-License-Identifier: AGPL-3.0-only

//! Slot-addressed batched DFlash capture and temporary commit-front packing.

use super::types::TransformerModel;
use anyhow::Result;

impl TransformerModel {
    /// Batched UNIFIED_CTX capture: seq i owns rows
    /// `[i * kmax, i * kmax + ks[i])` of `dflash_hidden_save`.
    pub(super) fn try_dflash_capture_batched(
        &self,
        layer_idx: usize,
        ks: &[usize],
        off: &[usize],
        stream: u64,
    ) -> Result<()> {
        Self::try_dflash_capture_batched_at(self, layer_idx, ks, off, None, stream)
    }

    /// Slot-indexed variant: `slots[i]` is sequence i's STABLE SSM slot. The
    /// capture region is `slots[i] * kmax`, not the batch position — after
    /// any mid-batch finish (churn/EOS/max_tokens) the pending set reorders,
    /// and a batch-position region would hand the re-propose another (or a
    /// dead) sequence's hiddens, poisoning the drafter ctx accumulator (the
    /// release-binary churn ILA; see the wave-12 session evidence). `None`
    /// keeps the historical batch-position layout for the single-seq path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_dflash_capture_batched_at(
        &self,
        layer_idx: usize,
        ks: &[usize],
        off: &[usize],
        slots: Option<&[usize]>,
        stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        let slot = match self
            .dflash_capture_layers
            .iter()
            .position(|&l| l == layer_idx)
        {
            Some(s) => s,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let ctx_slot_bytes = self.dflash_capture_layers.len() * h * bf16;
        let kmax = self.dflash_hidden_save_rows;
        let nseq = self.dflash_hidden_save_nseq;
        if let Some(slots) = slots {
            anyhow::ensure!(
                slots.len() == ks.len(),
                "DFlash batched capture owner-slot width {} != batch width {}",
                slots.len(),
                ks.len()
            );
        }
        for (i, &k) in ks.iter().enumerate() {
            let region = slots.map(|s| s[i]).unwrap_or(i);
            anyhow::ensure!(
                region < nseq,
                "DFlash batched capture owner slot {region} exceeds capacity {nseq}"
            );
            let seq_base = dst.offset(region * kmax * ctx_slot_bytes);
            self.dflash_capture_rows(
                off[i],
                k.min(kmax),
                seq_base.offset(slot * h * bf16),
                ctx_slot_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// Preserve the current C=1 front before any slot-addressed batch region is
    /// packed over it. The preserve area is the extra sequence-sized region
    /// allocated after all owner slots.
    pub(super) fn preserve_dflash_save_front(&self, k: usize, _stream: u64) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let ctx_slot_bytes = self.dflash_capture_layers.len() * self.config.hidden_size * 2;
        if ctx_slot_bytes == 0 {
            return Ok(());
        }
        let kmax = self.dflash_hidden_save_rows;
        let n = k.min(kmax);
        if n == 0 {
            return Ok(());
        }
        // default_stream, not the caller arg: the front is written by the
        // capture and read by commit_ctx on the model's (non-blocking)
        // default stream. The scheduler boundary passes the legacy 0 —
        // honoring it would leave this copy unordered with both sides.
        let stream = self.gpu.default_stream();
        let preserve = dst.offset(self.dflash_hidden_save_nseq * kmax * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(dst, preserve, n * ctx_slot_bytes, stream)
    }

    /// Compact a slot-addressed capture region to the C=1 front for the
    /// commit_ctx reader. The slot-0 region already is the front; its original
    /// rows are preserved once before the batch and restored after the loop.
    pub(super) fn pack_dflash_save_seq(&self, seq_i: usize, k: usize, stream: u64) -> Result<()> {
        Self::pack_dflash_save_region(self, seq_i, k, stream)
    }

    /// Slot-addressed pack: copies the sequence's slot-indexed capture
    /// region to the front for the commit_ctx reader.
    pub(super) fn pack_dflash_save_region(
        &self,
        slot: usize,
        k: usize,
        _stream: u64,
    ) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let h = self.config.hidden_size;
        let ctx_slot_bytes = self.dflash_capture_layers.len() * h * 2;
        if ctx_slot_bytes == 0 {
            return Ok(());
        }
        let kmax = self.dflash_hidden_save_rows;
        anyhow::ensure!(
            slot < self.dflash_hidden_save_nseq,
            "DFlash hidden-save owner slot {slot} exceeds capacity {}",
            self.dflash_hidden_save_nseq
        );
        let n = k.min(kmax);
        if slot == 0 || n == 0 {
            return Ok(());
        }
        // See preserve_dflash_save_front: must be ordered with the capture
        // (writer) and commit_ctx (reader), both on the default stream.
        let stream = self.gpu.default_stream();
        let src = dst.offset(slot * kmax * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(src, dst, n * ctx_slot_bytes, stream)
    }

    /// Restore the original C=1 front after the batched commit loop packed
    /// owner regions over it. The scheduler calls this only when owner slot 0
    /// was present and therefore was explicitly preserved before the loop.
    pub(super) fn restore_dflash_save_front(&self, k: usize, _stream: u64) -> Result<()> {
        let dst = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let ctx_slot_bytes = self.dflash_capture_layers.len() * self.config.hidden_size * 2;
        if ctx_slot_bytes == 0 {
            return Ok(());
        }
        let kmax = self.dflash_hidden_save_rows;
        let n = k.min(kmax);
        if n == 0 {
            return Ok(());
        }
        // See preserve_dflash_save_front: must be ordered with the capture
        // (writer) and commit_ctx (reader), both on the default stream.
        let stream = self.gpu.default_stream();
        let preserve = dst.offset(self.dflash_hidden_save_nseq * kmax * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(preserve, dst, n * ctx_slot_bytes, stream)
    }
}
