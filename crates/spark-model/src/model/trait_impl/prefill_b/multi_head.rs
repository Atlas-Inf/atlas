// SPDX-License-Identifier: AGPL-3.0-only

//! The logits of a multi-sequence prefill pass (`multi`): every sequence's
//! last row through the LM head in one weight pass, each row the bytes of
//! the single prefill's `lm_head` (`dense_gemv_bf16`):
//! `Qwen4ExpWideRows::dense_rows` runs the row-exact `dense_gemv_bf16_batchm`
//! / `qwen4exp_bf16_rows16/32` (the batched decode head's kernels), where the
//! single prefill re-streamed the 1.27 GB head once a sequence.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;

/// `(ptr, rows)` of the pass's logits rows (`vocab` BF16 each) followed by
/// the gathered last hidden rows, grown on demand.
static ROWS: std::sync::Mutex<(u64, usize)> = std::sync::Mutex::new((0, 0));

impl TransformerModel {
    /// Device room for `n` logits rows (and as many hidden rows after them):
    /// `(base, capacity in rows)`.
    pub(super) fn prefill_multi_rows(&self, n: usize, stream: u64) -> Result<(DevicePtr, usize)> {
        let row = (self.config.vocab_size + self.config.hidden_size) * 2;
        let mut buf = ROWS.lock().unwrap();
        if buf.1 < n {
            if buf.0 != 0 {
                self.gpu.synchronize(stream)?;
                self.gpu.free(DevicePtr(buf.0))?;
            }
            let rows = n.max(16);
            *buf = (self.gpu.alloc(rows * row)?.0, rows);
        }
        Ok((DevicePtr(buf.0), buf.1))
    }

    /// Logits row `i` of [`Self::prefill_multi_rows`].
    pub(super) fn prefill_multi_logits_row(&self, base: DevicePtr, i: usize) -> DevicePtr {
        base.offset(i * self.config.vocab_size * 2)
    }

    /// Project the last rows `last[i]` of `hidden` into logits rows `0..`,
    /// when the head is the plain BF16 one the batched kernels serve.
    /// `Ok(false)`: the caller's per-sequence finish projects them.
    pub(super) fn prefill_multi_lm_head(
        &self,
        base: DevicePtr,
        cap: usize,
        last: &[usize],
        stream: u64,
    ) -> Result<bool> {
        let n = last.len();
        if n < 2
            || self.lm_head_fp8.is_some()
            || self.lm_head_nvfp4.is_some()
            || self.overlays.is_some()
            || self.logit_softcap_kernel.0 != 0
            || self.dense_gemv_batchm_kernel.0 == 0
        {
            return Ok(false);
        }
        let (h, v) = (self.config.hidden_size, self.config.vocab_size);
        let gathered = base.offset(cap * v * 2);
        let hidden = self.buffers.hidden_states();
        for (i, &row) in last.iter().enumerate() {
            self.final_norm_rows(
                hidden.offset(row * h * 2),
                gathered.offset(i * h * 2),
                1,
                stream,
            )?;
        }
        self.qwen4exp_wide_rows.dense_rows(
            self.gpu.as_ref(),
            self.dense_gemv_batchm_kernel,
            gathered,
            &self.lm_head_weight,
            base,
            (n as u32, v as u32, h as u32, v as u32),
            stream,
        )?;
        Ok(true)
    }
}
