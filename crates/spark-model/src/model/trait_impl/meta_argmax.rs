// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::layers::ops;

impl TransformerModel {
    pub(super) fn argmax_batch_dispatch(
        &self,
        logits_ptr: DevicePtr,
        n: usize,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        let stream = self.gpu.default_stream();
        let v = self.config.vocab_size;
        let bf16 = 2usize;
        let out_ptr = self.buffers.scratch();
        // ONE launch, one block per row. The single-row `argmax_bf16` is a one-CTA
        // reduction (grid [1,1,1]), so n calls on the same stream serialise n
        // single-SM scans: measured 16 x 100.6 us = 1.6 ms per decode step at n=16.
        // The batched kernel runs the identical per-row body, so ties resolve the
        // same way — byte-identical. Falls back to the loop when the kernel set
        // lacks the batched entry.
        fn argmax_batch_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("ATLAS_NO_ARGMAX_BATCH").ok().as_deref() != Some("1"))
        }
        if self.argmax_batch_kernel.0 != 0 && argmax_batch_enabled() {
            ops::argmax_bf16_batch(
                self.gpu.as_ref(),
                self.argmax_batch_kernel,
                logits_ptr,
                out_ptr,
                v as u32,
                n as u32,
                v as u32,
                stream,
            )?;
        } else {
            for i in 0..n {
                let logits_i = logits_ptr.offset(i * v * bf16);
                let out_i = out_ptr.offset(i * 4);
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits_i,
                    out_i,
                    v as u32,
                    stream,
                )?;
            }
        }
        let mut buf = vec![0u8; n * 4];
        self.gpu.copy_d2h(out_ptr, &mut buf)?;
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            results.push(u32::from_le_bytes([
                buf[i * 4],
                buf[i * 4 + 1],
                buf[i * 4 + 2],
                buf[i * 4 + 3],
            ]));
        }
        Ok(results)
    }

    pub(super) fn hidden_after_norm_dispatch(&self) -> DevicePtr {
        // norm_output() holds the post-final-norm hidden state from the last decode
        self.buffers.norm_output()
    }
}
