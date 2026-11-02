// SPDX-License-Identifier: AGPL-3.0-only
//! Post-capture vocabulary-head diagnostic; preserves the former dispatch order.
use super::TransformerModel;
use crate::layers::ops;
use anyhow::Result;

fn glm_k5_bf16_lmhead_batchm_check_once() -> bool {
    static CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    std::env::var("ATLAS_GLM_K5_BF16_LMHEAD_BATCHM_CHECK")
        .ok()
        .as_deref()
        == Some("1")
        && !CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed)
}

impl TransformerModel {
    pub(super) fn check_glm_k5_bf16_head(&self, k: usize, out: &[u32], stream: u64) -> Result<()> {
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let out_ptr = self.buffers.scratch();
        // One-shot behavioral oracle for the GLM K=5 BF16 vocabulary-head
        // dispatch. The optimized graph has already produced `out`; rerun the
        // former dense GEMM outside capture, reduce its logits identically,
        // and require every target token ID to agree. The baseline logits may
        // differ below BF16 rounding because the kernels have different
        // accumulation trees; target argmax equality is the inference seam.
        if k == 5
            && self.config.model_type == "glm5_next"
            && self.lm_head_nvfp4.is_none()
            && self.lm_head_fp8.is_none()
            && glm_k5_bf16_lmhead_batchm_check_once()
        {
            let normed = self.buffers.norm_output();
            let logits = self.buffers.logits();
            let vocab = self.config.vocab_size;
            ops::dense_gemm(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                normed,
                &self.lm_head_weight,
                logits,
                k as u32,
                vocab as u32,
                h as u32,
                stream,
            )?;
            for t in 0..k {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    logits.offset(t * vocab * bf16),
                    out_ptr.offset(t * 4),
                    vocab as u32,
                    stream,
                )?;
            }
            let mut baseline_raw = vec![0u8; k * 4];
            self.gpu.copy_d2h(out_ptr, &mut baseline_raw)?;
            let baseline: Vec<u32> = baseline_raw
                .chunks_exact(4)
                .map(|raw| u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
                .collect();
            anyhow::ensure!(
                out == baseline,
                "GLM K5 BF16 LM-head batchm oracle mismatch: batchm={out:?}, dense={baseline:?}"
            );
            tracing::info!(
                "GLM K5 BF16 LM-head batchm oracle passed: all {} target argmax IDs match",
                k
            );
        }

        Ok(())
    }
}
