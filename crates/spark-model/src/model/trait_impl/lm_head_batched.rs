// SPDX-License-Identifier: AGPL-3.0-only

//! The decode LM head, batched — ONE source of truth for two call sites.
//!
//! `decode_a2.rs` (the pure-decode batch) and `decode_b2.rs`
//! (`mixed_final_norm_lm_head`, the prefill+decode co-dispatch head reached
//! from `decode_b.rs` via `mixed_forward_dispatch`) both finish a step with
//! RMS-norm then the vocab projection. They had drifted: `decode_a2` grew the
//! full ladder while `decode_b2` still looped `padded_n` times through
//! `ops::w4a16_gemv`, re-reading the whole vocab weight once per row —
//! ~N x 254 MB/step on live continuous-batching traffic.
//!
//! Credit for spotting the site: @rsafier in #332, which fixed it with a bare
//! default-OFF `batch16`. This lifts `decode_a2`'s ladder instead, so the two
//! heads cannot diverge NUMERICALLY at the same batch width — two
//! independently-maintained ladders would be a second source of truth for
//! which kernel a given `padded_n` lands on, and the first thing to go wrong
//! would be a silent accuracy difference between the pure-decode and
//! co-dispatch paths.
//!
//! HONESTY: this is a bandwidth-accounting argument, not a measurement. The
//! mixed path has never been A/B'd. Any throughput claim for it must be gated
//! spec-OFF or on reversed-order pairs (a single spec-ON pair drifts +/-2%).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::types::TransformerModel;
use crate::layers::ops;

/// Batched-GEMV decode lm_head: **ON by default**, disabled by
/// `ATLAS_NO_LM_HEAD_BATCH_GEMV=1`.
///
/// Strict `== "1"` on an `ATLAS_NO_*` name, not a presence check — presence
/// flags in this codebase are ENABLED by `=0`. Read once; this is a per-step
/// site.
pub(super) fn lm_head_batch_gemv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NO_LM_HEAD_BATCH_GEMV").as_deref() != Ok("1"))
}

#[derive(Debug, Clone, Copy)]
enum LmHeadNvfp4Route {
    BatchGemv(KernelHandle),
    TransposedBf16,
    TransposedOther,
    PlainGemm,
}

fn select_lm_head_nvfp4_route(
    padded_n: usize,
    batch_gemv_enabled: bool,
    batch: KernelHandle,
    has_transposed_bf16: bool,
    has_transposed_other: bool,
) -> LmHeadNvfp4Route {
    if batch_gemv_enabled && batch.0 != 0 {
        return LmHeadNvfp4Route::BatchGemv(batch);
    }
    if padded_n >= 5 && has_transposed_bf16 {
        LmHeadNvfp4Route::TransposedBf16
    } else if padded_n >= 5 && has_transposed_other {
        LmHeadNvfp4Route::TransposedOther
    } else {
        LmHeadNvfp4Route::PlainGemm
    }
}

impl TransformerModel {
    /// Project `normed` [padded_n, H] into `logits` [padded_n, V].
    ///
    /// `v` is read from `self.config.vocab_size` rather than passed: it is the
    /// same number at both call sites and a parameter would be a second place
    /// for it to be wrong.
    pub(super) fn lm_head_project_batched(
        &self,
        normed: DevicePtr,
        padded_n: usize,
        h: usize,
        bf16: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let logits = self.buffers.logits();
        let v = self.config.vocab_size;
        if let Some(ref fp8) = self.lm_head_fp8 {
            for i in 0..padded_n {
                ops::dense_gemv_fp8w(
                    self.gpu.as_ref(),
                    self.dense_gemv_fp8w_kernel,
                    normed.offset(i * h * bf16),
                    fp8,
                    logits.offset(i * v * bf16),
                    v as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4) = self.lm_head_nvfp4 {
            // The batch4/8/16 GEMV tiers are byte-identical to M independent
            // scalar GEMVs and therefore own every admitted decode width through
            // 16. Tensor-core/transposed GEMM changes the FP32 reduction order;
            // it is a capability fallback when an exact tier is unavailable or
            // explicitly disabled, not a lossless substitute.
            // Exact-M tier first (batch2..8/16/32 incl. the 5/6/7 tiers);
            // batch16 stays the capability fallback when no exact tier resolved.
            let narrow = self.w4a16_batchm.kernel(padded_n as u32);
            let batch_k = if narrow.0 != 0 {
                narrow
            } else if padded_n <= 16 {
                self.w4a16_gemv_batch16_kernel
            } else {
                KernelHandle(0)
            };
            let route = select_lm_head_nvfp4_route(
                padded_n,
                lm_head_batch_gemv_enabled(),
                batch_k,
                self.w4a16_gemm_t_bf16_kernel.0 != 0 && self.lm_head_nvfp4_t.is_some(),
                self.w4a16_gemm_t_kernel.0 != 0 && self.lm_head_nvfp4_t.is_some(),
            );
            match route {
                LmHeadNvfp4Route::BatchGemv(gemv_k) => ops::w4a16_gemv_batchm(
                    self.gpu.as_ref(),
                    gemv_k,
                    normed,
                    nvfp4,
                    logits,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    stream,
                )?,
                LmHeadNvfp4Route::TransposedBf16 => {
                    let (nvfp4_t, ldb) = self
                        .lm_head_nvfp4_t
                        .as_ref()
                        .expect("route requires transposed LM-head weights");
                    ops::w4a16_gemm_n128_m128_bf16_ldb(
                        self.gpu.as_ref(),
                        self.w4a16_gemm_t_bf16_kernel,
                        normed,
                        nvfp4_t,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        *ldb,
                        stream,
                    )?;
                }
                LmHeadNvfp4Route::TransposedOther => {
                    let (nvfp4_t, ldb) = self
                        .lm_head_nvfp4_t
                        .as_ref()
                        .expect("route requires transposed LM-head weights");
                    ops::w4a16_gemm_n128_ldb(
                        self.gpu.as_ref(),
                        self.w4a16_gemm_t_kernel,
                        normed,
                        nvfp4_t,
                        logits,
                        padded_n as u32,
                        v as u32,
                        h as u32,
                        *ldb,
                        stream,
                    )?;
                }
                LmHeadNvfp4Route::PlainGemm => ops::w4a16_gemm(
                    self.gpu.as_ref(),
                    self.w4a16_gemm_kernel,
                    normed,
                    nvfp4,
                    logits,
                    padded_n as u32,
                    v as u32,
                    h as u32,
                    stream,
                )?,
            }
        } else {
            ops::dense_gemm(
                self.gpu.as_ref(),
                self.dense_gemm_kernel,
                normed,
                &self.lm_head_weight,
                logits,
                padded_n as u32,
                v as u32,
                h as u32,
                stream,
            )?;
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::{LmHeadNvfp4Route, select_lm_head_nvfp4_route};
    use spark_runtime::gpu::KernelHandle;

    #[test]
    fn exact_batch_gemv_tiers_win_over_transposed_fallbacks() {
        // The exact-M tier LOOKUP now lives in the caller
        // (`self.w4a16_batchm.kernel(padded_n)`), which resolves batch2..16 and
        // passes a single handle in; this function no longer takes one handle
        // per tier. What it still owns is the PRECEDENCE, which is the part
        // that matters numerically: a resolved batch-GEMV handle beats both
        // transposed fallbacks at every admitted decode width, because the
        // tiers are byte-identical to M independent scalar GEMVs while
        // transposed GEMM changes the FP32 reduction order.
        for padded_n in [2usize, 4, 8, 16] {
            let route = select_lm_head_nvfp4_route(padded_n, true, KernelHandle(41), true, true);
            assert!(
                matches!(route, LmHeadNvfp4Route::BatchGemv(KernelHandle(41))),
                "padded_n={padded_n}: a resolved tier handle must win over the \
                 transposed fallbacks"
            );
        }
    }

    #[test]
    fn disabled_or_missing_batch_gemv_uses_transposed_fallback() {
        let disabled = select_lm_head_nvfp4_route(8, false, KernelHandle(82), true, true);
        assert!(matches!(disabled, LmHeadNvfp4Route::TransposedBf16));

        let missing_handle = select_lm_head_nvfp4_route(8, true, KernelHandle(0), true, true);
        assert!(matches!(missing_handle, LmHeadNvfp4Route::TransposedBf16));
    }

    #[test]
    fn transposed_fallback_prefers_bf16_then_other_for_wide_batches() {
        let bf16 = select_lm_head_nvfp4_route(17, true, KernelHandle(0), true, true);
        assert!(matches!(bf16, LmHeadNvfp4Route::TransposedBf16));

        let other = select_lm_head_nvfp4_route(17, true, KernelHandle(0), false, true);
        assert!(matches!(other, LmHeadNvfp4Route::TransposedOther));
    }

    #[test]
    fn missing_transposed_fallback_uses_plain_gemm() {
        let route = select_lm_head_nvfp4_route(17, true, KernelHandle(0), false, false);
        assert!(matches!(route, LmHeadNvfp4Route::PlainGemm));
    }

    #[test]
    fn small_batch_without_exact_gemv_keeps_plain_gemm_fallback() {
        let route = select_lm_head_nvfp4_route(4, false, KernelHandle(0), true, true);
        assert!(matches!(route, LmHeadNvfp4Route::PlainGemm));
    }
}
