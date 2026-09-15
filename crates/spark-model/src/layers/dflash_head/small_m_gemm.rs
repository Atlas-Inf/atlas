// SPDX-License-Identifier: AGPL-3.0-only

//! Small-M drafter GEMM dispatch: batched GEMV vs pipelined WMMA GEMM.
//!
//! Every drafter projection at decode runs at M=γ rows (γ ≤ 8). The
//! pipelined `dense_gemm_bf16_pipelined` kernel is a 128-row M-tile WMMA
//! GEMM, so at M=8 ~94% of its MMA work is padding — expensive on
//! gfx1151 (~2–3.5 TFLOPS BF16 WMMA) where the projections dominate
//! propose time. `dense_gemv_bf16_batchm` is bandwidth-bound and caps
//! at MAX_M=8, which covers every γ the drafter uses.
//!
//! Both arms accumulate in FP32 but do not produce bit-identical
//! results (different reduction order); the draft distribution and
//! therefore acceptance length can shift slightly.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::BlockDiffusionDraftHead;
use crate::layers::ops;
use crate::layers::ops::DENSE_GEMV_BATCHM_MAX_M;
use crate::weight_map::DenseWeight;

/// Policy: gfx1151 (`cfg!(atlas_scale)`) defaults ON, NVIDIA defaults
/// OFF; `ATLAS_DFLASH_SMALL_M_GEMV=1|0` overrides. Read once.
pub(super) fn small_m_gemv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = match std::env::var("ATLAS_DFLASH_SMALL_M_GEMV").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => cfg!(atlas_scale),
        };
        tracing::info!(
            "DFlash drafter small-M GEMM arm: {}",
            if on {
                "batched GEMV (gfx1151 default)"
            } else {
                "pipelined GEMM"
            }
        );
        on
    })
}

/// Pure decision: the batched GEMV iff enabled, the kernel resolved,
/// and `m` fits its compile-time MAX_M.
pub(super) fn use_small_m_gemv(enabled: bool, handle_nonzero: bool, m: u32) -> bool {
    enabled && handle_nonzero && (1..=DENSE_GEMV_BATCHM_MAX_M).contains(&m)
}

impl BlockDiffusionDraftHead {
    /// C[m,n] = A[m,k] · W[n,k]^T in BF16 — `dense_gemv_bf16_batchm`
    /// when `m ≤ 8` and the arm is enabled, `dense_gemm_bf16_pipelined`
    /// otherwise. `out_stride = n` on both arms.
    pub(super) fn drafter_dense_gemm(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        w: &DenseWeight,
        dst: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if use_small_m_gemv(
            small_m_gemv_enabled(),
            self.kernels.dense_gemv_batchm.0 != 0,
            m,
        ) {
            return ops::dense_gemv_batchm(
                gpu,
                self.kernels.dense_gemv_batchm,
                src,
                w,
                dst,
                m,
                n,
                k,
                n,
                stream,
            );
        }
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            src,
            w,
            dst,
            m,
            n,
            k,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_m_gemv_decision() {
        assert!(use_small_m_gemv(true, true, 8));
        assert!(!use_small_m_gemv(true, true, 9));
        assert!(!use_small_m_gemv(true, true, 0));
        assert!(!use_small_m_gemv(true, false, 8));
        assert!(!use_small_m_gemv(false, true, 8));
    }
}
