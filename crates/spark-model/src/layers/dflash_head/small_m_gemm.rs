// SPDX-License-Identifier: AGPL-3.0-only

//! Small-M drafter GEMM dispatch: batched GEMVs vs pipelined WMMA GEMM.
//!
//! Every drafter projection at decode runs at M=γ rows (γ ≤ 8), and a
//! batched proposal at B·γ rows. The pipelined `dense_gemm_bf16_pipelined`
//! kernel is a 128-row M-tile WMMA GEMM, so at M=8 ~94% of its MMA work is
//! padding — expensive on gfx1151 (~2–3.5 TFLOPS BF16 WMMA) and on GB10,
//! where its 32-CTA grids stream the weight at 50-165 GB/s. The GEMV arms
//! read each weight once at bandwidth:
//! - `dense_gemv_bf16_batchm` for 1..=8 rows;
//! - the tensor-core `dense_gemv_bf16_tc16/32` for 9..=32 rows, and two
//!   tensor-core pieces for 33..=64 (5+ sequences' blocks), on targets that
//!   ship them (the GLM-5.3 kernel set).
//!
//! The arms accumulate in FP32 but do not produce bit-identical results
//! (different reduction order; the tensor-core tiers are within 2 BF16 ulps
//! of batch-M), so the draft distribution and acceptance can shift slightly.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::BlockDiffusionDraftHead;
use crate::layers::ops;
use crate::layers::ops::{DENSE_GEMV_BATCHM_MAX_M, DENSE_GEMV_TC_MAX_M};
use crate::weight_map::DenseWeight;

/// Policy: `ATLAS_DFLASH_SMALL_M_GEMV=1|0` overrides; otherwise ON for
/// gfx1151 (`cfg!(atlas_scale)`) and for targets that ship the tensor-core
/// BF16 GEMV tiers (`tc_present`: the GLM-5.3 set, where the GEMVs took the
/// C1 propose from 28 to 17 ms), OFF elsewhere on NVIDIA. Resolved once at
/// head construction into `DflashKernels::small_m_gemv`.
pub(super) fn small_m_gemv_enabled(tc_present: bool) -> bool {
    let on = match std::env::var("ATLAS_DFLASH_SMALL_M_GEMV").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => cfg!(atlas_scale) || tc_present,
    };
    tracing::info!(
        "DFlash drafter small-M GEMM arm: {}",
        if on {
            "batched GEMV (tensor-core tiers up to 64 rows when present)"
        } else {
            "pipelined GEMM"
        }
    );
    on
}

/// Which kernel serves a BF16 `[m, k] · [n, k]ᵀ` drafter GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SmallMArm {
    /// `dense_gemv_bf16_batchm`, 1..=8 rows.
    BatchM,
    /// `dense_gemv_bf16_tc16/32`, 9..=32 rows.
    TensorCore,
    /// Two pieces (32 rows, then the rest), 33..=64 rows.
    TensorCoreSplit,
    /// `dense_gemm_bf16_pipelined`.
    Pipelined,
}

/// Pure decision: the GEMV arms iff enabled and resolved; the tensor-core
/// tiers also need `k % 8 == 0`.
pub(super) fn small_m_arm(enabled: bool, batchm: bool, tc: bool, m: u32, k: u32) -> SmallMArm {
    let tc = enabled && tc && k.is_multiple_of(8);
    if enabled && batchm && (1..=DENSE_GEMV_BATCHM_MAX_M).contains(&m) {
        SmallMArm::BatchM
    } else if tc && (1..=DENSE_GEMV_TC_MAX_M).contains(&m) {
        SmallMArm::TensorCore
    } else if tc && m <= 2 * DENSE_GEMV_TC_MAX_M {
        SmallMArm::TensorCoreSplit
    } else {
        SmallMArm::Pipelined
    }
}

/// Pure decision (kept for callers that only ask about the batch-M arm).
pub(super) fn use_small_m_gemv(enabled: bool, handle_nonzero: bool, m: u32) -> bool {
    small_m_arm(enabled, handle_nonzero, false, m, 0) == SmallMArm::BatchM
}

impl BlockDiffusionDraftHead {
    /// C[m,n] = A[m,k] · W[n,k]^T in BF16 through [`small_m_arm`]'s pick.
    /// `out_stride = n` on every arm.
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
        ensure_bf16_present(w, m, n, k)?;
        let kernels = &self.kernels;
        match small_m_arm(
            kernels.small_m_gemv,
            kernels.dense_gemv_batchm.0 != 0,
            kernels.dense_gemv_tc16.0 != 0 && kernels.dense_gemv_tc32.0 != 0,
            m,
            k,
        ) {
            SmallMArm::BatchM => ops::dense_gemv_batchm(
                gpu,
                kernels.dense_gemv_batchm,
                src,
                w,
                dst,
                m,
                n,
                k,
                n,
                stream,
            ),
            SmallMArm::TensorCore => {
                let kernel = if m <= 16 {
                    kernels.dense_gemv_tc16
                } else {
                    kernels.dense_gemv_tc32
                };
                ops::dense_gemv_bf16_tc(gpu, kernel, src, w, dst, m, n, k, n, stream)
            }
            SmallMArm::TensorCoreSplit => {
                // A few rows past one tier (many sequences' blocks): two
                // tensor-core pieces beat the 128-row tiled GEMM.
                let first = DENSE_GEMV_TC_MAX_M;
                self.drafter_dense_gemm(gpu, src, w, dst, first, n, k, stream)?;
                self.drafter_dense_gemm(
                    gpu,
                    src.offset(first as usize * k as usize * 2),
                    w,
                    dst.offset(first as usize * n as usize * 2),
                    m - first,
                    n,
                    k,
                    stream,
                )
            }
            SmallMArm::Pipelined => ops::dense_gemm_bf16_pipelined(
                gpu,
                kernels.dense_gemm_pipelined,
                src,
                w,
                dst,
                m,
                n,
                k,
                stream,
            ),
        }
    }

    /// The legacy (non-paged) path keeps the pipelined GEMM
    /// unconditionally; same signature shape as the `gemm` closure
    /// `conv::prepare`/`selector::select_candidates` take.
    pub(super) fn drafter_pipelined_gemm(
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
        ensure_bf16_present(w, m, n, k)?;
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

/// A projection whose BF16 copy `ATLAS_DFLASH_DROP_BF16` freed has a null
/// pointer; reaching a BF16 kernel with it must fail loudly, not fault.
fn ensure_bf16_present(w: &DenseWeight, m: u32, n: u32, k: u32) -> Result<()> {
    anyhow::ensure!(
        w.weight.0 != 0,
        "DFlash BF16 projection [{n}, {k}] was dropped (ATLAS_DFLASH_DROP_BF16) but {m} rows need it"
    );
    Ok(())
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

    #[test]
    fn tensor_core_tiers_take_nine_to_sixty_four_rows() {
        use SmallMArm::*;
        let arm = |m| small_m_arm(true, true, true, m, 4096);
        assert_eq!(arm(8), BatchM);
        assert_eq!(arm(9), TensorCore);
        assert_eq!(arm(32), TensorCore);
        assert_eq!(arm(33), TensorCoreSplit);
        assert_eq!(arm(64), TensorCoreSplit);
        assert_eq!(arm(65), Pipelined);
        // Tiers absent, disabled, or a K the tiers cannot take: tiled GEMM.
        assert_eq!(small_m_arm(true, true, false, 16, 4096), Pipelined);
        assert_eq!(small_m_arm(false, true, true, 16, 4096), Pipelined);
        assert_eq!(small_m_arm(false, true, true, 8, 4096), Pipelined);
        assert_eq!(small_m_arm(true, true, true, 16, 4100), Pipelined);
    }
}
