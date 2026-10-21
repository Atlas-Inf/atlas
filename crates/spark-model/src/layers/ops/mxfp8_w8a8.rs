// SPDX-License-Identifier: AGPL-3.0-only
//! MXFP8 W8A8 projections: a checkpoint's own MXFP8 weight times MXFP8
//! activations on cuBLASLt's block-scaled tensor cores
//! ([`spark_runtime::cublaslt::fp8_gemm_act_weight_t_mx`]), the prefill
//! GEMM for a projection that ships MXFP8.
//!
//! The weight operand is a decode twin ([`Mxfp8Weight`]: E4M3 `[n, k]` +
//! row-major UE8M0 `[n, k/32]`). Its E4M3 bytes are read where the twin keeps
//! them, not copied; only its scales are relaid, once at load, into
//! cuBLASLt's 128x4 tiles ([`MxW8a8::from_twin`], `n * k / 32` bytes, 3% of
//! the weight). The activation is quantized per row, 32 values a scale
//! ([`mxfp8_quant_act`]), so a row's bytes never depend on the rows beside
//! it; and the GEMM runs its row bucket's tuned in-order kernel, whose
//! outputs do not depend on the row count or the row's place in the pass
//! (`cublaslt::mx_algo`). So the projection is row-invariant end to end,
//! which `ATLAS_QWEN4EXP_PREFILL_ROWINV` needs.
//!
//! Numerics: the weights are exact (the checkpoint's); the activations are
//! rounded to E4M3 under a power-of-two scale per 32 values, which puts a
//! projection's output ~2-3e-2 (relative) from the same weights on BF16
//! activations.
//!
//! Ported from Enntity/atlas (branch `sparkqwen` @ 92dd543, AGPL-3.0-only),
//! `crates/spark-model/src/layers/ops/mxfp8_w8a8.rs`, with Atlas-Inf/atlas#198's integration changes.

use std::sync::Mutex;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{MXFP8_ACT_MODULE, MXFP8_BLOCK, mx_lt_scale_bytes, mxfp8_lt_tile_scales};
use crate::weight_map::Mxfp8Weight;

/// [`mxfp8_quant_act`]'s kernel in [`MXFP8_ACT_MODULE`].
pub const MX_QUANT_KERNEL: &str = "mxfp8_quant_act_bf16";
/// [`mxfp8_lt_tile_scales`]'s kernel in [`MXFP8_ACT_MODULE`].
pub const MX_TILE_KERNEL: &str = "mxfp8_lt_scale_tile";

/// An MXFP8 weight as the W8A8 GEMM reads it: the twin's E4M3 `[n, k]`
/// (borrowed: the twin owns it) and its scales in the tiled layout (owned).
#[derive(Debug, Clone, Copy)]
pub struct MxW8a8 {
    pub data: DevicePtr,
    pub sf: DevicePtr,
    pub n: usize,
    pub k: usize,
}

impl MxW8a8 {
    /// The W8A8 operand of an untiled `[n, k]` `twin`, its row buckets
    /// tuned up to `m_max` rows ([`mx_prepare`]; `0`, an unknown bound, leaves
    /// each bucket to be tuned at its first GEMM). Synchronizes `stream`.
    pub fn from_twin(
        gpu: &dyn GpuBackend,
        twin: &Mxfp8Weight,
        [n, k]: [usize; 2],
        m_max: usize,
        stream: u64,
    ) -> Result<Self> {
        // Our Mxfp8Weight has one untiled layout, so the twin is always
        // W8A8-ready.
        ensure!(
            n > 0 && k > 0 && k.is_multiple_of(MXFP8_BLOCK),
            "MXFP8 W8A8: a [{n}, {k}] twin is not an [n, k % 32 == 0] one"
        );
        let tile = gpu.kernel(MXFP8_ACT_MODULE, MX_TILE_KERNEL)?;
        let sf = gpu.alloc(mx_lt_scale_bytes(n, k))?;
        let built = (|| -> Result<()> {
            let ld = (k / MXFP8_BLOCK) as u32;
            mxfp8_lt_tile_scales(gpu, tile, twin.scales, ld, sf, n as u32, k as u32, stream)?;
            gpu.synchronize(stream)?;
            if m_max == 0 {
                return Ok(());
            }
            mx_prepare(n, k, m_max, stream)
        })();
        undo_on_err(built, || gpu.free(sf))?;
        Ok(Self {
            data: twin.data,
            sf,
            n,
            k,
        })
    }

    /// Free the tiled scales (the E4M3 bytes are the twin's).
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.sf)
    }

    /// `out[m, n]` (BF16, contiguous) `= act[m, k] x self^T`.
    pub fn gemm(&self, act: MxAct, out: DevicePtr, m: usize, stream: u64) -> Result<()> {
        spark_runtime::cublaslt::fp8_gemm_act_weight_t_mx(
            act.q.0,
            act.sf.0,
            self.data.0,
            self.sf.0,
            out.0,
            [m as u32, self.n as u32, self.k as u32],
            stream,
        )
    }
}

/// `r`, after running `undo` if it is an error. An `undo` that fails too is
/// logged, not returned: the caller gets the error that made it undo.
pub fn undo_on_err<T>(r: Result<T>, undo: impl FnOnce() -> Result<()>) -> Result<T> {
    if r.is_err()
        && let Err(e) = undo()
    {
        tracing::warn!("MXFP8 W8A8: cleanup after a failed install failed too: {e:#}");
    }
    r
}

/// An MXFP8 activation: E4M3 rows (`k` bytes each) from `q`, their scales
/// in the tiled layout from `sf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MxAct {
    pub q: DevicePtr,
    pub sf: DevicePtr,
}

/// `(n, k, m_max)` already tuned in this process, and the milliseconds the
/// tuning took in all.
static PREPARED: Mutex<(Vec<(usize, usize, usize)>, f64)> = Mutex::new((Vec::new(), 0.0));

/// Tune every row bucket of an `n x k` W8A8 GEMM up to `m_max` rows
/// ([`spark_runtime::cublaslt::fp8_gemm_mx_prepare`]), once per shape per
/// process, so serving never times a candidate. Synchronizes `stream`.
pub fn mx_prepare(n: usize, k: usize, m_max: usize, stream: u64) -> Result<()> {
    let key = (n, k, m_max);
    if PREPARED.lock().unwrap().0.contains(&key) {
        return Ok(());
    }
    let t = std::time::Instant::now();
    spark_runtime::cublaslt::fp8_gemm_mx_prepare(n as u32, k as u32, key.2 as u32, stream)?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let mut done = PREPARED.lock().unwrap();
    done.0.push(key);
    done.1 += ms;
    tracing::info!(
        "MXFP8 W8A8: {n}x{k} row buckets tuned to {} rows in {ms:.0} ms ({:.0} ms all shapes)",
        key.2,
        done.1
    );
    Ok(())
}

#[cfg(test)]
#[path = "mxfp8_w8a8_test_gpu.rs"]
mod gpu_tests;

#[cfg(test)]
mod tests {
    use super::undo_on_err;

    #[test]
    fn a_failed_undo_never_hides_the_error_that_caused_it() {
        let failed = || -> anyhow::Result<u8> { anyhow::bail!("tune") };
        let e = undo_on_err(failed(), || anyhow::bail!("free")).unwrap_err();
        assert_eq!(e.to_string(), "tune");
        let mut undone = false;
        let e = undo_on_err(failed(), || {
            undone = true;
            Ok(())
        })
        .unwrap_err();
        assert!(undone && e.to_string() == "tune");
        // Nothing to undo on success.
        let r = undo_on_err(Ok(7), || panic!("undo on success"));
        assert_eq!(r.unwrap(), 7);
    }
}
