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

use std::ops::Range;
use std::sync::Mutex;

use anyhow::{Result, ensure};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{
    MX_LT_TILE_ROWS, MXFP8_ACT_MODULE, MXFP8_BLOCK, mx_lt_inner_pad, mx_lt_scale_bytes,
    mxfp8_lt_tile_scales, mxfp8_quant_act,
};
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
    /// Every GEMM runs the one pinned in-order kernel of `(n, k)`
    /// ([`Self::pinned`]), not the tuned per-bucket pick.
    pub pinned: bool,
}

/// Rows per GEMM on the pinned kernel: the tallest height `mx_algo::pinned`
/// checks it takes (taller passes run in chunks of it; the kernel computes
/// each row alone, so the bytes are the same).
pub const MX_PINNED_MAX_ROWS: usize = 8192;
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
            pinned: false,
        })
    }

    /// This operand on the pinned kernel at every row count: one in-order
    /// cuBLASLt kernel per `(n, k)` chosen once without timing and never
    /// retuned (`fp8_gemm_act_weight_t_mx_pinned`), so a row's bytes do not
    /// depend on the pass height, the chunking or which heights ran first
    /// (the tuned pick of an unswept shape is retuned when it refuses a
    /// height). The GLM FP4-Local native paths (KDA, dense MLP) run pinned.
    pub fn pinned(self) -> Self {
        Self {
            pinned: true,
            ..self
        }
    }

    /// [`Self::pinned`], picking the shape's pinned kernel now (load time) by
    /// timing the eligible in-order candidates at `ref_m` rows
    /// (`fp8_gemm_mx_pinned_prepare`: the fastest, as the tuned entry would
    /// pick, kept for the run and logged). Where it cannot time (no cuBLASLt
    /// on this host) the first GEMM pins the heuristic's first eligible
    /// kernel instead, untimed; logged.
    pub fn pinned_timed(self, ref_m: usize, stream: u64) -> Self {
        if let Err(e) = spark_runtime::cublaslt::fp8_gemm_mx_pinned_prepare(
            self.n as u32,
            self.k as u32,
            ref_m.max(1) as u32,
            stream,
        ) {
            tracing::warn!(
                "MXFP8 W8A8 {}x{}: no timed pin ({e:#}); the first GEMM pins untimed",
                self.n,
                self.k
            );
        }
        self.pinned()
    }
    /// Bytes this operand allocated (the tiled scales).
    pub fn owned_bytes(&self) -> usize {
        mx_lt_scale_bytes(self.n, self.k)
    }

    /// Free the tiled scales (the E4M3 bytes are the twin's).
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.sf)
    }

    /// `out[m, n]` (BF16, contiguous) `= act[m, k] x self^T`.
    pub fn gemm(&self, act: MxAct, out: DevicePtr, m: usize, stream: u64) -> Result<()> {
        // A pinned operand, or a packed prefill (`glm_packed_scope`): one
        // kernel for every row count, chosen without timing, so a prompt's
        // rows do not depend on the pass height or on which heights were
        // tuned first; passes taller than the heights it was checked at run
        // in chunks.
        if self.pinned {
            for row0 in (0..m).step_by(MX_PINNED_MAX_ROWS) {
                let rows = (m - row0).min(MX_PINNED_MAX_ROWS);
                let a = act.rows_from(row0, self.k)?;
                spark_runtime::cublaslt::fp8_gemm_act_weight_t_mx_pinned(
                    a.q.0,
                    a.sf.0,
                    self.data.0,
                    self.sf.0,
                    out.offset(row0 * self.n * 2).0,
                    [rows as u32, self.n as u32, self.k as u32],
                    stream,
                )?;
            }
            return Ok(());
        }
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

/// What installing W8A8 prefill on a layer did: the bytes its operands
/// allocated, and the BF16 copies it took out of the layer (pointer, bytes),
/// which the caller frees.
#[derive(Debug, Default)]
pub struct W8a8Swap {
    pub allocated: usize,
    pub bf16: Vec<(DevicePtr, usize)>,
}

/// An MXFP8 activation: E4M3 rows (`k` bytes each) from `q`, their scales
/// in the tiled layout from `sf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MxAct {
    pub q: DevicePtr,
    pub sf: DevicePtr,
}

impl MxAct {
    /// The same operand from row `row0` on: a whole number of scale tiles,
    /// so `row0` must be a multiple of 128.
    pub fn rows_from(self, row0: usize, k: usize) -> Result<Self> {
        ensure!(
            row0.is_multiple_of(MX_LT_TILE_ROWS),
            "MXFP8 activation rows from {row0}: not on the 128-row scale tile grid"
        );
        Ok(Self {
            q: self.q.offset(row0 * k),
            sf: self.sf.offset(row0 * mx_lt_inner_pad(k)),
        })
    }
}

/// The arena's FP8 activation scratch (`fp8_act` / `fp8_act_scale`, which
/// no other qwen4_exp prefill op touches) as an `[m, k]` MXFP8 operand.
pub fn mx_act_scratch(buffers: &BufferArena, m: usize, k: usize) -> Result<MxAct> {
    let (q_cap, sf_cap) = (buffers.fp8_act_bytes(), buffers.sizes().fp8_act_scale);
    ensure!(
        m * k <= q_cap && mx_lt_scale_bytes(m, k) <= sf_cap,
        "MXFP8 W8A8: a [{m}, {k}] activation needs {} + {} B of FP8 scratch, the arena has \
         {q_cap} + {sf_cap}",
        m * k,
        mx_lt_scale_bytes(m, k),
    );
    Ok(MxAct {
        q: buffers.fp8_act(),
        sf: buffers.fp8_act_scale(),
    })
}

/// Quantize `rows` of BF16 `x` (`[m, k]`, contiguous) into the same rows of
/// `act`. `rows.start` must be on the 128-row tile grid. `kernel` is
/// [`MX_QUANT_KERNEL`].
pub fn mx_quant_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    act: MxAct,
    rows: Range<usize>,
    k: usize,
    stream: u64,
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let at = act.rows_from(rows.start, k)?;
    let (m, k32) = (rows.len() as u32, k as u32);
    mxfp8_quant_act(
        gpu,
        kernel,
        x.offset(rows.start * k * 2),
        k32,
        at.q,
        at.sf,
        m,
        k32,
        stream,
    )
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

/// Milliseconds [`mx_prepare`] has spent in this process.
pub fn mx_prepare_ms() -> f64 {
    PREPARED.lock().unwrap().1
}

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
