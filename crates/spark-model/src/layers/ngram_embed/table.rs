// SPDX-License-Identifier: AGPL-3.0-only

//! One n-gram lookup table on device: BF16 as shipped, FP8-quantized at
//! load, or NVMe-backed by a bounded row cache.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use crate::weight_map::{DenseWeight, Fp8DenseWeight};

/// One n-gram lookup table on device: BF16 as shipped, or FP8-quantized at
/// load (per-row E4M3 + f32 scale — halves the ~63 GB table footprint;
/// embeddings tolerate this well and the gather dequantizes on read).
pub enum NgramTable {
    Bf16(DenseWeight),
    Fp8(Fp8DenseWeight),
    /// NVMe-backed: only a bounded set of ROWS is resident, in a pinned
    /// GPU-addressable arena. The host resolves `row_id -> slot` (the ids are
    /// a pure function of token ids, so this is host-side anyway) and the
    /// SAME gather kernels then read the arena by slot index — no kernel
    /// change, no `cuMemcpyHtoD` on the fault path.
    ///
    /// This is what makes a 51 B-parameter embedding table serveable on a
    /// 121 GB box: the tables are the model's largest tensors and its least
    /// bandwidth-hungry (12 rows ~ 3 KB per token), so demoting them buys
    /// back tens of GB for KV.
    #[cfg(feature = "cuda")]
    Cached(Box<spark_storage::NgramRowCache>),
    /// FP4-Local's NVFP4 table held ENTIRELY on device (`ATLAS_PLE_RESIDENT`):
    /// packed U8 rows `[rows_total, 80]` and their E4M3 block scales
    /// `[rows_total, 10]`. No row cache, no faulting: the gather kernel
    /// indexes `rows`/`scales` by `row_id - row_base` exactly as it indexes a
    /// cache arena by slot. `row_base` exists so a later head-sharded variant
    /// (rank owns a contiguous head range) is a small change; it is 0 today.
    #[cfg(feature = "cuda")]
    ResidentNvfp4 {
        rows: spark_runtime::gpu::DevicePtr,
        scales: spark_runtime::gpu::DevicePtr,
        row_base: u64,
        rows_total: u64,
    },
}

impl NgramTable {
    /// Warm `row_ids` into the backing row cache, WITHOUT pinning — the
    /// prefetch half of a resolve, for a caller that computed its ids ahead
    /// of time (PLE: a pure function of the prompt). Returns the number of
    /// rows faulted in; `Ok(0)` for resident tables, which need no warm.
    pub fn prefetch(&mut self, row_ids: &[u64]) -> Result<usize> {
        #[cfg(feature = "cuda")]
        if let Self::Cached(cache) = self {
            return cache.prefetch(row_ids);
        }
        let _ = row_ids;
        Ok(0)
    }

    /// Whether this table is NVMe-backed with a bounded resident row set —
    /// the only kind with rows to warm into. Always false without `cuda`,
    /// where the `Cached` variant does not exist at all.
    pub fn is_row_cached(&self) -> bool {
        #[cfg(feature = "cuda")]
        if matches!(self, Self::Cached(_)) {
            return true;
        }
        false
    }

    /// Quantize a BF16 table to FP8 on the GPU (per-row E4M3 absmax +
    /// f32 scale via `quantize_bf16_to_fp8`) — the quantize-on-load
    /// lever. The caller frees the BF16 source afterwards; tables are
    /// loaded one at a time so peak overhead is a single table.
    pub fn quantize_bf16(
        w: &DenseWeight,
        rows: usize,
        dim: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Self> {
        let k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
        Ok(Self::Fp8(crate::weight_map::quantize_to_fp8(
            w, rows, dim, gpu, k, stream,
        )?))
    }
}
