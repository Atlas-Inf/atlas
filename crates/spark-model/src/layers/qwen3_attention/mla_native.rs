// SPDX-License-Identifier: AGPL-3.0-only

//! GLM MLA q_a / q_b / kv_a / o on the checkpoint's own FP8 block bytes
//! (`ATLAS_GLM_FP4L_MLA_FP8`, loaded by `weight_loader::glm5::layers`).
//!
//! Every projection call of a target layer whose weight carries a twin goes
//! here first ([`Qwen3AttentionLayer::mla_native_rows`]): the paged GLM
//! prefill / verify lane (`mla_prefill_dense`), the plain one-row decode, the
//! C2..C5 batched and per-row multi-sequence decode, and the K3 query / O
//! stages. Calls of up to 64 rows (32 a launch) run the
//! `fp8b_gemv_tc{8,16,32}` tiers, whose row bits do not depend on the launch
//! width: a decode row equals the same row verified. Longer prefill chunks
//! run the prefill GEMM `ATLAS_GLM_FP4L_FP8_PREFILL` picks (W8A8 by
//! default, NVIDIA's vLLM recipe; W8A16 exact; or the BF16 copy). Unless the
//! BF16 copy serves prefill, the loader frees it and leaves a unique, never
//! mapped address as the lookup key: a reader this module does not route
//! faults instead of reading freed memory.
//!
//! The fused front (`ATLAS_GLM_MLA_FUSE`) can read native q_a / kv_a on
//! TP2/TP4 through the same FP8 block CTA body. Their checkpoint bytes and
//! FP32 block-scale arithmetic are unchanged.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops::{self, Fp8bPrefill, MXFP8_MAX_ROWS};
use crate::weight_map::{DenseWeight, Fp8BlockWeight};

/// Rows up to which every call runs the GEMV tiers (32 a launch): decode,
/// DFlash verify and owner-batched verify blocks of up to two launches, so a
/// row's bits never depend on how many rows ride with it. Longer calls are
/// prefill chunks.
pub(crate) const NATIVE_GEMV_MAX_ROWS: usize = 2 * MXFP8_MAX_ROWS;

/// The native twins of one MLA layer and how its prefill runs.
#[derive(Debug, Default)]
pub(crate) struct MlaNative {
    /// `(lookup key, [n, k], twin)`: the key is the BF16 weight's pointer, or
    /// its stand-in once the BF16 copy is freed.
    pub twins: Vec<(DevicePtr, [usize; 2], Fp8BlockWeight)>,
    /// `None`: no twins (the layer is not native).
    pub prefill: Option<Fp8bPrefill>,
}

/// A stand-in key for a freed BF16 MLA weight: 16-byte aligned, non-null and
/// in no allocation (above the 48-bit device virtual range the driver hands
/// out), unique per (layer, projection).
pub(crate) fn freed_key(layer: usize, proj: usize) -> DevicePtr {
    DevicePtr(0xDEAD_0000_0000_0000 | ((layer as u64) << 12) | ((proj as u64) << 4))
}

impl Qwen3AttentionLayer {
    /// Install the native twins (`(key, [n, k], twin)`) and the prefill mode.
    pub(crate) fn install_mla_native(
        &mut self,
        twins: Vec<(DevicePtr, [usize; 2], Fp8BlockWeight)>,
        prefill: Fp8bPrefill,
    ) {
        self.mla_native = MlaNative {
            twins,
            prefill: Some(prefill),
        };
    }

    /// The native twin keyed by `weight`, with its `[n, k]`.
    pub(super) fn mla_native_twin(
        &self,
        weight: DevicePtr,
    ) -> Option<([usize; 2], &Fp8BlockWeight)> {
        self.mla_native
            .twins
            .iter()
            .find(|(key, ..)| *key == weight)
            .map(|(_, nk, w)| (*nk, w))
    }

    /// Whether `weight` is served natively (no BF16 kernel may read it).
    pub(crate) fn mla_is_native(&self, weight: &DenseWeight) -> bool {
        self.mla_native_twin(weight.weight).is_some()
    }

    /// `output[m, n] = input[m, k] · W^T` (rows contiguous, output rows `n`
    /// apart) on `weight`'s native twin; `false`, nothing launched, when it
    /// has none, or for a prefill the BF16 copy serves (`bf16` mode).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_native_rows(
        &self,
        ctx: &ForwardContext,
        weight: &DenseWeight,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(([tn, tk], w)) = self.mla_native_twin(weight.weight) else {
            return Ok(false);
        };
        let (m, n, k) = (m as usize, n as usize, k as usize);
        ensure!(
            [tn, tk] == [n, k],
            "GLM MLA native: twin [{tn}, {tk}] for a [{n}, {k}] projection"
        );
        let kernels = ops::native_kernels(ctx.gpu);
        if m <= NATIVE_GEMV_MAX_ROWS {
            ops::fp8b_gemv_rows(ctx.gpu, &kernels.fp8b, w, input, output, m, n, k, n, stream)?;
            return Ok(true);
        }
        let mode = self.mla_native.prefill.unwrap_or(Fp8bPrefill::W8a8);
        if mode.needs_bf16() {
            return Ok(false);
        }
        ops::fp8b_prefill_gemm(
            ctx.gpu,
            &kernels,
            ops::Fp8ActScratch::of(ctx.buffers),
            mode,
            w,
            input,
            output,
            m,
            n,
            k,
            stream,
        )?;
        Ok(true)
    }

    /// [`Self::mla_native_rows`] for a caller that never runs prefill-sized
    /// row counts (decode, multi-sequence, K3 stages: at most 32 rows).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_native_gemv(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        weight: &DenseWeight,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        self.mla_native_gemv_strided(gpu, weight, input, output, [m, n, k], n, stream)
    }

    /// [`Self::mla_native_gemv`] with output rows `out_stride` elements apart.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_native_gemv_strided(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        weight: &DenseWeight,
        input: DevicePtr,
        output: DevicePtr,
        [m, n, k]: [u32; 3],
        out_stride: u32,
        stream: u64,
    ) -> Result<bool> {
        let Some(([tn, tk], w)) = self.mla_native_twin(weight.weight) else {
            return Ok(false);
        };
        let (m, n, k) = (m as usize, n as usize, k as usize);
        ensure!(
            [tn, tk] == [n, k] && m <= MXFP8_MAX_ROWS,
            "GLM MLA native GEMV: twin [{tn}, {tk}] for {m} rows of a [{n}, {k}] projection"
        );
        let kernels = ops::native_kernels(gpu);
        let ld = out_stride as usize;
        ops::fp8b_gemv_rows(gpu, &kernels.fp8b, w, input, output, m, n, k, ld, stream)?;
        Ok(true)
    }
}

impl Qwen3AttentionLayer {
    /// One-row MLA projection: the native twin, else the BF16 decode GEMV.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mla_gemv1(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.mla_native_gemv(gpu, weight, input, output, 1, n, k, stream)? {
            return Ok(());
        }
        ops::dense_gemv(gpu, self.dense_gemv_k, input, weight, output, n, k, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freed_keys_are_aligned_unique_and_outside_device_memory() {
        let mut seen = std::collections::HashSet::new();
        for layer in [3usize, 7, 43] {
            for proj in 0..4 {
                let key = freed_key(layer, proj);
                assert!(key.0.is_multiple_of(16) && key.0 >= 1 << 48);
                assert!(seen.insert(key.0));
            }
        }
    }
}
