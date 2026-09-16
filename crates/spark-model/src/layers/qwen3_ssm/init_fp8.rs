// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 weight-install setters and the NVFP4→FP8 prefill pre-dequant for
//! [`Qwen3SsmLayer`]. Split out of `init.rs` (500-LoC cap).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3SsmLayer;
use crate::weight_map::Fp8Weight;

impl Qwen3SsmLayer {
    /// Install native FP8 block-scaled weights for the decode GEMV path.
    ///
    /// Inputs MUST be tagged `WeightQuantFormat::Fp8BlockScaled` — that is
    /// the canonical input format for the `w8a16_gemv` kernel
    /// (`out[n] = sum_k A[k] * E4M3_LUT[B[n,k]] * block_scale[n/BS, k/BS]`,
    /// see `kernels/gb10/common/w8a16_gemv.cu`). The kernel reads the
    /// scale buffer at `[N/BS, K/BS]` BF16 — exactly the shape produced
    /// by `load_fp8_block_scaled_as_fp8weight`.
    ///
    /// This setter does NOT install the raw FP8 DevicePtr fields used by
    /// the prefill `fp8_gemm_n128` kernel — that kernel takes no scale
    /// argument and assumes single-scale FP8 (baked-in scale) produced
    /// by `bf16_to_fp8`. Block-scaled bytes would silently produce wrong
    /// outputs there. For prefill, call `set_fp8_prefill_only_weights`
    /// separately with single-scale FP8 derived from a BF16 dequant.
    pub fn set_fp8_decode_weights(&mut self, qkvz: Option<Fp8Weight>, out_proj: Option<Fp8Weight>) {
        if let Some(ref w) = qkvz {
            w.scale_format.expect(
                crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "set_fp8_decode_weights::qkvz (w8a16_gemv expects [N/BS,K/BS] BF16 block scales)",
            );
        }
        if let Some(ref w) = out_proj {
            w.scale_format.expect(
                crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "set_fp8_decode_weights::out_proj (w8a16_gemv expects [N/BS,K/BS] BF16 block scales)",
            );
        }
        self.qkvz_fp8w = qkvz;
        self.out_proj_fp8w = out_proj;
    }

    /// Transpose the block-scaled FP8 weights for the coalesced `w8a16_gemm_t`
    /// prefill path. Must be called after [`Self::set_fp8_decode_weights`].
    /// Mirrors `qwen3_attention::prefill_weights::transpose_fp8_for_prefill`.
    ///
    /// Without the transposed copies the SSM prefill projections fall through
    /// to the strided non-pipelined `w8a16_gemm`, which reads B\[N,K\] with a
    /// stride-K single-byte access per thread — ~15x under the memory floor on
    /// gfx1151 where the cp.async pipelined variant is absent. The transpose
    /// kernels live in the `w8a16_gemm_t` module, so this is a no-op where that
    /// module is absent. Allocates new GPU buffers; keeps the non-transposed
    /// copies alive for the decode `w8a16_gemv` path.
    pub fn transpose_fp8_for_prefill(&mut self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        if self.w8a16_gemm_t_k.0 == 0 {
            return Ok(()); // transposed GEMM kernel absent on this target
        }
        let transpose_k = gpu.kernel("w8a16_gemm_t", "transpose_fp8")?;
        let transpose_scale_k = gpu.kernel("w8a16_gemm_t", "transpose_block_scale")?;
        if let Some(w) = self.qkvz_fp8w.as_ref() {
            self.qkvz_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        if let Some(w) = self.out_proj_fp8w.as_ref() {
            self.out_proj_fp8w_t =
                Some(w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?);
        }
        Ok(())
    }

    /// Install PER-ROW FP8 weights for the row-wise cuBLASLt PREFILL arm
    /// (`ATLAS_FP8_ROWWISE=1`, mixed-precision compressed-tensors
    /// checkpoints). Decode is untouched and keeps the NVFP4 copy.
    ///
    /// The `Fp8PerRow` assertion is the mirror of `set_fp8_decode_weights`'s
    /// `Fp8BlockScaled` one: each setter refuses the other's layout, so the
    /// two FP8 shapes cannot cross into each other's kernels. That crossing
    /// does not fault — the smaller buffer is read in-bounds — so an assert
    /// is the only thing that catches it.
    pub fn set_fp8_rowwise_prefill_weights(
        &mut self,
        qkvz: Option<Fp8Weight>,
        out_proj: Option<Fp8Weight>,
    ) {
        for (w, what) in [(&qkvz, "qkvz"), (&out_proj, "out_proj")] {
            if let Some(w) = w {
                w.scale_format.expect(
                    crate::weight_map::WeightQuantFormat::Fp8PerRow,
                    "set_fp8_rowwise_prefill_weights (cuBLASLt row-wise expects [N] f32)",
                );
                let _ = what;
            }
        }
        self.qkvz_fp8w_rowwise = qkvz;
        self.out_proj_fp8w_rowwise = out_proj;
    }

    /// Set raw FP8 DevicePtrs for the prefill GEMM path ONLY (no decode GEMV
    /// scale fields). Used by the Qwen3.6-27B-FP8 native-FP8 SSM prefill path:
    /// the FP8 buffer here is a single-scale FP8 (BF16 → FP8 truncation; values
    /// already in FP8 range) suitable for `fp8_gemm_n128`. Decode falls back to
    /// the NVFP4/BF16 paths via the existing `qkvz_nvfp4*` fields. PCND:
    /// caller decides whether to install — never set implicitly.
    pub fn set_fp8_prefill_only_weights(
        &mut self,
        qkvz_fp8: Option<DevicePtr>,
        out_proj_fp8: Option<DevicePtr>,
    ) {
        if qkvz_fp8.is_some() {
            self.qkvz_fp8 = qkvz_fp8;
        }
        if out_proj_fp8.is_some() {
            self.out_proj_fp8 = out_proj_fp8;
        }
    }

    /// Pre-dequant NVFP4 → FP8 for QKVZ and out_proj transposed weights.
    /// Eliminates per-inference dequant overhead in prefill GEMMs.
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        let h = config.hidden_size;
        let qkvz_size = config.ssm_qkvz_size();
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;

        // QKVZ FP8 predequant: tested at ISL=1019, FP8 is ~50% slower (1900µs vs 1228µs)
        // because weight matrix [12288, 2048] is bandwidth-dominated at M=1024 — the 2×
        // larger FP8 weights (25 MB vs 12.6 MB NVFP4) cost more than the dequant saves.
        let _ = qkvz_size; // suppress unused warning
        // Use NON-transposed out_proj (ssm.out_proj is [N, K/2] layout).
        // predequant_nvfp4_to_fp8 assumes [N, K/2] input layout.
        if self.out_proj_nvfp4_t.is_some() {
            self.out_proj_fp8 = Some(self.ssm.out_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                h,
                value_dim,
                stream,
            )?);
        }
        Ok(())
    }
}
