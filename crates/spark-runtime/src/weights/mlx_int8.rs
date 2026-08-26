// SPDX-License-Identifier: AGPL-3.0-only

//! MLX uint32-packed 8-bit weight format support.
//!
//! Models published as `mlx-community/<name>-MLX-8bit` ship safetensors
//! that pack each linear layer's weights into a triplet:
//!
//! - `{base}.weight` — `U32` of shape `[out_features, in_features / 4]`.
//!   Each `uint32` packs four unsigned 8-bit weight bytes (low byte =
//!   column 0, high byte = column 3 of the four-column block).
//! - `{base}.scales` — `BF16` of shape `[out_features, in_features / G]`.
//!   One scale per group of `G` (default 64) columns.
//! - `{base}.biases` — `BF16` of shape `[out_features, in_features / G]`.
//!   One additive bias per group.
//!
//! The dequantization formula is affine:
//!
//!   `w[r, c] = byte * scales[r, c/G] + biases[r, c/G]`
//!
//! where `byte` is the `c%4`-th byte of `packed[r, c/4]` (little-endian).
//!
//! `MlxInt8Weight` holds the triplet on the GPU; `dequantize_to` runs the
//! `mlx_int8_dequant` kernel to materialize a contiguous BF16 view, and
//! `gemv` / `gemm` run the fused dequant-and-multiply kernels for the
//! decode and prefill paths respectively.

use anyhow::{Context, Result, bail};
use safetensors::SafeTensors;
use serde_json::Value as JsonValue;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// Quantization metadata parsed from the model's `config.json`.
///
/// Both top-level `quantization` and `quantization_config` blocks are
/// recognised — MLX exports the same data under both keys.
#[derive(Debug, Clone, Copy)]
pub struct MlxQuantConfig {
    pub bits: u32,
    pub group_size: u32,
}

impl MlxQuantConfig {
    /// Look for `quantization` (or `quantization_config` as fallback)
    /// at the top level of `config.json`. Returns `None` if either
    /// the block is missing or the bits/group_size keys are absent —
    /// non-MLX checkpoints flow through other detection paths.
    pub fn from_config(config: &JsonValue) -> Option<Self> {
        let q = config
            .get("quantization")
            .or_else(|| config.get("quantization_config"))?;
        let bits = q.get("bits")?.as_u64()? as u32;
        let group_size = q.get("group_size")?.as_u64()? as u32;
        Some(Self { bits, group_size })
    }
}

/// One MLX-int8 quantized linear weight resident on the GPU.
///
/// The fields are public because the consumer (transformer layer
/// implementation) usually owns the pointers and frees them in batch
/// at model teardown — there's no per-weight Drop here.
pub struct MlxInt8Weight {
    /// `[out_features, in_features / 4]` packed bytes (uint32 words).
    pub packed: DevicePtr,
    /// `[out_features, in_features / group_size]` per-group BF16 scales.
    pub scales: DevicePtr,
    /// `[out_features, in_features / group_size]` per-group BF16 biases.
    pub biases: DevicePtr,
    pub out_features: u32,
    pub in_features: u32,
    pub group_size: u32,
}

impl MlxInt8Weight {
    /// Load a `(.weight, .scales, .biases)` triplet from a parsed
    /// safetensors blob and upload to the GPU. `base` is the tensor
    /// name minus the suffix (e.g. `"language_model.model.embed_tokens"`).
    pub fn load(
        gpu: &dyn GpuBackend,
        st: &SafeTensors,
        base: &str,
        group_size: u32,
    ) -> Result<Self> {
        let weight_name = format!("{base}.weight");
        let scales_name = format!("{base}.scales");
        let biases_name = format!("{base}.biases");

        let weight = st
            .tensor(&weight_name)
            .with_context(|| format!("missing tensor {weight_name}"))?;
        let scales = st
            .tensor(&scales_name)
            .with_context(|| format!("missing tensor {scales_name}"))?;
        let biases = st
            .tensor(&biases_name)
            .with_context(|| format!("missing tensor {biases_name}"))?;

        if weight.dtype() != safetensors::Dtype::U32 {
            bail!(
                "{weight_name}: expected U32 (MLX 8-bit packed), got {:?}",
                weight.dtype()
            );
        }
        if scales.dtype() != safetensors::Dtype::BF16 || biases.dtype() != safetensors::Dtype::BF16
        {
            bail!(
                "{base}.scales/biases: expected BF16, got scales={:?}, biases={:?}",
                scales.dtype(),
                biases.dtype()
            );
        }

        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            bail!(
                "{weight_name}: expected 2-D weight tensor, got rank {}",
                weight_shape.len()
            );
        }
        let out_features = weight_shape[0] as u32;
        let packed_cols = weight_shape[1] as u32;
        let in_features = packed_cols * 4;

        let groups_per_row = in_features
            .checked_div(group_size)
            .filter(|&g| g * group_size == in_features)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{base}: in_features {in_features} not divisible by group_size {group_size}"
                )
            })?;

        let expected = [out_features as usize, groups_per_row as usize];
        if scales.shape() != expected {
            bail!(
                "{}.scales: expected shape {:?}, got {:?}",
                base,
                expected,
                scales.shape()
            );
        }
        if biases.shape() != expected {
            bail!(
                "{}.biases: expected shape {:?}, got {:?}",
                base,
                expected,
                biases.shape()
            );
        }

        let packed_ptr = gpu.alloc(weight.data().len())?;
        gpu.copy_h2d(weight.data(), packed_ptr)?;
        let scales_ptr = gpu.alloc(scales.data().len())?;
        gpu.copy_h2d(scales.data(), scales_ptr)?;
        let biases_ptr = gpu.alloc(biases.data().len())?;
        gpu.copy_h2d(biases.data(), biases_ptr)?;

        Ok(Self {
            packed: packed_ptr,
            scales: scales_ptr,
            biases: biases_ptr,
            out_features,
            in_features,
            group_size,
        })
    }

    /// Materialize the full dequantized weight as BF16 into `out`,
    /// which must be a `DevicePtr` to a buffer of at least
    /// `out_features * in_features * 2` bytes. Runs the
    /// `mlx_int8_dequant` Metal kernel under the hood.
    pub fn dequantize_to(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_dequant", "mlx_int8_dequant")?;
        // 16×1 thread grid per (col_tile, row); covers all (r, c)
        // with bounds checks inside the kernel.
        let block_x: u32 = 16;
        let block_y: u32 = 1;
        let grid_x = self.in_features.div_ceil(block_x);
        let grid_y = self.out_features;
        gpu.launch_typed(
            kernel,
            [grid_x, grid_y, 1],
            [block_x, block_y, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(out),
            ],
        )
    }

    /// Decode-path matvec: `y = self_dequant @ x`. `x` must be BF16
    /// `[in_features]`; `y` must be a BF16 buffer with at least
    /// `out_features` slots. Runs the fused `mlx_int8_gemv` kernel.
    pub fn gemv(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_gemv", "mlx_int8_gemv")?;
        // 4 rows per threadgroup, one simdgroup (32 threads) per row.
        // Sharing `x[]` across 4 rows via L2 cache cuts input-side
        // bandwidth by 4×; row-local simd_sum avoids cross-simdgroup
        // reductions entirely.
        const ROWS_PER_TG: u32 = 4;
        const SIMDGROUP_SIZE: u32 = 32;
        let threads_per_tg: u32 = ROWS_PER_TG * SIMDGROUP_SIZE; // 128
        let row_groups = self.out_features.div_ceil(ROWS_PER_TG);
        gpu.launch_typed(
            kernel,
            [row_groups, 1, 1],
            [threads_per_tg, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(x),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// Like `gemv_silu_gate`, but additionally folds the residual
    /// stream addition into the same kernel:
    ///   `y[n] = x_resid[n] + sum_k self[n, k] * (silu(gate[k]) ⊙ up[k])`
    /// Eliminates the trailing `bf16_add` and the FFN-out staging
    /// buffer on the decoder layer's exit.
    pub fn gemv_silu_gate_resid(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_gemv_silu_gate", "mlx_int8_gemv_silu_gate_resid")?;
        const ROWS_PER_TG: u32 = 4;
        const SIMDGROUP_SIZE: u32 = 32;
        let threads_per_tg: u32 = ROWS_PER_TG * SIMDGROUP_SIZE;
        let row_groups = self.out_features.div_ceil(ROWS_PER_TG);
        gpu.launch_typed(
            kernel,
            [row_groups, 1, 1],
            [threads_per_tg, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(up),
                KernelArg::Buffer(x_resid),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// Decode-path FFN-residual fusion:
    ///   `y = self @ (silu(gate) ⊙ up)`
    /// Runs the fused `mlx_int8_gemv_silu_gate` kernel — replaces the
    /// `silu_gate → gemv(down_proj)` pair with a single launch and
    /// no INTERMEDIATE-sized staging buffer.
    pub fn gemv_silu_gate(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_gemv_silu_gate", "mlx_int8_gemv_silu_gate")?;
        const ROWS_PER_TG: u32 = 4;
        const SIMDGROUP_SIZE: u32 = 32;
        let threads_per_tg: u32 = ROWS_PER_TG * SIMDGROUP_SIZE;
        let row_groups = self.out_features.div_ceil(ROWS_PER_TG);
        gpu.launch_typed(
            kernel,
            [row_groups, 1, 1],
            [threads_per_tg, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(up),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// Prefill-path GEMM: `Y = X @ self_dequant^T`. `X` is BF16
    /// `[m, in_features]`; `Y` is BF16 `[m, out_features]`. Runs
    /// the fused `mlx_int8_gemm` kernel — straightforward correctness
    /// reference; tile-optimised replacement is a follow-on PR.
    pub fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        y: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        let kernel = gpu.kernel("mlx_int8_gemm", "mlx_int8_gemm")?;
        let block_x: u32 = 16;
        let block_y: u32 = 16;
        let grid_x = self.out_features.div_ceil(block_x);
        let grid_y = m.div_ceil(block_y);
        gpu.launch_typed(
            kernel,
            [grid_x, grid_y, 1],
            [block_x, block_y, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&m.to_le_bytes()),
                KernelArg::Bytes(&self.out_features.to_le_bytes()),
                KernelArg::Bytes(&self.in_features.to_le_bytes()),
                KernelArg::Bytes(&self.group_size.to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(self.packed),
                KernelArg::Buffer(self.scales),
                KernelArg::Buffer(self.biases),
                KernelArg::Buffer(y),
            ],
        )
    }

    /// Free the three GPU buffers backing this weight. Idempotent if
    /// the pointers are null. Call this at model teardown — there's
    /// no Drop because `MlxInt8Weight` is intentionally Copy-friendly
    /// (the `DevicePtr`s are u64 handles, not owners).
    pub fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.packed)?;
        gpu.free(self.scales)?;
        gpu.free(self.biases)?;
        Ok(())
    }
}

/// Dual-output GEMV: `gate_y = gate @ x` and `up_y = up @ x` in one
/// kernel launch. Halves the x-side memory bandwidth and removes one
/// kernel-launch round-trip per FFN. Both projections must share
/// `(out_features, in_features, group_size)`.
pub fn gemv_gate_up(
    gpu: &dyn GpuBackend,
    gate: &MlxInt8Weight,
    up: &MlxInt8Weight,
    x: DevicePtr,
    gate_y: DevicePtr,
    up_y: DevicePtr,
    stream: u64,
) -> Result<()> {
    debug_assert_eq!(gate.out_features, up.out_features);
    debug_assert_eq!(gate.in_features, up.in_features);
    debug_assert_eq!(gate.group_size, up.group_size);
    let kernel = gpu.kernel("mlx_int8_gemv_gate_up", "mlx_int8_gemv_gate_up")?;
    const ROWS_PER_TG: u32 = 4;
    const SIMDGROUP_SIZE: u32 = 32;
    let threads_per_tg: u32 = ROWS_PER_TG * SIMDGROUP_SIZE;
    let row_groups = gate.out_features.div_ceil(ROWS_PER_TG);
    gpu.launch_typed(
        kernel,
        [row_groups, 1, 1],
        [threads_per_tg, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&gate.out_features.to_le_bytes()),
            KernelArg::Bytes(&gate.in_features.to_le_bytes()),
            KernelArg::Bytes(&gate.group_size.to_le_bytes()),
            KernelArg::Buffer(gate.packed),
            KernelArg::Buffer(gate.scales),
            KernelArg::Buffer(gate.biases),
            KernelArg::Buffer(up.packed),
            KernelArg::Buffer(up.scales),
            KernelArg::Buffer(up.biases),
            KernelArg::Buffer(x),
            KernelArg::Buffer(gate_y),
            KernelArg::Buffer(up_y),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quant_config_from_mlx_layout() {
        let cfg: JsonValue = serde_json::json!({
            "quantization": { "bits": 8, "group_size": 64, "mode": "affine" },
            "model_type": "qwen3_5",
        });
        let q = MlxQuantConfig::from_config(&cfg).expect("expected quant block");
        assert_eq!(q.bits, 8);
        assert_eq!(q.group_size, 64);
    }

    #[test]
    fn parse_quant_config_falls_back_to_quantization_config() {
        let cfg: JsonValue = serde_json::json!({
            "quantization_config": { "bits": 8, "group_size": 64 },
        });
        let q = MlxQuantConfig::from_config(&cfg).expect("expected quant_config block");
        assert_eq!(q.bits, 8);
        assert_eq!(q.group_size, 64);
    }

    #[test]
    fn parse_quant_config_returns_none_when_absent() {
        let cfg: JsonValue = serde_json::json!({ "model_type": "qwen3_5" });
        assert!(MlxQuantConfig::from_config(&cfg).is_none());
    }
}
