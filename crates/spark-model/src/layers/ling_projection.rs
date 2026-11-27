// SPDX-License-Identifier: AGPL-3.0-only
//! Original BF16 or dynamic W8A8 projections for the Ling trunk on one GPU.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// BF16 rows: up to the batched GEMV's limit one kernel whose rows do not
/// depend on the row count (serial decode and a small verify or batch agree
/// byte for byte), a tensor-core GEMM past it.
pub(crate) fn bf16_prompt_rows(
    gpu: &dyn GpuBackend,
    batch_gemv: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if m <= ops::DENSE_GEMV_BATCHM_MAX_M {
        ops::dense_gemv_batchm(gpu, batch_gemv, input, weight, output, m, n, k, n, stream)
    } else {
        ops::bf16_gemm(input, weight.weight.0, output, m, n, k, stream)
    }
}

#[derive(Clone, Copy)]
pub struct LingProjectionScratch {
    act: DevicePtr,
    act_bytes: usize,
    scales: DevicePtr,
    scale_bytes: usize,
}

impl LingProjectionScratch {
    pub fn new(act: DevicePtr, act_bytes: usize, scales: DevicePtr, scale_bytes: usize) -> Self {
        Self {
            act,
            act_bytes,
            scales,
            scale_bytes,
        }
    }
}

#[derive(Clone, Copy)]
enum Format {
    Bf16 {
        batch_gemv: KernelHandle,
        decode_nvfp4: Option<(
            crate::weight_map::QuantizedWeight,
            KernelHandle,
            KernelHandle,
        )>,
    },
    Fp8 {
        scales: DevicePtr,
        gemm: KernelHandle,
        quant: KernelHandle,
        linear: KernelHandle,
    },
}

pub struct LingProjection {
    format: Format,
    weight: DevicePtr,
    n: u32,
    k: u32,
}

impl LingProjection {
    pub fn load(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        ensure!(rows > 0 && cols > 0, "empty Ling projection {name}");
        let w = store.get(&format!("{name}.weight"))?;
        ensure!(
            w.shape == [rows, cols],
            "{name} has shape {:?}; expected [{rows}, {cols}]",
            w.shape
        );
        let format = match w.dtype {
            WeightDtype::BF16 => {
                // kv_b_proj never runs at decode: MLA absorbs it into the
                // query and value sides of the latent cache (`ling_mla`).
                let pack_attention = name.contains(".attention.")
                    && !name.ends_with(".kv_b_proj")
                    && std::env::var("ATLAS_LING_NVFP4_DECODE").as_deref() == Ok("1");
                let pack_mlp = name.contains(".mlp.")
                    && std::env::var("ATLAS_LING_NVFP4_MLP_DECODE").as_deref() == Ok("1");
                let decode_nvfp4 = if (pack_attention || pack_mlp)
                    && rows >= 512
                    && cols >= 512
                    && store.contains("model.layers.2.mlp.experts.0.gate_proj.weight_packed")
                {
                    let packed = crate::weight_map::quantize_to_nvfp4(
                        &DenseWeight { weight: w.ptr },
                        rows,
                        cols,
                        gpu,
                        gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
                        gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
                        gpu.default_stream(),
                    )?;
                    Some((
                        packed,
                        gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
                        gpu.kernel("w4a16_gemv", "w4a16_gemv_batch8")?,
                    ))
                } else {
                    None
                };
                Format::Bf16 {
                    decode_nvfp4,
                    batch_gemv: gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
                }
            }
            WeightDtype::FP8E4M3 => {
                ensure!(
                    cols.is_multiple_of(128),
                    "{name}: input width must contain whole FP8 blocks"
                );
                let s = store.get(&format!("{name}.weight_scale_inv"))?;
                ensure!(
                    s.shape == [rows.div_ceil(128), cols / 128],
                    "{name}: invalid scale grid {:?}",
                    s.shape
                );
                ensure!(
                    matches!(s.dtype, WeightDtype::BF16 | WeightDtype::FP32),
                    "{name}: invalid block scale dtype {:?}",
                    s.dtype
                );
                Format::Fp8 {
                    scales: crate::weight_map::dense_keep_f32(
                        store,
                        &format!("{name}.weight_scale_inv"),
                        gpu,
                    )?
                    .weight,
                    gemm: gpu.kernel("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled")?,
                    quant: gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
                    linear: gpu.kernel("ling_fp8_linear", "ling_fp8_linear")?,
                }
            }
            other => bail!("{name}: unsupported Ling projection dtype {other:?}"),
        };
        Ok(Self {
            weight: w.ptr,
            format,
            n: rows.try_into()?,
            k: cols.try_into()?,
        })
    }

    /// The weight as BF16 `[n, k]`: the checkpoint's own tensor, or for an
    /// FP8 checkpoint a block-dequantized copy that lives as long as the model.
    pub fn bf16_weight(
        &self,
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        name: &str,
    ) -> Result<DevicePtr> {
        match self.format {
            Format::Bf16 { .. } => Ok(self.weight),
            Format::Fp8 { .. } => {
                Ok(crate::weight_map::dequant_fp8_blockscaled_to_bf16(store, name, gpu)?.weight)
            }
        }
    }

    /// [`Self::forward`] over consecutive pieces of at most `piece` rows. The
    /// BF16 GEMM's algorithm, and so its rounding, follows the row count: a
    /// fixed piece keeps every row's bytes whatever `rows` is.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_pieces(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: u32,
        piece: u32,
        scratch: LingProjectionScratch,
        stream: u64,
    ) -> Result<()> {
        for first in (0..rows).step_by(piece as usize) {
            let at = first as usize;
            self.forward(
                gpu,
                input.offset(at * self.k as usize * 2),
                output.offset(at * self.n as usize * 2),
                (rows - first).min(piece),
                scratch,
                stream,
            )?;
        }
        Ok(())
    }

    pub fn forward(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: u32,
        scratch: LingProjectionScratch,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=65535).contains(&rows),
            "invalid Ling projection row count {rows}"
        );
        match self.format {
            Format::Bf16 {
                batch_gemv,
                decode_nvfp4,
            } => {
                if let Some((weight, one, batch)) = decode_nvfp4 {
                    if rows == 1 {
                        return ops::w4a16_gemv(
                            gpu, one, input, &weight, output, self.n, self.k, stream,
                        );
                    }
                    if rows <= 8 {
                        return ops::w4a16_gemv_batchm(
                            gpu, batch, input, &weight, output, rows, self.n, self.k, stream,
                        );
                    }
                }
                let weight = DenseWeight {
                    weight: self.weight,
                };
                // One row too: `dense_gemv_batchm`'s rows do not depend on the
                // row count, so serial decode computes the bytes a two-row MTP
                // verify computes for the same row.
                bf16_prompt_rows(
                    gpu, batch_gemv, input, &weight, output, rows, self.n, self.k, stream,
                )
            }
            Format::Fp8 {
                scales,
                gemm,
                quant,
                linear,
            } => {
                let elems = (rows as usize)
                    .checked_mul(self.k as usize)
                    .ok_or_else(|| anyhow::anyhow!("Ling FP8 scratch size overflow"))?;
                ensure!(
                    elems <= scratch.act_bytes && elems / 128 * 4 <= scratch.scale_bytes,
                    "insufficient Ling FP8 projection scratch"
                );
                ops::per_token_group_quant_fp8(
                    gpu,
                    quant,
                    input,
                    scratch.act,
                    scratch.scales,
                    rows,
                    self.k,
                    stream,
                )?;
                if rows >= 16 {
                    return ops::fp8_gemm_t_blockscaled(
                        gpu,
                        gemm,
                        scratch.act,
                        scratch.scales,
                        self.weight,
                        scales,
                        output,
                        rows,
                        self.n,
                        self.k,
                        stream,
                    );
                }
                KernelLaunch::new(gpu, linear)
                    .grid([self.n.div_ceil(8), rows, 1])
                    .block([256, 1, 1])
                    .arg_ptr(scratch.act)
                    .arg_ptr(scratch.scales)
                    .arg_ptr(self.weight)
                    .arg_ptr(scales)
                    .arg_ptr(output)
                    .arg_u32(self.n)
                    .arg_u32(self.k)
                    .arg_u32(0)
                    .launch(stream)
            }
        }
    }
}
