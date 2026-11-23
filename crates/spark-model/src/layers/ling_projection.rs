// SPDX-License-Identifier: AGPL-3.0-only
//! Original BF16 or dynamic W8A8 projections for the Ling trunk on one GPU.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// The one-row decode and small concurrent batches use the original kernels.
pub(crate) fn bf16_prompt_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if m >= 32 {
        ops::bf16_gemm(input, weight.weight.0, output, m, n, k, stream)
    } else {
        ops::dense_gemm_tc(gpu, kernel, input, weight, output, m, n, k, stream)
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
        gemv: KernelHandle,
        gemm: KernelHandle,
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
            WeightDtype::BF16 => Format::Bf16 {
                gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
                gemm: gpu.kernel("gemm_tc", "dense_gemm_tc")?,
            },
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
            Format::Bf16 { gemv, gemm } => {
                let weight = DenseWeight {
                    weight: self.weight,
                };
                if rows == 1 {
                    ops::dense_gemv(gpu, gemv, input, &weight, output, self.n, self.k, stream)
                } else {
                    bf16_prompt_rows(
                        gpu, gemm, input, &weight, output, rows, self.n, self.k, stream,
                    )
                }
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
