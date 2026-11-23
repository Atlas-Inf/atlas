// SPDX-License-Identifier: AGPL-3.0-only
//! Original BF16 or dynamic W8A8 projections for the Ling trunk on one GPU.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

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
    Bf16,
    Fp8 { scales: DevicePtr, bf16: bool },
}

pub struct LingProjection {
    format: Format,
    weight: DevicePtr,
    n: u32,
    k: u32,
}

impl LingProjection {
    pub fn load(store: &WeightStore, name: &str, rows: usize, cols: usize) -> Result<Self> {
        ensure!(rows > 0 && cols > 0, "empty Ling projection {name}");
        let w = store.get(&format!("{name}.weight"))?;
        ensure!(
            w.shape == [rows, cols],
            "{name} has shape {:?}; expected [{rows}, {cols}]",
            w.shape
        );
        let format = match w.dtype {
            WeightDtype::BF16 => Format::Bf16,
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
                    scales: s.ptr,
                    bf16: s.dtype == WeightDtype::BF16,
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
            Format::Bf16 => {
                let weight = DenseWeight {
                    weight: self.weight,
                };
                if rows == 1 {
                    ops::dense_gemv(
                        gpu,
                        gpu.kernel("gemv", "dense_gemv_bf16")?,
                        input,
                        &weight,
                        output,
                        self.n,
                        self.k,
                        stream,
                    )
                } else {
                    ops::dense_gemm_tc(
                        gpu,
                        gpu.kernel("gemm_tc", "dense_gemm_tc")?,
                        input,
                        &weight,
                        output,
                        rows,
                        self.n,
                        self.k,
                        stream,
                    )
                }
            }
            Format::Fp8 { scales, bf16 } => {
                let elems = (rows as usize)
                    .checked_mul(self.k as usize)
                    .ok_or_else(|| anyhow::anyhow!("Ling FP8 scratch size overflow"))?;
                ensure!(
                    elems <= scratch.act_bytes && elems / 128 * 4 <= scratch.scale_bytes,
                    "insufficient Ling FP8 projection scratch"
                );
                ops::per_token_group_quant_fp8(
                    gpu,
                    gpu.kernel("per_token_group_quant_fp8", "per_token_group_quant_fp8")?,
                    input,
                    scratch.act,
                    scratch.scales,
                    rows,
                    self.k,
                    stream,
                )?;
                KernelLaunch::new(gpu, gpu.kernel("ling_fp8_linear", "ling_fp8_linear")?)
                    .grid([self.n.div_ceil(8), rows, 1])
                    .block([256, 1, 1])
                    .arg_ptr(scratch.act)
                    .arg_ptr(scratch.scales)
                    .arg_ptr(self.weight)
                    .arg_ptr(scales)
                    .arg_ptr(output)
                    .arg_u32(self.n)
                    .arg_u32(self.k)
                    .arg_u32(u32::from(bf16))
                    .launch(stream)
            }
        }
    }
}
