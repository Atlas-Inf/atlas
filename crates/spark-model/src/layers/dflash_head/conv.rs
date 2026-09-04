// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash2 2-tap grouped dynamic causal convolution.
//!
//! Wraps attention and MLP sublayers:
//! - `prepare`: projects input states through `kernel_projection` to generate
//!   dynamic kernel coefficients (2 taps input, 2 taps output) and applies
//!   input convolution on `input_buf`.
//! - `finish`: applies output convolution on `sublayer_out` using the saved
//!   output dynamic kernel coefficients.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::ops;
use crate::weight_map::DenseWeight;

#[derive(Clone)]
pub struct Dflash2Conv {
    pub base_kernel: DenseWeight,
    pub kernel_projection: DenseWeight,
    pub num_groups: usize,
    pub group_size: usize,
    pub kernel_size: usize,
    pub hidden_size: usize,
}

impl Dflash2Conv {
    pub fn new(
        base_kernel: DenseWeight,
        kernel_projection: DenseWeight,
        hidden_size: usize,
        group_size: usize,
        kernel_size: usize,
    ) -> Self {
        let num_groups = hidden_size / group_size.max(1);
        Self {
            base_kernel,
            kernel_projection,
            num_groups,
            group_size,
            kernel_size,
            hidden_size,
        }
    }

    /// Project input states to dynamic kernel delta and apply input-side convolution.
    /// Returns the device pointer to the output-side dynamic coefficients.
    pub fn prepare(
        &self,
        gpu: &dyn GpuBackend,
        dense_gemm_pipelined: KernelHandle,
        conv_kernel: Option<KernelHandle>,
        input_buf: DevicePtr,
        delta_buf: DevicePtr,
        out_buf: DevicePtr,
        gamma: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = self.hidden_size as u32;
        let total_dynamic_dims = (2 * self.kernel_size * self.num_groups) as u32;

        ops::dense_gemm_bf16_pipelined(
            gpu,
            dense_gemm_pipelined,
            input_buf,
            &self.kernel_projection,
            delta_buf,
            gamma,
            total_dynamic_dims,
            h,
            stream,
        )?;

        let input_base_kernel = self.base_kernel.weight;
        let input_delta = delta_buf;

        if let Some(k) = conv_kernel {
            let total_elems = gamma * h;
            let block_size = 256u32;
            let grid_size = (total_elems + block_size - 1) / block_size;
            KernelLaunch::new(gpu, k)
                .grid([grid_size, 1, 1])
                .block([block_size, 1, 1])
                .arg_ptr(out_buf)
                .arg_ptr(input_buf)
                .arg_ptr(input_delta)
                .arg_ptr(input_base_kernel)
                .arg_u32(gamma)
                .arg_u32(h)
                .arg_u32(self.group_size as u32)
                .arg_u32(self.num_groups as u32)
                .launch(stream)?;
        }

        let output_delta = delta_buf.offset(2 * self.num_groups * 2);
        Ok(output_delta)
    }

    /// Apply output-side convolution on sublayer output.
    pub fn finish(
        &self,
        gpu: &dyn GpuBackend,
        conv_kernel: Option<KernelHandle>,
        sublayer_out: DevicePtr,
        output_delta: DevicePtr,
        out_buf: DevicePtr,
        gamma: u32,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let output_base_kernel = self.base_kernel.weight.offset(2 * self.hidden_size * 2);

        if let Some(k) = conv_kernel {
            let total_elems = gamma * h;
            let block_size = 256u32;
            let grid_size = (total_elems + block_size - 1) / block_size;
            KernelLaunch::new(gpu, k)
                .grid([grid_size, 1, 1])
                .block([block_size, 1, 1])
                .arg_ptr(out_buf)
                .arg_ptr(sublayer_out)
                .arg_ptr(output_delta)
                .arg_ptr(output_base_kernel)
                .arg_u32(gamma)
                .arg_u32(h)
                .arg_u32(self.group_size as u32)
                .arg_u32(self.num_groups as u32)
                .launch(stream)?;
        }
        Ok(())
    }
}
