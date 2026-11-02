// SPDX-License-Identifier: AGPL-3.0-only

//! CUDA launch wrappers for the native GLM visual tower.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::ops::dense_gemm_bf16_pipelined;
use crate::weight_map::DenseWeight;

use super::{FLASH_QUERY_ROWS, GlmVisionEncoder};

/// Threads per row for the block-reduction norms.
const NORM_THREADS: u32 = 256;

fn dense(ptr: DevicePtr) -> DenseWeight {
    DenseWeight { weight: ptr }
}

impl GlmVisionEncoder {
    /// Tensor-core BF16 GEMM: `output[m, n] = input[m, k] @ weight[n, k]^T`.
    /// Every GLM visual `k` (1176, 1024, 4096, 10240) is a multiple of 8, the
    /// pipelined kernel's vectorised-load alignment.
    pub(super) fn gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        dense_gemm_bf16_pipelined(
            gpu,
            self.k_gemm,
            input,
            &dense(weight),
            output,
            m as u32,
            n as u32,
            k as u32,
            stream,
        )
    }

    pub(super) fn gemm_bias(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: DevicePtr,
        bias: DevicePtr,
        output: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<()> {
        self.gemm(gpu, input, weight, output, m, n, k, stream)?;
        KernelLaunch::new(gpu, self.k_add_bias)
            .grid([div_ceil((m * n) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(output)
            .arg_ptr(bias)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .launch(stream)
    }

    pub(super) fn rms_norm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
        rows: usize,
        hidden: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_rms_norm)
            .grid([rows as u32, 1, 1])
            .block([NORM_THREADS, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight)
            .arg_ptr(output)
            .arg_u32(rows as u32)
            .arg_u32(hidden as u32)
            .arg_f32(self.rms_norm_eps)
            .launch(stream)
    }

    /// Attention over one image sequence held in the fused `qkv` rows:
    /// q/k RMSNorm + RoPE in place, then tensor-core flash attention.
    pub(super) fn attention(
        &self,
        gpu: &dyn GpuBackend,
        qkv: DevicePtr,
        q_norm: DevicePtr,
        k_norm: DevicePtr,
        cos: DevicePtr,
        sin: DevicePtr,
        output: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_qk_norm_rope)
            .grid([rows as u32, self.num_heads as u32, 2])
            .block([self.head_dim as u32, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(q_norm)
            .arg_ptr(k_norm)
            .arg_ptr(cos)
            .arg_ptr(sin)
            .arg_u32(rows as u32)
            .arg_u32(self.num_heads as u32)
            .arg_u32(self.head_dim as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.k_attention)
            .grid([
                div_ceil(rows as u32, FLASH_QUERY_ROWS),
                self.num_heads as u32,
                1,
            ])
            .block([128, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(output)
            .arg_u32(rows as u32)
            .arg_u32(self.num_heads as u32)
            .launch(stream)
    }

    pub(super) fn swiglu(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        intermediate: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_swiglu)
            .grid([div_ceil((rows * intermediate) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(output)
            .arg_u32(rows as u32)
            .arg_u32(intermediate as u32)
            .arg_f32(self.swiglu_limit)
            .launch(stream)
    }

    pub(super) fn add(
        &self,
        gpu: &dyn GpuBackend,
        dst: DevicePtr,
        src: DevicePtr,
        elements: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(elements as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_u32(elements as u32)
            .launch(stream)
    }

    pub(super) fn layer_norm(
        &self,
        gpu: &dyn GpuBackend,
        data: DevicePtr,
        weight: DevicePtr,
        bias: DevicePtr,
        rows: usize,
        hidden: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_layer_norm)
            .grid([rows as u32, 1, 1])
            .block([NORM_THREADS, 1, 1])
            .arg_ptr(data)
            .arg_ptr(weight)
            .arg_ptr(bias)
            .arg_u32(rows as u32)
            .arg_u32(hidden as u32)
            .arg_f32(self.rms_norm_eps)
            .launch(stream)
    }

    pub(super) fn gelu(
        &self,
        gpu: &dyn GpuBackend,
        data: DevicePtr,
        elements: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(elements as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(data)
            .arg_u32(elements as u32)
            .launch(stream)
    }

    /// The merger's 2x2 stride-2 Conv2d as one GEMM: reorder each merge
    /// block's four rows into the weight's `(c, ih, iw)` K order, then
    /// multiply by the `[out, hidden * 4]` view of the weight. `scratch` holds
    /// `patches * hidden` BF16 elements.
    pub(super) fn conv2d(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        scratch: DevicePtr,
        output: DevicePtr,
        grid_h: usize,
        grid_w: usize,
        stream: u64,
    ) -> Result<()> {
        let tokens = (grid_h / 2) * (grid_w / 2);
        KernelLaunch::new(gpu, self.k_merge_reorder)
            .grid([div_ceil((tokens * 4 * self.hidden_size) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(scratch)
            .arg_u32(tokens as u32)
            .arg_u32(self.hidden_size as u32)
            .launch(stream)?;
        self.gemm_bias(
            gpu,
            scratch,
            self.downsample_w,
            self.downsample_b,
            output,
            tokens,
            self.out_hidden_size,
            4 * self.hidden_size,
            stream,
        )
    }
}
