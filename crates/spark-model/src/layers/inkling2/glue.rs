// SPDX-License-Identifier: AGPL-3.0-only

//! The small glue kernels of `inkling2_misc.cu` and the shared Atlas kernels
//! the Inkling 2 layer reuses: BF16 projections (the M=1 GEMVs for decode,
//! the BF16 GEMMs for prefill rows), fp32 router logits, batched fp32 sigmoid
//! top-k and the BF16 KV write. All buffers are contiguous `[rows, dim]`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::weights::Linear;
use crate::layers::ops;

#[derive(Clone, Copy, Debug)]
pub(crate) struct GlueKernels {
    pub rmsnorm: KernelHandle,
    pub gate_mul: KernelHandle,
    pub residual: KernelHandle,
    pub layer_out: KernelHandle,
    pub f32_to_bf16: KernelHandle,
    /// `dense_gemv_bf16`: y[N] = W[N, K] x[K], BF16
    pub dense_gemv: KernelHandle,
    /// `dense_gemv_bf16_fp32out`: one row of router logits
    pub dense_gemv_f32: KernelHandle,
    /// `moe_topk_sigmoid_batched_f32`
    pub topk: KernelHandle,
    /// `reshape_and_cache_flash` (BF16 KV)
    pub kv_write: KernelHandle,
    /// Spark NVFP4 projections: one row, many rows
    pub w4a16_gemv: KernelHandle,
    pub w4a16_gemv_b8: KernelHandle,
    pub w4a16_gemv_b16: KernelHandle,
}

impl GlueKernels {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let m = |f: &str| gpu.kernel("inkling2_misc", f);
        Ok(Self {
            rmsnorm: m("inkling2_rmsnorm")?,
            gate_mul: m("inkling2_gate_mul")?,
            residual: m("inkling2_residual")?,
            layer_out: m("inkling2_layer_out")?,
            f32_to_bf16: m("inkling2_f32_to_bf16")?,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemv_f32: gpu.kernel("gemv", "dense_gemv_bf16_fp32out")?,
            topk: gpu.kernel("moe_topk_sig", "moe_topk_sigmoid_batched_f32")?,
            kv_write: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_b8: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch8")?,
            w4a16_gemv_b16: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch16")?,
        })
    }
}

fn blocks(n: usize) -> u32 {
    n.div_ceil(256) as u32
}

impl GlueKernels {
    #[allow(clippy::too_many_arguments)]
    fn gemv_with(
        &self,
        gpu: &dyn GpuBackend,
        k: KernelHandle,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        n: usize,
        kk: usize,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, k)
            .grid([(n as u32).div_ceil(4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(y)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .launch(s)
    }

    /// `y[rows, n] = x[rows, k] W[n, k]^T`, BF16 out: a BF16 weight through
    /// the dense GEMV / BF16 GEMM; a Spark NVFP4 one through `w4a16_gemv` /
    /// `w4a16_gemv_batch8/16` for decode and verify rows, and its checkpoint BF16
    /// weight for prefill rows (from `MOE_GROUPED_MIN_ROWS`, where the experts
    /// take the checkpoint's too).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn proj(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        x: DevicePtr,
        w: Linear,
        y: DevicePtr,
        n: usize,
        k: usize,
        s: u64,
    ) -> Result<()> {
        match w {
            Linear::Bf16(w) if rows == 1 => self.gemv_with(gpu, self.dense_gemv, x, w, y, n, k, s),
            Linear::Bf16(w) => ops::bf16_gemm(x, w.0, y, rows as u32, n as u32, k as u32, s),
            Linear::Nvfp4 { q, .. } if rows == 1 => {
                ops::w4a16_gemv(gpu, self.w4a16_gemv, x, &q, y, n as u32, k as u32, s)
            }
            // a decode batch: Atlas's batched GEMV reads the weight once for all rows
            Linear::Nvfp4 { q, .. } if rows < super::moe_ops::MOE_GROUPED_MIN_ROWS => {
                let kernel = if rows <= 8 {
                    self.w4a16_gemv_b8
                } else {
                    self.w4a16_gemv_b16
                };
                ops::w4a16_gemv_batchm(gpu, kernel, x, &q, y, rows as u32, n as u32, k as u32, s)
            }
            Linear::Nvfp4 { prefill, .. } => {
                ops::bf16_gemm(x, prefill.0, y, rows as u32, n as u32, k as u32, s)
            }
        }
    }

    /// The same with fp32 outputs (router logits).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn proj_f32(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        n: usize,
        k: usize,
        s: u64,
    ) -> Result<()> {
        if rows == 1 {
            self.gemv_with(gpu, self.dense_gemv_f32, x, w, y, n, k, s)
        } else {
            ops::bf16_gemm_f32out(x, w.0, y, rows as u32, n as u32, k as u32, s)
        }
    }

    /// Exactly one of `x_f32` / `x_bf16` non-NULL.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rmsnorm(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        x_f32: DevicePtr,
        x_bf16: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        h: usize,
        eps: f32,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rmsnorm)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x_f32)
            .arg_ptr(x_bf16)
            .arg_ptr(w)
            .arg_ptr(y)
            .arg_u32(h as u32)
            .arg_u32(h as u32)
            .arg_u32(h as u32)
            .arg_f32(eps)
            .launch(s)
    }

    pub(crate) fn gate_mul(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        g: DevicePtr,
        out: DevicePtr,
        n: usize,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.gate_mul)
            .grid([blocks(n), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(g)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .launch(s)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn residual(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        c: DevicePtr,
        r: DevicePtr,
        n: usize,
        restart: bool,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.residual)
            .grid([blocks(n), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(c)
            .arg_ptr(r)
            .arg_u32(n as u32)
            .arg_u32(u32::from(restart))
            .launch(s)
    }

    pub(crate) fn layer_out(
        &self,
        gpu: &dyn GpuBackend,
        r: DevicePtr,
        m: DevicePtr,
        out: DevicePtr,
        n: usize,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.layer_out)
            .grid([blocks(n), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(r)
            .arg_ptr(m)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .launch(s)
    }

    pub(crate) fn cast_bf16(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        out: DevicePtr,
        n: usize,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.f32_to_bf16)
            .grid([blocks(n), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .launch(s)
    }

    /// Sigmoid top-k over fp32 logits `[rows, experts]`, normalised, x `scale`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn route(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        logits: DevicePtr,
        bias: DevicePtr,
        ids: DevicePtr,
        w: DevicePtr,
        experts: usize,
        top_k: usize,
        scale: f32,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.topk)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(bias)
            .arg_ptr(ids)
            .arg_ptr(w)
            .arg_u32(experts as u32)
            .arg_u32(top_k as u32)
            .arg_u32(1)
            .arg_f32(scale)
            .launch(s)
    }

    /// Write `rows` tokens' K and V into this visit's BF16 pages at `slots`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn kv_write(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        k: DevicePtr,
        v: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        slots: DevicePtr,
        kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        s: u64,
    ) -> Result<()> {
        let stride = (kv_heads * head_dim) as u32;
        KernelLaunch::new(gpu, self.kv_write)
            .grid([rows as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(k)
            .arg_ptr(v)
            .arg_ptr(k_pool)
            .arg_ptr(v_pool)
            .arg_ptr(slots)
            .arg_u32(kv_heads as u32)
            .arg_u32(head_dim as u32)
            .arg_u32(block_size as u32)
            .arg_u32(stride)
            .arg_u32(stride)
            .launch(s)
    }
}
