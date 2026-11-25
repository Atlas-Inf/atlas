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

mod route;

#[derive(Clone, Copy, Debug)]
pub(crate) struct GlueKernels {
    pub rmsnorm: KernelHandle,
    pub gate_mul: KernelHandle,
    pub residual: KernelHandle,
    pub layer_out: KernelHandle,
    pub f32_to_bf16: KernelHandle,
    /// `dense_gemv_bf16`: y[N] = W[N, K] x[K], BF16
    pub dense_gemv: KernelHandle,
    /// `dense_gemv_bf16_batchm`: up to 8 rows from one pass over W, each row
    /// bit-identical to `dense_gemv_bf16`
    pub dense_gemv_batchm: KernelHandle,
    /// `dense_gemv_bf16_batchm_f32out`: up to 8 rows of router logits
    pub dense_gemv_batchm_f32: KernelHandle,
    /// `dense_gemv_bf16_fp32out`: one row of router logits
    pub dense_gemv_f32: KernelHandle,
    /// `inkling2_proj_gemv`: a visit's BF16 input projections in one launch
    pub proj_gemv: KernelHandle,
    /// `inkling2_proj_gemv_mixed`: the same for one decode row with NVFP4
    /// projections among them (0 where the target does not ship it)
    pub proj_gemv_mixed: KernelHandle,
    /// `moe_topk_sigmoid_batched_f32`
    pub topk: KernelHandle,
    /// `reshape_and_cache_flash` (BF16 KV)
    pub kv_write: KernelHandle,
    /// Spark NVFP4 projections: one row, many rows
    pub w4a16_gemv: KernelHandle,
    /// AMD (strix-hip): the single-warp `w4a16_gemv_sw` (bit-identical to `w4a16_gemv`,
    /// 8 outputs a block); NULL elsewhere or when the kernel is absent
    pub w4a16_gemv_sw: KernelHandle,
    /// AMD (strix-hip): WMMA batched projection `w4a16_wmma` for rows 2..=16;
    /// NULL elsewhere or when the kernel is absent.
    pub w4a16_wmma: KernelHandle,
    /// Its split-K reduce pass (`w4a16_wmma_reduce`) and the shared
    /// `[8 splits, 16 rows, 4096]` f32 partials scratch; NULL/unset with the
    /// main kernel.
    pub w4a16_wmma_reduce: KernelHandle,
    pub w4a16_wmma_partials: DevicePtr,
    pub w4a16_gemv_b8: KernelHandle,
    pub w4a16_gemv_b16: KernelHandle,
}

impl GlueKernels {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let m = |f: &str| gpu.kernel("inkling2_misc", f);
        // The WMMA batched projection needs its 8 MB split-K scratch only when
        // the kernel is present; an allocation failure is a hard error, not a
        // silent NULL.
        let w4a16_wmma = if cfg!(atlas_scale) {
            crate::layers::try_kernel(gpu, "inkling2_w4a16_wmma", "w4a16_wmma")
        } else {
            KernelHandle(0)
        };
        let w4a16_wmma_partials = if w4a16_wmma.0 != 0 {
            gpu.alloc(8 * 16 * 4096 * 4)
                .map_err(|e| anyhow::anyhow!("w4a16_wmma split-K partials: {e}"))?
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            rmsnorm: m("inkling2_rmsnorm")?,
            gate_mul: m("inkling2_gate_mul")?,
            residual: m("inkling2_residual")?,
            layer_out: m("inkling2_layer_out")?,
            f32_to_bf16: m("inkling2_f32_to_bf16")?,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemv_batchm: gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
            dense_gemv_batchm_f32: gpu
                .kernel("inkling2_router_gemv", "dense_gemv_bf16_batchm_f32out")?,
            dense_gemv_f32: gpu.kernel("gemv", "dense_gemv_bf16_fp32out")?,
            proj_gemv: gpu.kernel("inkling2_proj_gemv", "inkling2_proj_gemv")?,
            proj_gemv_mixed: crate::layers::try_kernel(
                gpu,
                "inkling2_proj_gemv",
                "inkling2_proj_gemv_mixed",
            ),
            topk: gpu.kernel("moe_topk_sig", "moe_topk_sigmoid_batched_f32")?,
            kv_write: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw: if cfg!(atlas_scale) {
                crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw")
            } else {
                KernelHandle(0)
            },
            w4a16_wmma,
            w4a16_wmma_reduce: if cfg!(atlas_scale) {
                crate::layers::try_kernel(gpu, "inkling2_w4a16_wmma", "w4a16_wmma_reduce")
            } else {
                KernelHandle(0)
            },
            w4a16_wmma_partials,
            w4a16_gemv_b8: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch8")?,
            w4a16_gemv_b16: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch16")?,
        })
    }
}

/// `(first row, rows)` passes of at most `width` rows over `rows`.
pub(crate) fn passes(rows: usize, width: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows)
        .step_by(width)
        .map(move |first| (first, (rows - first).min(width)))
}

/// Rows `dense_gemv_bf16_batchm` takes at once (its MAX_M): a decode batch or a
/// verify of up to 8 rows reads each BF16 weight once, every row computed as the
/// one-row GEMV computes it. cuBLASLt took 2-6x longer on the small projections
/// (sinks, relative bias, router) at two rows; RigMark short code, Strict, one
/// Spark: C2 / C4 / C8 49.5 / 79.2 / 120.2 -> 52.6 / 81.7 / 122.1.
pub(crate) const BATCHM_MAX_ROWS: usize = 8;

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

    /// A `dense_gemv_bf16_batchm` launch (either output type) over `rows`
    /// contiguous rows.
    #[allow(clippy::too_many_arguments)]
    fn batchm_with(
        &self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        rows: usize,
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        n: usize,
        k: usize,
        s: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, kernel)
            .grid([(n as u32).div_ceil(4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(y)
            .arg_u32(rows as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_u32(n as u32)
            .launch(s)
    }

    /// `y[rows, n] = x[rows, k] W[n, k]^T`, BF16 out: a BF16 weight through
    /// the dense GEMV / BF16 GEMM; a Spark NVFP4 one through `w4a16_gemv` /
    /// `w4a16_gemv_batch8/16` for decode and verify rows, and its checkpoint BF16
    /// weight for prefill rows (from `MOE_GROUPED_MIN_ROWS`, where the experts
    /// take the checkpoint's too). `exact` (a verify run): every row through
    /// the decode GEMVs at any row count, in passes of their widest batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn proj(
        &self,
        gpu: &dyn GpuBackend,
        (rows, exact): (usize, bool),
        x: DevicePtr,
        w: Linear,
        y: DevicePtr,
        n: usize,
        k: usize,
        s: u64,
    ) -> Result<()> {
        let pass = match w {
            Linear::Bf16(_) => BATCHM_MAX_ROWS,
            Linear::Nvfp4 { .. } => super::moe_ops::MOE_GROUPED_MIN_ROWS - 1,
        };
        if exact && rows > pass {
            for (first, part) in passes(rows, pass) {
                let (xo, yo) = (x.offset(first * k * 2), y.offset(first * n * 2));
                self.proj(gpu, (part, true), xo, w, yo, n, k, s)?;
            }
            return Ok(());
        }
        match w {
            Linear::Bf16(w) if rows == 1 => self.gemv_with(gpu, self.dense_gemv, x, w, y, n, k, s),
            Linear::Bf16(w) if rows <= BATCHM_MAX_ROWS => {
                self.batchm_with(gpu, self.dense_gemv_batchm, rows, x, w, y, n, k, s)
            }
            Linear::Bf16(w) => ops::bf16_gemm(x, w.0, y, rows as u32, n as u32, k as u32, s),
            // AMD (strix-hip): the single-warp twin, bit-identical, for the occupancy (f370:
            // w4a16_gemv was 27% of Spark decode at ~133 GB/s over 288 small calls a token)
            Linear::Nvfp4 { q, .. } if rows == 1 && self.w4a16_gemv_sw.0 != 0 => {
                ops::w4a16_gemv_sw(gpu, self.w4a16_gemv_sw, x, &q, y, n as u32, k as u32, s)
            }
            Linear::Nvfp4 { q, .. } if rows == 1 => {
                ops::w4a16_gemv(gpu, self.w4a16_gemv, x, &q, y, n as u32, k as u32, s)
            }
            // AMD (strix-hip): one block per 128 outputs x K slice on the
            // pretiled weight replay — K64 software-pipelined WMMA with an
            // f32 partials reduce — replaces batch8/16's issue-bound rows.
            // Rows 2-3 and the narrow rel-bias projection stay on batch8:
            // WMMA's fixed per-launch cost only pays off from M ~4 / N ~1K.
            Linear::Nvfp4 {
                q,
                wmma_packed,
                wmma_scales,
                ..
            } if (4..=16).contains(&rows)
                && n >= 1024
                && self.w4a16_wmma.0 != 0
                && wmma_packed != DevicePtr::NULL
                && self.w4a16_wmma_partials != DevicePtr::NULL
                && n <= 4096 =>
            {
                let mn = (rows * n) as u32;
                // Fill ~40 CUs: small-N projections split K deeper. Each
                // slice needs >= 2 K64 tiles to keep the pipeline fed.
                let n_split = (64 / (n / 128)).clamp(1, (k / 64) / 2).min(8) as u32;
                KernelLaunch::new(gpu, self.w4a16_wmma)
                    .grid([(n as u32) / 128, n_split, 1])
                    .block([256, 1, 1])
                    .arg_ptr(x)
                    .arg_ptr(wmma_packed)
                    .arg_ptr(wmma_scales)
                    .arg_ptr(self.w4a16_wmma_partials)
                    .arg_u32(rows as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(s)?;
                KernelLaunch::new(gpu, self.w4a16_wmma_reduce)
                    .grid([mn.div_ceil(512), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(self.w4a16_wmma_partials)
                    .arg_f32(q.weight_scale_2)
                    .arg_ptr(y)
                    .arg_u32(mn)
                    .arg_u32(n_split)
                    .launch(s)
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
        (rows, exact): (usize, bool),
        x: DevicePtr,
        w: DevicePtr,
        y: DevicePtr,
        n: usize,
        k: usize,
        s: u64,
    ) -> Result<()> {
        if exact && rows > BATCHM_MAX_ROWS {
            for (first, part) in passes(rows, BATCHM_MAX_ROWS) {
                let (xo, yo) = (x.offset(first * k * 2), y.offset(first * n * 4));
                self.proj_f32(gpu, (part, true), xo, w, yo, n, k, s)?;
            }
            return Ok(());
        }
        if rows == 1 {
            self.gemv_with(gpu, self.dense_gemv_f32, x, w, y, n, k, s)
        } else if rows <= BATCHM_MAX_ROWS {
            self.batchm_with(gpu, self.dense_gemv_batchm_f32, rows, x, w, y, n, k, s)
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
        self.rmsnorms(gpu, rows, x_f32, x_bf16, 0, &[w], y, h, eps, s)
    }

    /// One norm per weight in `ws` (at most 3), in one launch: input `g` at
    /// `x + g * in_plane` elements, output at `y + g * rows * h`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rmsnorms(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        x_f32: DevicePtr,
        x_bf16: DevicePtr,
        in_plane: usize,
        ws: &[DevicePtr],
        y: DevicePtr,
        h: usize,
        eps: f32,
        s: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            (1..=3).contains(&ws.len()),
            "inkling2 rmsnorm: 1 to 3 inputs"
        );
        let w = |g: usize| ws[g.min(ws.len() - 1)];
        KernelLaunch::new(gpu, self.rmsnorm)
            .grid([rows as u32, ws.len() as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(x_f32)
            .arg_ptr(x_bf16)
            .arg_ptr(w(0))
            .arg_ptr(w(1))
            .arg_ptr(w(2))
            .arg_ptr(y)
            .arg_u32(h as u32)
            .arg_u32(h as u32)
            .arg_u32(h as u32)
            .arg_u64(in_plane as u64)
            .arg_u64((rows * h) as u64)
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
}
