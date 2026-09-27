// SPDX-License-Identifier: AGPL-3.0-only

//! Rust launchers for the `exl3` CUDA module (kernels/gb10/qwen3.8-flash-next/
//! exl3/exl3.cu) and the EXL3 linear composed from them.
//!
//! The kernels are the vendored exllamav3 reconstruct / Hadamard / conversion /
//! hgemm entry points; each launcher here mirrors one kernel signature exactly
//! and computes the grid/block geometry documented in `exl3.cu`.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::exl3::Exl3Weight;

/// 1/sqrt(128) as an f32 — the same value as `cpu_ref::HAD_SCALE`, which a CPU
/// test pins against `1.0f32 / (128.0f32).sqrt()`. Passed as a launch argument;
/// the pre/post passes fold it into the scales, `had_plain` into the butterflies.
const HAD_SCALE: f32 = 0.088_388_346;

/// The `exl3` module's kernels, resolved once per backend.
pub struct Exl3Kernels {
    /// The integer bit rates, mul1 codebook: index = `bits - 3` (k3..k6).
    pub reconstruct: [KernelHandle; 4],
    pub had_pre: KernelHandle,
    pub had_post: KernelHandle,
    /// The scale-free pass; the linear uses only pre/post, which fold the
    /// scales in. Read by the GPU parity tests.
    pub had_plain: KernelHandle,
    pub bf16_to_f16: KernelHandle,
    pub f16_to_bf16: KernelHandle,
    pub transpose: KernelHandle,
    /// The fused rebuild step: both 128-wide Hadamard passes, the transpose and
    /// the fp16 -> bf16 conversion over one shared-memory tile.
    pub had2_transpose_bf16: KernelHandle,
    pub hgemm: KernelHandle,
}

impl Exl3Kernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: &str| gpu.kernel("exl3", name);
        Ok(Self {
            reconstruct: [
                k("exl3_reconstruct_mul1_k3")?,
                k("exl3_reconstruct_mul1_k4")?,
                k("exl3_reconstruct_mul1_k5")?,
                k("exl3_reconstruct_mul1_k6")?,
            ],
            had_pre: k("exl3_had_r128_pre")?,
            had_post: k("exl3_had_r128_post")?,
            had_plain: k("exl3_had_r128_plain")?,
            bf16_to_f16: k("exl3_bf16_to_f16")?,
            f16_to_bf16: k("exl3_f16_to_bf16")?,
            transpose: k("exl3_transpose_f16")?,
            had2_transpose_bf16: k("exl3_had2_transpose_bf16")?,
            hgemm: k("exl3_hgemm_f16")?,
        })
    }
}

/// Expand a whole EXL3 trellis tensor into its `W_inner`: fp16 row-major
/// `[in_features, out_features]`, bit-exact against `decode_inner`.
///
/// The kernel writes `out_cols` columns per row from tile column
/// `packed_n_offset` of the packed trellis; this launcher covers the whole
/// tensor (`out_cols = out_features`, offset 0), so `out_features` must be a
/// multiple of the 128-wide Hadamard block the grid is tiled in.
pub fn exl3_reconstruct(
    gpu: &dyn GpuBackend,
    k: &Exl3Kernels,
    w: &Exl3Weight,
    out_f16: DevicePtr,
    stream: u64,
) -> Result<()> {
    let sh = &w.shape;
    anyhow::ensure!(
        (3..=6).contains(&sh.bits),
        "EXL3 reconstruct: bits {} has no kernel (only the integer rates 3..=6 are instantiated)",
        sh.bits
    );
    anyhow::ensure!(
        sh.out_features.is_multiple_of(128),
        "EXL3 reconstruct: out_features {} must be a multiple of 128 (the kernel's column tile)",
        sh.out_features
    );
    KernelLaunch::new(gpu, k.reconstruct[sh.bits as usize - 3])
        .grid([
            div_ceil(sh.out_features as u32, 128),
            div_ceil(sh.in_features as u32, 16),
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(out_f16)
        .arg_ptr(w.trellis)
        .arg_i32((sh.out_features / 16) as i32)
        .arg_i32(0)
        .launch(stream)
}

/// One 128-wide Hadamard pass over an fp16 `[rows, cols]` tensor: `kernel` is
/// `had_pre` / `had_post` / `had_plain`, `scale` the fp16 vector it applies
/// (`DevicePtr::NULL` for plain). In-place is allowed (`input == output`).
pub fn exl3_had_r128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    output: DevicePtr,
    scale: DevicePtr,
    rows: u32,
    cols: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        cols.is_multiple_of(128),
        "EXL3 had_r128: cols {cols} must be a multiple of the 128-wide Hadamard block"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([rows, cols / 128, 1])
        .block([32, 1, 1])
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_ptr(scale)
        .arg_f32(HAD_SCALE)
        .launch(stream)
}

/// One of the two elementwise fp16 <-> bf16 conversions, grid-stride.
pub fn exl3_convert(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    output: DevicePtr,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 256).min(1024), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_u32(n)
        .launch(stream)
}

/// `out[c * rows + r] = in[r * cols + c]` — the fp16 transpose. Block (32, 8),
/// grid (ceil(cols / 32), ceil(rows / 32)); rows and cols need not be multiples
/// of 32 (every access is guarded in the kernel).
pub fn exl3_transpose_f16(
    gpu: &dyn GpuBackend,
    k: &Exl3Kernels,
    input: DevicePtr,
    output: DevicePtr,
    rows: u32,
    cols: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.transpose)
        .grid([cols.div_ceil(32), rows.div_ceil(32), 1])
        .block([32, 8, 1])
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(cols)
        .launch(stream)
}

/// `c[m, n] = a[m, k] * b[k, n]`, row-major fp16 in and out, fp32 accumulation.
pub fn exl3_hgemm(
    gpu: &dyn GpuBackend,
    k: &Exl3Kernels,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    kdim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k.hgemm)
        .grid([div_ceil(n, 16), div_ceil(m, 16), 1])
        .block([16, 16, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(kdim)
        .launch(stream)
}

/// `y = x · diag(suh) H W_inner H diag(svh)` for bf16 activations — the
/// exllamav3 linear, in upstream's order:
///
/// ```text
/// xh = bf16_to_f16(x)             xh [m, in], fp16
/// xh = had_r128(xh * suh)         had_pre, in place
/// y  = xh · w_inner               hgemm, w_inner [in, out] fp16 from exl3_reconstruct
/// y  = had_r128(y) * svh          had_post, in place
/// y  = f16_to_bf16(y) -> out
/// ```
///
/// `w_inner_f16` holds the `in * out` fp16 output of `exl3_reconstruct` for
/// `w`; `xh_scratch` is `m * in` fp16, `y_scratch` `m * out` fp16. `cols % 128`
/// is enforced by the Hadamard launches; `in_features` must be a multiple of
/// 128 for the pre pass to tile the activation rows.
pub fn exl3_linear_bf16(
    gpu: &dyn GpuBackend,
    k: &Exl3Kernels,
    x_bf16: DevicePtr,
    m: u32,
    w: &Exl3Weight,
    w_inner_f16: DevicePtr,
    xh_scratch: DevicePtr,
    y_scratch: DevicePtr,
    out_bf16: DevicePtr,
    stream: u64,
) -> Result<()> {
    let (rows, cols) = (m as usize, w.shape.in_features);
    if cols % 128 != 0 {
        bail!("EXL3 linear: in_features {cols} must be a multiple of the 128-wide Hadamard block");
    }
    exl3_convert(
        gpu,
        k.bf16_to_f16,
        x_bf16,
        xh_scratch,
        (rows * cols) as u32,
        stream,
    )?;
    exl3_had_r128(
        gpu,
        k.had_pre,
        xh_scratch,
        xh_scratch,
        w.suh,
        rows as u32,
        cols as u32,
        stream,
    )?;
    exl3_hgemm(
        gpu,
        k,
        xh_scratch,
        w_inner_f16,
        y_scratch,
        rows as u32,
        w.shape.out_features as u32,
        cols as u32,
        stream,
    )?;
    exl3_had_r128(
        gpu,
        k.had_post,
        y_scratch,
        y_scratch,
        w.svh,
        rows as u32,
        w.shape.out_features as u32,
        stream,
    )?;
    exl3_convert(
        gpu,
        k.f16_to_bf16,
        y_scratch,
        out_bf16,
        (rows * w.shape.out_features) as u32,
        stream,
    )
}

/// `w` fully dequantized to BF16, row-major `[out_features, in_features]` — the
/// HF layout Atlas dense weights use.
///
/// `W = diag(suh) · H · W_inner · H · diag(svh)` is `[in, out]`, so this walks
/// the transpose `Wᵀ = diag(svh) · H · W_innerᵀ · H · diag(suh)`:
/// reconstruct → `had_post` over the rows (scale `svh`) → transpose →
/// `had_post` over the rows (scale `suh`) → `f16_to_bf16`.
///
/// `out_bf16` is caller-owned and must hold `in * out * 2` bytes. The two fp16
/// scratch buffers are allocated here and freed before returning, including on
/// the error path. Both feature counts must be multiples of the 128-wide
/// Hadamard block.
pub fn exl3_dense_bf16_nk(
    gpu: &dyn GpuBackend,
    k: &Exl3Kernels,
    w: &Exl3Weight,
    out_bf16: DevicePtr,
    stream: u64,
) -> Result<()> {
    let scratch_bytes = check_dense_shape(&w.shape)?;
    let a = gpu.alloc(scratch_bytes)?;
    let r = exl3_dense_bf16_nk_with_scratch(gpu, k, w, out_bf16, a, stream);
    // Scratch must outlive the queued kernels: sync before freeing rather than
    // relying on cuMemFree synchronizing implicitly.
    let r = r.and_then(|()| gpu.synchronize(stream));
    gpu.free(a)?;
    r
}

/// [`exl3_dense_bf16_nk`] on caller-owned scratch: `a` is `in * out * 2` bytes
/// of fp16. NO alloc, NO free and NO synchronize — the rebuild runs entirely as
/// stream-ordered work, so the caller's stream owns the scratch's lifetime (see
/// `exl3_bf16_pool`). Reconstruct, then [`exl3_had2_transpose_bf16`], which
/// fuses the two Hadamard passes, the transpose and the bf16 conversion and is
/// bitwise identical to running them as separate kernels.
pub fn exl3_dense_bf16_nk_with_scratch(
    gpu: &dyn GpuBackend,
    k: &Exl3Kernels,
    w: &Exl3Weight,
    out_bf16: DevicePtr,
    a: DevicePtr,
    stream: u64,
) -> Result<()> {
    check_dense_shape(&w.shape)?;
    exl3_reconstruct(gpu, k, w, a, stream)?;
    exl3_had2_transpose_bf16(gpu, k, a, w, out_bf16, stream)
}

/// `W^T [out, in]` bf16 from the reconstructed `W_inner [in, out]` fp16 in one
/// kernel over 128x128 tiles: `had_r128_post` with `svh` over the rows, the
/// transpose, `had_r128_post` with `suh`, then fp16 -> bf16 — the same
/// arithmetic and fp16 rounding points as those four separate launches.
pub fn exl3_had2_transpose_bf16(
    gpu: &dyn GpuBackend,
    k: &Exl3Kernels,
    w_inner: DevicePtr,
    w: &Exl3Weight,
    out_bf16: DevicePtr,
    stream: u64,
) -> Result<()> {
    check_dense_shape(&w.shape)?;
    let (i, o) = (w.shape.in_features as u32, w.shape.out_features as u32);
    KernelLaunch::new(gpu, k.had2_transpose_bf16)
        .grid([o / 128, i / 128, 1])
        .block([128, 1, 1])
        .shared_mem(2 * 128 * 136 * 2)
        .arg_ptr(w_inner)
        .arg_ptr(out_bf16)
        .arg_ptr(w.suh)
        .arg_ptr(w.svh)
        .arg_u32(i)
        .arg_u32(o)
        .arg_f32(HAD_SCALE)
        .launch(stream)
}

/// Shape gate for the dense rebuild; returns the per-buffer scratch bytes.
fn check_dense_shape(sh: &crate::weight_map::exl3::Exl3Shape) -> Result<usize> {
    let (i, o) = (sh.in_features, sh.out_features);
    if i % 128 != 0 || o % 128 != 0 {
        bail!(
            "EXL3 dense: in_features {i} and out_features {o} must both be multiples of \
             the 128-wide Hadamard block"
        );
    }
    Ok(i * o * 2)
}

#[cfg(all(test, feature = "cuda"))]
#[path = "exl3_gpu_tests.rs"]
mod gpu_tests;
