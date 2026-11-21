// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2 routed-expert launchers: the decode GEMVs over the stacked
//! checkpoint tensors and the grouped-GEMM prefill path.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::ops::{Inkling2Kernels, u32_of};

/// One layer's routed experts as stored in the checkpoint.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StackedExperts {
    pub mxfp8: bool,
    pub w13: DevicePtr,
    pub w13_scale: DevicePtr,
    /// NVFP4: per-expert `amax / denominator` (F32 `[E]`); NULL for MXFP8
    pub w13_global: DevicePtr,
    pub w2: DevicePtr,
    pub w2_scale: DevicePtr,
    pub w2_global: DevicePtr,
}

/// Largest hidden / intermediate size the MoE kernels stage in shared memory.
const MAX_MOE_DIM: usize = 4096;

/// Routed experts for `rows` tokens: gate_up + softcapped SwiGLU, down, weighted combine.
pub(crate) struct MoeDecode {
    pub x: DevicePtr,
    pub experts: StackedExperts,
    pub expert_ids: DevicePtr,
    pub weights: DevicePtr,
    pub act: DevicePtr,
    pub y: DevicePtr,
    pub out: DevicePtr,
    pub rows: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub inter: usize,
    pub out_stride: usize,
    pub gate_softcap: f32,
    pub up_softcap: f32,
}

pub(crate) fn moe_decode(
    gpu: &dyn GpuBackend,
    kernels: &Inkling2Kernels,
    a: &MoeDecode,
    stream: u64,
) -> Result<()> {
    ensure!(
        a.hidden.is_multiple_of(32)
            && a.inter.is_multiple_of(32)
            && a.hidden.max(a.inter) <= MAX_MOE_DIM,
        "inkling2 MoE dims must be multiples of 32 up to {MAX_MOE_DIM}"
    );
    let e = a.experts;
    // the kernels read code rows 16 bytes and NVFP4 scales 2 bytes at a time
    ensure!(
        e.w13.0.is_multiple_of(16)
            && e.w2.0.is_multiple_of(16)
            && e.w13_scale.0.is_multiple_of(2)
            && e.w2_scale.0.is_multiple_of(2),
        "inkling2: expert tensors not aligned for 16-byte loads"
    );
    let (gate_up, down) = if e.mxfp8 {
        (kernels.moe_gate_up_mxfp8, kernels.moe_down_mxfp8)
    } else {
        (kernels.moe_gate_up_nvfp4, kernels.moe_down_nvfp4)
    };
    let (rows, top_k) = (u32_of(a.rows, "rows")?, u32_of(a.top_k, "top_k")?);
    KernelLaunch::new(gpu, gate_up)
        .grid([u32_of(a.inter.div_ceil(8), "inter")?, top_k, rows])
        .block([256, 1, 1])
        .arg_ptr(a.x)
        .arg_ptr(e.w13)
        .arg_ptr(e.w13_scale)
        .arg_ptr(e.w13_global)
        .arg_ptr(a.expert_ids)
        .arg_ptr(a.act)
        .arg_u32(u32_of(a.inter, "inter")?)
        .arg_u32(u32_of(a.hidden, "hidden")?)
        .arg_u32(top_k)
        .arg_f32(a.gate_softcap)
        .arg_f32(a.up_softcap)
        .launch(stream)?;
    KernelLaunch::new(gpu, down)
        .grid([u32_of(a.hidden.div_ceil(32), "hidden")?, top_k, rows])
        .block([256, 1, 1])
        .arg_ptr(a.act)
        .arg_ptr(e.w2)
        .arg_ptr(e.w2_scale)
        .arg_ptr(e.w2_global)
        .arg_ptr(a.expert_ids)
        .arg_ptr(a.y)
        .arg_u32(u32_of(a.hidden, "hidden")?)
        .arg_u32(u32_of(a.inter, "inter")?)
        .arg_u32(top_k)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.moe_combine)
        .grid([u32_of(a.hidden.div_ceil(256), "hidden")?, rows, 1])
        .block([256, 1, 1])
        .arg_ptr(a.y)
        .arg_ptr(a.weights)
        .arg_ptr(a.out)
        .arg_u32(u32_of(a.hidden, "hidden")?)
        .arg_u32(top_k)
        .arg_u32(u32_of(a.out_stride, "out stride")?)
        .launch(stream)
}

/// Routed experts for many rows (prefill): sort the rows' expert slots by
/// expert, one grouped GEMM per projection over all experts (each expert's
/// weights read once per tile of its rows, not once per row), the softcapped
/// SwiGLU between, then the top-k weighted sum back in row order.
pub(crate) struct MoePrefill {
    pub x: DevicePtr,
    pub experts: StackedExperts,
    pub expert_ids: DevicePtr,
    pub weights: DevicePtr,
    /// `[rows * top_k, 2 * inter]` BF16
    pub gu: DevicePtr,
    /// `[rows * top_k, inter]` BF16
    pub act: DevicePtr,
    /// `[rows * top_k, hidden]` BF16
    pub y: DevicePtr,
    /// `[rows, hidden]` BF16
    pub out: DevicePtr,
    /// [`moe_sort_bytes`] of scratch
    pub sort: DevicePtr,
    pub rows: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub inter: usize,
    pub num_experts: usize,
    pub gate_softcap: f32,
    pub up_softcap: f32,
}

/// Output columns per CTA of `inkling2_moe_gemm_nvfp4` (its GN).
const GEMM_N_TILE: u32 = 128;

/// Rows from which the routed experts run as grouped GEMMs (fewer rows read
/// each expert once per row through the decode GEMVs).
pub(crate) const MOE_GROUPED_MIN_ROWS: usize = 16;

/// Sort scratch for `expanded` expert slots: sorted rows, their experts, the
/// expert offsets and each slot's sorted position.
pub(crate) fn moe_sort_bytes(expanded: usize, experts: usize) -> usize {
    (3 * expanded + experts + 1) * 4
}

pub(crate) fn moe_prefill(
    gpu: &dyn GpuBackend,
    k: &Inkling2Kernels,
    a: &MoePrefill,
    stream: u64,
) -> Result<()> {
    let e = a.experts;
    let te = a.rows * a.top_k;
    let (sorted_rows, sorted_experts) = (a.sort, a.sort.offset(te * 4));
    let offsets = a.sort.offset(te * 8);
    let to_perm = offsets.offset((a.num_experts + 1) * 4);
    let (ne, h, i) = (
        u32_of(a.num_experts, "experts")?,
        u32_of(a.hidden, "hidden")?,
        u32_of(a.inter, "inter")?,
    );
    super::super::ops::moe_sort_by_expert(
        gpu,
        k.moe_sort,
        a.expert_ids,
        sorted_rows,
        sorted_experts,
        offsets,
        to_perm,
        u32_of(te, "slots")?,
        ne,
        u32_of(a.top_k, "top_k")?,
        stream,
    )?;
    // an expert takes each row at most once
    let m_tiles = u32_of(a.rows.div_ceil(64).max(1), "m tiles")?;
    let gemm = |x: DevicePtr,
                w: DevicePtr,
                sc: DevicePtr,
                g: DevicePtr,
                y: DevicePtr,
                gather: DevicePtr,
                n: u32,
                kk: u32| {
        // the checkpoint's stacked tensors, on this target's tensor-core kernel
        if e.mxfp8 {
            KernelLaunch::new(gpu, k.moe_gemm_mxfp8)
                .grid([n.div_ceil(GEMM_N_TILE), m_tiles, ne])
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_ptr(w)
                .arg_ptr(sc)
                .arg_ptr(y)
                .arg_ptr(offsets)
                .arg_ptr(gather)
                .arg_u32(n)
                .arg_u32(kk)
                .launch(stream)
        } else {
            KernelLaunch::new(gpu, k.moe_gemm_nvfp4)
                .grid([n.div_ceil(GEMM_N_TILE), m_tiles, ne])
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_ptr(w)
                .arg_ptr(sc)
                .arg_ptr(g)
                .arg_ptr(y)
                .arg_ptr(offsets)
                .arg_ptr(gather)
                .arg_u32(n)
                .arg_u32(kk)
                .launch(stream)
        }
    };
    // gate and up rows stay interleaved in the output, as the checkpoint stores them
    gemm(
        a.x,
        e.w13,
        e.w13_scale,
        e.w13_global,
        a.gu,
        sorted_rows,
        2 * i,
        h,
    )?;
    KernelLaunch::new(gpu, k.softcap_swiglu)
        .grid([i.div_ceil(256), u32_of(te, "slots")?, 1])
        .block([256, 1, 1])
        .arg_ptr(a.gu)
        .arg_ptr(a.act)
        .arg_u32(i)
        .arg_f32(a.gate_softcap)
        .arg_f32(a.up_softcap)
        .launch(stream)?;
    gemm(
        a.act,
        e.w2,
        e.w2_scale,
        e.w2_global,
        a.y,
        DevicePtr::NULL,
        h,
        i,
    )?;
    super::super::ops::moe_unpermute_reduce_indexed(
        gpu,
        k.moe_unpermute,
        a.y,
        a.out,
        to_perm,
        a.weights,
        h,
        u32_of(a.rows, "rows")?,
        u32_of(a.top_k, "top_k")?,
        stream,
    )
}

/// Routed experts for a decode batch of a few rows: the rows' expert slots
/// sorted by expert, each expert's weights read once for every row that chose
/// it (the per-row GEMVs read them once per row), then the weighted sum back in
/// row order. Same buffers as [`MoePrefill`] (`gu` unused).
pub(crate) fn moe_batched(
    gpu: &dyn GpuBackend,
    k: &Inkling2Kernels,
    a: &MoePrefill,
    stream: u64,
) -> Result<()> {
    ensure!(
        a.hidden <= GROUPED_MAX_K && a.inter <= GROUPED_MAX_K,
        "inkling2 grouped expert GEMVs stage rows of at most {GROUPED_MAX_K}"
    );
    let e = a.experts;
    let te = a.rows * a.top_k;
    let (sorted_rows, sorted_experts) = (a.sort, a.sort.offset(te * 4));
    let offsets = a.sort.offset(te * 8);
    let to_perm = offsets.offset((a.num_experts + 1) * 4);
    let (ne, h, i) = (
        u32_of(a.num_experts, "experts")?,
        u32_of(a.hidden, "hidden")?,
        u32_of(a.inter, "inter")?,
    );
    super::super::ops::moe_sort_by_expert(
        gpu,
        k.moe_sort,
        a.expert_ids,
        sorted_rows,
        sorted_experts,
        offsets,
        to_perm,
        u32_of(te, "slots")?,
        ne,
        u32_of(a.top_k, "top_k")?,
        stream,
    )?;
    let (gate_up, down) = if e.mxfp8 {
        (k.moe_grouped_gate_up_mxfp8, k.moe_grouped_down_mxfp8)
    } else {
        (k.moe_grouped_gate_up_nvfp4, k.moe_grouped_down_nvfp4)
    };
    KernelLaunch::new(gpu, gate_up)
        .grid([i.div_ceil(8 * GROUPED_TILES), ne, 1])
        .block([256, 1, 1])
        .arg_ptr(a.x)
        .arg_ptr(e.w13)
        .arg_ptr(e.w13_scale)
        .arg_ptr(e.w13_global)
        .arg_ptr(sorted_rows)
        .arg_ptr(offsets)
        .arg_ptr(a.act)
        .arg_u32(i)
        .arg_u32(h)
        .arg_f32(a.gate_softcap)
        .arg_f32(a.up_softcap)
        .launch(stream)?;
    KernelLaunch::new(gpu, down)
        .grid([h.div_ceil(32 * GROUPED_TILES), ne, 1])
        .block([256, 1, 1])
        .arg_ptr(a.act)
        .arg_ptr(e.w2)
        .arg_ptr(e.w2_scale)
        .arg_ptr(e.w2_global)
        .arg_ptr(offsets)
        .arg_ptr(a.y)
        .arg_u32(h)
        .arg_u32(i)
        .launch(stream)?;
    super::super::ops::moe_unpermute_reduce_indexed(
        gpu,
        k.moe_unpermute,
        a.y,
        a.out,
        to_perm,
        a.weights,
        h,
        u32_of(a.rows, "rows")?,
        u32_of(a.top_k, "top_k")?,
        stream,
    )
}

/// `inkling2_moe_decode.cu` INK_GR_MAX_K: the longest row the grouped GEMVs stage.
const GROUPED_MAX_K: usize = 2304;
/// `inkling2_moe_decode.cu` INK_GR_TILES: output tiles a grouped block covers.
const GROUPED_TILES: u32 = 4;
