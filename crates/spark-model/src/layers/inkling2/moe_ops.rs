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

impl StackedExperts {
    /// Code rows are read 16 bytes and scale rows `scale_align` bytes at a time.
    pub(super) fn ensure_aligned(&self, scale_align: u64) -> Result<()> {
        ensure!(
            self.w13.0.is_multiple_of(16)
                && self.w2.0.is_multiple_of(16)
                && self.w13_scale.0.is_multiple_of(scale_align)
                && self.w2_scale.0.is_multiple_of(scale_align),
            "inkling2: expert tensors not aligned for their loads ({scale_align}-byte scales)"
        );
        Ok(())
    }
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
    /// false: leave `y` per expert slot for `inkling2_mlp_tail`'s own combine
    pub combine: bool,
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
    // the GEMVs read code rows 16 bytes and scale rows 4 bytes at a time
    e.ensure_aligned(4)?;
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
    if !a.combine {
        return Ok(());
    }
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
/// SwiGLU between, then the top-k weighted sum back in row order. Prefill takes
/// the checkpoint's experts at its declared numerics ([`moe_prefill_lowp`]).
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

/// Output columns per CTA of `inkling2_moe_gemm_w4a4` / `_w8a8` (and of the strix-hip
/// BF16-row `inkling2_moe_gemm_nvfp4` / `_mxfp8`).
pub(super) const GEMM_N_TILE: u32 = 128;

/// Rows from which the routed experts run as grouped GEMMs (fewer rows read
/// each expert once per row through the decode GEMVs).
pub(crate) const MOE_GROUPED_MIN_ROWS: usize = 16;

/// Sort scratch for `expanded` expert slots: sorted rows, their experts, the
/// expert offsets, each slot's sorted position, then the low-precision GEMMs'
/// row-tile list (`inkling2_moe_tiles`: a count and at most one partial tile per
/// expert beyond the full ones).
pub(crate) fn moe_sort_bytes(expanded: usize, experts: usize) -> usize {
    (3 * expanded + experts + 1 + 1 + max_row_tiles(expanded, experts)) * 4
}

fn max_row_tiles(expanded: usize, experts: usize) -> usize {
    expanded.div_ceil(LOWP_M_TILE) + experts
}

pub(crate) fn moe_prefill(
    gpu: &dyn GpuBackend,
    k: &Inkling2Kernels,
    a: &MoePrefill,
    stream: u64,
) -> Result<()> {
    let te = a.rows * a.top_k;
    let (sorted_rows, sorted_experts) = (a.sort, a.sort.offset(te * 4));
    let offsets = a.sort.offset(te * 8);
    let to_perm = offsets.offset((a.num_experts + 1) * 4);
    let ne = u32_of(a.num_experts, "experts")?;
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
    if cfg!(atlas_scale) {
        return super::moe_prefill_hip::moe_prefill_bf16(
            gpu,
            k,
            a,
            sorted_rows,
            offsets,
            to_perm,
            stream,
        );
    }
    moe_prefill_lowp(gpu, k, a, sorted_rows, offsets, to_perm, stream)
}

/// `inkling2_situ_quant_fp4`: 32 lanes x INK_SITU_BLOCKS blocks of 16.
const SITU_MAX_INTER: usize = 32 * 3 * 16;

/// Tile rows of `inkling2_moe_gemm_w4a4` / `_w8a8` (nvfp4_w4a4_tile.cuh `W4_BM`).
const LOWP_M_TILE: usize = 128;

/// The grouped experts of [`moe_prefill`] at the checkpoint's declared numerics: each
/// row quantized once (NVFP4 for NVFP4 experts, MXFP8 for MXFP8 ones; staged in
/// `y`, which the down projection fills only later), the gate_up GEMM on the
/// block-scaled tensor cores (W4A4 or W8A8), the softcapped SwiGLU quantized per
/// expert row the same way (staged in `act`), the down GEMM likewise, then the
/// weighted combine.
#[allow(clippy::too_many_arguments)]
fn moe_prefill_lowp(
    gpu: &dyn GpuBackend,
    k: &Inkling2Kernels,
    a: &MoePrefill,
    sorted_rows: DevicePtr,
    offsets: DevicePtr,
    to_perm: DevicePtr,
    stream: u64,
) -> Result<()> {
    let e = a.experts;
    let mx = e.mxfp8;
    let te = a.rows * a.top_k;
    ensure!(
        a.hidden.is_multiple_of(128) && a.inter.is_multiple_of(128) && a.inter <= SITU_MAX_INTER,
        "inkling2 low-precision experts need dims that are multiples of 128 (inter up to {SITU_MAX_INTER})"
    );
    // the tile copies each row's scales 4 (W8A8) or 8 (W4A4) bytes at a time
    e.ensure_aligned(if mx { 4 } else { 8 })?;
    let (ne, h, i) = (
        u32_of(a.num_experts, "experts")?,
        u32_of(a.hidden, "hidden")?,
        u32_of(a.inter, "inter")?,
    );
    // codes, block scales (and NVFP4 outer scales) of the rows, then of the SwiGLU
    // rows: NVFP4 packs two values a byte, one scale per 16; MXFP8 one, per 32
    let (per_byte, group) = if mx { (1, 32) } else { (2, 16) };
    let plane = |base: DevicePtr, n: usize, width: usize| {
        let s = base.offset(n * width / per_byte);
        (base, s, s.offset(n * width / group))
    };
    let (xq, xs, xo) = plane(a.y, a.rows, a.hidden);
    let (aq, as_, ao) = plane(a.act, te, a.inter);
    let rows = u32_of(a.rows, "rows")?;
    let slots = u32_of(te, "slots")?;
    if mx {
        KernelLaunch::new(gpu, k.quant_mxfp8)
            .grid([u32_of((a.rows * a.hidden).div_ceil(256), "x")?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.x)
            .arg_ptr(xq)
            .arg_ptr(xs)
            .arg_u64((a.rows * a.hidden) as u64)
            .launch(stream)?;
    } else {
        KernelLaunch::new(gpu, k.quant_fp4)
            .grid([rows.div_ceil(8), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.x)
            .arg_ptr(xq)
            .arg_ptr(xs)
            .arg_ptr(xo)
            .arg_u32(rows)
            .arg_u32(h)
            .launch(stream)?;
    }
    // the experts' row tiles, one CTA row each
    ensure!(a.num_experts <= 1024, "inkling2: over 1024 experts");
    let tiles = to_perm.offset(te * 4);
    KernelLaunch::new(gpu, k.moe_tiles)
        .grid([1, 1, 1])
        .block([u32_of(a.num_experts.next_multiple_of(32), "experts")?, 1, 1])
        .arg_ptr(offsets)
        .arg_ptr(tiles)
        .arg_u32(ne)
        .launch(stream)?;
    let m_tiles = u32_of(max_row_tiles(te, a.num_experts), "row tiles")?;
    let gemm = |q: DevicePtr,
                qs: DevicePtr,
                outer: DevicePtr,
                w: DevicePtr,
                sc: DevicePtr,
                g: DevicePtr,
                y: DevicePtr,
                gather: DevicePtr,
                n: u32,
                kk: u32| {
        let l = KernelLaunch::new(gpu, if mx { k.moe_gemm_w8a8 } else { k.moe_gemm_w4a4 })
            .grid([n.div_ceil(GEMM_N_TILE), m_tiles, 1])
            .block([256, 1, 1])
            .arg_ptr(q)
            .arg_ptr(qs);
        let l = if mx { l } else { l.arg_ptr(outer) };
        let l = l.arg_ptr(w).arg_ptr(sc);
        let l = if mx { l } else { l.arg_ptr(g) };
        l.arg_ptr(y)
            .arg_ptr(offsets)
            .arg_ptr(tiles)
            .arg_ptr(gather)
            .arg_u32(n)
            .arg_u32(kk)
            .launch(stream)
    };
    gemm(
        xq,
        xs,
        xo,
        e.w13,
        e.w13_scale,
        e.w13_global,
        a.gu,
        sorted_rows,
        2 * i,
        h,
    )?;
    if mx {
        KernelLaunch::new(gpu, k.situ_quant_mxfp8)
            .grid([slots, i.div_ceil(256), 1])
            .block([256, 1, 1])
            .arg_ptr(a.gu)
            .arg_ptr(aq)
            .arg_ptr(as_)
            .arg_u32(i)
            .arg_f32(a.gate_softcap)
            .arg_f32(a.up_softcap)
            .launch(stream)?;
    } else {
        KernelLaunch::new(gpu, k.situ_quant_fp4)
            .grid([slots.div_ceil(8), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.gu)
            .arg_ptr(aq)
            .arg_ptr(as_)
            .arg_ptr(ao)
            .arg_u32(slots)
            .arg_u32(i)
            .arg_f32(a.gate_softcap)
            .arg_f32(a.up_softcap)
            .launch(stream)?;
    }
    gemm(
        aq,
        as_,
        ao,
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
        rows,
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
    e.ensure_aligned(4)?;
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
