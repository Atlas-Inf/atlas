// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the Inkling 2 kernels in `kernels/gb10/inkling-2-mini/nvfp4/`.
//! Each takes a plain argument struct (device pointers, dimensions) so call
//! sites read as data, and checks the limits the kernels compile in.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::weights::MessageGate;

pub(crate) use super::attn_ops::*;
pub(crate) use super::moe_ops::*;

/// Compile-time limits of the kernels (keep in step with the `.cu` files).
pub(crate) const MAX_REL_EXTENT: usize = 1024;
pub(crate) const MAX_D_REL: usize = 32;
pub(crate) const MAX_CONV_K: usize = 4;
pub(crate) const MAX_BLOCK_STATES: usize = 9;
pub(crate) const MAX_RES_QUERIES: usize = 4;
pub(crate) const MAX_HEAD_DIM: usize = 128;
pub(crate) const MAX_GATE_RANK: usize = 64;

/// Every Inkling kernel handle, resolved once at load.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Inkling2Kernels {
    pub decode_attn: KernelHandle,
    pub decode_attn_combine: KernelHandle,
    pub prefill_attn: KernelHandle,
    pub rel_bias_table: KernelHandle,
    pub flash_prefill: KernelHandle,
    pub sconv: KernelHandle,
    pub sconv_state: KernelHandle,
    pub sconv_rewind: KernelHandle,
    pub block_res_sums: KernelHandle,
    pub block_res_out: KernelHandle,
    pub block_res_row: KernelHandle,
    pub head_norm: KernelHandle,
    pub msg_gate_z: KernelHandle,
    pub msg_gate_out: KernelHandle,
    pub msg_gate_fused: KernelHandle,
    pub moe_gate_up_nvfp4: KernelHandle,
    pub moe_gate_up_mxfp8: KernelHandle,
    pub moe_down_nvfp4: KernelHandle,
    pub moe_down_mxfp8: KernelHandle,
    /// decode batches: per-expert GEMVs over the rows' sorted expert slots
    pub moe_grouped_gate_up_nvfp4: KernelHandle,
    pub moe_grouped_gate_up_mxfp8: KernelHandle,
    pub moe_grouped_down_nvfp4: KernelHandle,
    pub moe_grouped_down_mxfp8: KernelHandle,
    pub moe_combine: KernelHandle,
    /// prefill: Atlas's sort, grouped GEMMs (NVFP4 pointer table, BF16) and
    /// weighted unpermute, around this target's softcapped SwiGLU
    pub moe_sort: KernelHandle,
    pub moe_unpermute: KernelHandle,
    pub moe_gemm_w4a4: KernelHandle,
    pub moe_gemm_w8a8: KernelHandle,
    pub moe_tiles: KernelHandle,
    pub quant_fp4: KernelHandle,
    pub situ_quant_fp4: KernelHandle,
    pub quant_mxfp8: KernelHandle,
    pub situ_quant_mxfp8: KernelHandle,
    /// AMD (strix-hip) only, from kernels/strix-hip/inkling-2-mini/nvfp4/inkling2_moe_gemm.cu:
    /// the BF16-row grouped expert GEMMs and the softcapped SwiGLU between them, which prefill
    /// takes there instead of the gb10 block-scaled W4A4 / W8A8 path (NULL elsewhere)
    pub moe_gemm_nvfp4: KernelHandle,
    pub moe_gemm_mxfp8: KernelHandle,
    pub softcap_swiglu: KernelHandle,
}

/// A strix-hip-only kernel of this target's `inkling2_moe_gemm` module; NULL on other targets.
fn amd_kernel(gpu: &dyn GpuBackend, name: &str) -> Result<KernelHandle> {
    if cfg!(atlas_scale) {
        gpu.kernel("inkling2_moe_gemm", name)
    } else {
        Ok(KernelHandle(0))
    }
}

impl Inkling2Kernels {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            decode_attn: gpu.kernel("inkling2_decode_attn", "inkling2_decode_attn")?,
            decode_attn_combine: gpu
                .kernel("inkling2_decode_attn", "inkling2_decode_attn_combine")?,
            prefill_attn: gpu.kernel("inkling2_prefill_attn", "inkling2_prefill_attn")?,
            rel_bias_table: gpu.kernel("inkling2_prefill_attn", "inkling2_rel_bias_table")?,
            flash_prefill: gpu.kernel("inkling2_flash_prefill", "inkling2_flash_prefill")?,
            sconv: gpu.kernel("inkling2_sconv", "inkling2_sconv")?,
            sconv_state: gpu.kernel("inkling2_sconv", "inkling2_sconv_state")?,
            sconv_rewind: gpu.kernel("inkling2_sconv", "inkling2_sconv_rewind")?,
            block_res_sums: gpu.kernel("inkling2_block_res", "inkling2_block_res_sums")?,
            block_res_out: gpu.kernel("inkling2_block_res", "inkling2_block_res_out")?,
            block_res_row: gpu.kernel("inkling2_block_res", "inkling2_block_res_row")?,
            head_norm: gpu.kernel("inkling2_head_ops", "inkling2_head_norm")?,
            msg_gate_z: gpu.kernel("inkling2_head_ops", "inkling2_msg_gate_z")?,
            msg_gate_out: gpu.kernel("inkling2_head_ops", "inkling2_msg_gate_out")?,
            msg_gate_fused: gpu.kernel("inkling2_head_ops", "inkling2_msg_gate_fused")?,
            moe_gate_up_nvfp4: gpu.kernel("inkling2_moe_decode", "inkling2_moe_gate_up_nvfp4")?,
            moe_gate_up_mxfp8: gpu.kernel("inkling2_moe_decode", "inkling2_moe_gate_up_mxfp8")?,
            moe_down_nvfp4: gpu.kernel("inkling2_moe_decode", "inkling2_moe_down_nvfp4")?,
            moe_down_mxfp8: gpu.kernel("inkling2_moe_decode", "inkling2_moe_down_mxfp8")?,
            moe_grouped_gate_up_nvfp4: gpu
                .kernel("inkling2_moe_decode", "inkling2_moe_grouped_gate_up_nvfp4")?,
            moe_grouped_gate_up_mxfp8: gpu
                .kernel("inkling2_moe_decode", "inkling2_moe_grouped_gate_up_mxfp8")?,
            moe_grouped_down_nvfp4: gpu
                .kernel("inkling2_moe_decode", "inkling2_moe_grouped_down_nvfp4")?,
            moe_grouped_down_mxfp8: gpu
                .kernel("inkling2_moe_decode", "inkling2_moe_grouped_down_mxfp8")?,
            moe_combine: gpu.kernel("inkling2_moe_decode", "inkling2_moe_combine")?,
            moe_sort: gpu.kernel("moe", "moe_sort_by_expert")?,
            moe_unpermute: gpu.kernel("moe", "moe_unpermute_reduce_indexed")?,
            moe_gemm_w4a4: gpu.kernel("inkling2_moe_gemm", "inkling2_moe_gemm_w4a4")?,
            moe_gemm_w8a8: gpu.kernel("inkling2_moe_gemm", "inkling2_moe_gemm_w8a8")?,
            moe_tiles: gpu.kernel("inkling2_moe_gemm", "inkling2_moe_tiles")?,
            quant_fp4: gpu.kernel("inkling2_moe_prefill", "inkling2_quant_fp4")?,
            situ_quant_fp4: gpu.kernel("inkling2_moe_prefill", "inkling2_situ_quant_fp4")?,
            quant_mxfp8: gpu.kernel("inkling2_moe_prefill", "inkling2_quant_mxfp8")?,
            situ_quant_mxfp8: gpu.kernel("inkling2_moe_prefill", "inkling2_situ_quant_mxfp8")?,
            moe_gemm_nvfp4: amd_kernel(gpu, "inkling2_moe_gemm_nvfp4")?,
            moe_gemm_mxfp8: amd_kernel(gpu, "inkling2_moe_gemm_mxfp8")?,
            softcap_swiglu: amd_kernel(gpu, "inkling2_softcap_swiglu")?,
        })
    }
}

pub(super) fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| anyhow::anyhow!("inkling2 {what} {v} does not fit u32"))
}

/// Short convolution over flat rows, then the state roll.
pub(crate) struct ShortConv {
    pub x: DevicePtr,
    pub out: DevicePtr,
    /// `[num_seqs]` device pointers, one state base per sequence
    pub states: DevicePtr,
    /// fp32 elements from a base to this conv's `[channels, k - 1]` state
    pub state_offset: usize,
    /// sequence `s` owns rows `s * rows_per_seq ..`
    pub rows_per_seq: usize,
    /// static `[C, k]`, or dynamic static taps `[k, C]`
    pub weight: DevicePtr,
    /// dynamic per-token taps `[rows, dyn_stride]`; NULL for a static conv
    pub dyn_taps: DevicePtr,
    pub rows: usize,
    pub num_seqs: usize,
    pub channels: usize,
    pub k: usize,
    pub x_stride: usize,
    pub out_stride: usize,
    pub dyn_stride: usize,
    pub head_dim: usize,
}

pub(crate) fn short_conv(
    gpu: &dyn GpuBackend,
    kernels: &Inkling2Kernels,
    a: &ShortConv,
    stream: u64,
) -> Result<()> {
    ensure!(
        (2..=MAX_CONV_K).contains(&a.k),
        "inkling2 conv kernel size {}",
        a.k
    );
    ensure!(a.head_dim > 0, "inkling2 conv head_dim");
    let cblocks = u32_of(a.channels.div_ceil(256), "channels")?;
    // decode rows: the conv kernel rolls its own state (one row a sequence)
    let roll = a.rows_per_seq == 1;
    KernelLaunch::new(gpu, kernels.sconv)
        .grid([cblocks, u32_of(a.rows, "rows")?, 1])
        .block([256, 1, 1])
        .arg_ptr(a.x)
        .arg_ptr(a.out)
        .arg_ptr(a.states)
        .arg_u32(u32_of(a.state_offset, "state offset")?)
        .arg_u32(u32_of(a.rows_per_seq, "rows per seq")?)
        .arg_ptr(a.weight)
        .arg_ptr(a.dyn_taps)
        .arg_u32(u32_of(a.rows, "rows")?)
        .arg_u32(u32_of(a.channels, "channels")?)
        .arg_u32(u32_of(a.k, "k")?)
        .arg_u32(u32_of(a.x_stride, "x stride")?)
        .arg_u32(u32_of(a.out_stride, "out stride")?)
        .arg_u32(u32_of(a.dyn_stride, "dyn stride")?)
        .arg_u32(u32_of(a.head_dim, "head dim")?)
        .arg_u32(u32::from(roll))
        .launch(stream)?;
    if roll {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernels.sconv_state)
        .grid([cblocks, u32_of(a.num_seqs, "seqs")?, 1])
        .block([256, 1, 1])
        .arg_ptr(a.x)
        .arg_ptr(a.states)
        .arg_u32(u32_of(a.state_offset, "state offset")?)
        .arg_u32(u32_of(a.rows_per_seq, "rows per seq")?)
        .arg_u32(u32_of(a.num_seqs, "seqs")?)
        .arg_u32(u32_of(a.channels, "channels")?)
        .arg_u32(u32_of(a.k, "k")?)
        .arg_u32(u32_of(a.x_stride, "x stride")?)
        .launch(stream)
}

/// Verify rollback of one sequence's conv state (already restored to its
/// pre-verify value): roll it forward over the first `rows` journaled inputs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sconv_rewind(
    gpu: &dyn GpuBackend,
    kernels: &Inkling2Kernels,
    inputs: DevicePtr,
    state: DevicePtr,
    channels: usize,
    k: usize,
    rows: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernels.sconv_rewind)
        .grid([u32_of(channels.div_ceil(256), "channels")?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(inputs)
        .arg_ptr(state)
        .arg_u32(u32_of(channels, "channels")?)
        .arg_u32(u32_of(k, "k")?)
        .arg_u32(u32_of(rows, "rows")?)
        .launch(stream)
}

/// Block attention residual: `nq` fp32 outputs `[nq, rows, hidden]`.
pub(crate) struct BlockRes {
    pub blocks: DevicePtr,
    /// exactly one of the two partial pointers is non-NULL
    pub partial_f32: DevicePtr,
    pub partial_bf16: DevicePtr,
    pub queries: DevicePtr,
    pub out: DevicePtr,
    pub rows: usize,
    pub hidden: usize,
    pub num_blocks: usize,
    /// elements between rows of one block
    pub row_stride: usize,
    /// elements between blocks
    pub block_stride: usize,
    pub num_queries: usize,
    /// per-row chunk sums scratch, [`block_res_sums_per_row`] fp32 each
    pub sums: DevicePtr,
    pub eps: f32,
}

pub(crate) fn block_res(
    gpu: &dyn GpuBackend,
    kernels: &Inkling2Kernels,
    a: &BlockRes,
    stream: u64,
) -> Result<()> {
    ensure!(
        a.num_blocks < MAX_BLOCK_STATES && (1..=MAX_RES_QUERIES).contains(&a.num_queries),
        "inkling2 block residual: {} blocks / {} queries over the kernel limits",
        a.num_blocks,
        a.num_queries
    );
    ensure!(
        a.partial_f32.is_null() != a.partial_bf16.is_null(),
        "inkling2 block residual needs exactly one partial"
    );
    if a.rows > RES_ROW_CTA_MIN_ROWS {
        // prefill rows: both passes in one CTA a row
        return KernelLaunch::new(gpu, kernels.block_res_row)
            .grid([u32_of(a.rows, "rows")?, 1, 1])
            .block([RES_CHUNK as u32, 1, 1])
            .arg_ptr(a.blocks)
            .arg_ptr(a.partial_f32)
            .arg_ptr(a.partial_bf16)
            .arg_ptr(a.queries)
            .arg_ptr(a.out)
            .arg_u32(u32_of(a.rows, "rows")?)
            .arg_u32(u32_of(a.hidden, "hidden")?)
            .arg_u32(u32_of(a.num_blocks, "blocks")?)
            .arg_u32(u32_of(a.row_stride, "row stride")?)
            .arg_u64(a.block_stride as u64)
            .arg_u32(u32_of(a.num_queries, "queries")?)
            .arg_f32(a.eps)
            .launch(stream);
    }
    // decode rows spread their channels over the GPU
    let chunks = a.hidden.div_ceil(RES_CHUNK);
    let grid = [u32_of(chunks, "chunks")?, u32_of(a.rows, "rows")?, 1];
    KernelLaunch::new(gpu, kernels.block_res_sums)
        .grid(grid)
        .block([RES_CHUNK as u32, 1, 1])
        .arg_ptr(a.blocks)
        .arg_ptr(a.partial_f32)
        .arg_ptr(a.partial_bf16)
        .arg_ptr(a.queries)
        .arg_ptr(a.sums)
        .arg_u32(u32_of(a.hidden, "hidden")?)
        .arg_u32(u32_of(a.num_blocks, "blocks")?)
        .arg_u32(u32_of(a.row_stride, "row stride")?)
        .arg_u64(a.block_stride as u64)
        .arg_u32(u32_of(a.num_queries, "queries")?)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.block_res_out)
        .grid(grid)
        .block([RES_CHUNK as u32, 1, 1])
        .arg_ptr(a.blocks)
        .arg_ptr(a.partial_f32)
        .arg_ptr(a.partial_bf16)
        .arg_ptr(a.sums)
        .arg_ptr(a.out)
        .arg_u32(u32_of(a.rows, "rows")?)
        .arg_u32(u32_of(a.hidden, "hidden")?)
        .arg_u32(u32_of(a.num_blocks, "blocks")?)
        .arg_u32(u32_of(a.row_stride, "row stride")?)
        .arg_u64(a.block_stride as u64)
        .arg_u32(u32_of(a.num_queries, "queries")?)
        .arg_f32(a.eps)
        .launch(stream)
}

/// Channels per block-residual CTA (the kernels' INK_THREADS).
const RES_CHUNK: usize = 256;
/// Rows above which each row takes one block-residual CTA.
const RES_ROW_CTA_MIN_ROWS: usize = 64;

/// fp32 elements of `inkling2_block_res_sums` output per row: per hidden chunk,
/// per state (blocks + partial), the sum of squares and each query's dot.
pub(crate) fn block_res_sums_per_row(hidden: usize) -> usize {
    hidden.div_ceil(RES_CHUNK) * MAX_BLOCK_STATES * (1 + MAX_RES_QUERIES)
}

/// Per-head epilogue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum HeadNorm {
    /// L2 clip after an optional learned scale; log scaling when positions are given.
    Clip { threshold: f32 },
    /// gamma-free RMSNorm
    Rms { eps: f32 },
}

pub(crate) struct HeadNormArgs {
    pub x: DevicePtr,
    pub y: DevicePtr,
    pub scale: DevicePtr,
    pub positions: DevicePtr,
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub in_stride: usize,
    pub out_stride: usize,
    pub norm: HeadNorm,
    pub log_n_floor: f32,
    pub log_alpha: f32,
}

pub(crate) fn head_norm(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: &HeadNormArgs,
    stream: u64,
) -> Result<()> {
    ensure!(
        a.head_dim <= MAX_HEAD_DIM,
        "inkling2 head_dim {} over 128",
        a.head_dim
    );
    let (mode, value) = match a.norm {
        HeadNorm::Clip { threshold } => (0u32, threshold),
        HeadNorm::Rms { eps } => (1u32, eps),
    };
    let warps = a.rows * a.heads;
    KernelLaunch::new(gpu, k)
        .grid([u32_of((warps * 32).div_ceil(256), "warps")?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a.x)
        .arg_ptr(a.y)
        .arg_ptr(a.scale)
        .arg_ptr(a.positions)
        .arg_u32(u32_of(a.rows, "rows")?)
        .arg_u32(u32_of(a.heads, "heads")?)
        .arg_u32(u32_of(a.head_dim, "head dim")?)
        .arg_u32(u32_of(a.in_stride, "in stride")?)
        .arg_u32(u32_of(a.out_stride, "out stride")?)
        .arg_u32(mode)
        .arg_f32(value)
        .arg_f32(a.log_n_floor)
        .arg_f32(a.log_alpha)
        .launch(stream)
}

/// Up to three of a visit's loop message gates over fp32 rows: gate `i` reads
/// `h + i * h_stride`, `m + i * m_stride` (elements) and writes
/// `out + i * out_stride`, which may alias `h`.
pub(crate) struct MsgGate<'a> {
    pub gates: &'a [MessageGate],
    pub h: DevicePtr,
    pub h_stride: usize,
    pub m: DevicePtr,
    pub m_stride: usize,
    pub out: DevicePtr,
    pub out_stride: usize,
    /// `[gates, rows, rank]` fp32 scratch
    pub z: DevicePtr,
    pub rows: usize,
    pub hidden: usize,
    pub rank: usize,
    pub eps: f32,
}

const MAX_GATES: usize = 3;
const Z_WARPS: usize = 4;
/// `inkling2_head_ops.cu` INK_Z_ROWS / INK_OUT_ROWS: rows a gate block covers.
const Z_ROWS: u32 = 8;
const OUT_ROWS: u32 = 16;
/// `inkling2_msg_gate_fused`: rows a CTA takes, and the rows from which a
/// launch takes it (its 16-row MMA tiles; decode rows keep the two kernels).
const FUSED_ROWS: usize = 16;

pub(crate) fn msg_gate(
    gpu: &dyn GpuBackend,
    kernels: &Inkling2Kernels,
    a: &MsgGate<'_>,
    stream: u64,
) -> Result<()> {
    let n = a.gates.len();
    ensure!(
        (1..=MAX_GATES).contains(&n),
        "inkling2: {n} message gates in one launch"
    );
    ensure!(
        a.rank <= MAX_GATE_RANK && a.rank.is_multiple_of(8) && a.hidden.is_multiple_of(8),
        "inkling2 loop gate rank {} / hidden {}",
        a.rank,
        a.hidden
    );
    // both kernels read the weights 16 bytes at a time
    ensure!(
        a.gates.iter().all(|g| [g.down, g.forget, g.gain]
            .iter()
            .all(|p| p.0.is_multiple_of(16))),
        "inkling2: message gate weights not 16-byte aligned"
    );
    let pick = |f: fn(&MessageGate) -> DevicePtr, i: usize| f(&a.gates[i.min(n - 1)]);
    let rows = u32_of(a.rows, "rows")?;
    // AMD (strix-hip): the fused gate is mma.sync PTX (a stub there); the two-kernel path runs.
    if !cfg!(atlas_scale)
        && a.rows >= FUSED_ROWS
        && a.rank == MAX_GATE_RANK
        && a.hidden.is_multiple_of(64)
    {
        let mut f = KernelLaunch::new(gpu, kernels.msg_gate_fused)
            .grid([rows.div_ceil(FUSED_ROWS as u32), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a.h)
            .arg_ptr(a.m);
        for w in [
            |g: &MessageGate| g.down,
            |g: &MessageGate| g.forget,
            |g: &MessageGate| g.gain,
        ] {
            for i in 0..MAX_GATES {
                f = f.arg_ptr(pick(w, i));
            }
        }
        return f
            .arg_ptr(a.out)
            .arg_u32(rows)
            .arg_u32(u32_of(a.hidden, "hidden")?)
            .arg_u32(n as u32)
            .arg_u64(a.h_stride as u64)
            .arg_u64(a.m_stride as u64)
            .arg_u64(a.out_stride as u64)
            .arg_f32(a.eps)
            .launch(stream);
    }
    let mut z = KernelLaunch::new(gpu, kernels.msg_gate_z)
        .grid([
            u32_of(a.rank.div_ceil(Z_WARPS), "rank")?,
            rows.div_ceil(Z_ROWS),
            n as u32,
        ])
        .block([(32 * Z_WARPS) as u32, 1, 1])
        .arg_ptr(a.h)
        .arg_ptr(a.m);
    for i in 0..MAX_GATES {
        z = z.arg_ptr(pick(|g| g.down, i));
    }
    z.arg_ptr(a.z)
        .arg_u32(rows)
        .arg_u32(u32_of(a.hidden, "hidden")?)
        .arg_u32(u32_of(a.rank, "rank")?)
        .arg_u64(a.h_stride as u64)
        .arg_u64(a.m_stride as u64)
        .arg_f32(a.eps)
        .launch(stream)?;
    let mut out = KernelLaunch::new(gpu, kernels.msg_gate_out)
        .grid([
            u32_of(a.hidden.div_ceil(32), "hidden")?,
            rows.div_ceil(OUT_ROWS),
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a.h)
        .arg_ptr(a.m);
    for i in 0..MAX_GATES {
        out = out.arg_ptr(pick(|g| g.forget, i));
    }
    for i in 0..MAX_GATES {
        out = out.arg_ptr(pick(|g| g.gain, i));
    }
    out.arg_ptr(a.z)
        .arg_ptr(a.out)
        .arg_u32(rows)
        .arg_u32(u32_of(a.hidden, "hidden")?)
        .arg_u32(u32_of(a.rank, "rank")?)
        .arg_u32(n as u32)
        .arg_u64(a.h_stride as u64)
        .arg_u64(a.m_stride as u64)
        .arg_u64(a.out_stride as u64)
        .launch(stream)
}
