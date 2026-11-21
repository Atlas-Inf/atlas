// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for the Inkling 2 kernels in `kernels/gb10/inkling-2-mini/nvfp4/`.
//! Each takes a plain argument struct (device pointers, dimensions) so call
//! sites read as data, and checks the limits the kernels compile in.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::weights::MessageGate;

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
    pub sconv: KernelHandle,
    pub sconv_state: KernelHandle,
    pub block_res: KernelHandle,
    pub head_norm: KernelHandle,
    pub msg_gate_z: KernelHandle,
    pub msg_gate_out: KernelHandle,
    pub moe_gate_up_nvfp4: KernelHandle,
    pub moe_gate_up_mxfp8: KernelHandle,
    pub moe_down_nvfp4: KernelHandle,
    pub moe_down_mxfp8: KernelHandle,
    pub moe_combine: KernelHandle,
}

impl Inkling2Kernels {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            decode_attn: gpu.kernel("inkling2_decode_attn", "inkling2_decode_attn")?,
            sconv: gpu.kernel("inkling2_sconv", "inkling2_sconv")?,
            sconv_state: gpu.kernel("inkling2_sconv", "inkling2_sconv_state")?,
            block_res: gpu.kernel("inkling2_block_res", "inkling2_block_res")?,
            head_norm: gpu.kernel("inkling2_head_ops", "inkling2_head_norm")?,
            msg_gate_z: gpu.kernel("inkling2_head_ops", "inkling2_msg_gate_z")?,
            msg_gate_out: gpu.kernel("inkling2_head_ops", "inkling2_msg_gate_out")?,
            moe_gate_up_nvfp4: gpu.kernel("inkling2_moe_decode", "inkling2_moe_gate_up_nvfp4")?,
            moe_gate_up_mxfp8: gpu.kernel("inkling2_moe_decode", "inkling2_moe_gate_up_mxfp8")?,
            moe_down_nvfp4: gpu.kernel("inkling2_moe_decode", "inkling2_moe_down_nvfp4")?,
            moe_down_mxfp8: gpu.kernel("inkling2_moe_decode", "inkling2_moe_down_mxfp8")?,
            moe_combine: gpu.kernel("inkling2_moe_decode", "inkling2_moe_combine")?,
        })
    }
}

fn u32_of(v: usize, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| anyhow::anyhow!("inkling2 {what} {v} does not fit u32"))
}

/// Paged decode attention with relative bias and per-token sinks.
pub(crate) struct DecodeAttn {
    pub q: DevicePtr,
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    pub out: DevicePtr,
    pub block_tables: DevicePtr,
    /// `[num_seqs]` query positions; a row attends keys 0 ..= its position
    pub positions: DevicePtr,
    pub rel: DevicePtr,
    pub rel_proj: DevicePtr,
    pub sinks: DevicePtr,
    pub max_blocks_per_seq: usize,
    pub num_seqs: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub q_stride: usize,
    pub sliding_window: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
}

pub(crate) fn decode_attn(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: &DecodeAttn,
    stream: u64,
) -> Result<()> {
    ensure!(
        a.head_dim == 128,
        "inkling2 decode attention is built for head_dim 128"
    );
    ensure!(
        a.d_rel <= MAX_D_REL && a.rel_extent <= MAX_REL_EXTENT,
        "inkling2 rel bias d_rel {} / extent {} over the kernel limits",
        a.d_rel,
        a.rel_extent
    );
    ensure!(
        a.kv_heads > 0 && a.heads.is_multiple_of(a.kv_heads),
        "inkling2 GQA heads"
    );
    KernelLaunch::new(gpu, k)
        .grid([u32_of(a.heads, "heads")?, u32_of(a.num_seqs, "seqs")?, 1])
        .block([256, 1, 1])
        .arg_ptr(a.q)
        .arg_ptr(a.k_cache)
        .arg_ptr(a.v_cache)
        .arg_ptr(a.out)
        .arg_ptr(a.block_tables)
        .arg_ptr(a.positions)
        .arg_ptr(a.rel)
        .arg_ptr(a.rel_proj)
        .arg_ptr(a.sinks)
        .arg_u32(u32_of(a.max_blocks_per_seq, "max blocks")?)
        .arg_u32(u32_of(a.heads, "heads")?)
        .arg_u32(u32_of(a.kv_heads, "kv heads")?)
        .arg_u32(u32_of(a.block_size, "block size")?)
        .arg_f32((a.head_dim as f32).powf(-0.5))
        .arg_u32(u32_of(a.q_stride, "q stride")?)
        .arg_u32(u32_of(a.sliding_window, "window")?)
        .arg_u32(u32_of(a.d_rel, "d_rel")?)
        .arg_u32(u32_of(a.rel_extent, "rel extent")?)
        .launch(stream)
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
        .launch(stream)?;
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
    pub eps: f32,
}

pub(crate) fn block_res(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
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
    KernelLaunch::new(gpu, k)
        .grid([u32_of(a.rows, "rows")?, 1, 1])
        .block([256, 1, 1])
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
        .launch(stream)
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
    let mut z = KernelLaunch::new(gpu, kernels.msg_gate_z)
        .grid([u32_of(a.rank.div_ceil(Z_WARPS), "rank")?, rows, n as u32])
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
        .grid([u32_of(a.hidden.div_ceil(128), "hidden")?, rows, 1])
        .block([128, 1, 1])
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
