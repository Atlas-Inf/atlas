// SPDX-License-Identifier: AGPL-3.0-only

//! Launchers for `kernels/strix-hip/inkling-2-mini/nvfp4/inkling2_fused.cu`:
//! elementwise decode fusions that keep the unfused kernels' arithmetic bit for
//! bit (NULL handles elsewhere — the call sites keep the unfused chain).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::ops::{Inkling2Kernels, MAX_CONV_K, MAX_D_REL, MAX_HEAD_DIM, u32_of};

/// k/v short convs + the q/k/v/rel head norms + the paged KV write in one
/// launch (`inkling2_attn_prologue`). Covers `inkling2_sconv` (roll when
/// `rows_per_seq == 1`, both static and dynamic taps), `inkling2_head_norm`
/// modes 0 and 1, and `reshape_and_cache_flash`.
pub(crate) struct AttnPrologue {
    pub k_lin: DevicePtr,
    pub v_lin: DevicePtr,
    pub q_in: DevicePtr,
    pub r_in: DevicePtr,
    pub kc: DevicePtr,
    pub vc: DevicePtr,
    pub qn: DevicePtr,
    pub kn: DevicePtr,
    pub vn: DevicePtr,
    pub rn: DevicePtr,
    /// `[num_seqs]` device pointers, one state base per sequence
    pub states: DevicePtr,
    /// fp32 elements from a base to each conv's `[channels, k - 1]` state
    pub state_off_k: usize,
    pub state_off_v: usize,
    pub k_w: DevicePtr,
    pub v_w: DevicePtr,
    /// dynamic per-token taps `[rows, dyn_stride]`; NULL for static convs
    pub k_dyn: DevicePtr,
    pub v_dyn: DevicePtr,
    pub q_scale: DevicePtr,
    pub rel_scale: DevicePtr,
    /// the rows' positions for the log-length scale; NULL for SWA layers
    pub positions: DevicePtr,
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    pub slots: DevicePtr,
    pub rows: usize,
    pub rows_per_seq: usize,
    pub num_seqs: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub conv_k: usize,
    pub dyn_stride: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub eps: f32,
    pub log_n_floor: f32,
    pub log_alpha: f32,
}

pub(crate) fn attn_prologue(
    gpu: &dyn GpuBackend,
    kernels: &Inkling2Kernels,
    a: &AttnPrologue,
    stream: u64,
) -> Result<()> {
    ensure!(
        a.head_dim <= MAX_HEAD_DIM && a.d_rel <= MAX_D_REL && (2..=MAX_CONV_K).contains(&a.conv_k),
        "inkling2 fused prologue dims over the kernel limits"
    );
    ensure!(
        a.kv_heads <= 8,
        "inkling2 fused prologue needs a warp a kv head ({} > 8)",
        a.kv_heads
    );
    KernelLaunch::new(gpu, kernels.attn_prologue)
        .grid([u32_of(a.rows, "rows")?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a.k_lin)
        .arg_ptr(a.v_lin)
        .arg_ptr(a.q_in)
        .arg_ptr(a.r_in)
        .arg_ptr(a.kc)
        .arg_ptr(a.vc)
        .arg_ptr(a.qn)
        .arg_ptr(a.kn)
        .arg_ptr(a.vn)
        .arg_ptr(a.rn)
        .arg_ptr(a.states)
        .arg_u32(u32_of(a.state_off_k, "k state offset")?)
        .arg_u32(u32_of(a.state_off_v, "v state offset")?)
        .arg_ptr(a.k_w)
        .arg_ptr(a.v_w)
        .arg_ptr(a.k_dyn)
        .arg_ptr(a.v_dyn)
        .arg_ptr(a.q_scale)
        .arg_ptr(a.rel_scale)
        .arg_ptr(a.positions)
        .arg_ptr(a.k_cache)
        .arg_ptr(a.v_cache)
        .arg_ptr(a.slots)
        .arg_u32(u32_of(a.rows_per_seq, "rows per seq")?)
        .arg_u32(u32_of(a.heads, "heads")?)
        .arg_u32(u32_of(a.kv_heads, "kv heads")?)
        .arg_u32(u32_of(a.head_dim, "head dim")?)
        .arg_u32(u32_of(a.d_rel, "d_rel")?)
        .arg_u32(u32_of(a.conv_k, "conv k")?)
        .arg_u32(u32_of(a.dyn_stride, "dyn stride")?)
        .arg_f32(a.qk_clip)
        .arg_f32(a.rel_clip)
        .arg_f32(a.eps)
        .arg_f32(a.log_n_floor)
        .arg_f32(a.log_alpha)
        .launch(stream)?;
    if a.rows_per_seq > 1 {
        // verify rows: the fused conv left the roll out, so each sequence's
        // state rolls over its rows exactly as `inkling2_sconv_state` would
        let state = |x: DevicePtr, off: usize| {
            KernelLaunch::new(gpu, kernels.sconv_state)
                .grid([
                    u32_of((a.kv_heads * a.head_dim).div_ceil(256), "channels")?,
                    u32_of(a.num_seqs, "seqs")?,
                    1,
                ])
                .block([256, 1, 1])
                .arg_ptr(x)
                .arg_ptr(a.states)
                .arg_u32(u32_of(off, "state offset")?)
                .arg_u32(u32_of(a.rows_per_seq, "rows per seq")?)
                .arg_u32(u32_of(a.num_seqs, "seqs")?)
                .arg_u32(u32_of(a.kv_heads * a.head_dim, "channels")?)
                .arg_u32(u32_of(a.conv_k, "k")?)
                .arg_u32(u32_of(a.kv_heads * a.head_dim, "x stride")?)
                .launch(stream)
        };
        state(a.k_lin, a.state_off_k)?;
        state(a.v_lin, a.state_off_v)?;
    }
    Ok(())
}

/// `moe_combine` + the static post-MLP conv (with its state roll) + `layer_out`
/// in one launch (`inkling2_mlp_tail`), decode rows only: a sequence's conv
/// reads its own state alone, which only holds when each row is its own
/// sequence (`rows_per_seq == 1`).
pub(crate) struct MlpTail {
    pub y: DevicePtr,
    pub weights: DevicePtr,
    pub moe_out: DevicePtr,
    pub mlp_conv: DevicePtr,
    pub states: DevicePtr,
    pub state_off: usize,
    pub conv_w: DevicePtr,
    pub resid: DevicePtr,
    pub hidden: DevicePtr,
    pub rows: usize,
    pub hidden_dim: usize,
    pub top_k: usize,
    pub conv_k: usize,
}

pub(crate) fn mlp_tail(
    gpu: &dyn GpuBackend,
    kernels: &Inkling2Kernels,
    a: &MlpTail,
    stream: u64,
) -> Result<()> {
    ensure!(
        (2..=MAX_CONV_K).contains(&a.conv_k),
        "inkling2 fused mlp tail conv k {}",
        a.conv_k
    );
    KernelLaunch::new(gpu, kernels.mlp_tail)
        .grid([
            u32_of(a.hidden_dim.div_ceil(256), "hidden")?,
            u32_of(a.rows, "rows")?,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a.y)
        .arg_ptr(a.weights)
        .arg_ptr(a.moe_out)
        .arg_ptr(a.mlp_conv)
        .arg_ptr(a.states)
        .arg_u32(u32_of(a.state_off, "state offset")?)
        .arg_ptr(a.conv_w)
        .arg_ptr(a.resid)
        .arg_ptr(a.hidden)
        .arg_u32(u32_of(a.hidden_dim, "hidden")?)
        .arg_u32(u32_of(a.top_k, "top k")?)
        .arg_u32(u32_of(a.conv_k, "conv k")?)
        .launch(stream)
}

/// The MLP block residual with `inkling2_residual` and the boundary block push
/// folded into `inkling2_block_res_sums` on its own (chunks, rows) grid
/// (`inkling2_block_res_sums_resid`), then the unchanged `_out` pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_res_mlp(
    gpu: &dyn GpuBackend,
    kernels: &Inkling2Kernels,
    a: &super::ops::BlockRes,
    resid_x: DevicePtr,
    resid_c: DevicePtr,
    resid: DevicePtr,
    push: bool,
    stream: u64,
) -> Result<()> {
    ensure!(
        a.num_blocks < super::ops::MAX_BLOCK_STATES
            && (1..=super::ops::MAX_RES_QUERIES).contains(&a.num_queries),
        "inkling2 block residual: {} blocks / {} queries over the kernel limits",
        a.num_blocks,
        a.num_queries
    );
    let chunks = a.hidden.div_ceil(256);
    let grid = [u32_of(chunks, "chunks")?, u32_of(a.rows, "rows")?, 1];
    KernelLaunch::new(gpu, kernels.block_res_resid)
        .grid(grid)
        .block([256, 1, 1])
        .arg_ptr(a.blocks)
        .arg_ptr(resid_x)
        .arg_ptr(resid_c)
        .arg_ptr(resid)
        .arg_ptr(a.queries)
        .arg_ptr(a.sums)
        .arg_u32(u32_of(a.hidden, "hidden")?)
        .arg_u32(u32_of(a.num_blocks, "blocks")?)
        .arg_u32(u32_of(a.row_stride, "row stride")?)
        .arg_u64(a.block_stride as u64)
        .arg_u32(u32_of(a.num_queries, "queries")?)
        .arg_u32(u32::from(push))
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.block_res_out)
        .grid(grid)
        .block([256, 1, 1])
        .arg_ptr(a.blocks)
        .arg_ptr(resid)
        .arg_ptr(DevicePtr::NULL)
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
