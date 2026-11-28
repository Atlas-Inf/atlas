// SPDX-License-Identifier: AGPL-3.0-only

//! The visit's loop message gates, split out of `ops.rs`.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::ops::{Inkling2Kernels, MAX_GATE_RANK, u32_of};
use super::weights::MessageGate;

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
    /// The `h` and `m` rows' sums of squares, `[plane, rows]` fp32 with a plane
    /// a `stride / hidden` rows on: their `block_res` wrote them when
    /// [`fused_takes`] the rows (NULL otherwise)
    pub sq_h: DevicePtr,
    pub sq_m: DevicePtr,
    /// gb10's fused gate rebuilds the messages from these instead of reading
    /// `m` (which may then be unwritten)
    pub msg: Option<MsgSources>,
    /// `[gates, rows, rank]` fp32 scratch
    pub z: DevicePtr,
    pub rows: usize,
    /// Decode or verify rows: the two decode kernels at any row count, so each
    /// row is the one serial decode computes.
    pub decode_rows: bool,
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

/// A loop message stack and its rows' mix weights: message `q` of a row is
/// `sum_j p[row][q][j] state_j` over the stack's blocks, then the partial
/// (`inkling2_block_res_row`'s own operation order).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MsgSources {
    pub blocks: DevicePtr,
    /// elements between blocks
    pub block_stride: usize,
    pub num_blocks: usize,
    /// BF16 `[rows, hidden]`
    pub partial: DevicePtr,
    /// `[rows, 4, 9]` fp32 (`BlockRes::p_out`)
    pub p: DevicePtr,
    /// the launch's first gate's message
    pub first: usize,
}

/// `inkling2_msg_gate_fused`'s dynamic shared memory on gb10 (INK_FG_SMEM):
/// two stages of three gates' inputs and D tiles, z, the rows' mix weights
/// and rms.
const FUSED_SMEM: u32 =
    (2 * 3 * 16 * 72 * 2 + 2 * 3 * 64 * 72 * 2 + 3 * 16 * 72 * 2 + 16 * 4 * 9 * 4 + 3 * 16 * 2 * 4)
        as u32;

/// Whether `rows` take `inkling2_msg_gate_fused`, which reads the rows' sums
/// of squares from their producer (prefill rows: decode and verify rows keep
/// the two decode kernels).
pub(crate) fn fused_takes(rows: usize, decode_rows: bool, rank: usize, hidden: usize) -> bool {
    !decode_rows && rows >= FUSED_ROWS && rank == MAX_GATE_RANK && hidden.is_multiple_of(64)
}

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
    // (on AMD the fused gate is the WMMA port in inkling2_head_ops.cu's AMD branch)
    if fused_takes(a.rows, a.decode_rows, a.rank, a.hidden) {
        ensure!(
            !a.sq_h.is_null() && !a.sq_m.is_null(),
            "inkling2: fused message gate without its rows' sums of squares"
        );
        // gb10 rebuilds the messages (`msg`); the AMD WMMA port reads `m`
        let ms = if cfg!(atlas_scale) {
            None
        } else {
            Some(
                a.msg
                    .context("inkling2: fused message gate without its messages' sources")?,
            )
        };
        let mut f = KernelLaunch::new(gpu, kernels.msg_gate_fused)
            .grid([rows.div_ceil(FUSED_ROWS as u32), 1, 1])
            .block([256, 1, 1])
            .shared_mem(if cfg!(atlas_scale) { 0 } else { FUSED_SMEM })
            .arg_ptr(a.h)
            .arg_ptr(a.m)
            .arg_ptr(a.sq_h)
            .arg_ptr(a.sq_m)
            .arg_ptr(ms.map_or(DevicePtr::NULL, |m| m.blocks))
            .arg_ptr(ms.map_or(DevicePtr::NULL, |m| m.partial))
            .arg_ptr(ms.map_or(DevicePtr::NULL, |m| m.p))
            .arg_u64(ms.map_or(0, |m| m.block_stride as u64))
            .arg_u32(u32_of(ms.map_or(0, |m| m.num_blocks), "message blocks")?)
            .arg_u32(u32_of(ms.map_or(0, |m| m.first), "first message")?);
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
