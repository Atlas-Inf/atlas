// SPDX-License-Identifier: AGPL-3.0-only

//! The visit's loop message gates, split out of `ops.rs`.

use anyhow::{Result, ensure};
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
    // (on AMD the fused gate is the WMMA port in inkling2_head_ops.cu's AMD branch)
    if a.rows >= FUSED_ROWS && a.rank == MAX_GATE_RANK && a.hidden.is_multiple_of(64) {
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
