// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_BATCH_SMALL` (`model/qwen4exp_batch_fast.rs`): the exact
//! fused GDN step of `qwen4exp_decode_fuse` over the rows of a batched decode
//! (`qwen4exp_gdn_decode_fused_rows`, one token a sequence) and of an exact
//! MTP verify (`qwen4exp_gdn_verify_fused_rows`, up to four tokens a
//! sequence), each sequence's state passed by value. The layer side is
//! `qwen3_ssm/gdn_fused_rows.rs`; every byte is checked against the chains
//! replaced by `scripts/dev/qwen4exp_batch_small_bench.cu`.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::qwen4exp_decode_fuse::{GDN_D, aligned, gdn_geometry_fits};

/// Sequences one `qwen4exp_gdn_decode_fused_rows` launch takes
/// (`QDF_ROWS_MAX`: the by-value state table's length).
pub const GDN_ROWS_MAX: usize = 8;

/// A batched decode step's GDN buffers between the projections: row `r` is
/// sequence `r`'s token, against `states[r]` (recurrence, conv window).
pub struct GdnDecodeRows<'a> {
    pub states: &'a [(DevicePtr, DevicePtr)],
    /// `[rows, qkvz_stride]` BF16: `[Q | K | V | Z]` per row.
    pub qkvz: DevicePtr,
    pub qkvz_stride: u32,
    pub conv_w: DevicePtr,
    /// `[rows, ba_k]` BF16: the mixer input rows.
    pub ba_in: DevicePtr,
    pub ba_w: DevicePtr,
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    /// `[rows, 2 * nv]` FP32: gate then beta per row.
    pub gates: DevicePtr,
    pub norm_w: DevicePtr,
    /// `[rows, nv * 128]` BF16.
    pub out: DevicePtr,
}

/// `ATLAS_QWEN4EXP_BATCH_SMALL`: the GDN step of every row of a batched
/// decode (one token per sequence) as `qwen4exp_gdn_decode_fused_rows`
/// launches of up to [`GDN_ROWS_MAX`] sequences, in place of four launches
/// per sequence. Each row's bytes are the four kernels' (the fused step's
/// contract), whether or not the decode-fuse tier is on. Returns whether it
/// launched; on `false` nothing was launched. The caller checks the lever and
/// that its per-sequence arm is the four-kernel FP32 one.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_rows(
    gpu: &dyn GpuBackend,
    b: &GdnDecodeRows<'_>,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    d_conv: u32,
    ba_k: u32,
    l2_eps: f32,
    eps: f32,
    stream: u64,
) -> Result<bool> {
    static K: OnceLock<KernelHandle> = OnceLock::new();
    let k = *K.get_or_init(|| {
        crate::layers::try_kernel(
            gpu,
            "qwen4exp_decode_fuse",
            "qwen4exp_gdn_decode_fused_rows",
        )
    });
    let conv_dim = 2 * nk * kd + nv * vd;
    let fits = k.0 != 0
        && !b.states.is_empty()
        && gdn_geometry_fits(nk, nv, kd, vd, d_conv, ba_k)
        && b.states.iter().all(|&(_, conv)| conv.0.is_multiple_of(16))
        && aligned(&[b.ba_in, b.ba_w], 16)
        && b.qkvz_stride.is_multiple_of(4)
        && aligned(&[b.qkvz.offset(conv_dim as usize * 2), b.norm_w, b.out], 8);
    if !fits {
        return Ok(false);
    }
    let (bf16, fp32) = (2usize, 4usize);
    for (chunk, states) in b.states.chunks(GDN_ROWS_MAX).enumerate() {
        let first = chunk * GDN_ROWS_MAX;
        // QdfRowStates: h[8], then conv[8]; unused slots stay null.
        let mut table = [0u64; 2 * GDN_ROWS_MAX];
        for (i, &(h, conv)) in states.iter().enumerate() {
            table[i] = h.0;
            table[GDN_ROWS_MAX + i] = conv.0;
        }
        KernelLaunch::new(gpu, k)
            .grid([nv, states.len() as u32, 1])
            .block([GDN_D, 1, 1])
            .arg_words(&table)
            .arg_ptr(b.qkvz.offset(first * b.qkvz_stride as usize * bf16))
            .arg_ptr(b.conv_w)
            .arg_ptr(b.ba_in.offset(first * ba_k as usize * bf16))
            .arg_ptr(b.ba_w)
            .arg_ptr(b.a_log)
            .arg_ptr(b.dt_bias)
            .arg_ptr(b.gates.offset(first * 2 * nv as usize * fp32))
            .arg_ptr(b.norm_w)
            .arg_ptr(b.out.offset(first * (nv * vd) as usize * bf16))
            .arg_u32(nk)
            .arg_u32(nv)
            .arg_u32(ba_k)
            .arg_u32(kd)
            .arg_u32(b.qkvz_stride)
            .arg_f32(l2_eps)
            .arg_f32(eps)
            .launch(stream)?;
    }
    Ok(true)
}

/// Tokens a sequence may bring to `qwen4exp_gdn_verify_fused_rows`
/// (`QDF_VERIFY_KMAX`): the exact lane's 8-row window (7 drafts).
pub const GDN_VERIFY_KMAX: usize = 8;

/// One sequence of a verify step: its recurrence and conv state, the rollback
/// slots for tokens `0..k-1` (`h_snap[t]` / `conv_snap[t]` after token `t`),
/// and its rows `row0..row0 + k` of the step.
#[derive(Clone, Copy)]
pub struct GdnVerifySeq {
    pub h: DevicePtr,
    pub conv: DevicePtr,
    pub h_snap: [DevicePtr; GDN_VERIFY_KMAX - 1],
    pub conv_snap: [DevicePtr; GDN_VERIFY_KMAX - 1],
    pub row0: u32,
    pub k: u32,
}

/// A verify step's GDN buffers between the projections, rows as in
/// [`GdnDecodeRows`]; `gates` holds the step's gate rows (the exact arm's
/// input), `out` receives the normed rows.
pub struct GdnVerifyRows<'a> {
    pub seqs: &'a [GdnVerifySeq],
    pub qkvz: DevicePtr,
    pub qkvz_stride: u32,
    pub conv_w: DevicePtr,
    pub gates: DevicePtr,
    pub norm_w: DevicePtr,
    pub out: DevicePtr,
}

/// `ATLAS_QWEN4EXP_BATCH_SMALL`: the exact MTP verify's per-token GDN chain
/// (conv, conv rollback copy, recurrence, gated norm, H rollback copy, for
/// each token of each sequence) as `qwen4exp_gdn_verify_fused_rows` launches
/// of up to [`GDN_ROWS_MAX`] sequences, every byte the chain writes. Returns
/// whether it launched; on `false` nothing was launched. The caller checks the
/// lever and that the exact arm is the four-kernel FP32 one.
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_rows(
    gpu: &dyn GpuBackend,
    b: &GdnVerifyRows<'_>,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    d_conv: u32,
    l2_eps: f32,
    eps: f32,
    stream: u64,
) -> Result<bool> {
    static K: OnceLock<KernelHandle> = OnceLock::new();
    let k = *K.get_or_init(|| {
        crate::layers::try_kernel(
            gpu,
            "qwen4exp_decode_fuse",
            "qwen4exp_gdn_verify_fused_rows",
        )
    });
    let conv_dim = 2 * nk * kd + nv * vd;
    let fits = k.0 != 0
        && !b.seqs.is_empty()
        && gdn_geometry_fits(nk, nv, kd, vd, d_conv, 8)
        && b.seqs.iter().all(|q| {
            (1..=GDN_VERIFY_KMAX as u32).contains(&q.k)
                && aligned(&[q.conv], 16)
                && aligned(&q.conv_snap[..q.k as usize - 1], 16)
        })
        && b.qkvz_stride.is_multiple_of(4)
        && aligned(&[b.qkvz.offset(conv_dim as usize * 2), b.norm_w, b.out], 8);
    if !fits {
        return Ok(false);
    }
    // QdfVerifySeq: h, conv, h_snap[KMAX-1], conv_snap[KMAX-1], then row0 | k << 32.
    const WORDS: usize = 3 + 2 * (GDN_VERIFY_KMAX - 1);
    for seqs in b.seqs.chunks(GDN_ROWS_MAX) {
        let mut table = [0u64; WORDS * GDN_ROWS_MAX];
        for (q, w) in seqs.iter().zip(table.chunks_mut(WORDS)) {
            w[0] = q.h.0;
            w[1] = q.conv.0;
            for t in 0..GDN_VERIFY_KMAX - 1 {
                w[2 + t] = q.h_snap[t].0;
                w[2 + GDN_VERIFY_KMAX - 1 + t] = q.conv_snap[t].0;
            }
            w[WORDS - 1] = u64::from(q.row0) | u64::from(q.k) << 32;
        }
        KernelLaunch::new(gpu, k)
            .grid([nv, seqs.len() as u32, 1])
            .block([GDN_D, 1, 1])
            .arg_words(&table)
            .arg_ptr(b.qkvz)
            .arg_ptr(b.conv_w)
            .arg_ptr(b.gates)
            .arg_ptr(b.norm_w)
            .arg_ptr(b.out)
            .arg_u32(nk)
            .arg_u32(nv)
            .arg_u32(kd)
            .arg_u32(b.qkvz_stride)
            .arg_f32(l2_eps)
            .arg_f32(eps)
            .launch(stream)?;
    }
    Ok(true)
}
