// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_EXACT_DEFER=1` (default off; needs
//! `ATLAS_QWEN4EXP_EXACT_VERIFY` and `ATLAS_QWEN4EXP_BATCH_SMALL`): the exact
//! MTP verify's GDN step stores no recurrence state, and the accepted prefix
//! is replayed from H0 after the verdict.
//!
//! The storing verify (`qwen4exp_gdn_verify_fused_rows`, `qwen4exp_gdn_rows`)
//! writes, per sequence and GDN layer, the state after each of the first
//! `k - 1` tokens and the final one, so the commit of an `n`-token prefix is a
//! copy of slot `n - 1` (`commit_accepted_prefix`). At C8 x K=4 on TP=EP=2
//! (nsys, rank 0) that was 10.3 ms of verify GDN kernel a step, most of it
//! those stores, plus ~5.3 pitched copies of 56.6 MB + 2.9 MB (~3.1 ms) on
//! the commit, which the next step's drafts wait for.
//!
//! Deferred (`qwen4exp_gdn_verify_defer_rows`), the verify reads H0 and the
//! conv windows and writes only its outputs and a few KiB of staged inputs
//! (each token's raw Q|K|V conv channels and its gate and beta) into the
//! sequence's existing deferred-commit staging (`SsmLayerState::gdn_commit_*`,
//! allocated under `ATLAS_GDN_DEFERRED_COMMIT`, on by default). After the
//! verdict `qwen4exp_gdn_commit_layers` replays the accepted `n` tokens from
//! H0 over all of the sequence's GDN layers in one launch, with the functions
//! the verify ran on the same inputs, so the committed recurrence and conv
//! bytes are the storing kernel's slot `n - 1` (its final state when
//! `n == k`). `scripts/dev/qwen4exp_gdn_defer_bench.cu` (defer-check) checks
//! that for 1..12 sequences, every k in 1..8 and every accepted length.
//! GB10, TP2 shapes, 8 sequences: the verify 351.5 -> 87.6 us a layer at
//! K=4; the commit ~0.6 ms a sequence (every sequence, full accepts too)
//! against ~0.5 ms of copies per partially accepted one.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::qwen4exp_decode_fuse::{GDN_D, aligned, gdn_geometry_fits};
use super::qwen4exp_gdn_rows::{GDN_ROWS_MAX, GDN_VERIFY_KMAX};

/// `ATLAS_QWEN4EXP_EXACT_DEFER=1`, read once.
pub fn exact_defer_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_EXACT_DEFER").as_deref() == Ok("1"))
}

/// `(verify_defer, commit)` kernel handles, 0 where the target lacks them.
pub fn defer_kernels(gpu: &dyn GpuBackend) -> (KernelHandle, KernelHandle) {
    static K: OnceLock<(KernelHandle, KernelHandle)> = OnceLock::new();
    *K.get_or_init(|| {
        let k = |name| crate::layers::try_kernel(gpu, "qwen4exp_decode_fuse", name);
        (
            k("qwen4exp_gdn_verify_defer_rows"),
            k("qwen4exp_gdn_commit_layers"),
        )
    })
}

/// One sequence of a deferred verify: its H0 and conv windows (read only),
/// its staging (`[k, conv_dim]` BF16 inputs, `[k, 2 * nv]` FP32 gate | beta)
/// and its rows `row0..row0 + k` of the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnDeferSeq {
    pub h: DevicePtr,
    pub conv: DevicePtr,
    pub stage_qkv: DevicePtr,
    pub stage_gb: DevicePtr,
    pub row0: u32,
    pub k: u32,
    /// `ATLAS_QWEN4EXP_GDN_COMMIT_FUSE`: the sequence's pending-commit word
    /// (the previous step's accepted tokens, replayed and stored first;
    /// `model/.../gdn_commit_fuse.rs`), NULL when off.
    pub fuse_n: DevicePtr,
}

/// The step buffers, as [`super::qwen4exp_gdn_rows::GdnVerifyRows`].
pub struct GdnDeferRows<'a> {
    pub seqs: &'a [GdnDeferSeq],
    pub qkvz: DevicePtr,
    pub qkvz_stride: u32,
    pub conv_w: DevicePtr,
    pub gates: DevicePtr,
    pub norm_w: DevicePtr,
    pub out: DevicePtr,
}

/// `QdfDeferRows`: six words a sequence (`h, conv, stage_qkv, stage_gb,
/// row0 | k << 32, fuse_n`), `GDN_ROWS_MAX` sequences, unused ones zero.
pub(crate) fn defer_table(seqs: &[GdnDeferSeq]) -> [u64; 6 * GDN_ROWS_MAX] {
    let mut t = [0u64; 6 * GDN_ROWS_MAX];
    for (q, w) in seqs.iter().zip(t.chunks_mut(6)) {
        w.copy_from_slice(&[
            q.h.0,
            q.conv.0,
            q.stage_qkv.0,
            q.stage_gb.0,
            u64::from(q.row0) | u64::from(q.k) << 32,
            q.fuse_n.0,
        ]);
    }
    t
}

/// The deferred verify of `b.seqs` in launches of up to [`GDN_ROWS_MAX`]
/// sequences. Returns whether it launched; on `false` (kernel missing, a
/// shape or alignment the kernel does not take) nothing was launched and the
/// caller must not leave its sequences pending.
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_defer_rows(
    gpu: &dyn GpuBackend,
    b: &GdnDeferRows<'_>,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    d_conv: u32,
    l2_eps: f32,
    eps: f32,
    stream: u64,
) -> Result<bool> {
    let (k, _) = defer_kernels(gpu);
    let conv_dim = 2 * nk * kd + nv * vd;
    let fits = k.0 != 0
        && !b.seqs.is_empty()
        && gdn_geometry_fits(nk, nv, kd, vd, d_conv, 8)
        && b.seqs.iter().all(|q| {
            (1..=GDN_VERIFY_KMAX as u32).contains(&q.k)
                && aligned(&[q.conv], 16)
                && !q.stage_qkv.is_null()
                && aligned(&[q.stage_gb], 4)
                && !q.stage_gb.is_null()
        })
        && b.qkvz_stride.is_multiple_of(4)
        && aligned(&[b.qkvz.offset(conv_dim as usize * 2), b.norm_w, b.out], 8);
    if !fits {
        return Ok(false);
    }
    for seqs in b.seqs.chunks(GDN_ROWS_MAX) {
        KernelLaunch::new(gpu, k)
            .grid([nv, seqs.len() as u32, 1])
            .block([GDN_D, 1, 1])
            .arg_words(&defer_table(seqs))
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

/// Layers one `qwen4exp_gdn_commit_layers` launch takes (`QDF_COMMIT_LAYERS`).
pub const GDN_COMMIT_LAYERS: usize = 48;

/// One GDN layer of the sequence being committed: its live state (updated in
/// place), what the deferred verify staged for it, and its conv weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnCommitLayer {
    pub h: DevicePtr,
    pub conv: DevicePtr,
    pub stage_qkv: DevicePtr,
    pub stage_gb: DevicePtr,
    pub conv_w: DevicePtr,
}

/// `QdfCommitLayers`: five words a layer, `GDN_COMMIT_LAYERS` layers.
pub(crate) fn commit_table(layers: &[GdnCommitLayer]) -> Vec<u64> {
    let mut t = vec![0u64; 5 * GDN_COMMIT_LAYERS];
    for (l, w) in layers.iter().zip(t.chunks_mut(5)) {
        w.copy_from_slice(&[l.h.0, l.conv.0, l.stage_qkv.0, l.stage_gb.0, l.conv_w.0]);
    }
    t
}

/// Replay the first `n_tokens` staged tokens into every layer of `layers`, in
/// launches of up to [`GDN_COMMIT_LAYERS`]. Errors rather than skipping: a
/// sequence whose verify deferred has no other way to its committed state.
#[allow(clippy::too_many_arguments)]
pub fn gdn_commit_layers(
    gpu: &dyn GpuBackend,
    layers: &[GdnCommitLayer],
    n_tokens: u32,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    d_conv: u32,
    l2_eps: f32,
    stream: u64,
) -> Result<()> {
    let (_, k) = defer_kernels(gpu);
    anyhow::ensure!(
        k.0 != 0 && gdn_geometry_fits(nk, nv, kd, vd, d_conv, 8),
        "deferred GDN commit: qwen4exp_gdn_commit_layers unavailable for nk={nk} nv={nv}"
    );
    anyhow::ensure!(
        (1..=GDN_VERIFY_KMAX as u32).contains(&n_tokens),
        "deferred GDN commit: {n_tokens} tokens past the staged window"
    );
    for chunk in layers.chunks(GDN_COMMIT_LAYERS) {
        KernelLaunch::new(gpu, k)
            .grid([nv, chunk.len() as u32, 1])
            .block([GDN_D, 1, 1])
            .arg_words(&commit_table(chunk))
            .arg_u32(n_tokens)
            .arg_u32(nk)
            .arg_u32(nv)
            .arg_u32(kd)
            .arg_f32(l2_eps)
            .launch(stream)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "qwen4exp_gdn_defer_tests.rs"]
mod tests;
