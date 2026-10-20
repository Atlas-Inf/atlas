// SPDX-License-Identifier: AGPL-3.0-only
//! Exact TP2 vocabulary split of the qwen4_exp (Qwen3.8-Flash-Next) LM head
//! (`ATLAS_QWEN4EXP_LMHEAD_SPLIT=1`, default off).
//!
//! The untied BF16 head is `[vocab, hidden]` = 248077 x 2560 (1.27 GB) and
//! every rank holds all of it: at TP2 both ranks projected the full
//! vocabulary each step (`dense_gemv_bf16`, ~5.2 ms a row on GB10). Here
//! each rank projects one contiguous half of the vocabulary, through a
//! pointer offset into the resident head (no copy, no extra weight memory),
//! with the kernel the unsplit head runs for that row count, and the two
//! halves are swapped over the pair (`CommBackend::peer_exchange_async`, the
//! RDMA one-shot when it is up). Both ranks then hold the full logits rows,
//! so sampling, penalties, logprobs, grammar masks and the verify picks read
//! the same buffer they always did.
//!
//! Exactness. Every head kernel here computes output column `n` from weight
//! row `n` and the activation row alone: `dense_gemv_bf16` /
//! `dense_gemv_bf16_batchm` (fixed lane split and reduction tree per
//! output) and `dense_gemm_bf16` (strict `k = 0..K` per element). Neither
//! the column's position in the grid nor the total column count enters
//! its arithmetic, so a shard projected from `weight + start * hidden` is
//! bit-identical to columns `start..` of the full projection. Precondition:
//! the final-normed hidden rows are bitwise identical on both ranks (they
//! are replicated: every TP/EP reduction lands the same sum on both).
//! `ATLAS_QWEN4EXP_LMHEAD_SPLIT_CHECK=1` checks the assembled rows against a
//! local full projection every step (rank-local, eager steps only).
//!
//! Geometry ([`Split`]). The vocabulary is odd, so the halves cannot be
//! equal, and the exchange must be: both ranks project `width` columns,
//! rank 0 from column 0 and rank 1 from `cut` (half the vocabulary rounded
//! down to 8 columns), and swap `rows x width` BF16 values. Rank 0 keeps its
//! first `cut` columns (its last `width - cut` are projected and ignored),
//! rank 1 keeps all of its. Each side lands into disjoint column ranges.
//!
//! Collectives. One `peer_exchange_async` per [`Split::chunk_rows`] rows
//! (one per row count up to 4 rows at the default 1 MiB one-shot) is added
//! to every step that runs a split head. Only call sites both ranks run in
//! lockstep take the split (`*_tp` wrappers): the single-row decode, the
//! K=2/3/4 graphed verifies and the batched decode. Every other head (the
//! prefill last-token head, prompt logprobs, the mixed co-dispatch head,
//! the rank-0-only MTP drafter head) keeps the full projection on whichever
//! ranks run it today. Every decline is configuration both ranks share
//! (`startup_parity` carries the switch), so both ranks decline together.
//!
//! Prior art: SparkGLM's argmax-only verify split (`glm_vocab_split`) shards
//! the same way and swaps `(value, id)` pairs; this one swaps the logits so
//! every sampler stays exact.
use std::sync::OnceLock;

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::types::TransformerModel;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

const BF16: usize = 2;
/// Staging rows above the batch width: the K=4 graphed verify.
const MIN_STAGING_ROWS: usize = 4;
/// Wider heads keep the full projection (~0.5 MB of staging a row).
const MAX_STAGING_ROWS: usize = 64;

/// `ATLAS_QWEN4EXP_LMHEAD_SPLIT=1`, read once. Both ranks must agree
/// (`startup_parity`).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_LMHEAD_SPLIT").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_LMHEAD_BATCHM=1`: project several GEMV rows with ONE pass
/// over the head (`dense_gemv_bf16_batchm`, byte-identical to a
/// `dense_gemv_bf16` per row: same per-row K order and reduction tree,
/// `--fmad=false`; `bf16_batch_bitparity_microtest`). Today the K=2 verify
/// head reads the whole BF16 head once per row. Rank-local: the bytes do not
/// change, so ranks may differ.
pub(crate) fn batchm_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_LMHEAD_BATCHM").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_LMHEAD_SPLIT_VERIFY=1`: stage the split for the batched
/// verify's rows too (`verify_e`: every sequence's K = depth + 1 rows, C8 x
/// K=4 = 32), not only for `--max-batch-size` rows. Without it a batched
/// verify wider than the batch declines ("rows") and both ranks project the
/// whole 1.27 GB head, ~7 ms a C8 step on GB10 at 32 rows. Staging grows to
/// `max_batch_size x K` rows (<= 64, ~0.5 MB a row). The bytes do not
/// change (the module docs' exactness); both ranks must agree
/// (`startup_parity`), since the rows a split serves decide whether a step
/// exchanges.
pub(crate) fn verify_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_LMHEAD_SPLIT_VERIFY").as_deref() == Ok("1"))
}

/// Staging rows: the batch width, or with [`verify_requested`] the batched
/// verify's `max_batch_size x (depth + 1)`, within the module bounds.
fn staging_rows(max_batch_size: usize, verify: bool, depth: usize) -> usize {
    let k = if verify { depth.max(1) + 1 } else { 1 };
    (max_batch_size * k).clamp(MIN_STAGING_ROWS, MAX_STAGING_ROWS)
}

/// `ATLAS_QWEN4EXP_LMHEAD_SPLIT_CHECK=1`: compare every eager split head with
/// a local full projection (a full head pass and a host sync per step).
fn check_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_LMHEAD_SPLIT_CHECK").as_deref() == Ok("1"))
}

/// The arithmetic the unsplit BF16 head runs for a call shape, which a shard
/// must reproduce column for column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeadArith {
    /// `dense_gemv_bf16` a row (or its byte-identical batch-M form).
    Gemv,
    /// `dense_gemm_bf16`, the scalar tile GEMM, over all rows.
    Gemm,
}

impl HeadArith {
    /// What `lm_head_batched` runs for `rows` on a BF16 head without
    /// overlays: two GEMVs at 2 rows, else `glm_k3_head::project`, which is
    /// `dense_gemm` off GLM (`impl_a3.rs`). Under the qwen4_exp exact verify
    /// (`exact`) every verify width projects GEMV rows: serial decode's head
    /// is `dense_gemv_bf16`, and the scalar tile GEMM sums K sequentially.
    pub(crate) fn batched(rows: usize, exact: bool) -> Self {
        if rows == 2 || (exact && rows > 1) {
            Self::Gemv
        } else {
            Self::Gemm
        }
    }
}

/// The column split of a `vocab`-wide head over two ranks (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Split {
    vocab: usize,
    cut: usize,
}

impl Split {
    pub(crate) fn new(vocab: usize) -> Option<Self> {
        let cut = (vocab / 2) & !7;
        (cut > 0).then_some(Self { vocab, cut })
    }

    /// Columns each rank projects and sends a row.
    pub(crate) fn width(&self) -> usize {
        self.vocab - self.cut
    }

    /// First column `rank` projects.
    pub(crate) fn start(&self, rank: usize) -> usize {
        if rank == 0 { 0 } else { self.cut }
    }

    /// Columns of `rank`'s shard that land in the logits, from its start.
    pub(crate) fn owned(&self, rank: usize) -> usize {
        if rank == 0 { self.cut } else { self.width() }
    }

    /// Rows a single exchange carries: as many as fit `max_bytes` (the
    /// one-shot cap, so each exchange stays on it), else all of them.
    pub(crate) fn chunk_rows(&self, rows: usize, max_bytes: usize) -> usize {
        let row = self.width() * BF16;
        if max_bytes >= row {
            (max_bytes / row).min(rows).max(1)
        } else {
            rows.max(1)
        }
    }
}

/// Fixed staging for the split head, allocated at construction.
pub(crate) struct HeadSplit {
    geom: Split,
    /// `[rows, width]` BF16: this rank's shard.
    send: DevicePtr,
    /// `[rows, width]` BF16: the peer's shard.
    recv: DevicePtr,
    rows: usize,
    /// Full-vocabulary reference rows for the check (grow-only).
    check: Mutex<(DevicePtr, usize)>,
}

impl HeadSplit {
    /// Staging when the switch is on for a qwen4_exp pair, else `None`.
    pub(crate) fn alloc(
        model_type: &str,
        vocab: usize,
        comm: Option<&dyn CommBackend>,
        max_batch_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<Self>> {
        if !enabled() || model_type != "qwen4_exp" || comm.is_none_or(|c| c.world_size() != 2) {
            return Ok(None);
        }
        let geom = Split::new(vocab)
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp LM-head split: vocabulary {vocab}"))?;
        let rows = staging_rows(
            max_batch_size,
            verify_requested(),
            super::qwen4exp_mtp_depth::requested(),
        );
        let bytes = rows * geom.width() * BF16;
        let send = gpu.alloc(bytes)?;
        let recv = gpu.alloc(bytes)?;
        tracing::info!(
            cut = geom.cut,
            width = geom.width(),
            rows,
            mib = (2 * bytes) >> 20,
            "qwen4_exp LM-head vocab split staged (ATLAS_QWEN4EXP_LMHEAD_SPLIT=1)"
        );
        Ok(Some(Self {
            geom,
            send,
            recv,
            rows,
            check: Mutex::new((DevicePtr::NULL, 0)),
        }))
    }
}

/// Swap `rows` staged rows with the peer, in one-shot-sized chunks. Inside a
/// capture only the capturable one-shot may run.
pub(crate) fn exchange(
    comm: &dyn CommBackend,
    gpu: &dyn GpuBackend,
    geom: Split,
    (send, recv): (DevicePtr, DevicePtr),
    rows: usize,
    stream: u64,
) -> Result<()> {
    let row = geom.width() * BF16;
    let per = geom.chunk_rows(rows, comm.capturable_all_reduce_max_bytes());
    let capturing = gpu.stream_is_capturing(stream);
    for first in (0..rows).step_by(per) {
        let (s, r) = (send.offset(first * row), recv.offset(first * row));
        let bytes = per.min(rows - first) * row;
        if capturing {
            ensure!(
                comm.peer_exchange_capturable(s.0, r.0, bytes, stream)?,
                "qwen4_exp LM-head split: a captured step needs the RDMA one-shot \
                 (ATLAS_RDMA_ONESHOT=1) for {bytes} bytes"
            );
        } else {
            comm.peer_exchange_async(s.0, r.0, bytes, stream)?;
        }
    }
    Ok(())
}

/// Land both shards into `logits` rows `[0, rows)` (row pitch = vocabulary):
/// this rank's owned columns, then the peer's. The ranges are disjoint.
pub(crate) fn assemble(
    gpu: &dyn GpuBackend,
    geom: Split,
    rank: usize,
    (send, recv): (DevicePtr, DevicePtr),
    logits: DevicePtr,
    rows: usize,
    stream: u64,
) -> Result<()> {
    let (pitch, width) = (geom.vocab * BF16, geom.width() * BF16);
    for (q, src) in [(rank, send), (1 - rank, recv)] {
        gpu.copy_d2d_2d_async(
            src,
            width,
            logits.offset(geom.start(q) * BF16),
            pitch,
            geom.owned(q) * BF16,
            rows,
            stream,
        )?;
    }
    Ok(())
}

#[path = "qwen4exp_lmhead_split/model_impl.rs"]
mod model_impl;

#[cfg(test)]
#[path = "qwen4exp_lmhead_split_tests.rs"]
mod tests;
