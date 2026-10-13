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

impl TransformerModel {
    /// Project `rows` rows of `input` [rows, H] onto head columns
    /// `first..first + n` into `out` (row pitch `pitch` elements) with
    /// `arith`. The single source of the BF16 head's GEMV/GEMM launches for
    /// the split, its check and the unsplit 2-row head.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn head_project(
        &self,
        arith: HeadArith,
        input: DevicePtr,
        first: usize,
        n: usize,
        rows: usize,
        out: DevicePtr,
        pitch: usize,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let weight = DenseWeight {
            weight: self.lm_head_weight.weight.offset(first * h * BF16),
        };
        let gpu = self.gpu.as_ref();
        let (rows32, n32, h32) = (rows as u32, n as u32, h as u32);
        match arith {
            HeadArith::Gemm => {
                ensure!(pitch == n, "dense_gemm head writes rows {n} apart");
                ops::dense_gemm(
                    gpu,
                    self.dense_gemm_kernel,
                    input,
                    &weight,
                    out,
                    rows32,
                    n32,
                    h32,
                    stream,
                )
            }
            HeadArith::Gemv
                if rows > 1
                    && self.dense_gemv_batchm_kernel.0 != 0
                    && self.config.model_type == "qwen4_exp"
                    && (batchm_requested()
                        || self.levers.qwen4exp_exact_verify
                        || self.levers.qwen4exp_batch_fast)
                    && !self.levers.batch_bisect(ops::BISECT_HEAD) =>
            {
                // Up to 32 rows a pass over the head (8 without the lane).
                self.qwen4exp_wide_rows.dense_rows(
                    gpu,
                    self.dense_gemv_batchm_kernel,
                    input,
                    &weight,
                    out,
                    (rows32, n32, h32, pitch as u32),
                    stream,
                )
            }
            HeadArith::Gemv => (0..rows).try_for_each(|r| {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_kernel,
                    input.offset(r * h * BF16),
                    &weight,
                    out.offset(r * pitch * BF16),
                    n32,
                    h32,
                    stream,
                )
            }),
        }
    }

    /// Why the split would not serve `rows` rows, if it would not. Every term
    /// is configuration both ranks share.
    fn lmhead_split_decline(&self, split: &HeadSplit, rows: usize) -> Option<&'static str> {
        let Some(comm) = self.comm.as_ref() else {
            return Some("no_comm");
        };
        [
            (!comm.supports_peer_exchange_async(), "peer_exchange"),
            (self.lm_head_fp8.is_some(), "lm_head_fp8"),
            (self.lm_head_nvfp4.is_some(), "lm_head_nvfp4"),
            (self.lm_head_weight.weight.is_null(), "no_bf16_head"),
            (self.overlays.is_some(), "overlays"),
            (
                self.logit_softcap_kernel.0 != 0 || self.logit_softcap_fp32_kernel.0 != 0,
                "softcap",
            ),
            (self.use_fp32_logits, "fp32_logits"),
            (
                Split::new(self.config.vocab_size) != Some(split.geom),
                "vocab",
            ),
            (rows == 0 || rows > split.rows, "rows"),
        ]
        .into_iter()
        .find_map(|(hit, why)| hit.then_some(why))
    }

    /// The split head: `normed` [rows, H] -> `self.buffers.logits()` rows
    /// `[0, rows)`, bit-identical to the unsplit head of shape `arith`.
    /// `Ok(false)`, nothing launched: the caller runs the full head.
    fn qwen4exp_split_head(
        &self,
        normed: DevicePtr,
        rows: usize,
        arith: HeadArith,
        stream: u64,
    ) -> Result<bool> {
        let Some(split) = self.lmhead_split.as_ref() else {
            return Ok(false);
        };
        if let Some(why) = self.lmhead_split_decline(split, rows) {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::warn!("qwen4_exp LM-head split declined ({why}, rows={rows}): full head")
            });
            return Ok(false);
        }
        let comm = self.comm.as_deref().expect("declined without comm");
        let (geom, rank) = (split.geom, comm.rank());
        let staging = (split.send, split.recv);
        let logits = self.buffers.logits();
        let width = geom.width();
        self.head_project(
            arith,
            normed,
            geom.start(rank),
            width,
            rows,
            split.send,
            width,
            stream,
        )?;
        exchange(comm, self.gpu.as_ref(), geom, staging, rows, stream)?;
        assemble(self.gpu.as_ref(), geom, rank, staging, logits, rows, stream)?;
        static ACTIVE: std::sync::Once = std::sync::Once::new();
        ACTIVE.call_once(|| {
            tracing::info!(
                rank,
                start = geom.start(rank),
                width,
                "qwen4_exp LM-head vocab split active"
            )
        });
        if check_requested() && !self.gpu.stream_is_capturing(stream) {
            self.check_split_head(split, arith, normed, rows, stream)?;
        }
        Ok(true)
    }

    /// `ATLAS_QWEN4EXP_LMHEAD_SPLIT_CHECK=1`: the full projection, locally,
    /// against the assembled rows.
    fn check_split_head(
        &self,
        split: &HeadSplit,
        arith: HeadArith,
        normed: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let v = self.config.vocab_size;
        let bytes = rows * v * BF16;
        let mut check = split.check.lock();
        if check.1 < bytes {
            if !check.0.is_null() {
                self.gpu.free(check.0)?;
            }
            *check = (self.gpu.alloc(bytes)?, bytes);
        }
        self.head_project(arith, normed, 0, v, rows, check.0, v, stream)?;
        self.gpu.synchronize(stream)?;
        let (mut got, mut want) = (vec![0u8; bytes], vec![0u8; bytes]);
        self.gpu.copy_d2h(self.buffers.logits(), &mut got)?;
        self.gpu.copy_d2h(check.0, &mut want)?;
        let differ = got
            .chunks_exact(BF16)
            .zip(want.chunks_exact(BF16))
            .enumerate()
            .filter(|(_, (a, b))| a != b);
        let (count, first) = differ.fold((0usize, None), |(n, first), (i, _)| {
            (n + 1, first.or(Some((i / v, i % v))))
        });
        static STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let steps = STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        match first {
            Some((row, col)) => tracing::error!(
                rows,
                ?arith,
                differ = count,
                row,
                col,
                "qwen4_exp LM-head split CHECK: assembled logits differ from the full head \
                 (hidden not replicated across ranks, or a shard kernel is not column-local)"
            ),
            None if steps.is_power_of_two() => tracing::info!(
                steps,
                rows,
                ?arith,
                "qwen4_exp LM-head split CHECK: bit-identical to the full head"
            ),
            None => {}
        }
        Ok(())
    }

    /// `lm_head` (single-row decode), split when admitted.
    pub(super) fn lm_head_tp(&self, normed: DevicePtr, stream: u64) -> Result<()> {
        if !self.qwen4exp_split_head(normed, 1, HeadArith::Gemv, stream)? {
            self.lm_head(normed, stream)?;
        }
        Ok(())
    }

    /// `lm_head_batched` into `self.buffers.logits()`, split when admitted.
    pub(super) fn lm_head_batched_tp(
        &self,
        normed: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        // ATLAS_QWEN4EXP_BATCH_FAST: every row the single-row head's bytes.
        let exact = self.levers.qwen4exp_batch_fast;
        let arith = if exact {
            HeadArith::Gemv
        } else {
            HeadArith::batched(rows, self.levers.qwen4exp_exact_verify)
        };
        if self.qwen4exp_split_head(normed, rows, arith, stream)? {
            return Ok(());
        }
        if exact && self.plain_bf16_head() {
            let (logits, v) = (self.buffers.logits(), self.config.vocab_size);
            return self.head_project(arith, normed, 0, v, rows, logits, v, stream);
        }
        self.lm_head_batched(normed, rows as u32, self.buffers.logits(), stream)?;
        Ok(())
    }

    /// `lm_head_project_batched` (the batched decode head: `dense_gemm` on a
    /// BF16 head), split when admitted.
    ///
    /// `ATLAS_QWEN4EXP_BATCH_FAST`: the rows take the single-row decode head's
    /// arithmetic instead (`dense_gemv_bf16` a row, 8 rows a pass over the
    /// head), split or not, so each row's logits are C1's byte for byte.
    pub(super) fn lm_head_project_batched_tp(
        &self,
        normed: DevicePtr,
        padded_n: usize,
        h: usize,
        bf16: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let exact = self.levers.qwen4exp_batch_fast;
        let arith = if exact {
            HeadArith::Gemv
        } else {
            HeadArith::Gemm
        };
        if self.qwen4exp_split_head(normed, padded_n, arith, stream)? {
            return Ok(self.buffers.logits());
        }
        if exact && self.plain_bf16_head() {
            let (logits, v) = (self.buffers.logits(), self.config.vocab_size);
            self.head_project(arith, normed, 0, v, padded_n, logits, v, stream)?;
            return Ok(logits);
        }
        self.lm_head_project_batched(normed, padded_n, h, bf16, stream)
    }

    /// The head is the plain resident BF16 projection `lm_head` runs with
    /// `dense_gemv_bf16` (no FP8/NVFP4 copy, overlay, softcap or FP32 logits).
    fn plain_bf16_head(&self) -> bool {
        self.lm_head_fp8.is_none()
            && self.lm_head_nvfp4.is_none()
            && !self.lm_head_weight.weight.is_null()
            && self.overlays.is_none()
            && self.logit_softcap_kernel.0 == 0
            && self.logit_softcap_fp32_kernel.0 == 0
            && !self.use_fp32_logits
    }
}

#[cfg(test)]
#[path = "qwen4exp_lmhead_split_tests.rs"]
mod tests;
