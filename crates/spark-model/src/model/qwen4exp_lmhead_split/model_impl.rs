// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use super::{
    BF16, HeadArith, HeadSplit, Split, assemble, batchm_requested, check_requested, exchange,
    staging_rows, verify_requested,
};
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl TransformerModel {
    /// Project `rows` rows of `input` [rows, H] onto head columns
    /// `first..first + n` into `out` (row pitch `pitch` elements) with
    /// `arith`. The single source of the BF16 head's GEMV/GEMM launches for
    /// the split, its check and the unsplit 2-row head.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::model) fn head_project(
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
    pub(in crate::model) fn lm_head_tp(&self, normed: DevicePtr, stream: u64) -> Result<()> {
        if !self.qwen4exp_split_head(normed, 1, HeadArith::Gemv, stream)? {
            self.lm_head(normed, stream)?;
        }
        Ok(())
    }

    /// `lm_head_batched` into `self.buffers.logits()`, split when admitted.
    pub(in crate::model) fn lm_head_batched_tp(
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
    pub(in crate::model) fn lm_head_project_batched_tp(
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
