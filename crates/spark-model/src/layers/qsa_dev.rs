// SPDX-License-Identifier: AGPL-3.0-only
//! Device-resident QSA decode/verify: the default (`ATLAS_QSA_DEVICE=0` = host arm).
//!
//! The host arm (`decode_select`) takes each row's position as a launch
//! argument, sizes its grids from it and, by default, sorts the block scores
//! on the host: at 131K that is a D2H + sort + H2D per layer per verify row,
//! and none of it can live in a CUDA graph. This arm does the whole step on
//! the device for R consecutive rows of ONE sequence:
//!
//!   ingest   per row: qk GEMV (M=1, as before) + `qsa_decode_ingest`
//!            (position from `pos_dev`; stores the raw key, pools a block
//!            the row closes — `qsa_block_pool`'s arithmetic)
//!   q prep   `qsa_qprep_rows_dev`   — `qsa_qprep`'s expressions
//!   score    `qsa_score_rows_devN`  — exact-tree contraction, bit-identical
//!            to `qsa_score`; the block keys are read once for all R rows
//!   select   `qsa_select_rows_dev`  — identity while inert, else the host
//!            arm's selection (radix select, same total order)
//!   attend   `paged_decode_attn_sel` — `paged_decode_attn` walked over the
//!            selection straight from the paged cache (no gather), with the
//!            gather arm's scratch-space batching
//!
//! Positions come from device memory and every grid is fixed, so one
//! captured graph serves every position, inert or active. Output is the
//! host arm's, bit for bit: same scores, same selected set in the same
//! (ascending) order, same attention arithmetic in the same order — and
//! while inert, the dense kernel's.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{QsaIndexer, QsaSeqState};
use crate::layers::ops;

/// Most rows one call serves (a K=8 verify window).
pub const QSA_DEV_ROWS_MAX: usize = 8;
/// Fixed grid of the scorer: grid-stride over 64-block tiles.
const QSA_DEV_SCORE_GRID: u32 = 192;

/// On unless `ATLAS_QSA_DEVICE=0` (the host arm, kept for bisects) — read once.
pub fn device_mode() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QSA_DEVICE").as_deref() != Ok("0"))
}

pub struct QsaDev {
    k_ingest: KernelHandle,
    k_qprep: KernelHandle,
    k_score: [KernelHandle; 3], // bm = 1, 2, 4
    k_select: KernelHandle,
    k_attn: KernelHandle,
    qk_rows: DevicePtr, // [ROWS_MAX, qkw] BF16
    q_rows: DevicePtr,  // [ROWS_MAX, n_heads, hd] F32
    scores: DevicePtr,  // [ROWS_MAX, score_stride] F32
    sel: DevicePtr,     // [ROWS_MAX, sel_stride] i32
    nsel: DevicePtr,    // [ROWS_MAX] i32
    score_stride: u32,
    sel_stride: u32,
}

impl QsaDev {
    /// `None` (with a log line) when this geometry is not served: the
    /// selected-attention kernel is built for 256-wide attention heads and
    /// the scorer replays a 32-lane reduction tree.
    pub(super) fn new(ix: &QsaIndexer, gpu: &dyn GpuBackend) -> Result<Option<Self>> {
        if !device_mode() {
            return Ok(None);
        }
        let (n_heads, hd) = (ix.n_heads, ix.hd);
        if ix.hd_attn != 256 || !hd.is_multiple_of(32) {
            tracing::warn!(
                "QSA device arm: attention head_dim {} / indexer hd {} not served; host arm kept",
                ix.hd_attn,
                hd
            );
            return Ok(None);
        }
        let qkw = ix.qk_width();
        let score_stride = (ix.max_tokens / ix.ratio as usize).max(1);
        let sel_stride = (ix.budget + ix.ratio) as usize;
        let r = QSA_DEV_ROWS_MAX;
        let k = |m: &str, f: &str| gpu.kernel(m, f);
        let dev = Self {
            k_ingest: k("qsa_indexer", "qsa_decode_ingest")?,
            k_qprep: k("qsa_indexer", "qsa_qprep_rows_dev")?,
            k_score: [
                k("qsa_indexer", "qsa_score_rows_dev1")?,
                k("qsa_indexer", "qsa_score_rows_dev2")?,
                k("qsa_indexer", "qsa_score_rows_dev4")?,
            ],
            k_select: k("qsa_indexer", "qsa_select_rows_dev")?,
            k_attn: k("qsa_sel_attn", "paged_decode_attn_sel")?,
            qk_rows: gpu.alloc(r * qkw * 2)?,
            q_rows: gpu.alloc(r * (n_heads * hd) as usize * 4)?,
            scores: gpu.alloc(r * score_stride * 4)?,
            sel: gpu.alloc(r * sel_stride * 4)?,
            nsel: gpu.alloc(r * 4)?,
            score_stride: score_stride as u32,
            sel_stride: sel_stride as u32,
        };
        tracing::info!(
            "QSA device arm (default; ATLAS_QSA_DEVICE=0 = host arm): capturable selection, \
             {} rows x {} blocks of scores ({:.1} MB)",
            r,
            score_stride,
            (r * score_stride * 4) as f64 / 1e6
        );
        Ok(Some(dev))
    }
}

/// Where the attention reads and writes, for [`QsaIndexer::dev_decode_rows`].
pub struct QsaDevAttn {
    /// Row 0's Q; row i at `q + i * q_stride` elements.
    pub q: DevicePtr,
    pub q_stride: u32,
    /// Row i's output at `out + i * nq * 256` elements.
    pub out: DevicePtr,
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    /// Row i's table at `block_table + i * max_blocks_per_seq` entries.
    pub block_table: DevicePtr,
    pub max_blocks_per_seq: u32,
    pub nq: u32,
    pub nkv: u32,
    pub block_size: u32,
    pub inv_sqrt_d: f32,
}

impl QsaIndexer {
    pub(super) fn with_dev(mut self, gpu: &dyn GpuBackend) -> Result<Self> {
        self.dev = QsaDev::new(&self, gpu)?;
        Ok(self)
    }

    pub fn dev_enabled(&self) -> bool {
        self.dev.is_some()
    }

    /// Ingest, select and attend `rows` consecutive positions
    /// `first_pos..first_pos + rows` of ONE sequence. `normed` row i at
    /// `normed + i * normed_stride` bytes; `pos_dev[i]` must hold
    /// `first_pos + i` (the attention metadata's positions).
    #[allow(clippy::too_many_arguments)]
    pub fn dev_decode_rows(
        &self,
        st: &mut QsaSeqState,
        normed: DevicePtr,
        normed_stride: usize,
        first_pos: usize,
        rows: usize,
        pos_dev: DevicePtr,
        a: &QsaDevAttn,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let dev = self
            .dev
            .as_ref()
            .context("QSA device arm not initialised")?;
        anyhow::ensure!(
            (1..=QSA_DEV_ROWS_MAX).contains(&rows),
            "QSA device arm: {rows} rows (max {QSA_DEV_ROWS_MAX})"
        );
        // A re-run of the same step after a failed capture is idempotent.
        anyhow::ensure!(
            first_pos == st.ingested || first_pos + rows == st.ingested,
            "QSA device arm at pos {first_pos} but {} tokens ingested",
            st.ingested
        );
        anyhow::ensure!(
            first_pos + rows <= self.max_tokens,
            "QSA: pos {} >= indexer capacity {} — it derives from --max-seq-len",
            first_pos + rows - 1,
            self.max_tokens
        );
        let hd = self.hd as usize;
        let qkw = self.qk_width();
        for i in 0..rows {
            let qk_i = dev.qk_rows.offset(i * qkw * 2);
            ops::cublas_bf16_proj_dense(
                normed.offset(i * normed_stride),
                self.qk_proj_w,
                qk_i,
                1,
                qkw as u32,
                self.hidden,
                stream,
            )
            .context("QSA qk projection (device arm)")?;
            ops::qsa_decode_ingest(
                gpu,
                dev.k_ingest,
                qk_i.offset(self.n_heads as usize * hd * 2),
                pos_dev.offset(i * 4),
                st.raw_keys,
                self.k_norm_w,
                st.block_keys,
                self.ratio,
                self.hd,
                self.rot,
                self.theta,
                self.eps,
                stream,
            )?;
        }
        ops::qsa_qprep_rows_dev(
            gpu,
            dev.k_qprep,
            dev.qk_rows,
            self.q_norm_w,
            dev.q_rows,
            pos_dev,
            rows as u32,
            qkw as u32,
            self.n_heads,
            self.hd,
            self.rot,
            self.theta,
            self.eps,
            stream,
        )?;
        let qrow = (self.n_heads * self.hd) as usize * 4;
        let mut r0 = 0usize;
        while r0 < rows {
            let c = (rows - r0).min(4);
            let (bm, ki) = match c {
                1 => (1u32, 0usize),
                2 => (2, 1),
                _ => (4, 2),
            };
            ops::qsa_score_rows_dev(
                gpu,
                dev.k_score[ki],
                bm,
                QSA_DEV_SCORE_GRID,
                dev.q_rows.offset(r0 * qrow),
                st.block_keys,
                dev.scores.offset(r0 * dev.score_stride as usize * 4),
                pos_dev.offset(r0 * 4),
                dev.score_stride,
                self.ratio,
                self.n_heads,
                self.hd,
                c as u32,
                self.block_topk,
                stream,
            )?;
            r0 += c;
        }
        ops::qsa_select_rows_dev(
            gpu,
            dev.k_select,
            dev.scores,
            dev.sel,
            dev.nsel,
            pos_dev,
            rows as u32,
            dev.score_stride,
            dev.sel_stride,
            self.block_topk,
            self.ratio,
            stream,
        )?;
        ops::paged_decode_attn_sel(
            gpu,
            dev.k_attn,
            a.q,
            a.k_pool,
            a.v_pool,
            a.out,
            a.block_table,
            dev.sel,
            dev.nsel,
            dev.sel_stride,
            a.max_blocks_per_seq,
            rows as u32,
            a.nq,
            a.nkv,
            self.hd_attn,
            a.block_size,
            a.inv_sqrt_d,
            a.q_stride,
            stream,
        )?;
        st.ingested = first_pos + rows;
        st.pooled = st.ingested / self.ratio as usize;
        Ok(())
    }
}
