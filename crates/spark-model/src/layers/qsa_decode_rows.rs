// SPDX-License-Identifier: AGPL-3.0-only

//! QSA decode at long context: two exact switches and the batched-rows
//! selection (`kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_decode_rows.cu`).
//!
//! `ATLAS_QWEN4EXP_QSA_TOPK_WIDE=1` (default off): the device radix top-k
//! serves every width. Without it, a selection over more than
//! `QSA_SELECT_MAX_BLOCKS` (16384 blocks = 65,536 tokens) took the host arm —
//! a D2H of the scores, a host sort and an H2D, i.e. a stream drain — once per
//! attention layer per decode ROW: 12 x (K+1) drains a verify step, ~1.07 ms of
//! host sort each at 77K (`scripts/dev/qwen4exp_qsa_decode_bench.cu`). The
//! kernel never had a width bound (scores are re-read from global memory per
//! pass); only the dispatch did. Same selection by construction, and pinned
//! against the host sort by the bench and `ATLAS_QSA_TOPK_VERIFY=1`.
//!
//! `ATLAS_QWEN4EXP_QSA_DECODE_ROWS=1` (default off): the R rows of one
//! sequence that a verify step serves past the bound (`multi_seq/qsa_rows.rs`)
//! select and attend in one launch per stage instead of R: q prep
//! (`qsa_qprep_rows`), scores (`qsa_score_rows_dec`), top-k
//! (`qsa_select_topk_radix_rows`) and the selected-set attention straight from
//! the paged cache (`qsa_sparse_decode_attn`, no gather). Every stage is
//! bit-identical, row for row, to what the serial `decode_select` + bs=1
//! attention computes (the bench checks each byte), and the ingest stays
//! per row, in row order, through the same projection and pooling — so the
//! indexer state after the step is the serial state and verify equals serial
//! decode. The single-row `decode_select` also scores with
//! `qsa_score_rows_dec` under this switch (bit-identical to `qsa_score`,
//! 10.5 vs 68 us at 77K). One layer's QSA work for 4 rows at 77K: 5.14 ms
//! (host arm) -> 0.77 ms (device radix, per row) -> 0.14 ms (this path).

//!
//! `ATLAS_QWEN4EXP_MTP_INDEX_SHARE=1`: `qsa_draft_share.rs`.

use std::sync::atomic::AtomicU64;

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{QsaIndexer, QsaSeqState};
use crate::layers::ops;

/// Most rows one batched selection serves; wider runs split.
pub const QSA_ROWS_MAX: usize = 16;
/// `QSD_MAX_SEL` in qsa_decode_rows.cu: the widest selection a row stages.
const QSA_SPARSE_MAX_SEL: usize = 4096;
/// Selection slots past `budget + ratio` that a shared draft selection may
/// use: its tail runs from the shared tail start (up to ratio - 1 tokens) plus
/// one token per later draft.
pub(in crate::layers) const SHARE_MARGIN: usize = 16;
/// Scores-scratch growth granule, in blocks (64 KiB of F32 per row).
const ROWS_BLOCK_GRANULE: usize = 16384;

pub(super) fn env_on(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

/// The switches and kernels of this module, resolved once per indexer.
pub(super) struct RowsPath {
    pub(super) decode_rows: bool,
    pub(super) topk_wide: bool,
    /// Bumped by every fresh `decode_select` selection into `sel_dev`.
    pub(super) sel_generation: AtomicU64,
    k_score_dec: KernelHandle,
    k_radix_rows: KernelHandle,
    k_sparse_attn: KernelHandle,
    scratch: Mutex<RowsScratch>,
}

/// Layer-owned, like every other QSA launch buffer: valid until the next
/// selection on this layer.
struct RowsScratch {
    /// [QSA_ROWS_MAX, n_heads, hd] F32 prepped queries.
    q: DevicePtr,
    /// [QSA_ROWS_MAX, sel_stride] I32 token ids.
    sel: DevicePtr,
    /// [QSA_ROWS_MAX, blocks_cap] F32, grown on demand.
    scores: DevicePtr,
    blocks_cap: usize,
}

/// One batched selection: row `r` (position `first_pos + r`) attends the
/// `n_sel(pos)` token ids at `sel + r * sel_stride * 4`.
pub struct QsaRowsSelection {
    pub sel: DevicePtr,
    pub sel_stride: u32,
    pub first_pos: u32,
    pub rows: u32,
}

impl RowsPath {
    /// The switches from the environment (module docs).
    pub(super) fn new(
        n_heads: usize,
        hd: usize,
        sel_cap: usize,
        hd_attn: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        Self::with_switches(
            env_on("ATLAS_QWEN4EXP_QSA_DECODE_ROWS"),
            env_on("ATLAS_QWEN4EXP_QSA_TOPK_WIDE"),
            [n_heads, hd, sel_cap, hd_attn],
            gpu,
        )
    }

    /// `geo`: indexer heads, indexer head_dim, selection slots, attention
    /// head_dim.
    pub(super) fn with_switches(
        decode_rows: bool,
        topk_wide: bool,
        geo: [usize; 4],
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let [n_heads, hd, sel_cap, hd_attn] = geo;
        anyhow::ensure!(
            !decode_rows
                || (sel_cap <= QSA_SPARSE_MAX_SEL
                    && hd_attn == 256
                    && n_heads <= 4
                    && hd.is_multiple_of(32)),
            "ATLAS_QWEN4EXP_QSA_DECODE_ROWS: geometry outside the kernels' envelope \
             (selection {sel_cap} <= {QSA_SPARSE_MAX_SEL}, attention head_dim {hd_attn} == 256, \
             indexer heads {n_heads} <= 4, indexer head_dim {hd} % 32 == 0)"
        );
        let [k_score_dec, k_radix_rows, k_sparse_attn] = if decode_rows {
            [
                gpu.kernel("qsa_decode_rows", "qsa_score_rows_dec")?,
                gpu.kernel("qsa_decode_rows", "qsa_select_topk_radix_rows")?,
                gpu.kernel("qsa_decode_rows", "qsa_sparse_decode_attn")?,
            ]
        } else {
            [KernelHandle(0); 3]
        };
        let scratch = if decode_rows {
            RowsScratch {
                q: gpu.alloc(QSA_ROWS_MAX * n_heads * hd * 4)?,
                sel: gpu.alloc(QSA_ROWS_MAX * sel_cap * 4)?,
                scores: DevicePtr(0),
                blocks_cap: 0,
            }
        } else {
            RowsScratch {
                q: DevicePtr(0),
                sel: DevicePtr(0),
                scores: DevicePtr(0),
                blocks_cap: 0,
            }
        };
        Ok(Self {
            decode_rows,
            topk_wide,
            sel_generation: AtomicU64::new(0),
            k_score_dec,
            k_radix_rows,
            k_sparse_attn,
            scratch: Mutex::new(scratch),
        })
    }
}

impl QsaIndexer {
    /// `ATLAS_QWEN4EXP_QSA_DECODE_ROWS`: whether the batched rows path is on.
    pub fn decode_rows_on(&self) -> bool {
        self.rows.decode_rows
    }

    /// Ingest `rows` consecutive positions of ONE sequence (row `i` at
    /// `first_pos + i`, its attention input at `normed + i * row_bytes`) and
    /// select for every row with one launch per stage. Every row must be
    /// ACTIVE (`is_active_at(first_pos)`; activity is monotone in position).
    ///
    /// The ingest is `decode_select`'s, per row and in row order (the same
    /// M=1 projection, raw-key park and pooling), so the state afterwards is
    /// the serial state. Selection then reads, for row `i`, exactly the
    /// blocks complete at its position — pooled keys never change once
    /// pooled, so pooling the later rows' blocks first changes nothing row
    /// `i` reads.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_select_rows(
        &self,
        st: &mut QsaSeqState,
        normed: DevicePtr,
        row_bytes: usize,
        first_pos: usize,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<QsaRowsSelection> {
        anyhow::ensure!(self.rows.decode_rows, "QSA rows path is off");
        anyhow::ensure!(
            (1..=QSA_ROWS_MAX).contains(&rows),
            "QSA rows: {rows} rows (1..={QSA_ROWS_MAX})"
        );
        anyhow::ensure!(
            self.is_active_at(first_pos),
            "QSA rows: first row at pos {first_pos} is inside the inert bound"
        );
        anyhow::ensure!(
            first_pos == st.ingested,
            "QSA: decode at pos {first_pos} but {} tokens ingested — the indexer \
             cache lost sync (prefix-cache skip or a rewound sequence)",
            st.ingested
        );
        let hd = self.hd as usize;
        let qkw = self.qk_width();
        for i in 0..rows {
            let pos = first_pos + i;
            self.reserve(st, pos + 1, gpu, stream)?;
            let qk_row = self.qk_scratch.offset(i * qkw * 2);
            ops::cublas_bf16_proj_dense(
                normed.offset(i * row_bytes),
                self.qk_proj_w,
                qk_row,
                1,
                qkw as u32,
                self.hidden,
                stream,
            )?;
            self.raw_room(st, 1, gpu, stream)?;
            gpu.copy_d2d_async(
                qk_row.offset(self.n_heads as usize * hd * 2),
                self.raw_slot(st, pos),
                hd * 2,
                stream,
            )?;
            st.ingested = pos + 1;
            self.pool_new_blocks(st, gpu, stream)?;
        }

        let complete_last = (first_pos + rows) / self.ratio as usize;
        let mut s = self.rows.scratch.lock();
        if s.blocks_cap < complete_last {
            let cap = complete_last.next_multiple_of(ROWS_BLOCK_GRANULE);
            if s.scores.0 != 0 {
                // An earlier step's launches may still read the old buffer.
                gpu.synchronize(stream)?;
                gpu.free(s.scores)?;
                s.scores = DevicePtr(0);
            }
            s.scores = gpu.alloc(QSA_ROWS_MAX * cap * 4)?;
            s.blocks_cap = cap;
        }
        ops::qsa_qprep_rows(
            gpu,
            self.k_qprep_rows_k,
            self.qk_scratch,
            self.q_norm_w,
            s.q,
            rows as u32,
            first_pos as u32,
            qkw as u32,
            self.n_heads,
            self.hd,
            self.rot,
            self.theta,
            self.eps,
            stream,
        )?;
        ops::qsa_score_rows_dec(
            gpu,
            self.rows.k_score_dec,
            s.q,
            st.block_keys,
            s.scores,
            rows as u32,
            complete_last as u32,
            first_pos as u32,
            s.blocks_cap as u32,
            self.ratio,
            self.n_heads,
            self.hd,
            stream,
        )?;
        let sel_stride = self.budget + self.ratio;
        ops::qsa_select_topk_radix_rows(
            gpu,
            self.rows.k_radix_rows,
            s.scores,
            s.sel,
            rows as u32,
            s.blocks_cap as u32,
            sel_stride,
            first_pos as u32,
            self.block_topk,
            self.ratio,
            stream,
        )?;
        Ok(QsaRowsSelection {
            sel: s.sel,
            sel_stride,
            first_pos: first_pos as u32,
            rows: rows as u32,
        })
    }

    /// `decode_select`'s block scores for the query at `pos` through
    /// `qsa_score_rows_dec` (one row): the same bytes as `qsa_score`.
    pub(super) fn score_dec_one(
        &self,
        st: &QsaSeqState,
        pos: usize,
        complete: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ops::qsa_score_rows_dec(
            gpu,
            self.rows.k_score_dec,
            self.q_post,
            st.block_keys,
            self.scores_dev,
            1,
            complete as u32,
            pos as u32,
            complete as u32,
            self.ratio,
            self.n_heads,
            self.hd,
            stream,
        )
    }

    /// Attention of `sel`'s rows over their selections, read straight from
    /// the paged cache: bit-identical to `qsa_gather` + the bs=1 BF16 paged
    /// decode attention per row. Row `r`'s query at `q + r * q_stride`
    /// elements; output `[rows, nq, hd]` at `out`; `nq`/`nkv` this rank's.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_rows(
        &self,
        sel: &QsaRowsSelection,
        q: DevicePtr,
        q_stride: u32,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        block_table: DevicePtr,
        out: DevicePtr,
        nq: u32,
        nkv: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ops::qsa_sparse_decode_attn(
            gpu,
            self.rows.k_sparse_attn,
            [q, k_pool, v_pool, out, block_table, sel.sel],
            [
                sel.sel_stride,
                sel.first_pos,
                self.ratio,
                self.block_topk,
                nq,
                nkv,
                self.hd_attn,
                block_size,
            ],
            inv_sqrt_d,
            q_stride,
            sel.rows,
            stream,
        )
    }
}

#[path = "qsa_draft_share.rs"]
mod qsa_draft_share;
pub use qsa_draft_share::{DraftShare, draft_share_set};

#[cfg(all(test, feature = "cuda"))]
#[path = "qsa_decode_rows_gpu_tests.rs"]
mod gpu_tests;
