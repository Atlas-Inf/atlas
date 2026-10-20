// SPDX-License-Identifier: AGPL-3.0-only

//! The Qwen3.8-Flash-Next QSA indexer — decode-side sparse-attention
//! selection (#753 phase G).
//!
//! Reference: `Qwen4ExpTextQSAIndexer`. The attention layer's INPUT (the
//! hc_pre mixed output) is projected to 4 query heads + 1 raw key per token;
//! the visible prefix is grouped into 4-token blocks whose keys are
//! mean-pooled, k_layernormed and roped at the block's first position; each
//! query attends the top-512 blocks by `sum_h relu(q_h . k_b)/sqrt(128)`,
//! plus the incomplete tail. At or below `budget + ratio - 1` (2051) visible
//! tokens the selection is PROVABLY all-visible — the inert regime the port
//! served in until now.
//!
//! SCOPE: raw keys are ingested during prefill and decode. At DECODE steps
//! past the inert bound, `decode_select` feeds the EXISTING paged decode
//! attention through a gathered contiguous scratch + identity block table;
//! PREFILL rows past the bound get a per-query selection from
//! `prefill_select` (`qsa_select.rs`, both prefill paths). BF16 KV only. The
//! launch scratch is layer-owned and steps serialize on one stream, so a
//! `QsaSelection` is valid only until the next `decode_select` on this layer.
//!
//! CUDA graphs: the ingest counter is host state and launch parameters depend
//! on the position, so an indexer vetoes decode-graph capture, except for an
//! inert step whose ingest is staged (`qsa_staged.rs`). Top-k runs on the
//! device by default (`qsa_select_topk_radix`, identical selection to the host
//! `rank_cmp`) to `QSA_SELECT_MAX_BLOCKS` (every width with
//! `ATLAS_QWEN4EXP_QSA_TOPK_WIDE=1`); `ATLAS_QSA_DEVICE_TOPK=0`: host sort.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;

#[path = "qsa_aux.rs"]
mod qsa_aux;
#[path = "qsa_decode_rows.rs"]
mod qsa_decode_rows;
pub use qsa_decode_rows::{QSA_ROWS_MAX, QsaRowsSelection, draft_share_set};
#[path = "qsa_decode_select.rs"]
mod qsa_decode_select;
#[path = "qsa_free.rs"]
mod qsa_free;
#[path = "qsa_key_only.rs"]
mod qsa_key_only;
#[path = "qsa_select.rs"]
mod qsa_select;
#[path = "qsa_select_sp.rs"]
pub mod qsa_select_sp;
#[path = "qsa_staged.rs"]
mod qsa_staged;
pub use qsa_staged::{CommitEntry, CommitTable, StagedIngest, staged_ingest};
#[path = "qsa_select_stages.rs"]
mod qsa_select_stages;
#[path = "qsa_window.rs"]
mod qsa_window;
#[cfg(all(test, feature = "cuda"))]
#[path = "qsa_tests.rs"]
mod tests;

/// Kill switch `ATLAS_QSA_NO_PREFILL_SELECT=1` (read once): stage-1 behavior,
/// dense prefill past the inert bound (decode still selects). It skips the
/// `ATLAS_QWEN4EXP_PREFILL_QSA_SPLIT` exchanges, so both ranks must agree
/// (`model::startup_parity`).
pub fn no_prefill_select() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("ATLAS_QSA_NO_PREFILL_SELECT").as_deref() == Ok("1"))
}

/// One decode step's selection: contiguous NHD `k/v` scratch + identity table.
pub struct QsaSelection {
    pub k_scratch: DevicePtr,
    pub v_scratch: DevicePtr,
    pub table_dev: DevicePtr,
    pub seq_len_dev: DevicePtr,
    pub n_sel: u32,
    pub max_blocks: u32,
}

pub struct QsaSeqState {
    /// Tokens whose raw keys have been ingested (contiguous from 0).
    ingested: usize,
    /// Complete 4-token blocks pooled into `block_keys`.
    pooled: usize,
    /// Identity block table upload done (needs block_size, known lazily).
    table_len: usize,
    /// Positions `block_keys` can hold; grown on demand (`qsa_free.rs`).
    cap: usize,
    /// [cap/ratio, hd] BF16 — this sequence's pooled block keys.
    block_keys: DevicePtr,
    /// Raw keys still to be pooled, plus a rewind margin (`qsa_window.rs`).
    raw: qsa_window::RawWindow,
    /// MTP drafter selection reuse (`qsa_draft_share.rs`); `Off` elsewhere.
    share: qsa_decode_rows::DraftShare,
}

pub struct QsaIndexer {
    qk_proj_w: DevicePtr, // [ (n_heads+1)*hd, hidden ] BF16 row-major
    q_norm_w: DevicePtr,  // [hd]
    k_norm_w: DevicePtr,  // [hd]

    n_heads: u32,
    hd: u32,
    ratio: u32,
    budget: u32,
    block_topk: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    hidden: u32,
    nkv_attn: u32,
    hd_attn: u32,
    max_tokens: usize,

    k_pool_k: KernelHandle,
    /// `qsa_commit_table` (`qsa_staged.rs`), 0 where the target lacks it.
    k_commit_table_k: KernelHandle,
    k_qprep_k: KernelHandle,
    k_score_k: KernelHandle,
    k_gather_k: KernelHandle,
    k_qprep_rows_k: KernelHandle,
    k_score_rows_k: KernelHandle,
    /// Per-row top-k block selection, on the GPU. Replaces a D2H of the
    /// whole score matrix plus a host sort per row; see `qsa_topk_rows`.
    k_topk_rows_k: KernelHandle,
    /// Tiled scorer: QSA_SR_B outputs per block, bit-identical to
    /// `k_score_rows_k`. See `qsa_score_rows_b`.
    #[allow(dead_code)] // kept as the fallback below the exact-tree scorer
    k_score_rows_b_k: KernelHandle,
    /// One thread per score, BIT-IDENTICAL to `k_score_rows_k`: it replays
    /// the reference reduction tree locally. See `qsa_score_rows_exact`.
    k_score_rows_exact_k: KernelHandle,
    k_score_rows_gemm_k: KernelHandle,
    k_score_rows_tc_k: KernelHandle,
    k_prefill_attn_k: KernelHandle,
    k_select_k: KernelHandle,
    /// `ATLAS_QSA_DEVICE_TOPK=1`: select on the device (no per-layer host
    /// round trip); `ATLAS_QSA_TOPK_VERIFY=1` also runs the host reference
    /// and fails on the first mismatch.
    device_topk: bool,
    topk_verify: bool,
    /// `ATLAS_QWEN4EXP_QSA_TOPK_WIDE` / `_DECODE_ROWS` (`qsa_decode_rows.rs`).
    rows: qsa_decode_rows::RowsPath,
    /// `QSA_PA_G` q-heads per block. Same math, one K/V read per group
    /// instead of per head; see `ops::qsa_prefill_attn_grouped_ok`.
    k_prefill_attn_g_k: KernelHandle,
    k_prefill_attn_l8_k: KernelHandle,
    k_prefill_attn_tc_k: KernelHandle,
    k_prefill_attn_tc2_k: KernelHandle,
    k_prefill_attn_tc3_k: KernelHandle,

    qk_scratch: DevicePtr, // [INGEST_SLAB, (n_heads+1)*hd] BF16
    q_post: DevicePtr,     // [n_heads, hd] F32
    scores_dev: DevicePtr, // [max_tokens/ratio] F32
    sel_dev: DevicePtr,    // [budget + ratio] i32
    k_scratch: DevicePtr,  // [budget+ratio, nkv_attn, hd_attn] BF16
    v_scratch: DevicePtr,
    table_dev: DevicePtr,   // [ceil((budget+ratio)/8)] i32 (any block_size >= 8)
    seq_len_dev: DevicePtr, // [1] i32
    /// The sequence's REAL block table, uploaded per prefill-select call —
    /// chunk-0 metadata carries no device table (cache-skip attention is
    /// contiguous), so the host Vec is the source of truth.
    prefill_table_dev: DevicePtr, // [ceil(max_tokens/8)] i32
}

/// Prefill ingest GEMM slab (rows), bounding `qk_scratch`.
const INGEST_SLAB: usize = 2048;

impl QsaIndexer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qk_proj_w: DevicePtr,
        q_norm_w: DevicePtr,
        k_norm_w: DevicePtr,
        n_heads: usize,
        hd: usize,
        ratio: usize,
        budget: usize,
        max_seq_len: usize,
        rot: usize,
        theta: f32,
        eps: f32,
        hidden: usize,
        nkv_attn: usize,
        hd_attn: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        anyhow::ensure!(
            ratio > 0 && budget.is_multiple_of(ratio),
            "QSA: budget % ratio != 0"
        );
        // Capacity derives from the served context; ATLAS_QSA_MAX_TOKENS can
        // only raise it, never below `max_seq_len` (that is what killed decode
        // at 32768 on a --max-seq-len 65536 serve).
        let env_max = std::env::var("ATLAS_QSA_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        // A config built outside `serve` carries 0; keep the historical
        // capacity there rather than allocating a zero-token indexer.
        let max_seq_len = if max_seq_len == 0 { 32768 } else { max_seq_len };
        let max_tokens: usize = match env_max {
            Some(n) if n < max_seq_len => {
                tracing::warn!(
                    "QSA: ATLAS_QSA_MAX_TOKENS={n} < max_seq_len={max_seq_len}, clamped up"
                );
                max_seq_len
            }
            Some(n) => n,
            None => max_seq_len,
        };
        tracing::info!(
            "QSA: indexer capacity {max_tokens} tokens (max_seq_len={max_seq_len}); \
             per-seq pooled keys grow on demand, {} B/token",
            (hd * 2).div_ceil(ratio)
        );
        let block_topk = budget / ratio;
        let qk_width = (n_heads + 1) * hd;
        // + room for a shared draft selection's longer tail.
        let sel_cap = budget + ratio + qsa_decode_rows::SHARE_MARGIN;
        Ok(Self {
            qk_proj_w,
            q_norm_w,
            k_norm_w,
            n_heads: n_heads as u32,
            hd: hd as u32,
            ratio: ratio as u32,
            budget: budget as u32,
            block_topk: block_topk as u32,
            rot: rot as u32,
            theta,
            eps,
            hidden: hidden as u32,
            nkv_attn: nkv_attn as u32,
            hd_attn: hd_attn as u32,
            max_tokens,
            k_pool_k: gpu.kernel("qsa_indexer", "qsa_block_pool")?,
            k_commit_table_k: super::try_kernel(gpu, "qsa_indexer", "qsa_commit_table"),
            k_qprep_k: gpu.kernel("qsa_indexer", "qsa_qprep")?,
            k_score_k: gpu.kernel("qsa_indexer", "qsa_score")?,
            k_gather_k: gpu.kernel("qsa_indexer", "qsa_gather")?,
            k_qprep_rows_k: gpu.kernel("qsa_indexer", "qsa_qprep_rows")?,
            k_score_rows_k: gpu.kernel("qsa_indexer", "qsa_score_rows")?,
            k_topk_rows_k: gpu.kernel("qsa_indexer", "qsa_topk_rows")?,
            k_score_rows_b_k: gpu.kernel("qsa_indexer", "qsa_score_rows_b")?,
            k_score_rows_exact_k: gpu.kernel("qsa_indexer", "qsa_score_rows_exact")?,
            k_score_rows_gemm_k: super::try_kernel(gpu, "qsa_indexer", "qsa_score_rows_gemm"),
            k_score_rows_tc_k: super::try_kernel(gpu, "qsa_score_tc", "qsa_score_rows_tc"),
            k_prefill_attn_k: gpu.kernel("qsa_indexer", "qsa_prefill_attn")?,
            k_select_k: gpu.kernel("qsa_indexer", "qsa_select_topk_radix")?,
            device_topk: std::env::var("ATLAS_QSA_DEVICE_TOPK").ok().as_deref() != Some("0"),
            topk_verify: std::env::var("ATLAS_QSA_TOPK_VERIFY").ok().as_deref() == Some("1"),
            rows: qsa_decode_rows::RowsPath::new(n_heads, hd, sel_cap, hd_attn, gpu)?,
            k_prefill_attn_g_k: gpu.kernel("qsa_indexer", "qsa_prefill_attn_g")?,
            k_prefill_attn_l8_k: gpu.kernel("qsa_indexer", "qsa_prefill_attn_l8")?,
            k_prefill_attn_tc_k: super::try_kernel(gpu, "qsa_attn_tc", "qsa_prefill_attn_tc"),
            k_prefill_attn_tc2_k: super::try_kernel(gpu, "qsa_attn_tc2", "qsa_prefill_attn_tc2"),
            k_prefill_attn_tc3_k: super::try_kernel(gpu, "qsa_attn_tc3", "qsa_prefill_attn_tc3"),
            qk_scratch: gpu.alloc(INGEST_SLAB * qk_width * 2)?,
            q_post: gpu.alloc(n_heads * hd * 4)?,
            scores_dev: gpu.alloc(max_tokens / ratio * 4)?,
            sel_dev: gpu.alloc(sel_cap * 4)?,
            k_scratch: gpu.alloc(sel_cap * nkv_attn * hd_attn * 2)?,
            v_scratch: gpu.alloc(sel_cap * nkv_attn * hd_attn * 2)?,
            table_dev: gpu.alloc(sel_cap.div_ceil(8) * 4)?,
            seq_len_dev: gpu.alloc(4)?,
            prefill_table_dev: gpu.alloc(max_tokens.div_ceil(8) * 4)?,
        })
    }

    /// The largest visible prefix whose selection is provably all-visible.
    pub fn inert_bound(&self) -> usize {
        (self.budget + self.ratio - 1) as usize
    }

    /// Whether the token at 0-based `pos` decodes with an ACTIVE selection,
    /// i.e. whether `decode_select(pos)` returns `Some`. The one place the
    /// inert/active boundary is decided: `pos + 1` visible tokens hold more
    /// than `block_topk` complete blocks exactly when `pos >= inert_bound()`.
    pub fn is_active_at(&self, pos: usize) -> bool {
        qsa_decode_select::select_geometry(pos, self.ratio as usize, self.block_topk as usize)
            .is_some()
    }

    fn qk_width(&self) -> usize {
        (self.n_heads as usize + 1) * self.hd as usize
    }

    /// Ingest `num_tokens` prefill tokens starting at `seq_start`: project
    /// qk, park the raw keys, pool freshly complete blocks. `seq_start == 0`
    /// resets the sequence (single-seq v1, PLE-style).
    pub fn prefill_ingest(
        &self,
        st: &mut QsaSeqState,
        hidden: DevicePtr,
        num_tokens: usize,
        seq_start: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if seq_start == 0 {
            st.ingested = 0;
            st.pooled = 0;
            st.raw.restart_at(0);
        }
        anyhow::ensure!(
            seq_start == st.ingested,
            "QSA: prefill chunk starts at {seq_start} but {} tokens are \
             ingested — a prefix-cache skip bypassed the indexer. Serve \
             qwen4_exp with the prefix cache disabled until QSA learns to \
             re-ingest cached prefixes.",
            st.ingested
        );
        self.reserve(st, seq_start + num_tokens, gpu, stream)?;

        let hd = self.hd as usize;
        let qkw = self.qk_width();
        let mut off = 0usize;
        while off < num_tokens {
            let ts = INGEST_SLAB.min(num_tokens - off);
            ops::cublas_bf16_proj_dense(
                hidden.offset((off) * self.hidden as usize * 2),
                self.qk_proj_w,
                self.qk_scratch,
                ts as u32,
                qkw as u32,
                self.hidden,
                stream,
            )
            .context("QSA qk projection (prefill)")?;
            // Raw key = the last hd columns of each row; pooled per slab
            // (blocks pool independently), so the window holds one slab.
            self.raw_room(st, ts, gpu, stream)?;
            gpu.copy_d2d_2d_async(
                self.qk_scratch.offset(self.n_heads as usize * hd * 2),
                qkw * 2,
                self.raw_slot(st, seq_start + off),
                hd * 2,
                hd * 2,
                ts,
                stream,
            )?;
            off += ts;
            st.ingested = seq_start + off;
            self.pool_new_blocks(st, gpu, stream)?;
        }
        Ok(())
    }

    // `prefill_select`: see `qsa_select.rs`; teardown: `qsa_free.rs`.

    /// Decode-step ingest + selection for the token at `pos` (0-based;
    /// `pos + 1` visible). `None` inside the inert bound (dense is exact).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_select(
        &self,
        st: &mut QsaSeqState,
        normed: DevicePtr,
        pos: usize,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        block_table_dev: DevicePtr,
        block_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<QsaSelection>> {
        anyhow::ensure!(
            pos == st.ingested,
            "QSA: decode at pos {pos} but {} tokens ingested — the indexer \
             cache lost sync (prefix-cache skip or a rewound sequence)",
            st.ingested
        );
        self.reserve(st, pos + 1, gpu, stream)?;

        let hd = self.hd as usize;
        let qkw = self.qk_width();
        // qk GEMV for this token; row 0 of the scratch.
        ops::cublas_bf16_proj_dense(
            normed,
            self.qk_proj_w,
            self.qk_scratch,
            1,
            qkw as u32,
            self.hidden,
            stream,
        )
        .context("QSA qk projection (decode)")?;
        self.raw_room(st, 1, gpu, stream)?;
        gpu.copy_d2d_async(
            self.qk_scratch.offset(self.n_heads as usize * hd * 2),
            self.raw_slot(st, pos),
            hd * 2,
            stream,
        )?;
        st.ingested = pos + 1;
        self.pool_new_blocks(st, gpu, stream)?;

        let visible = pos + 1;
        let Some(geo) =
            qsa_decode_select::select_geometry(pos, self.ratio as usize, self.block_topk as usize)
        else {
            return Ok(None); // provably all-visible: dense path is exact
        };
        let (complete, tail_start, n_sel) = (geo.complete, geo.tail_start, geo.n_sel);
        // ATLAS_QWEN4EXP_MTP_INDEX_SHARE: a draft reuses its propose's blocks.
        if let Some(n) = self.draft_share_tail(st, visible, gpu, stream)? {
            let kv = [k_pool, v_pool, block_table_dev];
            return self
                .gather_selection(st, n, kv, block_size, gpu, stream)
                .map(Some);
        }

        // q prep + block scores.
        ops::qsa_qprep(
            gpu,
            self.k_qprep_k,
            self.qk_scratch,
            self.q_norm_w,
            self.q_post,
            self.n_heads,
            self.hd,
            self.rot,
            pos as u32,
            self.theta,
            self.eps,
            stream,
        )?;
        if self.rows.decode_rows {
            // `qsa_score_rows_dec` at one row: the same bytes as qsa_score.
            self.score_dec_one(st, pos, complete, gpu, stream)?;
        } else {
            ops::qsa_score(
                gpu,
                self.k_score_k,
                self.q_post,
                st.block_keys,
                self.scores_dev,
                complete as u32,
                self.n_heads,
                self.hd,
                stream,
            )?;
        }

        // Block selection. `n_sel` never depends on the scores, only the
        // CONTENT of `sel_dev` does. Device arm (default): one radix-select
        // kernel writes `sel_dev` (no D2H, sort or H2D). Host arm (rollback,
        // and wider than QSA_SELECT_MAX_BLOCKS): D2H + sort + H2D — decode
        // graphs are vetoed whenever an indexer is present, so neither arm
        // ever runs inside a capture.
        if self.device_topk
            && (complete <= qsa_decode_select::QSA_SELECT_MAX_BLOCKS || self.rows.topk_wide)
        {
            ops::qsa_select_topk(
                gpu,
                self.k_select_k,
                self.scores_dev,
                self.sel_dev,
                complete as u32,
                self.block_topk,
                self.ratio,
                tail_start as u32,
                visible as u32,
                stream,
            )?;
            if self.topk_verify {
                self.verify_device_selection(gpu, complete, visible, pos, stream)?;
            }
        } else {
            let sel = self.host_select(gpu, complete, visible, stream)?;
            debug_assert_eq!(sel.len() as u32, n_sel);
            let sel_bytes: Vec<u8> = sel.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d_async(&sel_bytes, self.sel_dev, stream)?;
        }
        self.draft_share_record(st, tail_start);
        self.gather_selection(
            st,
            n_sel,
            [k_pool, v_pool, block_table_dev],
            block_size,
            gpu,
            stream,
        )
        .map(Some)
    }
}
