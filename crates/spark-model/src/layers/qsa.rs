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
//! v1 SCOPE (decode-side): raw keys are ingested during prefill and decode;
//! selection runs at DECODE steps once the visible prefix exceeds the inert
//! bound, and feeds the EXISTING paged decode attention through a gathered
//! contiguous scratch + identity block table. Prefill queries beyond the
//! inert bound still run dense (a one-time WARN documents the divergence;
//! per-query prefill selection is stage 2). Single sequence, BF16 KV only.
//!
//! CUDA graphs: selection does a host top-k on the scores (D2H), which can
//! never sit inside a captured graph — a layer carrying an indexer vetoes
//! decode-graph capture entirely (graphs measured speed-NEUTRAL on GB10, so
//! this costs nothing).

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;

#[cfg(test)]
#[path = "qsa_tests.rs"]
mod tests;

/// One decode step's selection, ready for the paged decode attention:
/// `k/v` are contiguous NHD scratch and `table` is the identity mapping.
pub struct QsaSelection {
    pub k_scratch: DevicePtr,
    pub v_scratch: DevicePtr,
    pub table_dev: DevicePtr,
    pub seq_len_dev: DevicePtr,
    pub n_sel: u32,
    pub max_blocks: u32,
}

struct QsaState {
    /// Tokens whose raw keys are in `raw_keys` (contiguous from 0).
    ingested: usize,
    /// Complete 4-token blocks pooled into `block_keys`.
    pooled: usize,
    /// Identity block table upload done (needs block_size, known lazily).
    table_len: usize,
    warned_prefill_dense: bool,
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
    k_qprep_k: KernelHandle,
    k_score_k: KernelHandle,
    k_gather_k: KernelHandle,
    k_qprep_rows_k: KernelHandle,
    k_score_rows_k: KernelHandle,
    k_prefill_attn_k: KernelHandle,

    raw_keys: DevicePtr,   // [max_tokens, hd] BF16
    block_keys: DevicePtr, // [max_tokens/ratio, hd] BF16
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

    state: std::sync::Mutex<QsaState>,
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
        let max_tokens: usize = std::env::var("ATLAS_QSA_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32768);
        let block_topk = budget / ratio;
        let qk_width = (n_heads + 1) * hd;
        let sel_cap = budget + ratio;
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
            k_qprep_k: gpu.kernel("qsa_indexer", "qsa_qprep")?,
            k_score_k: gpu.kernel("qsa_indexer", "qsa_score")?,
            k_gather_k: gpu.kernel("qsa_indexer", "qsa_gather")?,
            k_qprep_rows_k: gpu.kernel("qsa_indexer", "qsa_qprep_rows")?,
            k_score_rows_k: gpu.kernel("qsa_indexer", "qsa_score_rows")?,
            k_prefill_attn_k: gpu.kernel("qsa_indexer", "qsa_prefill_attn")?,
            raw_keys: gpu.alloc(max_tokens * hd * 2)?,
            block_keys: gpu.alloc(max_tokens / ratio * hd * 2)?,
            qk_scratch: gpu.alloc(INGEST_SLAB * qk_width * 2)?,
            q_post: gpu.alloc(n_heads * hd * 4)?,
            scores_dev: gpu.alloc(max_tokens / ratio * 4)?,
            sel_dev: gpu.alloc(sel_cap * 4)?,
            k_scratch: gpu.alloc(sel_cap * nkv_attn * hd_attn * 2)?,
            v_scratch: gpu.alloc(sel_cap * nkv_attn * hd_attn * 2)?,
            table_dev: gpu.alloc(sel_cap.div_ceil(8) * 4)?,
            seq_len_dev: gpu.alloc(4)?,
            prefill_table_dev: gpu.alloc(max_tokens.div_ceil(8) * 4)?,
            state: std::sync::Mutex::new(QsaState {
                ingested: 0,
                pooled: 0,
                table_len: 0,
                warned_prefill_dense: false,
            }),
        })
    }

    /// The largest visible prefix whose selection is provably all-visible.
    pub fn inert_bound(&self) -> usize {
        (self.budget + self.ratio - 1) as usize
    }

    fn qk_width(&self) -> usize {
        (self.n_heads as usize + 1) * self.hd as usize
    }

    /// Ingest `num_tokens` prefill tokens starting at `seq_start`: project
    /// qk, park the raw keys, pool freshly complete blocks. `seq_start == 0`
    /// resets the sequence (single-seq v1, PLE-style).
    pub fn prefill_ingest(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        seq_start: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let mut st = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("QSA state mutex poisoned"))?;
        if seq_start == 0 {
            st.ingested = 0;
            st.pooled = 0;
        }
        anyhow::ensure!(
            seq_start == st.ingested,
            "QSA: prefill chunk starts at {seq_start} but {} tokens are \
             ingested — a prefix-cache skip bypassed the indexer. Serve \
             qwen4_exp with the prefix cache disabled until QSA learns to \
             re-ingest cached prefixes.",
            st.ingested
        );
        anyhow::ensure!(
            seq_start + num_tokens <= self.max_tokens,
            "QSA: {} tokens exceeds ATLAS_QSA_MAX_TOKENS={}",
            seq_start + num_tokens,
            self.max_tokens
        );

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
            // Raw key = the last hd columns of each row.
            gpu.copy_d2d_2d_async(
                self.qk_scratch.offset(self.n_heads as usize * hd * 2),
                qkw * 2,
                self.raw_keys.offset((seq_start + off) * hd * 2),
                hd * 2,
                hd * 2,
                ts,
                stream,
            )?;
            off += ts;
        }
        st.ingested = seq_start + num_tokens;
        self.pool_new_blocks(&mut st, gpu, stream)
    }

    fn pool_new_blocks(&self, st: &mut QsaState, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let complete = st.ingested / self.ratio as usize;
        if complete > st.pooled {
            ops::qsa_block_pool(
                gpu,
                self.k_pool_k,
                self.raw_keys,
                self.k_norm_w,
                self.block_keys,
                st.pooled as u32,
                (complete - st.pooled) as u32,
                self.ratio,
                self.hd,
                self.rot,
                self.theta,
                self.eps,
                stream,
            )?;
            st.pooled = complete;
        }
        Ok(())
    }

    /// Stage 2: per-query prefill selection for chunk-0 prefills
    /// (`seq_len_start == 0`, i.e. global pos == chunk row). Overwrites the
    /// ATTENTION CONTEXT rows (pre-gate, pre-o_proj) of every selective
    /// query — pos >= the inert bound — with attention over exactly its
    /// reference-selected set, read straight from the paged KV cache. Rows
    /// below the bound keep the dense flash output, which is provably
    /// identical there. Requires `prefill_ingest` to have run for this chunk
    /// (it does — the ingest hook precedes the attention call).
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_select_chunk0(
        &self,
        normed: DevicePtr,
        q_roped: DevicePtr,
        attn_ctx: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        seq_block_table: &[u32],
        num_tokens: usize,
        nq: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        scratch: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let bound = self.inert_bound(); // first selective position
        if num_tokens <= bound {
            return Ok(());
        }
        // Kill switch: ATLAS_QSA_NO_PREFILL_SELECT=1 keeps stage-1 behavior
        // (dense prefill past the bound; decode still selects).
        static S2_OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *S2_OFF
            .get_or_init(|| std::env::var("ATLAS_QSA_NO_PREFILL_SELECT").as_deref() == Ok("1"))
        {
            return Ok(());
        }
        let diag = {
            static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *D.get_or_init(|| std::env::var("ATLAS_QSA_S2_DIAG").as_deref() == Ok("1"))
        };
        // Diagnostic: park the DENSE context of the LAST row before the
        // overwrite; log cosine(dense, selected) after. Selected attends
        // 2048 of the visible tokens, so a healthy overwrite is close to
        // dense (cos ~0.9+); garbage means a layout/addressing defect.
        let q_row = nq as usize * self.hd_attn as usize;
        let mut dense_last = Vec::new();
        if diag {
            dense_last = vec![0u8; q_row * 2];
            gpu.copy_d2h_on_stream(
                attn_ctx.offset((num_tokens - 1) * q_row * 2),
                &mut dense_last,
                stream,
            )?;
            // Norm probes: an INERT row (dense output must be real there no
            // matter what), the first selective row, and the last row —
            // separates wrong-buffer from wrong-offset in one run.
            let probe = |row: usize| -> Result<f64> {
                let mut b = vec![0u8; q_row * 2];
                gpu.copy_d2h_on_stream(attn_ctx.offset(row * q_row * 2), &mut b, stream)?;
                Ok(b.chunks_exact(2)
                    .map(|c| {
                        let v =
                            f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt())
            };
            tracing::warn!(
                "QSA S2 DIAG norms: row100={:.3} first_sel(row {bound})={:.3} last={:.3} q_row={q_row}",
                probe(100)?,
                probe(bound)?,
                probe(num_tokens - 1)?
            );
            // Boundary bisect: dense-ctx and roped-q norms across 2040..2056.
            let probe_at = |base: DevicePtr, row: usize| -> Result<f64> {
                let mut b = vec![0u8; q_row * 2];
                gpu.copy_d2h_on_stream(base.offset(row * q_row * 2), &mut b, stream)?;
                Ok(b.chunks_exact(2)
                    .map(|c| {
                        let v =
                            f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64;
                        v * v
                    })
                    .sum::<f64>()
                    .sqrt())
            };
            let mut ctx_line = String::new();
            let mut q_line = String::new();
            for row in [128usize, 256, 512, 768, 1024, 1280, 1536, 1792, 1900, 2000] {
                ctx_line += &format!(" {row}:{:.2}", probe_at(attn_ctx, row)?);
            }
            tracing::warn!("QSA S2 DIAG wide:{ctx_line}");
            ctx_line = String::new();
            for row in (2040..2056).step_by(2) {
                ctx_line += &format!(" {row}:{:.2}", probe_at(attn_ctx, row)?);
                q_line += &format!(" {row}:{:.2}", probe_at(q_roped, row)?);
            }
            tracing::warn!("QSA S2 DIAG ctx rows:{ctx_line}");
            tracing::warn!("QSA S2 DIAG   q rows:{q_line}");
        }
        // Upload the real physical block table for the token range.
        let pages_needed = num_tokens.div_ceil(block_size as usize);
        anyhow::ensure!(
            seq_block_table.len() >= pages_needed,
            "QSA: block table has {} pages for {} tokens",
            seq_block_table.len(),
            pages_needed
        );
        let tbytes: Vec<u8> = seq_block_table[..pages_needed]
            .iter()
            .flat_map(|b| (*b as i32).to_le_bytes())
            .collect();
        gpu.copy_h2d_async(&tbytes, self.prefill_table_dev, stream)?;
        let block_table_dev = self.prefill_table_dev;
        const ROWS: usize = 2048; // must match sizes.rs qsa_select_scratch
        let ratio = self.ratio as usize;
        let topk = self.block_topk as usize;
        let heads = self.n_heads as usize;
        let hd = self.hd as usize;
        let hd_attn = self.hd_attn as usize;
        let qkw = self.qk_width();
        let q_row = nq as usize * hd_attn;

        // Scratch layout (per-call score stride; always <= the sizes.rs
        // allowance because a chunk never exceeds max_seq_len).
        let stride = num_tokens.div_ceil(ratio);
        let qk_buf = scratch;
        let qpost = scratch.offset(ROWS * qkw * 2);
        let scores = qpost.offset(ROWS * heads * hd * 4);
        let lists = scores.offset(ROWS * stride * 4);

        let n_sel_total = num_tokens - bound;
        let mut slab = 0usize;
        while slab < n_sel_total {
            let rows = ROWS.min(n_sel_total - slab);
            let first_pos = bound + slab;

            ops::cublas_bf16_proj_dense(
                normed.offset(first_pos * self.hidden as usize * 2),
                self.qk_proj_w,
                qk_buf,
                rows as u32,
                qkw as u32,
                self.hidden,
                stream,
            )
            .context("QSA qk projection (prefill select)")?;
            ops::qsa_qprep_rows(
                gpu,
                self.k_qprep_rows_k,
                qk_buf,
                self.q_norm_w,
                qpost,
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
            let n_blocks_max = (first_pos + rows) / ratio; // last row's complete
            ops::qsa_score_rows(
                gpu,
                self.k_score_rows_k,
                qpost,
                self.block_keys,
                scores,
                rows as u32,
                n_blocks_max as u32,
                first_pos as u32,
                stride as u32,
                self.ratio,
                self.n_heads,
                self.hd,
                stream,
            )?;

            // Host top-k per row (sync D2H drains the stream first). Torch
            // tie-break: larger score first, lower index on ties.
            let mut raw = vec![0u8; rows * stride * 4];
            gpu.copy_d2h_on_stream(scores, &mut raw, stream)?;
            let sc: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut host_lists = vec![0u8; rows * topk * 4];
            for r in 0..rows {
                let complete = (first_pos + r + 1) / ratio;
                let row_sc = &sc[r * stride..r * stride + complete];
                let mut order: Vec<u32> = (0..complete as u32).collect();
                order.sort_by(|&a, &b| {
                    row_sc[b as usize]
                        .partial_cmp(&row_sc[a as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });
                for (i, b) in order[..topk].iter().enumerate() {
                    host_lists[(r * topk + i) * 4..(r * topk + i) * 4 + 4]
                        .copy_from_slice(&(*b as i32).to_le_bytes());
                }
            }
            gpu.copy_h2d_async(&host_lists, lists, stream)?;

            ops::qsa_prefill_attn(
                gpu,
                self.k_prefill_attn_k,
                q_roped.offset(first_pos * q_row * 2),
                k_pool,
                v_pool,
                block_table_dev,
                lists,
                attn_ctx.offset(first_pos * q_row * 2),
                rows as u32,
                first_pos as u32,
                topk as u32,
                self.ratio,
                block_size,
                nq,
                self.nkv_attn,
                self.hd_attn,
                inv_sqrt_d,
                stream,
            )?;
            slab += rows;
        }
        if diag {
            let mut sel_last = vec![0u8; q_row * 2];
            gpu.copy_d2h_on_stream(
                attn_ctx.offset((num_tokens - 1) * q_row * 2),
                &mut sel_last,
                stream,
            )?;
            let f = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect()
            };
            let (a, b) = (f(&dense_last), f(&sel_last));
            let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            tracing::warn!(
                "QSA S2 DIAG: last-row ctx dense-vs-selected cos={:.6} |dense|={:.3} |sel|={:.3}",
                dot / (na * nb).max(1e-30),
                na,
                nb
            );
        }
        tracing::debug!(
            "QSA prefill select: {} selective rows over {} tokens",
            n_sel_total,
            num_tokens
        );
        Ok(())
    }
    /// One-time WARN when a prefill query range extends past the inert
    /// bound: those queries run DENSE until stage 2 (per-query prefill
    /// selection) lands.
    pub fn warn_if_prefill_diverges(&self, seq_start: usize, num_tokens: usize) {
        if seq_start + num_tokens > self.inert_bound() + 1
            && let Ok(mut st) = self.state.lock()
            && !st.warned_prefill_dense
        {
            st.warned_prefill_dense = true;
            tracing::warn!(
                "QSA: prefill extends to {} tokens (> inert bound {}) — \
                 prefill queries beyond the bound run DENSE attention until \
                 per-query prefill selection lands; decode steps DO select.",
                seq_start + num_tokens,
                self.inert_bound()
            );
        }
    }

    /// Decode-step ingest + selection for the token at `pos` (0-based;
    /// `pos + 1` tokens are visible including this one). Returns `None`
    /// while the visible prefix is within the inert bound (dense path is
    /// exact there).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_select(
        &self,
        normed: DevicePtr,
        pos: usize,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        block_table_dev: DevicePtr,
        block_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<QsaSelection>> {
        let mut st = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("QSA state mutex poisoned"))?;
        anyhow::ensure!(
            pos == st.ingested,
            "QSA: decode at pos {pos} but {} tokens ingested — the indexer \
             cache lost sync (prefix-cache skip or a rewound sequence)",
            st.ingested
        );
        anyhow::ensure!(
            pos < self.max_tokens,
            "QSA: pos {pos} >= ATLAS_QSA_MAX_TOKENS"
        );

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
        gpu.copy_d2d_async(
            self.qk_scratch.offset(self.n_heads as usize * hd * 2),
            self.raw_keys.offset(pos * hd * 2),
            hd * 2,
            stream,
        )?;
        st.ingested = pos + 1;
        self.pool_new_blocks(&mut st, gpu, stream)?;

        let visible = pos + 1;
        let complete = visible / self.ratio as usize;
        if complete <= self.block_topk as usize {
            return Ok(None); // provably all-visible: dense path is exact
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
        ops::qsa_score(
            gpu,
            self.k_score_k,
            self.q_post,
            self.block_keys,
            self.scores_dev,
            complete as u32,
            self.n_heads,
            self.hd,
            stream,
        )?;

        // Host top-k over the block scores (D2H — decode graphs are vetoed
        // whenever an indexer is present, so this is never inside a capture).
        let mut raw = vec![0u8; complete * 4];
        gpu.copy_d2h_on_stream(self.scores_dev, &mut raw, stream)?;
        let scores: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut order: Vec<u32> = (0..complete as u32).collect();
        // torch.topk returns the k largest, ties broken by LOWER index —
        // sort by (-score, index) and take the first k for identical sets.
        order.sort_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut blocks: Vec<u32> = order[..self.block_topk as usize].to_vec();
        blocks.sort_unstable();

        let ratio = self.ratio as usize;
        let mut sel: Vec<i32> = Vec::with_capacity(self.budget as usize + ratio);
        for b in &blocks {
            let base = *b as i32 * self.ratio as i32;
            for r in 0..self.ratio as i32 {
                sel.push(base + r);
            }
        }
        for t in complete * ratio..visible {
            sel.push(t as i32);
        }
        let n_sel = sel.len() as u32;

        let sel_bytes: Vec<u8> = sel.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d_async(&sel_bytes, self.sel_dev, stream)?;
        ops::qsa_gather(
            gpu,
            self.k_gather_k,
            k_pool,
            v_pool,
            block_table_dev,
            self.sel_dev,
            self.k_scratch,
            self.v_scratch,
            n_sel,
            block_size,
            self.nkv_attn,
            self.hd_attn,
            stream,
        )?;

        // Identity table + seq_len for the scratch-as-paged-cache view.
        let pages = (n_sel as usize).div_ceil(block_size as usize);
        if st.table_len < pages {
            let ident: Vec<u8> = (0..pages as i32).flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d_async(&ident, self.table_dev, stream)?;
            st.table_len = pages;
        }
        gpu.copy_h2d_async(&(n_sel as i32).to_le_bytes(), self.seq_len_dev, stream)?;

        Ok(Some(QsaSelection {
            k_scratch: self.k_scratch,
            v_scratch: self.v_scratch,
            table_dev: self.table_dev,
            seq_len_dev: self.seq_len_dev,
            n_sel,
            max_blocks: pages as u32,
        }))
    }
}
