// SPDX-License-Identifier: AGPL-3.0-only

//! Per-query PREFILL selection for the QSA indexer (#753 stage 2), split
//! from `qsa.rs` for the ≤500 LoC cap. Child module of `qsa` (via
//! `#[path]`) so the indexer's private fields and `QsaState` stay
//! reachable without widening their visibility.

use anyhow::{Context, Result};
use spark_runtime::buffers::QSA_SELECT_SCRATCH_ROWS;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{QsaIndexer, QsaSeqState};
use crate::layers::ops;

impl QsaIndexer {
    /// Stage 2: per-query prefill selection for ANY prefill chunk. Chunk
    /// rows whose GLOBAL position (`seq_start + row`) is at or past the
    /// inert bound get their ATTENTION CONTEXT rows (pre-gate, pre-o_proj)
    /// overwritten with attention over exactly their reference-selected
    /// set, read straight from the paged KV cache — which at this point
    /// holds every prior chunk plus this one (section-7 writes precede
    /// attention). Rows below the bound keep the dense output, which is
    /// provably identical there. Requires `prefill_ingest` to have run for
    /// this chunk (the ingest hook precedes the attention call).
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_select(
        &self,
        st: &mut QsaSeqState,
        normed: DevicePtr,
        q_roped: DevicePtr,
        attn_ctx: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        seq_block_table: &[u32],
        seq_start: usize,
        num_tokens: usize,
        nq: u32,
        block_size: u32,
        inv_sqrt_d: f32,
        scratch: DevicePtr,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let bound = self.inert_bound(); // first selective GLOBAL position
        let total = seq_start + num_tokens;
        if total <= bound {
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
        // Upload the real physical block table for the FULL context (a
        // selective query attends blocks from every prior chunk).
        let pages_needed = total.div_ceil(block_size as usize);
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
        let row_cap = QSA_SELECT_SCRATCH_ROWS; // shared with sizes.rs qsa_select_scratch
        let ratio = self.ratio as usize;
        let topk = self.block_topk as usize;
        let heads = self.n_heads as usize;
        let hd = self.hd as usize;
        let hd_attn = self.hd_attn as usize;
        let qkw = self.qk_width();
        let q_row = nq as usize * hd_attn;

        // Scratch layout (per-call score stride; always <= the sizes.rs
        // allowance because total context never exceeds max_seq_len).
        let stride = total.div_ceil(ratio);
        let qk_buf = scratch;
        let qpost = scratch.offset(row_cap * qkw * 2);
        let scores = qpost.offset(row_cap * heads * hd * 4);
        let lists = scores.offset(row_cap * stride * 4);

        // First selective GLOBAL position, and its chunk-local row.
        let first_sel_pos = bound.max(seq_start);
        let n_sel_total = total - first_sel_pos;
        let mut slab = 0usize;
        while slab < n_sel_total {
            let rows = row_cap.min(n_sel_total - slab);
            let first_pos = first_sel_pos + slab; // GLOBAL position
            let first_row = first_pos - seq_start; // chunk-local buffer row

            ops::cublas_bf16_proj_dense(
                normed.offset(first_row * self.hidden as usize * 2),
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
            // One 128-thread block per OUTPUT SCALAR is 1.397 BILLION blocks
            // over a 30k prefill, for 512 MACs each. The tiled scorer gives a
            // block QSA_SR_B consecutive `b` values and stages the row's `q` in
            // shared once. Identical arithmetic -- same reduction, same order --
            // which matters because these scores feed a top-k, so a shifted
            // score changes WHICH blocks are attended.
            // SCORER. `_b` gives each CTA QSA_SR_B b-values and stages the
            // row's `q` in shared, bit-identically; the original is one CTA per
            // output scalar. The fallback is a shared-memory bound, not policy.
            //
            // A third arm exists and is NOT wired here: `qsa_score_rows_gemm`
            // drops the four block-wide reductions per output (one thread per
            // score, serial `d` contraction) and is worth **3.2 s of a 29.5 s
            // 30k prefill** — but it reassociates the contraction, and these
            // scores pick which blocks a query reads. Gated and REJECTED:
            //
            //   needle recall (scripts/lc_check.py)     12/12, unchanged
            //   kl_drift --precision-change             top-1 69.3%
            //   score drift vs reference                4.2e-7 rel, 44% bit-exact
            //
            // The kernel is correct — 4.2e-7 is ordinary FP32 reassociation over
            // 128 terms. The top-k amplifies it: near-ties either side of the
            // 512th block flip, and the attended set genuinely changes. The
            // project's own QSA bar is **>=98% top-1 agreement** (Phase 4
            // acceptance, vs the llama.cpp reference), so 69.3% is not close,
            // and needle recall passing is exactly why it is not the deciding
            // metric. Kept in-tree, exercised by
            // `qsa_score_rows_gemm_vs_reference_drift`.
            //
            // THE REAL QUALITY GATE HAS NOW BEEN RUN, and it says the rejection
            // above was measuring the wrong thing. `scripts/ppl.py` -- written
            // after that verdict, precisely because agreement-with-our-own-build
            // is meaningless for a scorer whose own summation order is arbitrary
            // -- puts this arm at **+0.036% above-bound perplexity**, with the
            // dense control at 0.000%. So the change that scored 69.3% top-1
            // agreement costs 0.036% of the only absolute measure available.
            // Calibrate future QSA verdicts against that pair.
            //
            // It still does NOT ship, for a different and simpler reason: the
            // 3.2 s it was worth in that note was against the OLD scorer.
            // `qsa_score_rows_exact` has since taken that win bit-identically,
            // and against it the GEMM arm measures **-0.06 s** at ctx 31481 --
            // inside run-to-run noise. There is no speed left to trade for even
            // a 0.036% regression, so it stays off (`ATLAS_QSA_SCORE_GEMM=1`).
            // `ATLAS_QSA_SCORE_GEMM=1` wires that third arm. The comment above
            // says "dispatch-disabled until someone runs the real quality gate",
            // and `scripts/ppl.py` -- written AFTER that rejection, precisely
            // because agreement-with-our-own-build is the wrong question for a
            // scorer whose own summation order is arbitrary -- is that gate.
            // `ATLAS_QSA_SCORE_TC=1`: the same scores on tensor cores. See
            // TTFT_GAP.md 34 -- the BF16-Q half was priced with a probe BEFORE
            // this kernel was written, and measured BETTER than F32 Q.
            let score_tc = matches!(
                std::env::var("ATLAS_QSA_SCORE_TC").as_deref(),
                Ok("1") | Ok("true")
            ) && self.k_score_rows_tc_k.0 != 0
                && ops::qsa_score_rows_tc_ok(self.n_heads, self.hd);
            if score_tc {
                ops::qsa_score_rows_tc(
                    gpu,
                    self.k_score_rows_tc_k,
                    qpost,
                    st.block_keys,
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
            }
            let score_gemm = !score_tc
                && matches!(
                    std::env::var("ATLAS_QSA_SCORE_GEMM").as_deref(),
                    Ok("1") | Ok("true")
                )
                && self.k_score_rows_gemm_k.0 != 0
                && ops::qsa_score_rows_gemm_ok(self.n_heads, self.hd);
            if score_tc {
                // already dispatched above
            } else if score_gemm {
                ops::qsa_score_rows_gemm(
                    gpu,
                    self.k_score_rows_gemm_k,
                    qpost,
                    st.block_keys,
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
            } else if ops::qsa_score_rows_exact_ok(self.n_heads, self.hd) {
                ops::qsa_score_rows_exact(
                    gpu,
                    self.k_score_rows_exact_k,
                    qpost,
                    st.block_keys,
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
            } else {
                ops::qsa_score_rows(
                    gpu,
                    self.k_score_rows_k,
                    qpost,
                    st.block_keys,
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
            }
            // SELECTION. On the GPU when the shape allows it: `qsa_topk_rows`
            // produces byte-for-byte the same list, in the same order, without
            // moving the score matrix anywhere. The host path below is the
            // fallback for a `topk` wider than the kernel's running best-K, and
            // it is what the GPU kernel is tested against
            // (`qsa_topk_rows_matches_host_selection`).
            //
            // What this is worth: the host round-trip is a full stream drain
            // per attention layer per slab, and no kernel-time profile can see
            // it because while it runs no kernel is running. Measuring GPU IDLE
            // instead (scripts/gaps.py), on a 30k prefill:
            //
            //   qsa_score_rows -> qsa_prefill_attn_g
            //       7279 ms over 179 gaps, 40.7 ms each -- 19% of the window,
            //       and the largest single item in it.
            let host_select = !ops::qsa_topk_rows_ok(topk as u32);
            if !host_select {
                ops::qsa_topk_rows(
                    gpu,
                    self.k_topk_rows_k,
                    scores,
                    lists,
                    rows as u32,
                    first_pos as u32,
                    stride as u32,
                    self.ratio,
                    topk as u32,
                    stream,
                )?;
            }
            // ── ATLAS_QSA_UNION_DIAG: how much do neighbouring rows agree? ──
            // Decides whether an EXACT block-sparse tensor-core attention is
            // possible here. A TC kernel needs a TILE of query rows to share one
            // K/V set; QSA selects per ROW. Attending the UNION of a tile's
            // selections and masking each row back to its own list is exactly
            // equivalent -- so the only question is how big that union is.
            // union/topk == 1.0 means free; == tile size means no sharing at all.
            if std::env::var("ATLAS_QSA_UNION_DIAG").as_deref() == Ok("1") {
                let mut host = vec![0u8; rows * topk * 4];
                gpu.synchronize(stream)?;
                gpu.copy_d2h_on_stream(lists, &mut host, stream)?;
                let ids: Vec<i32> = host
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let mut line = String::new();
                for tile in [16usize, 32, 64, 128] {
                    let (mut tot, mut n) = (0usize, 0usize);
                    let mut r0 = 0usize;
                    while r0 + tile <= rows {
                        let mut set = std::collections::HashSet::new();
                        for r in r0..r0 + tile {
                            for k in 0..topk {
                                let v = ids[r * topk + k];
                                if v >= 0 {
                                    set.insert(v);
                                }
                            }
                        }
                        tot += set.len();
                        n += 1;
                        r0 += tile;
                    }
                    if n > 0 {
                        line.push_str(&format!(
                            " tile{tile}: union={} ({:.2}x topk)",
                            tot / n,
                            (tot / n) as f64 / topk as f64
                        ));
                    }
                }
                tracing::info!("QSA union rows={rows} topk={topk}{line}");
            }
            if host_select {
                self.prefill_host_select(
                    gpu, scores, lists, rows, stride, first_pos, ratio, topk, stream,
                )?;
            }

            self.prefill_attend_slab(
                gpu,
                q_roped,
                attn_ctx,
                k_pool,
                v_pool,
                block_table_dev,
                lists,
                first_row,
                q_row,
                rows,
                first_pos,
                topk,
                block_size,
                nq,
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
}
