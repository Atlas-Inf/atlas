// SPDX-License-Identifier: AGPL-3.0-only

//! Batched cross-sequence propose for the qwen4_exp MTP drafter: ONE drafter
//! forward per draft position for n sequences, chained on the device.
//!
//! The per-sequence path (`qwen4exp_mtp_forward.rs`) runs a full drafter
//! forward per draft per sequence — every drafter weight and the draft head
//! (100k BF16 rows by default, ~512 MB) re-read each time — and ends each one
//! in a blocking 4-byte `copy_d2h` of the argmax, so the next draft's
//! embedding can be looked up on the host. At C4 with 3 drafts that is 12
//! forwards and 12 stream drains per verify step.
//!
//! Here draft position j runs once for all n rows:
//!
//! - the per-stream grouped norm keeps `forward_one`'s launch per (row,
//!   stream) — it is tiny and per-row identical by construction — or, under
//!   `ATLAS_QWEN4EXP_MTP_GROUPED_NORM=1`, one `rms_norm_f32_grouped` launch
//!   whose blocks are those launches' (`stream_norms`);
//! - `fc_hidden` (n*hc rows), `fc_embedding` and the draft head go through
//!   `dense_gemv_bf16_batchm` (or the scalar `w4a16_gemv_batch{M}` tiers for
//!   the NVFP4 head), whose rows are byte-identical to the single-row GEMVs;
//! - the body is the layer's own multi-sequence decode (`decode_multi_seq`),
//!   the same machinery the exact batched verify lane runs
//!   (`ATLAS_QWEN4EXP_BATCH_FAST`, which this path requires);
//! - the argmax is `argmax_bf16_batch` (same index and tie rule as
//!   `argmax_bf16`), written straight into the device token chain, and the
//!   next position embeds from it with `batched_embed`.
//!
//! Host traffic per CALL, whatever n and the depth: one H2D (the n input
//! tokens and every position's attention metadata) and one blocking D2H (all
//! drafts, plus confidences when wanted). The batched verify consumes host
//! draft tokens (`decode_verify_batched` takes `&[u32]`, and the verdict
//! compares on the host), so that one readback is the floor without
//! restructuring the verify.
//!
//! The confidence stop (`ATLAS_QWEN4EXP_MTP_CONFIDENCE`) is applied after the
//! readback: every row drafts the full depth, and a sequence whose draft i>0
//! falls below the threshold keeps drafts `0..i` and rewinds its drafter
//! `seq_len` to match — exactly the state the per-sequence early `break`
//! leaves, since later positions never feed earlier ones. The probability is
//! the device log-softmax of `argmax_bf16_batch_lp` (FP32 `__expf`) instead of
//! the host `top1_prob_bf16` (f64): a stop decision can differ only for a
//! probability within float rounding of the threshold.

use super::*;
use crate::layers::ops::DENSE_GEMV_BATCHM_MAX_M;

/// Sequences one batched propose carries: the row ceiling of the head's
/// `dense_gemv_bf16_batchm` / scalar `w4a16_gemv_batch{M}` tiers, and of the
/// small-T mHC collapse the head mixer runs at (`HC_DECODE_MAX_T`, also 8).
pub(super) const PROPOSE_BATCH_MAX: usize = DENSE_GEMV_BATCHM_MAX_M as usize;

/// Draft positions one call carries (the slab holds one metadata header per
/// position). qwen4_exp verifies at most 3; a deeper ask falls back.
pub(super) const PROPOSE_BATCH_MAX_DRAFTS: usize = 8;

// ── The slab: one device allocation per proposer ──
//
// [0 ..)          u32 token rows, n apart: row 0 the callers' tokens, row
//                 j+1 position j's drafts (written by the argmax)
// [lp_off ..)     f32 top-1 log-probabilities, row j = position j
// [HDR_OFF ..)    one attention header per position, HDR_BYTES apart:
//                 positions u32[8] | slots i64[8] @HDR_SLOT | seq_lens i32[8]
//                 @HDR_SEQ_LEN
// [BT_OFF ..)     the block tables, `max_blocks` i32 a row (shared by every
//                 position: all blocks are allocated up front)
const HDR_OFF: usize = 1024;
const HDR_BYTES: usize = 128;
const HDR_SLOT: usize = 32;
const HDR_SEQ_LEN: usize = 96;
const BT_OFF: usize = HDR_OFF + PROPOSE_BATCH_MAX_DRAFTS * HDR_BYTES;
const _: () = {
    let (n, d) = (PROPOSE_BATCH_MAX, PROPOSE_BATCH_MAX_DRAFTS);
    assert!(((d + 1) * n * 4).next_multiple_of(16) + d * n * 4 <= HDR_OFF);
    assert!(HDR_SLOT >= n * 4 && HDR_SEQ_LEN >= HDR_SLOT + n * 8);
    assert!(HDR_SEQ_LEN + n * 4 <= HDR_BYTES);
};

/// Device bytes of a slab whose rows hold `blocks` block-table entries.
pub(super) fn slab_bytes(blocks: usize) -> usize {
    BT_OFF + PROPOSE_BATCH_MAX * blocks * 4
}

/// Offset of the log-probability rows for `n` rows and `d` positions.
pub(super) fn lp_off(n: usize, d: usize) -> usize {
    ((d + 1) * n * 4).next_multiple_of(16)
}

/// Host image of a slab, built before any GPU work.
pub(super) struct Slab {
    pub bytes: Vec<u8>,
    pub max_blocks: usize,
}

/// Pack the slab for n rows and `num_drafts` positions. Row i drafts at
/// sequence positions `positions[i] + j` (RoPE) into drafter KV rows
/// `kv_lens[i] + j`, whose blocks `block_tables[i]` must already hold.
pub(super) fn pack_slab(
    tokens: &[u32],
    positions: &[usize],
    kv_lens: &[usize],
    block_tables: &[Vec<u32>],
    num_drafts: usize,
    block_size: usize,
    cap_blocks: usize,
) -> Result<Slab> {
    let n = tokens.len();
    anyhow::ensure!(
        (1..=PROPOSE_BATCH_MAX).contains(&n)
            && (1..=PROPOSE_BATCH_MAX_DRAFTS).contains(&num_drafts)
            && positions.len() == n
            && kv_lens.len() == n
            && block_tables.len() == n,
        "qwen4_exp batched propose: slab shape n={n} drafts={num_drafts}"
    );
    let max_blocks = block_tables.iter().map(Vec::len).max().unwrap_or(0);
    anyhow::ensure!(
        max_blocks <= cap_blocks,
        "qwen4_exp batched propose: {max_blocks} drafter blocks exceed the slab's {cap_blocks}"
    );
    let mut bytes = vec![0u8; BT_OFF + n * max_blocks * 4];
    for (i, &t) in tokens.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&t.to_le_bytes());
    }
    for j in 0..num_drafts {
        let hdr = HDR_OFF + j * HDR_BYTES;
        for i in 0..n {
            let row = kv_lens[i] + j;
            let block = *block_tables[i].get(row / block_size).ok_or_else(|| {
                anyhow::anyhow!("qwen4_exp batched propose: drafter row {row} has no block")
            })?;
            let slot = block as i64 * block_size as i64 + (row % block_size) as i64;
            let pos = (positions[i] + j) as u32;
            bytes[hdr + i * 4..][..4].copy_from_slice(&pos.to_le_bytes());
            bytes[hdr + HDR_SLOT + i * 8..][..8].copy_from_slice(&slot.to_le_bytes());
            let seen = (row + 1) as i32;
            bytes[hdr + HDR_SEQ_LEN + i * 4..][..4].copy_from_slice(&seen.to_le_bytes());
        }
    }
    for (i, bt) in block_tables.iter().enumerate() {
        for (b, &block) in bt.iter().enumerate() {
            let at = BT_OFF + (i * max_blocks + b) * 4;
            bytes[at..at + 4].copy_from_slice(&(block as i32).to_le_bytes());
        }
    }
    Ok(Slab { bytes, max_blocks })
}

/// Drafts a sequence keeps under the confidence stop: the first always, then
/// up to the first later draft whose probability is below `stop` — the
/// per-sequence `propose` loop's `break`. `stop <= 0` keeps them all.
pub(super) fn kept_drafts(probs: &[f32], stop: f32) -> usize {
    if stop <= 0.0 {
        return probs.len();
    }
    (1..probs.len())
        .find(|&i| probs[i] < stop)
        .unwrap_or(probs.len())
}

/// Read `rows x cols` little-endian 4-byte words from `buf` at `off`, row
/// `r` = position, column `i` = sequence, into per-sequence vectors.
fn per_seq<T>(buf: &[u8], off: usize, rows: usize, n: usize, f: fn([u8; 4]) -> T) -> Vec<Vec<T>> {
    (0..n)
        .map(|i| {
            (0..rows)
                .map(|j| {
                    let at = off + (j * n + i) * 4;
                    f(buf[at..at + 4].try_into().expect("4-byte word"))
                })
                .collect()
        })
        .collect()
}

impl Qwen4ExpMtpHead {
    /// The widest batch [`DraftProposer::propose_batch`] runs in one drafter
    /// forward per position; 1 = per-sequence only. Every bound is a resolved
    /// kernel or an arena capacity, not an assumption.
    pub(super) fn batch_width(
        &self,
        buffers: &spark_runtime::buffers::BufferArena,
        config: &atlas_core::config::ModelConfig,
    ) -> usize {
        let kernels = [
            self.dense_gemv_batchm_k,
            self.batched_embed_k,
            self.argmax_batch_k,
        ];
        let lowrank = self
            .module
            .hc_head
            .as_ref()
            .is_some_and(|h| h.lowrank.is_some());
        // The drafter QSA indexer (`ATLAS_MTP_DRAFTER_QSA=1`) keeps a
        // per-sequence watermark the per-sequence path owns.
        if kernels.iter().any(|k| k.0 == 0) || !lowrank || self.module.body.has_aux_state() {
            return 1;
        }
        let h = config.hidden_size;
        let hc = config.hc_mult.max(1);
        let sizes = buffers.sizes();
        // The body runs at the TP=1 view (every head) on an arena sized for
        // this rank's share, so its per-row buffers may be tp x wider.
        let mut cap = PROPOSE_BATCH_MAX
            .min(sizes.hc_streams / (hc * h * 4))
            .min(sizes.logits / (self.draft.rows() as usize * 2))
            .min(buffers.max_batch_tokens() / config.tp_world_size.max(1));
        while cap > 1 && !self.draft.rows_batchable(cap, self.dense_gemv_batchm_k) {
            cap -= 1;
        }
        cap.max(1)
    }

    /// Whether this call can batch: width, depth, the exact batching lane the
    /// body's multi-sequence decode needs for per-row parity with serial
    /// decode, and the confidence kernel when the stop is armed.
    pub(super) fn batch_admits(&self, n: usize, num_drafts: usize, ctx: &ForwardContext) -> bool {
        ctx.levers.qwen4exp_batch_fast
            && (1..=PROPOSE_BATCH_MAX_DRAFTS).contains(&num_drafts)
            && (self.conf_stop <= 0.0 || self.argmax_batch_lp_k.0 != 0)
            && (2..=self.batch_width(ctx.buffers, ctx.config)).contains(&n)
    }

    /// The batched propose. Allocates every drafter block the call writes,
    /// uploads the slab, runs `num_drafts` positions with no host sync, reads
    /// every draft back once, then applies the confidence stop per sequence.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_batch_impl(
        &self,
        last_tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        sts: &mut [&mut Qwen4ExpMtpProposerState],
        ctx: &ForwardContext,
        stream: u64,
        out_conf: Option<&mut Vec<Vec<f32>>>,
        tp: Option<&draft_tp::TpRun<'_>>,
    ) -> Result<Vec<Vec<u32>>> {
        let n = last_tokens.len();
        static LOGGED_N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let bit = 1u32 << (n & 31);
        if LOGGED_N.fetch_or(bit, std::sync::atomic::Ordering::Relaxed) & bit == 0 {
            tracing::info!(
                "qwen4_exp MTP propose_batch active: n={n} drafts={num_drafts} \
                 (kill switch ATLAS_NO_MTP_BATCH_PROPOSE)"
            );
        }
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        for st in sts.iter_mut() {
            super::qwen4exp_mtp_kv::settle_unverified(st);
            let need = (st.seq_len + num_drafts - 1) / bs + 1;
            while st.block_table.len() < need {
                st.block_table.push(kv_cache.alloc_block()?);
            }
        }
        let kv_lens: Vec<usize> = sts.iter().map(|st| st.seq_len).collect();
        let tables: Vec<Vec<u32>> = sts.iter().map(|st| st.block_table.clone()).collect();
        let slab = pack_slab(
            last_tokens,
            positions,
            &kv_lens,
            &tables,
            num_drafts,
            bs,
            self.batch_slab_blocks,
        )?;
        ctx.gpu
            .copy_h2d_async(&slab.bytes, self.batch_slab, stream)?;

        let want_lp = (self.conf_stop > 0.0 || out_conf.is_some()) && self.argmax_batch_lp_k.0 != 0;
        let banned: Vec<u32> = sts
            .iter()
            .zip(positions)
            .map(|(st, &p)| crate::traits::EosBan::banned_draft_depth(st.end_floor, p, num_drafts))
            .collect();
        for j in 0..num_drafts {
            let seq_lens: Vec<usize> = kv_lens.iter().map(|&l| l + j).collect();
            self.forward_rows(
                j,
                n,
                num_drafts,
                slab.max_blocks,
                &seq_lens,
                &tables,
                sts,
                &mut kv_cache,
                ctx,
                stream,
                want_lp,
                tp,
                &banned,
            )?;
        }
        drop(kv_cache);

        // The one sync of the call.
        let lp_at = lp_off(n, num_drafts);
        let mut buf = vec![0u8; lp_at + if want_lp { num_drafts * n * 4 } else { 0 }];
        ctx.gpu.copy_d2h(self.batch_slab, &mut buf)?;
        let mut drafts = per_seq(&buf, n * 4, num_drafts, n, u32::from_le_bytes);
        let lps = if want_lp {
            per_seq(&buf, lp_at, num_drafts, n, f32::from_le_bytes)
        } else {
            vec![vec![0.0; num_drafts]; n]
        };
        for (i, st) in sts.iter_mut().enumerate() {
            let probs: Vec<f32> = lps[i].iter().map(|lp| lp.exp()).collect();
            let kept = kept_drafts(&probs, self.conf_stop);
            if kept < num_drafts {
                tracing::debug!(
                    "qwen4_exp MTP confidence stop at draft {kept}: p={:.3} < {}",
                    probs[kept],
                    self.conf_stop
                );
            }
            drafts[i].truncate(kept);
            st.seq_len = kv_lens[i] + kept;
            st.last_num_drafted = kept;
            st.awaiting_verdict = true;
        }
        if let Some(c) = out_conf {
            *c = lps
                .into_iter()
                .zip(&drafts)
                .map(|(mut lp, d)| {
                    lp.truncate(d.len());
                    lp
                })
                .collect();
        }
        Ok(drafts)
    }

    /// Draft position `j` for all n rows: `forward_one`'s six steps at M = n,
    /// reading the tokens of slab row j and writing the argmax to row j+1.
    #[allow(clippy::too_many_arguments)]
    fn forward_rows(
        &self,
        j: usize,
        n: usize,
        num_drafts: usize,
        max_blocks: usize,
        seq_lens: &[usize],
        tables: &[Vec<u32>],
        sts: &mut [&mut Qwen4ExpMtpProposerState],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        want_lp: bool,
        tp: Option<&draft_tp::TpRun<'_>>,
        banned: &[u32],
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        let hc = ctx.config.hc_mult.max(1);
        let eps = ctx.config.rms_norm_eps as f32;
        let streams = ctx.buffers.hc_streams();
        let (h32, rows32, n32) = (h as u32, (n * hc) as u32, n as u32);

        // ── 1. Per-stream grouped norm of each row's incoming residual ──
        self.stream_norms(ctx, streams, n * hc, stream)?;
        // ── 2. Per-stream projection, every (row, stream) in one weight pass ──
        self.wide_rows.dense_rows(
            gpu,
            self.dense_gemv_batchm_k,
            self.normed_h,
            &self.module.fc_hidden,
            self.h_streams,
            (rows32, h32, h32, h32),
            stream,
        )?;
        // ── 3. Embedding branch from the device token row ──
        ops::batched_embed(
            gpu,
            self.batched_embed_k,
            self.batch_slab.offset(j * n * 4),
            self.embed_tokens.weight,
            self.embed_buf,
            n32,
            h32,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            self.embed_buf,
            &self.module.pre_fc_norm_embedding,
            self.normed_e,
            n32,
            h32,
            eps,
            stream,
        )?;
        self.wide_rows.dense_rows(
            gpu,
            self.dense_gemv_batchm_k,
            self.normed_e,
            &self.module.fc_embedding,
            self.e_branch,
            (n32, h32, h32, h32),
            stream,
        )?;
        ops::hc_expand(
            gpu,
            self.hc_expand_k,
            self.e_branch,
            streams,
            n32,
            h32,
            hc as u32,
            stream,
        )?;
        self.f32_add_bf16(gpu, streams, self.h_streams, rows32 * h32, stream)?;

        // ── 4. Body: the layer's multi-sequence decode, one row a sequence ──
        let hdr = self.batch_slab.offset(HDR_OFF + j * HDR_BYTES);
        let meta = AttnMetadataDev {
            positions: hdr,
            positions_h: hdr,
            positions_w: hdr,
            slot: hdr.offset(HDR_SLOT),
            seq_len: hdr.offset(HDR_SEQ_LEN),
            block_table: self.batch_slab.offset(BT_OFF),
            max_blocks_per_seq: max_blocks as u32,
            num_seqs: n32,
            seq_slot: DevicePtr(0),
            moe_row_adapter: DevicePtr::NULL,
        };
        let body_ctx = ForwardContext {
            ssm_batch: None,
            buffers: ctx.buffers,
            gpu,
            // The body's own (TP=1) config, as in `forward_one`.
            config: &self.module.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(meta),
            profile: ctx.profile,
            // Rank 0 only, every head and expert local: no collective.
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
        let mut states: Vec<&mut (dyn LayerState + 'static)> =
            sts.iter_mut().map(|st| st.body_state.as_mut()).collect();
        self.module.body.decode_multi_seq(
            ctx.buffers.hidden_states(),
            ctx.buffers.residual(),
            n,
            n,
            &mut states,
            kv_cache,
            seq_lens,
            tables,
            &body_ctx,
            stream,
        )?;

        // ── 5. Head mixer: collapse + final norm, n rows ──
        let lowrank = self
            .module
            .hc_head
            .as_ref()
            .and_then(|head| head.lowrank.as_ref())
            .ok_or_else(|| anyhow::anyhow!("qwen4exp_mtp: no low-rank mHC head"))?;
        ops::hc_head_lowrank(
            gpu,
            self.hc_head_k,
            streams,
            lowrank,
            self.h_out,
            ctx.buffers.hc_lowrank_scratch(),
            n32,
            h32,
            hc as u32,
            eps,
            stream,
        )?;

        // ── 6. Draft head, end-token ban, argmax (`qwen4exp_mtp_ban.rs`) ──
        self.draft_argmax_rows(j, n, num_drafts, banned, ctx, stream, want_lp, tp)
    }
}

#[cfg(test)]
#[path = "qwen4exp_mtp_batch_tests.rs"]
mod tests;

#[cfg(all(test, feature = "cuda"))]
#[path = "qwen4exp_mtp_batch_gpu_tests.rs"]
mod gpu_tests;
