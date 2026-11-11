// SPDX-License-Identifier: AGPL-3.0-only
//!
//! Ported from Enntity/atlas (AGPL-3.0-only, sparkqwen drafter series).

//! The drafter absorbs the target's verified rows
//! (`ATLAS_QWEN4EXP_MTP_ABSORB=1`, default off: an acceptance switch; drafts
//! only propose, the target verifies every one, so no output can change).
//!
//! A propose at position `Q` writes drafter KV rows at `Q .. Q + d - 1`: row
//! 0 from the target's stream `h_{Q-1}` and the committed token `t_Q`, row
//! `i > 0` from the drafter's OWN output stream of row `i - 1` and draft
//! `d_{i-1}`. The verify then runs the target over `t_Q, d_0, ..`, so its row
//! `j` leaves the target's `h_{Q+j}` on the stream highway. After a verdict
//! accepting `na` drafts, drafter rows `1 ..= na` pair committed tokens with
//! the drafter's chained streams: not what the target computed there, and
//! never corrected later. vLLM's and TensorFold's drafters do not keep them:
//! after each verify they re-run the MTP layer over the verified window with
//! the target's streams (TensorFold `absorb`), so the drafter's context is
//! exact for every committed token. This does the same:
//!
//! 1. `Qwen4ExpMtpHead::stage_absorb_impl`, after the verify and before any
//!    forward reuses its buffers: copy verify rows `0 .. na` of the target's
//!    final streams and the `na` accepted drafts into the sequence's stage.
//! 2. The verdict keeps at least row 0 (the switch implies
//!    `ATLAS_QWEN4EXP_MTP_KV_KEEP`'s row rule, `qwen4exp_mtp_kv.rs`).
//! 3. `Qwen4ExpMtpHead::absorb_before_propose`, at the next propose: rewind
//!    the drafter to row 0 and write positions `Q + 1 ..= Q + na` from the
//!    staged `(h_{Q+j}, d_j)` pairs, KV only
//!    (`TransformerLayer::decode_rows_kv_only`: a one-layer drafter's KV row
//!    is a function of its own input row, so a full forward's attention
//!    read, MoE and head would change nothing it keeps). The propose then
//!    drafts as before, its row 0 at `Q + na + 1` from verify row `na`.
//!
//! The stage, the verdict and the last propose's row-0 record must agree
//! (same count, after that propose); otherwise the stage is dropped and the
//! drafter keeps the keep-committed rows, as without the switch.
//!
//! Rank-local: only rank 0 drafts (the body runs every head and expert with
//! `comm: None`), so this adds nothing the TP worker must mirror.

use super::*;

/// Failures of the absorb so far (`warn_sparsely`).
static ABSORB_FAILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Log the 1st, 2nd, 4th, 8th ... occurrence counted by `count`: a failure
/// that repeats every step stays visible without a line a step.
pub(crate) fn warn_sparsely(count: &std::sync::atomic::AtomicU64, msg: impl FnOnce() -> String) {
    let n = count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::warn!("{} ({n} so far)", msg());
    }
}

/// `ATLAS_QWEN4EXP_MTP_ABSORB=1`, read once.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MTP_ABSORB").as_deref() == Ok("1"))
}

/// Verified rows a stage holds at most: the deepest verify's drafts.
const STAGE_ROWS: usize = qwen4exp_mtp_batch::PROPOSE_BATCH_MAX_DRAFTS;

/// A sequence's absorb state.
pub struct Absorb {
    /// The last propose's row 0: (drafter KV row, sequence position).
    row0: Option<(usize, usize)>,
    /// The accepted drafts staged with their verify rows (rows of `buf`), or
    /// the prompt tail's token.
    pub(super) tokens: Vec<u32>,
    /// The accepted count of the verdict applied since that propose.
    pub(super) verdict: Option<usize>,
    /// `[STAGE_ROWS, hc, hidden]` FP32 from the head's pool; NULL until the
    /// first stage.
    pub(super) buf: DevicePtr,
    /// This prefill chunk's highway rows (`qwen4exp_mtp_prompt.rs`):
    /// `(first position, rows, first highway row)`.
    pub(super) prompt: Option<(usize, usize, usize)>,
    /// The position of the prompt row staged in `buf` row 0 whose token was
    /// not known yet (`qwen4exp_mtp_prompt.rs`).
    pub(super) tail: Option<usize>,
}

impl Default for Absorb {
    fn default() -> Self {
        Self {
            row0: None,
            tokens: Vec::new(),
            verdict: None,
            buf: DevicePtr::NULL,
            prompt: None,
            tail: None,
        }
    }
}

/// What the next propose absorbs: `(first drafter row, first position, rows)`,
/// when the stage, the verdict and the row-0 record agree.
pub(super) fn plan(a: &Absorb, seq_len: usize) -> Option<(usize, usize, usize)> {
    // The prompt's last row, its token known by now: after the prompt rows.
    if let Some(pos) = a.tail {
        return (a.tokens.len() == 1).then_some((seq_len, pos, 1));
    }
    let (row0, pos0) = a.row0?;
    let na = a.tokens.len();
    (na > 0 && a.verdict == Some(na) && seq_len > row0).then_some((row0 + 1, pos0 + 1, na))
}

/// The head's absorb buffers.
pub(super) struct AbsorbBufs {
    /// The propose's own input rows, kept across the absorb's use of the
    /// highway (`PROPOSE_BATCH_MAX` rows; NULL with the switch off).
    save: DevicePtr,
    /// Stage buffers of finished sequences, for reuse.
    pool: Mutex<Vec<DevicePtr>>,
    /// One highway row: `hc * hidden` FP32.
    row_bytes: usize,
    /// `ATLAS_QWEN4EXP_MTP_PREFILL` (`qwen4exp_mtp_prompt.rs`).
    pub(super) prompt: Option<super::qwen4exp_mtp_prompt::PromptBufs>,
}

impl AbsorbBufs {
    pub(super) fn new(gpu: &dyn GpuBackend, hc: usize, h: usize) -> Result<Self> {
        let row_bytes = hc * h * 4;
        let save = if enabled() || super::qwen4exp_mtp_prompt::enabled() {
            gpu.alloc(qwen4exp_mtp_batch::PROPOSE_BATCH_MAX * row_bytes)?
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            save,
            pool: Mutex::new(Vec::new()),
            row_bytes,
            prompt: super::qwen4exp_mtp_prompt::PromptBufs::new(gpu, hc, h)?,
        })
    }

    pub(super) fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// A stage buffer: a finished sequence's, else a new one.
    pub(super) fn take_buf(&self, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
        match self.pool.lock().pop() {
            Some(buf) => Ok(buf),
            None => gpu.alloc(STAGE_ROWS * self.row_bytes),
        }
    }

    /// Return a finished sequence's stage buffer to the pool.
    pub(super) fn release(&self, a: &mut Absorb) {
        if !a.buf.is_null() {
            self.pool.lock().push(a.buf);
        }
        *a = Absorb::default();
    }
}

impl Qwen4ExpMtpHead {
    /// Stage the verify rows at `streams` (`tokens.len()` highway rows, the
    /// target's) with the accepted drafts `tokens`. Any earlier stage goes.
    pub(super) fn stage_absorb_impl(
        &self,
        state: &mut dyn ProposerState,
        streams: DevicePtr,
        tokens: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some(st) = state
            .as_any_mut()
            .downcast_mut::<Qwen4ExpMtpProposerState>()
        else {
            return Ok(());
        };
        st.absorb.tokens.clear();
        if !enabled() || tokens.is_empty() || tokens.len() > STAGE_ROWS {
            return Ok(());
        }
        let rb = self.absorb.row_bytes;
        if st.absorb.buf.is_null() {
            st.absorb.buf = self.absorb.take_buf(gpu)?;
        }
        gpu.copy_d2d_async(streams, st.absorb.buf, tokens.len() * rb, stream)?;
        st.absorb.tokens.extend_from_slice(tokens);
        Ok(())
    }

    /// Before a propose of `sts` at `positions`, whose input rows are highway
    /// rows `0..keep`: absorb each agreed stage (or the prompt tail), then
    /// record each propose's row 0. A no-op with both switches off.
    pub(super) fn absorb_before_propose(
        &self,
        sts: &mut [&mut Qwen4ExpMtpProposerState],
        positions: &[usize],
        keep: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !(enabled() || super::qwen4exp_mtp_prompt::enabled()) || !self.absorb_supported() {
            return Ok(());
        }
        let plans: Vec<_> = sts.iter().map(|st| plan(&st.absorb, st.seq_len)).collect();
        if plans.iter().any(Option::is_some)
            && let Err(e) = self.absorb_rows(sts, &plans, keep, ctx, stream)
        {
            warn_sparsely(&ABSORB_FAILED, || {
                format!("qwen4_exp MTP absorb failed (drafting without it): {e:#}")
            });
        }
        for (st, &position) in sts.iter_mut().zip(positions) {
            st.absorb.tokens.clear();
            st.absorb.verdict = None;
            st.absorb.tail = None;
            st.absorb.row0 = Some((st.seq_len, position));
        }
        Ok(())
    }

    /// Whether the absorb and the prompt rows can run: the batched row
    /// kernels, the save buffer, and a body with a KV-only decode (the
    /// drafter indexer of `ATLAS_MTP_DRAFTER_QSA=1` has none). Said once
    /// when not; the switches then leave the drafter as without them.
    pub(super) fn absorb_supported(&self) -> bool {
        let ok = self.dense_gemv_batchm_k.0 != 0
            && self.batched_embed_k.0 != 0
            && !self.absorb.save.is_null()
            && !self.module.body.has_aux_state();
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ok && !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(
                "qwen4_exp MTP absorb / prompt rows inactive: no batched row kernels or no \
                 KV-only drafter decode (ATLAS_MTP_DRAFTER_QSA=1 gives the drafter an indexer)"
            );
        }
        ok
    }

    /// Write every planned row, at most `PROPOSE_BATCH_MAX` per pass. The
    /// propose's input rows come back whatever happens.
    fn absorb_rows(
        &self,
        sts: &mut [&mut Qwen4ExpMtpProposerState],
        plans: &[Option<(usize, usize, usize)>],
        keep: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            keep <= qwen4exp_mtp_batch::PROPOSE_BATCH_MAX
                && self.dense_gemv_batchm_k.0 != 0
                && self.batched_embed_k.0 != 0
                && !self.absorb.save.is_null(),
            "qwen4_exp MTP absorb: unsupported here (keep={keep}, batch kernels)"
        );
        let (gpu, rb) = (ctx.gpu, self.absorb.row_bytes);
        let streams = ctx.buffers.hc_streams();
        let mut rows = Vec::new();
        for (i, (st, p)) in sts.iter_mut().zip(plans).enumerate() {
            if let Some((first, _, na)) = *p {
                // Back to row 0: a failed pass leaves no chained row behind.
                st.seq_len = first;
                rows.extend((0..na).map(|j| (i, j)));
            }
        }
        gpu.copy_d2d_async(streams, self.absorb.save, keep * rb, stream)?;
        let done = self.absorb_passes(sts, plans, &rows, ctx, stream);
        gpu.copy_d2d_async(self.absorb.save, streams, keep * rb, stream)?;
        done?;
        for (st, p) in sts.iter_mut().zip(plans) {
            if let Some((first, _, na)) = *p {
                st.seq_len = first + na;
            }
        }
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "qwen4_exp MTP absorb active: {} verified rows rewritten from the target",
                rows.len()
            );
        }
        Ok(())
    }

    /// The passes of [`Self::absorb_rows`]: `rows` = (state, staged row).
    fn absorb_passes(
        &self,
        sts: &mut [&mut Qwen4ExpMtpProposerState],
        plans: &[Option<(usize, usize, usize)>],
        rows: &[(usize, usize)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (gpu, rb) = (ctx.gpu, self.absorb.row_bytes);
        let streams = ctx.buffers.hc_streams();
        let mut kv = self.kv_cache.lock();
        let bs = kv.block_size();
        for chunk in rows.chunks(qwen4exp_mtp_batch::PROPOSE_BATCH_MAX) {
            let n = chunk.len();
            let (mut tokens, mut pos, mut kv_rows, mut tables) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (r, &(i, j)) in chunk.iter().enumerate() {
                let (first, pos0, _) = plans[i].expect("planned row");
                let st = &mut *sts[i];
                while st.block_table.len() < (first + j) / bs + 1 {
                    st.block_table.push(kv.alloc_block()?);
                }
                let src = st.absorb.buf.offset(j * rb);
                gpu.copy_d2d_async(src, streams.offset(r * rb), rb, stream)?;
                tokens.push(st.absorb.tokens[j]);
                pos.push(pos0 + j);
                kv_rows.push(first + j);
                tables.push(st.block_table.clone());
            }
            let slab = qwen4exp_mtp_batch::pack_slab(
                &tokens,
                &pos,
                &kv_rows,
                &tables,
                1,
                bs,
                self.batch_slab_blocks,
            )?;
            gpu.copy_h2d_async(&slab.bytes, self.batch_slab, stream)?;
            self.stage_rows_input(&self.own_scratch(), self.batch_slab, n, ctx, stream)?;
            let meta = qwen4exp_mtp_batch::slab_meta(self.batch_slab, 0, n, slab.max_blocks);
            let written = self.module.body.decode_rows_kv_only(
                crate::layer::KvRows {
                    hidden: ctx.buffers.hidden_states(),
                    residual: ctx.buffers.residual(),
                    n,
                    seq_lens: &kv_rows,
                    stream,
                },
                &mut kv,
                &self.body_ctx(ctx, meta),
            )?;
            anyhow::ensure!(
                written,
                "qwen4_exp MTP absorb: the body has no KV-only decode"
            );
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "qwen4exp_mtp_kv_only_gpu_tests.rs"]
mod kv_only_gpu_tests;

#[cfg(test)]
mod tests {
    use super::{Absorb, plan};

    fn absorb(row0: Option<(usize, usize)>, tokens: &[u32], verdict: Option<usize>) -> Absorb {
        Absorb {
            row0,
            tokens: tokens.to_vec(),
            verdict,
            ..Absorb::default()
        }
    }

    #[test]
    fn absorbs_the_accepted_drafts_after_row_zero() {
        // Propose at Q = 40 wrote drafter row 9; 3 of its drafts accepted.
        let a = absorb(Some((9, 40)), &[7, 8, 9], Some(3));
        assert_eq!(plan(&a, 13), Some((10, 41, 3)));
        // Full accept of a 3-draft propose: the kept rows are 9..=11.
        assert_eq!(plan(&a, 12), Some((10, 41, 3)));
    }

    #[test]
    fn nothing_to_absorb_unless_stage_verdict_and_propose_agree() {
        // Full reject: nothing accepted, row 0 stays.
        assert_eq!(plan(&absorb(Some((9, 40)), &[], Some(0)), 10), None);
        // Stage without its verdict (or a different count).
        assert_eq!(plan(&absorb(Some((9, 40)), &[7, 8], None), 11), None);
        assert_eq!(plan(&absorb(Some((9, 40)), &[7, 8], Some(1)), 11), None);
        // No propose recorded (a sequence's first propose).
        assert_eq!(plan(&absorb(None, &[7], Some(1)), 1), None);
        // The drafter lost row 0 (a reset): nothing to append after.
        assert_eq!(plan(&absorb(Some((9, 40)), &[7], Some(1)), 9), None);
    }

    #[test]
    fn the_prompt_tail_appends_once_its_token_is_known() {
        let mut a = absorb(None, &[], None);
        a.tail = Some(57);
        // Token not handed over yet: nothing.
        assert_eq!(plan(&a, 56), None);
        a.tokens = vec![9];
        // After the 56 prompt rows, at the prompt's last position.
        assert_eq!(plan(&a, 56), Some((56, 57, 1)));
    }
}
