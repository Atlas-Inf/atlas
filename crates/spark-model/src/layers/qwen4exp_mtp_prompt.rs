// SPDX-License-Identifier: AGPL-3.0-only

//! The drafter's KV over the prompt (`ATLAS_QWEN4EXP_MTP_PREFILL=<rows>`,
//! default off: an acceptance switch; drafts only propose).
//!
//! Without it the drafter's KV starts empty at each request and holds only
//! the rows its own decode-time proposes write, so its attention never sees
//! the prompt. vLLM and TensorFold run the MTP layer over the prompt at the
//! end of prefill, pairing the target's final streams `h_i` with the next
//! token `t_{i+1}`. This does the same, KV only (`qwen4exp_mtp_absorb.rs`
//! has why that suffices), over the last `<rows>` prompt positions:
//!
//! 1. At each prefill chunk's capture point the model notes which highway
//!    rows hold the chunk's final streams ([`Qwen4ExpMtpHead::note_prompt_rows_impl`]).
//! 2. After the chunk's prefill returns (its logits computed: the pass
//!    reuses the model's activation buffers) the drafter writes position
//!    `p + 1` from row `p` and prompt token `t_{p+1}`
//!    ([`Qwen4ExpMtpHead::absorb_prompt_impl`]), in passes of [`PASS_ROWS`].
//! 3. The chunk's last row has no next token yet when it is the prompt's
//!    last: it is staged, and the first propose (whose sequence holds the
//!    sampled token) appends it ([`Qwen4ExpMtpHead::prompt_tail_token_impl`]).
//!    A middle chunk's last row pairs with the next chunk's first token.
//!
//! What it cannot see, and leaves blind: a restored prefix (prefix cache,
//! warm turn) computed nothing this request, so it has no streams; only the
//! computed suffix is written. Under the sequence-parallel prefill
//! (`ATLAS_QWEN4EXP_PREFILL_SP`) rank 0 holds the final streams of its half
//! of a split chunk only, so only those rows are written.
//!
//! Cost: per prompt row, the drafter's row-wise front (two GEMMs, the mHC
//! pre-site, QKV) — no attention, MoE or head. The drafter KV pool is one
//! sequence's worth (`max_seq_len` rows) shared by every sequence, so `<rows>`
//! bounds what each prompt takes, and a chunk that would leave less than an
//! eighth of the pool free is not written.
//!
//! Rank-local like the absorb: only rank 0 holds the drafter.

use super::qwen4exp_mtp_rows::RowScratch;
use super::*;

/// `ATLAS_QWEN4EXP_MTP_PREFILL=<rows>`: the prompt positions the drafter
/// takes (the last ones); 0 / unset = off. Read once.
pub(crate) fn window() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_MTP_PREFILL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

pub(crate) fn enabled() -> bool {
    window() > 0
}

/// Rows per drafter pass over the prompt.
pub(super) const PASS_ROWS: usize = 64;

/// Prompt chunks the drafter KV pool was short for (`warn_sparsely`).
static POOL_SHORT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `ATLAS_QWEN4EXP_MTP_PREFILL` replaces the drafter-context prefill's
/// obsolete opt-in, which asks for drafter KV over the prompt too: refuse the
/// pair at startup instead of leaving an operator unsure which one ran. (The
/// default drafter context never writes this proposer's KV: it implements no
/// `prefill_drafter` / `install_drafter_kv`.)
pub fn check_exclusive() -> Result<()> {
    anyhow::ensure!(
        !(enabled() && std::env::var_os("ATLAS_MTP_DRAFTER_PREFILL").is_some()),
        "ATLAS_QWEN4EXP_MTP_PREFILL and ATLAS_MTP_DRAFTER_PREFILL both ask for drafter KV \
         over the prompt; unset ATLAS_MTP_DRAFTER_PREFILL"
    );
    Ok(())
}

/// Where a prompt pair's stream row is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Src {
    /// The stage buffer's row 0: the previous chunk's last row, held for the
    /// next chunk's first token. Rare: only a prompt longer than one chunk
    /// takes it, once per chunk boundary.
    Stage,
    /// A highway row of this chunk.
    Row(usize),
}

/// The pairs a chunk writes, `(stream row, position, token)`, and the row
/// left for a later token: positions below `lo` are skipped.
pub(super) fn prompt_pairs(
    prompt: &[u32],
    tail: Option<usize>,
    (first_pos, rows, hc_row): (usize, usize, usize),
    lo: usize,
) -> (Vec<(Src, usize, u32)>, Option<(usize, usize)>) {
    let mut pairs = Vec::new();
    if let Some(pos) = tail.filter(|&p| p >= lo && p < prompt.len()) {
        pairs.push((Src::Stage, pos, prompt[pos]));
    }
    let mut left = None;
    for r in 0..rows {
        let p = first_pos + r + 1;
        if p < lo {
            continue;
        }
        match prompt.get(p) {
            Some(&t) => pairs.push((Src::Row(hc_row + r), p, t)),
            None if p == prompt.len() => left = Some((hc_row + r, p)),
            None => {}
        }
    }
    (pairs, left)
}

/// A pass's input copies: `(destination row, source, rows)` runs of
/// consecutive source rows, none crossing `split` (rows below it read from
/// the save).
pub(super) fn copy_runs(chunk: &[(Src, usize, u32)], split: usize) -> Vec<(usize, Src, usize)> {
    let mut runs: Vec<(usize, Src, usize)> = Vec::new();
    for (i, &(src, _, _)) in chunk.iter().enumerate() {
        if let (Some((_, Src::Row(r0), n)), Src::Row(r)) = (runs.last_mut(), src)
            && *r0 + *n == r
            && (r < split) == (*r0 < split)
        {
            *n += 1;
            continue;
        }
        runs.push((i, src, 1));
    }
    runs
}

/// The head's prompt buffers (allocated with the switch on).
pub(super) struct PromptBufs {
    scratch: RowScratch,
    /// Highway rows `0..PASS_ROWS` across the passes.
    save: DevicePtr,
    /// positions u32 `[PASS_ROWS]` | slots i64 `[PASS_ROWS]`.
    meta: DevicePtr,
    /// Token ids u32 `[PASS_ROWS]`.
    tokens: DevicePtr,
}

impl PromptBufs {
    pub(super) fn new(gpu: &dyn GpuBackend, hc: usize, h: usize) -> Result<Option<Self>> {
        if !enabled() {
            return Ok(None);
        }
        Ok(Some(Self {
            scratch: RowScratch::alloc(gpu, PASS_ROWS, h, hc)?,
            save: gpu.alloc(PASS_ROWS * hc * h * 4)?,
            meta: gpu.alloc(PASS_ROWS * 12)?,
            tokens: gpu.alloc(PASS_ROWS * 4)?,
        }))
    }
}

fn downcast(state: &mut dyn ProposerState) -> Option<&mut Qwen4ExpMtpProposerState> {
    state
        .as_any_mut()
        .downcast_mut::<Qwen4ExpMtpProposerState>()
}

impl Qwen4ExpMtpHead {
    /// Highway rows `hc_row ..` hold the final streams of prompt positions
    /// `first_pos ..` (`rows` of them) until the chunk's next forward.
    pub(super) fn note_prompt_rows_impl(
        &self,
        state: &mut dyn ProposerState,
        first_pos: usize,
        rows: usize,
        hc_row: usize,
    ) {
        if let (Some(st), true) = (downcast(state), self.absorb.prompt.is_some()) {
            st.absorb.prompt = (rows > 0).then_some((first_pos, rows, hc_row));
        }
    }

    /// At a propose: the staged prompt tail's token, once `tokens` (the
    /// sequence's) holds it. A note no prefill consumed is stale: dropped.
    pub(super) fn prompt_tail_token_impl(&self, state: &mut dyn ProposerState, tokens: &[u32]) {
        let Some(st) = downcast(state) else {
            return;
        };
        st.absorb.prompt = None;
        if let Some(&t) = st.absorb.tail.and_then(|p| tokens.get(p)) {
            st.absorb.tokens = vec![t];
        }
    }

    /// After a prefill chunk returns: write the noted rows of `prompt` (the
    /// whole prompt's tokens) into the drafter KV.
    pub(super) fn absorb_prompt_impl(
        &self,
        state: &mut dyn ProposerState,
        prompt: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (Some(st), Some(bufs)) = (downcast(state), self.absorb.prompt.as_ref()) else {
            return Ok(());
        };
        let Some(note) = st.absorb.prompt.take() else {
            return Ok(());
        };
        if !self.absorb_supported() {
            return Ok(());
        }
        let lo = prompt.len().saturating_sub(window());
        let (pairs, left) = prompt_pairs(prompt, st.absorb.tail.take(), note, lo);
        let (gpu, rb) = (ctx.gpu, self.absorb.row_bytes());
        let streams = ctx.buffers.hc_streams();
        // The passes save and restore highway rows `0..PASS_ROWS` and read the
        // chunk's rows `hc_row ..`, all inside the highway.
        let (_, rows, hc_row) = note;
        anyhow::ensure!(
            PASS_ROWS.max(hc_row + rows) * rb <= ctx.buffers.sizes().hc_streams,
            "qwen4_exp MTP prefill: rows {hc_row}+{rows} (or the {PASS_ROWS}-row save) exceed hc_streams"
        );
        if !pairs.is_empty() {
            let mut kv = self.kv_cache.lock();
            let bs = kv.block_size();
            let need = (st.seq_len + pairs.len()) / bs + 1;
            let fresh = need.saturating_sub(st.block_table.len());
            if kv.num_free_blocks() < fresh + kv.num_blocks() / 8 {
                let n = pairs.len();
                super::qwen4exp_mtp_absorb::warn_sparsely(&POOL_SHORT, || {
                    format!(
                        "qwen4_exp MTP prefill: {n} prompt rows skipped, the drafter KV pool is short"
                    )
                });
            } else {
                while st.block_table.len() < need {
                    st.block_table.push(kv.alloc_block()?);
                }
                gpu.copy_d2d_async(streams, bufs.save, PASS_ROWS * rb, stream)?;
                let done = self.prompt_passes(st, bufs, &pairs, &mut kv, ctx, stream);
                gpu.copy_d2d_async(bufs.save, streams, PASS_ROWS * rb, stream)?;
                done?;
            }
        }
        // The last row waits for its token (the next chunk's, or the first
        // propose's sampled one), staged now: the next forward reuses the row.
        if let Some((row, pos)) = left {
            if st.absorb.buf.is_null() {
                st.absorb.buf = self.absorb.take_buf(gpu)?;
            }
            gpu.copy_d2d_async(streams.offset(row * rb), st.absorb.buf, rb, stream)?;
            st.absorb.tokens.clear();
            st.absorb.tail = Some(pos);
        }
        Ok(())
    }

    /// The passes over `pairs`: highway rows `0..n` take each pass's input
    /// (rows below `PASS_ROWS` read from the save, which the caller made).
    fn prompt_passes(
        &self,
        st: &mut Qwen4ExpMtpProposerState,
        bufs: &PromptBufs,
        pairs: &[(Src, usize, u32)],
        kv: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (gpu, rb) = (ctx.gpu, self.absorb.row_bytes());
        let streams = ctx.buffers.hc_streams();
        let bs = kv.block_size();
        for chunk in pairs.chunks(PASS_ROWS) {
            let n = chunk.len();
            let mut meta = vec![0u8; PASS_ROWS * 12];
            let mut toks = Vec::with_capacity(n * 4);
            let mut kv_rows = Vec::with_capacity(n);
            for (dst, from, rows) in copy_runs(chunk, PASS_ROWS) {
                let from = match from {
                    Src::Stage => st.absorb.buf,
                    Src::Row(r) if r < PASS_ROWS => bufs.save.offset(r * rb),
                    Src::Row(r) => streams.offset(r * rb),
                };
                gpu.copy_d2d_async(from, streams.offset(dst * rb), rows * rb, stream)?;
            }
            for (i, &(_, pos, tok)) in chunk.iter().enumerate() {
                let row = st.seq_len + i;
                let slot = st.block_table[row / bs] as i64 * bs as i64 + (row % bs) as i64;
                meta[i * 4..i * 4 + 4].copy_from_slice(&(pos as u32).to_le_bytes());
                let at = PASS_ROWS * 4 + i * 8;
                meta[at..at + 8].copy_from_slice(&slot.to_le_bytes());
                toks.extend_from_slice(&tok.to_le_bytes());
                kv_rows.push(row);
            }
            gpu.copy_h2d_async(&meta, bufs.meta, stream)?;
            gpu.copy_h2d_async(&toks, bufs.tokens, stream)?;
            self.stage_rows_input(&bufs.scratch, bufs.tokens, n, ctx, stream)?;
            // Only the attention input, RoPE and the cache write read
            // metadata on this path: positions and slots.
            let meta = AttnMetadataDev {
                positions: bufs.meta,
                positions_h: bufs.meta,
                positions_w: bufs.meta,
                slot: bufs.meta.offset(PASS_ROWS * 4),
                seq_len: DevicePtr::NULL,
                block_table: DevicePtr::NULL,
                max_blocks_per_seq: 0,
                num_seqs: n as u32,
                seq_slot: DevicePtr(0),
                moe_row_adapter: DevicePtr::NULL,
            };
            let written = self.module.body.decode_rows_kv_only(
                ctx.buffers.hidden_states(),
                ctx.buffers.residual(),
                n,
                kv,
                &kv_rows,
                &self.body_ctx(ctx, meta),
                stream,
            )?;
            anyhow::ensure!(
                written,
                "qwen4_exp MTP prefill: the body has no KV-only decode"
            );
            st.seq_len += n;
        }
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "qwen4_exp MTP prefill active: {} prompt rows written to the drafter KV",
                pairs.len()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Src, prompt_pairs};

    #[test]
    fn a_chunk_pairs_each_row_with_the_next_prompt_token() {
        let prompt = [10, 11, 12, 13, 14, 15];
        // Chunk 0..4 at highway row 0: rows 0..3 -> positions 1..4.
        let (pairs, left) = prompt_pairs(&prompt, None, (0, 4, 0), 0);
        assert_eq!(
            pairs,
            vec![
                (Src::Row(0), 1, 11),
                (Src::Row(1), 2, 12),
                (Src::Row(2), 3, 13),
                (Src::Row(3), 4, 14)
            ]
        );
        assert_eq!(left, None);
        // The last chunk: its last row waits for the sampled token.
        let (pairs, left) = prompt_pairs(&prompt, None, (4, 2, 5), 0);
        assert_eq!(pairs, vec![(Src::Row(5), 5, 15)]);
        assert_eq!(left, Some((6, 6)));
    }

    #[test]
    fn copies_run_over_consecutive_rows_and_split_at_the_save() {
        use super::copy_runs;
        let p = |s| (s, 0, 0);
        let chunk = [
            p(Src::Stage),
            p(Src::Row(62)),
            p(Src::Row(63)),
            p(Src::Row(64)),
            p(Src::Row(65)),
        ];
        assert_eq!(
            copy_runs(&chunk, 64),
            vec![
                (0, Src::Stage, 1),
                (1, Src::Row(62), 2),
                (3, Src::Row(64), 2)
            ]
        );
    }

    #[test]
    fn a_staged_row_goes_first_and_the_window_skips_old_positions() {
        let prompt = [10, 11, 12, 13, 14, 15];
        let (pairs, _) = prompt_pairs(&prompt, Some(3), (3, 2, 0), 0);
        assert_eq!(pairs[0], (Src::Stage, 3, 13));
        assert_eq!(pairs.len(), 3);
        // A window of the prompt's last position only (`lo` = 5): the staged
        // row and positions 4 go, position 5 is written, 6 waits.
        let (pairs, left) = prompt_pairs(&prompt, Some(3), (3, 3, 0), 5);
        assert_eq!(pairs, vec![(Src::Row(1), 5, 15)]);
        assert_eq!(left, Some((2, 6)));
    }
}
