// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched long verify diagnostics: the per-layer serial fallback
//! (`ATLAS_GLM_LONG_BATCH_SERIAL`) and the batched-vs-serial oracle
//! (`ATLAS_GLM_LONG_BATCH_ORACLE`).

use super::*;

/// `ATLAS_GLM_LONG_BATCH_ALIAS_CHECK`: every physical block one owner writes
/// this step that is held a second time, by another owner of the batch or at
/// another index of the writer's own table (a prefix adopted twice), as
/// `(block, (writer, its block index), (holder, its block index))`.
///
/// `owners` is each owner's `(seq_len, block_table)`. An owner writes rows
/// `seq_len..seq_len + rows`, so only its blocks from `seq_len / block_size`
/// on are written. The blocks below that are read-only, and owners that
/// matched one cached prefix share them by design: reporting those drowned
/// the real aliases.
pub(super) fn write_window_aliases(
    owners: &[(usize, &[u32])],
    rows: usize,
    block_size: usize,
) -> Vec<(u32, (usize, usize), (usize, usize))> {
    let used = |&(seq_len, table): &(usize, &[u32])| {
        (seq_len + rows).div_ceil(block_size).min(table.len())
    };
    let mut holders = std::collections::HashMap::<u32, Vec<(usize, usize)>>::new();
    for (o, owner) in owners.iter().enumerate() {
        for (idx, &block) in owner.1[..used(owner)].iter().enumerate() {
            holders.entry(block).or_default().push((o, idx));
        }
    }
    let mut aliases = Vec::new();
    for (w, owner) in owners.iter().enumerate() {
        for w_idx in owner.0 / block_size..used(owner) {
            let block = owner.1[w_idx];
            for &(o, idx) in &holders[&block] {
                // Two writing entries of one block are reported once, by the
                // first.
                let reported = (o, idx) < (w, w_idx) && idx >= owners[o].0 / block_size;
                if (o, idx) != (w, w_idx) && !reported {
                    aliases.push((block, (w, w_idx), (o, idx)));
                }
            }
        }
    }
    aliases
}

impl TransformerModel {
    /// Diagnostic (`ATLAS_GLM_LONG_BATCH_SERIAL`): run one layer through its
    /// ordinary single-owner verify, owner by owner, with each owner's hidden
    /// and highway rows moved to arena rows [0, rows) and back.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_long_serial_layer(
        &self,
        li: usize,
        rows: usize,
        seqs: &mut [&mut SequenceState],
        metas: &[AttnMetadataDev],
        stage: &GlmLongStage,
        kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let b = &self.buffers;
        let spans = [
            (b.hidden_states(), stage.hidden, stage.rows.hidden),
            (b.hc_streams(), stage.highway, stage.rows.highway),
        ];
        let total = seqs.len() * rows;
        let gpu = self.gpu.as_ref();
        stage.copy(gpu, &spans, 0, 0, total, true, stream)?;
        let layer = &self.layers[li];
        let attention = self.config.layer_type(li) == atlas_core::config::LayerType::FullAttention;
        for (o, seq) in seqs.iter_mut().enumerate() {
            stage.copy(gpu, &spans, 0, o * rows, rows, false, stream)?;
            let owner_ctx = ForwardContext {
                attn_metadata: Some(metas[o]),
                midchunk_capture: None,
                ..*ctx
            };
            if attention {
                let positions: Vec<usize> = (seq.seq_len..seq.seq_len + rows).collect();
                let tables = vec![seq.block_table.clone(); rows];
                let mut states: [&mut (dyn crate::layer::LayerState + 'static); 1] =
                    [seq.layer_states[li].as_mut()];
                layer.decode_multi_seq_rows(
                    b.hidden_states(),
                    b.residual(),
                    rows,
                    &mut states,
                    &vec![0; rows],
                    kv_cache,
                    &positions,
                    &tables,
                    &owner_ctx,
                    stream,
                )?;
            } else {
                let seq = &mut **seq;
                layer.decode_batched(
                    b.hidden_states(),
                    b.residual(),
                    rows,
                    seq.layer_states[li].as_mut(),
                    kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    &owner_ctx,
                    stream,
                )?;
            }
            stage.copy(gpu, &spans, 0, o * rows, rows, true, stream)?;
        }
        stage.copy(gpu, &spans, 0, 0, total, false, stream)
    }

    /// Oracle helper: copy every owner's KDA recurrent/conv state into a fresh
    /// buffer (`save`), or back from `buffer` (restore).
    pub(super) fn glm_long_snapshot(
        &self,
        seqs: &[&mut SequenceState],
        save: bool,
        buffer: Option<DevicePtr>,
    ) -> Result<DevicePtr> {
        let conv = self.config.ssm_conv_state_bytes();
        let h = self.ssm_pool.h_stored_bytes;
        let spans: Vec<(DevicePtr, usize)> = seqs
            .iter()
            .flat_map(|seq| seq.layer_states.iter())
            .filter_map(|s| s.as_any().downcast_ref::<crate::layer::SsmLayerState>())
            .flat_map(|s| [(s.h_state, h), (s.conv_state, conv)])
            .collect();
        let total: usize = spans.iter().map(|&(_, b)| b).sum();
        let base = match buffer {
            Some(b) => b,
            None => self.gpu.alloc(total)?,
        };
        let stream = self.gpu.default_stream();
        let mut at = 0usize;
        for (ptr, bytes) in spans {
            let (src, dst) = if save {
                (ptr, base.offset(at))
            } else {
                (base.offset(at), ptr)
            };
            self.gpu.copy_d2d_async(src, dst, bytes, stream)?;
            at += bytes;
        }
        Ok(base)
    }

    /// `ATLAS_GLM_LONG_BATCH_ORACLE=1`: compare the batched verify against the
    /// ordinary serial verify of every owner from the same state, then continue
    /// with the SERIAL results. Logs argmax agreement, max |logit delta| and the
    /// serial top-2 margin per row.
    pub(super) fn glm_long_oracle(
        &self,
        rows: usize,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stage: &GlmLongStage,
        snapshot: DevicePtr,
        batched_ids: &[u32],
    ) -> Result<Vec<u32>> {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static ROWS_SEEN: AtomicU64 = AtomicU64::new(0);
        static MISMATCH: AtomicU64 = AtomicU64::new(0);
        let vocab = self.config.vocab_size;
        let row_bytes = vocab * 2;
        let n = seqs.len();
        let mut batched = vec![0u8; n * rows * row_bytes];
        self.gpu.copy_d2h(self.buffers.logits(), &mut batched)?;
        self.glm_long_snapshot(seqs, false, Some(snapshot))?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        self.gpu.free(snapshot)?;
        let bf = |b: &[u8], i: usize| {
            f32::from_bits((u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16)
        };
        let mut out = Vec::with_capacity(n * rows);
        for ((o, seq), t) in seqs.iter_mut().enumerate().zip(tokens.chunks_exact(rows)) {
            let ids = self.decode_verify_graphed_kgamma(t, seq, 0)?;
            ensure!(
                ids.len() == rows,
                "GLM long oracle serial verify returned {} ids",
                ids.len()
            );
            let mut serial = vec![0u8; rows * row_bytes];
            self.gpu.copy_d2h(self.buffers.logits(), &mut serial)?;
            stage.copy(
                self.gpu.as_ref(),
                &self.glm_long_final_spans(stage),
                0,
                o * rows,
                rows,
                true,
                self.gpu.default_stream(),
            )?;
            for r in 0..rows {
                let b = &batched[(o * rows + r) * row_bytes..][..row_bytes];
                let s = &serial[r * row_bytes..][..row_bytes];
                let (mut diff, mut top, mut second) = (0f32, f32::MIN, f32::MIN);
                for i in 0..vocab {
                    let (x, y) = (bf(b, i), bf(s, i));
                    diff = diff.max((x - y).abs());
                    if y > top {
                        second = top;
                        top = y;
                    } else if y > second {
                        second = y;
                    }
                }
                ROWS_SEEN.fetch_add(1, Relaxed);
                let batched_id = batched_ids[o * rows + r];
                if batched_id != ids[r] {
                    MISMATCH.fetch_add(1, Relaxed);
                    tracing::warn!(
                        "GLM long oracle MISMATCH owner={o} row={r} pos={} batched={batched_id} \
                         serial={} max_dlogit={diff:.4} serial_margin={:.4}",
                        seq.seq_len - rows + r,
                        ids[r],
                        top - second
                    );
                } else if diff > 0.5 {
                    tracing::info!(
                        "GLM long oracle owner={o} row={r} max_dlogit={diff:.4} margin={:.4}",
                        top - second
                    );
                }
            }
            out.extend(ids);
        }
        let seen = ROWS_SEEN.load(Relaxed);
        if seen % 600 < (n * rows) as u64 {
            tracing::info!(
                "GLM long oracle summary: rows={seen} argmax_mismatch={}",
                MISMATCH.load(Relaxed)
            );
        }
        Ok(out)
    }
}

/// `ATLAS_GLM_LONG_BATCH_SERIAL=kda|mla|all`: bisection aid that keeps the
/// named layer kind on its single-owner verify inside the batched step. Both
/// ranks must run the same value (`startup_parity`): a serial layer issues
/// its collectives an owner at a time.
pub(in crate::model) fn serial_diagnostic(attention: bool) -> bool {
    static MODE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let mode =
        MODE.get_or_init(|| std::env::var("ATLAS_GLM_LONG_BATCH_SERIAL").unwrap_or_default());
    match mode.as_str() {
        "all" => true,
        "mla" => attention,
        "kda" => !attention,
        _ => false,
    }
}

/// `ATLAS_GLM_LONG_BATCH_ORACLE=1` ([`TransformerModel::glm_long_oracle`]).
/// Both ranks must run the same value (`startup_parity`): the oracle adds
/// every owner's serial verify, collectives included, to the step.
pub(in crate::model) fn oracle_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_LONG_BATCH_ORACLE").as_deref() == Ok("1"))
}
