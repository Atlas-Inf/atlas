// SPDX-License-Identifier: AGPL-3.0-only

//! Getting packs for a tick (`Model::score_packs`): each is held already,
//! restored from the pack store (`score_store.rs`), or built; the missing
//! ones are built together, as many to a pass as one pass takes (rows, free
//! sequences, the tower's vision rows), so a library of new photos is
//! encoded at the merged-pass rate rather than one pass a photo.

use anyhow::{Result, anyhow, ensure};
use spark_runtime::kv_cache::KvCacheDtype;

use super::super::super::types::TransformerModel;
use super::score_pack::{pack_id, touch};
use super::score_plan::ScorePlan;
use super::score_store::PackMeta;
use crate::traits::{ScorePackInfo, ScorePackSpec, SequenceState};

impl TransformerModel {
    /// See [`crate::traits::Model::score_packs`].
    pub(in crate::model) fn score_packs_impl(
        &self,
        specs: &[ScorePackSpec<'_>],
        limit: usize,
        stream: u64,
    ) -> Vec<Result<ScorePackInfo>> {
        let v = self.config.vocab_size as u32;
        let hashes: Vec<Vec<u128>> = specs
            .iter()
            .map(|s| s.images.iter().map(|it| it.content_hash).collect())
            .collect();
        let ids: Vec<u64> = specs
            .iter()
            .zip(&hashes)
            .map(|(s, h)| pack_id(s.tokens, h, s.prefix_bidir))
            .collect();
        let info = |i: usize, built: bool| ScorePackInfo {
            id: ids[i],
            tokens: specs[i].tokens.len(),
            built,
        };
        let mut out: Vec<Option<Result<ScorePackInfo>>> = specs.iter().map(|_| None).collect();
        let mut guard = self.score_packs.lock();
        let mut build: Vec<usize> = Vec::new();
        for (i, s) in specs.iter().enumerate() {
            let checked = (|| -> Result<bool> {
                ensure!(
                    limit > 0,
                    "score: packs are off on this server (start it with --score-packs N)"
                );
                ensure!(!s.tokens.is_empty(), "score: an empty pack");
                if let Some(t) = s.tokens.iter().find(|&&t| t >= v) {
                    anyhow::bail!(
                        "score: pack token id {t} outside the model's vocabulary ({v} rows)"
                    );
                }
                if let Ok(p) = touch(&mut guard, ids[i]) {
                    ensure!(
                        p.tokens == s.tokens && p.hashes == hashes[i],
                        "score: pack address collision ({:016x})",
                        ids[i]
                    );
                    return Ok(true);
                }
                if let Some((meta, seq)) = self.store_restore(ids[i], stream)? {
                    ensure!(
                        meta.tokens == s.tokens && meta.hashes == hashes[i],
                        "score: pack address collision ({:016x})",
                        ids[i]
                    );
                    self.hold_pack(&mut guard, ids[i], meta, seq, limit)?;
                    return Ok(true);
                }
                Ok(false)
            })();
            match checked {
                Ok(true) => out[i] = Some(Ok(info(i, false))),
                // The first of equal packs builds; the rest find it held.
                Ok(false) if !build.iter().any(|&j| ids[j] == ids[i]) => build.push(i),
                Ok(false) => {}
                Err(e) => out[i] = Some(Err(e)),
            }
        }
        let merge = self
            .config
            .vision
            .as_ref()
            .map_or(1, |c| c.spatial_merge_size.max(1));
        let tower = self.vision_encoder.as_ref().map_or(0, |e| e.output_rows());
        let vision_rows =
            |i: usize| -> usize { specs[i].images.iter().map(|it| it.pad_count(merge)).sum() };
        let trace = std::env::var("ATLAS_SCORE_TRACE").is_ok_and(|v| v == "1");
        let t0 = std::time::Instant::now();
        let mut passes: Vec<usize> = Vec::new();
        let mut next = 0;
        while next < build.len() {
            // As many as one pass takes, at least one (which then says why not).
            let (max_rows, max_streams) = self.score_capacity_held(guard.0.len());
            let (mut rows, mut vrows, mut end) = (0, 0, next);
            while end < build.len() && end - next < max_streams {
                let i = build[end];
                if end > next
                    && (rows + specs[i].tokens.len() > max_rows || vrows + vision_rows(i) > tower)
                {
                    break;
                }
                rows += specs[i].tokens.len();
                vrows += vision_rows(i);
                end += 1;
            }
            let chunk = &build[next..end.max(next + 1)];
            passes.push(chunk.len());
            next = end.max(next + 1);
            let built = self.score_build_packs(
                chunk.iter().map(|&i| specs[i]).collect(),
                guard.0.len(),
                stream,
            );
            match built {
                Ok(seqs) => {
                    for (&i, mut seq) in chunk.iter().zip(seqs) {
                        // The held sequence keeps no images (their pixels and
                        // admission permits): only their geometry and hashes.
                        let meta = PackMeta {
                            tokens: specs[i].tokens.to_vec(),
                            bidir: specs[i].prefix_bidir,
                            grids: std::mem::take(&mut seq.vision.grids),
                            hashes: hashes[i].clone(),
                        };
                        Self::clear_seq_vision_state(&mut seq);
                        let kept = self
                            .score_pack_roundtrip(&seq, stream)
                            .and_then(|()| self.store_insert(ids[i], &meta, &seq, stream));
                        out[i] = Some(match kept {
                            Ok(()) => self
                                .hold_pack(&mut guard, ids[i], meta, seq, limit)
                                .map(|()| info(i, true)),
                            Err(e) => self.free_sequence_dispatch(&mut seq, false).and(Err(e)),
                        });
                    }
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    for &i in chunk {
                        out[i] = Some(Err(anyhow!("{msg}")));
                    }
                }
            }
        }
        if trace {
            tracing::info!(
                "score packs: {} asked, {} built in passes {passes:?} ({} us)",
                specs.len(),
                build.len(),
                t0.elapsed().as_micros()
            );
        }
        // Equal packs after the first: held now, or the first's error.
        (0..specs.len())
            .map(|i| {
                out[i]
                    .take()
                    .unwrap_or_else(|| match touch(&mut guard, ids[i]) {
                        Ok(_) => Ok(info(i, false)),
                        Err(e) => Err(e),
                    })
            })
            .collect()
    }

    /// Encode `specs` in one pass, one stream each from position 0, and keep
    /// the sequences (`held`: packs already held, the registry being locked
    /// by the caller).
    fn score_build_packs(
        &self,
        specs: Vec<ScorePackSpec<'_>>,
        held: usize,
        stream: u64,
    ) -> Result<Vec<SequenceState>> {
        ensure!(
            self.score_supported(),
            "score: single-rank models with BF16 logits only"
        );
        ensure!(
            !specs.iter().any(|s| s.prefix_bidir)
                || self
                    .layers
                    .iter()
                    .all(|l| l.is_ssm_layer() || l.honours_state_end()),
            "score: attention \"prefix_bidir\" is not supported by this model's attention layers"
        );
        {
            let kv = self.kv_cache.lock();
            ensure!(
                (0..kv.num_layers()).all(|l| matches!(
                    kv.dtype_for_layer(l),
                    KvCacheDtype::Bf16 | KvCacheDtype::Fp8
                )),
                "score: packs need a bf16 or fp8 KV cache"
            );
        }
        let (max_rows, max_streams) = self.score_capacity_held(held);
        let rows: usize = specs.iter().map(|s| s.tokens.len()).sum();
        ensure!(
            rows <= max_rows && specs.len() <= max_streams,
            "score: {} packs of {rows} tokens exceed one pass ({max_rows} rows, {max_streams} free sequences)",
            specs.len()
        );
        let plan = ScorePlan::packs(
            &specs
                .iter()
                .map(|s| (s.tokens, s.prefix_bidir))
                .collect::<Vec<_>>(),
        );
        let mut seqs: Vec<SequenceState> = Vec::with_capacity(specs.len());
        let run = (|| -> Result<()> {
            for _ in &specs {
                let seq = self.alloc_sequence_reset(false)?;
                let slot = seq.slot_idx;
                seqs.push(seq);
                self.ssm_pool.zero_slot(slot, self.gpu.as_ref(), stream)?;
            }
            self.score_packs_vision(&specs, &mut seqs, stream)?;
            self.score_forward(&plan, &mut seqs, &[], stream)?;
            self.gpu.synchronize(stream)
        })();
        if let Err(e) = run {
            if let Err(s) = self.gpu.synchronize(stream) {
                tracing::error!("score pack: synchronize after a failed build: {s:#}");
            }
            for seq in seqs.iter_mut() {
                Self::clear_seq_vision_state(seq);
                if let Err(f) = self.free_sequence_dispatch(seq, false) {
                    tracing::error!("score pack: free_sequence: {f:#}");
                }
            }
            return Err(e);
        }
        Ok(seqs)
    }

    /// Encode the packs' images in one tower batch and point each pack's
    /// stream at its slice of the output (as `prepare_score_vision` does for
    /// requests; every pack stream is computed from position 0).
    fn score_packs_vision(
        &self,
        specs: &[ScorePackSpec<'_>],
        seqs: &mut [SequenceState],
        stream: u64,
    ) -> Result<()> {
        if specs.iter().all(|s| s.images.is_empty()) {
            return Ok(());
        }
        ensure!(
            self.vision_encoder.is_some(),
            "score: this model has no vision tower"
        );
        let per_pack: Vec<Vec<crate::VisionItem>> =
            specs.iter().map(|s| s.images.to_vec()).collect();
        let slices = self.prepare_vision_embed_batched_on_stream(&per_pack, stream)?;
        ensure!(
            slices.len() == specs.len(),
            "score: vision encode returned {} slices for {} packs",
            slices.len(),
            specs.len()
        );
        let resident: usize = slices.iter().map(|slice| slice.3).sum();
        for ((seq, spec), (row_base, grid_base, owned, rows)) in
            seqs.iter_mut().zip(specs).zip(slices)
        {
            if spec.images.is_empty() {
                continue;
            }
            self.stage_vision_grids_dispatch(seq, spec.images);
            seq.vision.items = spec.images.to_vec();
            seq.vision.rows_encoded = resident;
            seq.vision.rows_base = 0;
            seq.vision.skip_rows = 0;
            seq.vision.computed_from = 0;
            seq.vision.deferred = false;
            Self::set_vision_slice_base_entry(seq, row_base, grid_base, owned, rows);
        }
        Ok(())
    }
}
