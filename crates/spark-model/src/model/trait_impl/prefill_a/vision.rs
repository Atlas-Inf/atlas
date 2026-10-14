// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill phase A — vision-embed dispatch helpers.
//!
//! Extracted from `prefill_a.rs` to keep each file under the 500-LoC
//! file-size cap. These methods drive the ViT encoder (per-request and
//! cross-request batched) and stage the packed patch embeddings + grids
//! that the chunk-0 splice/MRoPE later consume.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(in crate::model) fn prepare_vision_embed_dispatch(
        &self,
        seq: &mut SequenceState,
        images: &[crate::VisionItem],
    ) -> Result<()> {
        let ve = match &self.vision_encoder {
            Some(ve) => ve,
            None => return Ok(()),
        };
        let stream = self.gpu.default_stream();
        // ONE batched ViT forward over all images in this request — block GEMM
        // weights read once over Σpatches instead of N× (the per-image loop also
        // overwrote buf_out row 0 every call, corrupting multi-image requests).
        // Each returned (post_h, post_w, merged_p) preserves group order, so
        // the packed buf_out matches the pad-token splice order downstream.
        // Keep items intact for the native GLM path. It emits one attention
        // sequence per temporal group and returns one geometry tuple per
        // group, while the splice/MRoPE bookkeeping below collapses those
        // tuples back to the original item boundaries.
        let mut item_refs: Vec<&crate::VisionItem> = Vec::with_capacity(images.len());
        let mut item_groups: Vec<usize> = Vec::with_capacity(images.len());
        for it in images {
            item_groups.push(it.t_len());
            item_refs.push(it);
        }
        let _vt0 = std::time::Instant::now();
        let per_image = ve.forward_items(&item_refs, self.gpu.as_ref(), stream)?;
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            self.gpu.synchronize(stream).ok();
            tracing::info!(
                "VIT_TIMING self-encode {} imgs: {:.1}ms",
                images.len(),
                _vt0.elapsed().as_secs_f64() * 1000.0
            );
        }
        // Collapse the encoder's per-GROUP output back to per-ITEM grids,
        // carrying each item's temporal extent. A still yields (1, h, w),
        // which is what this produced before video existed.
        let post_merge_grids: Vec<(usize, usize, usize)> = {
            let mut out = Vec::with_capacity(item_groups.len());
            let mut row = 0usize;
            for t_len in &item_groups {
                let (h, w, _) = per_image[row];
                out.push((*t_len, h, w));
                row += t_len;
            }
            out
        };
        let total_merged: usize = per_image.iter().map(|(_, _, mp)| *mp).sum();
        // The whole prompt is encoded in one pass: every row is resident from
        // physical 0, nothing is cached or deferred.
        seq.vision.items = images.to_vec();
        seq.vision.grids = post_merge_grids;
        seq.vision.hashes = images.iter().map(|it| it.content_hash).collect();
        seq.vision.rows_encoded = total_merged;
        seq.vision.rows_base = 0;
        seq.vision.skip_rows = 0;
        seq.vision.computed_from = 0;
        seq.vision.deferred = false;
        tracing::info!(
            "Vision encoder (batched): {} images, {} merged patches encoded",
            images.len(),
            total_merged
        );
        Ok(())
    }

    /// Cross-request batched encode: flatten every request's images into ONE
    /// `forward_batched` call so block GEMM weights are read once over Σpatches
    /// across the whole tick (the concurrent-image win). `per_request[i]` holds
    /// request i's images. Fills the shared packed `buf_out` and returns one
    /// `(patch_row_offset, grid_index_offset, num_images, patch_row_count)` per
    /// request locating its slice. Each request's chunk-0 splice/MRoPE then
    /// reads its slice via `set_vision_slice_base`; the scheduler stamps each
    /// request's own grids/hashes/row window into its `SequenceState` from the
    /// same descriptor (per-sequence vision state — the model holds none).
    pub(in crate::model) fn prepare_vision_embed_batched_dispatch(
        &self,
        per_request: &[Vec<crate::VisionItem>],
    ) -> Result<Vec<(usize, usize, usize, usize)>> {
        let ve = match &self.vision_encoder {
            Some(ve) => ve,
            None => return Ok(Vec::new()),
        };
        let stream = self.gpu.default_stream();
        // Flatten all requests' images, recording each request's (start, count).
        let mut flat_items: Vec<&crate::VisionItem> = Vec::new();
        let mut req_bounds: Vec<(usize, usize)> = Vec::with_capacity(per_request.len());
        let mut per_req_groups: Vec<Vec<usize>> = Vec::with_capacity(per_request.len());
        let mut group_cursor = 0usize;
        for imgs in per_request {
            let start = group_cursor;
            let mut groups = Vec::with_capacity(imgs.len());
            let mut group_count = 0usize;
            for it in imgs {
                groups.push(it.t_len());
                group_count += it.t_len();
                flat_items.push(it);
            }
            // Bounds are in per-group encoder rows, which is what buf_out and
            // the returned geometry vector index.
            req_bounds.push((start, group_count));
            group_cursor += group_count;
            per_req_groups.push(groups);
        }
        let per_image = ve.forward_items(&flat_items, self.gpu.as_ref(), stream)?;
        let total_merged: usize = per_image.iter().map(|(_, _, mp)| *mp).sum();
        // Per-request slice descriptors (request order matches the flatten order,
        // so row offsets accumulate Σ merged_p of earlier requests).
        // (buf_out row offset, GRID index offset, grid count, merged rows).
        // The grid offset/count are in ITEMS — one entry per item, not per
        // encoder row — while the row offset stays in merged rows, which is
        // what the splice indexes.
        let mut out = Vec::with_capacity(per_request.len());
        let mut row_cursor = 0usize;
        let mut grid_cursor = 0usize;
        for ((enc_start, n_rows), groups) in req_bounds.iter().zip(&per_req_groups) {
            let row_count: usize = per_image[*enc_start..*enc_start + *n_rows]
                .iter()
                .map(|(_, _, mp)| *mp)
                .sum();
            out.push((row_cursor, grid_cursor, groups.len(), row_count));
            row_cursor += row_count;
            grid_cursor += groups.len();
        }
        tracing::info!(
            "Vision co-dispatch: {} requests, {} images, {} merged patches",
            per_request.len(),
            flat_items.len(),
            total_merged,
        );
        Ok(out)
    }
}
