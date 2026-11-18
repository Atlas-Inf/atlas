// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};

use super::super::super::types::TransformerModel;
use super::score_plan::ScorePlan;
use crate::traits::{ScoreInput, SequenceState};

/// `encoder`: the vision tower's (merged-row capacity, spatial merge size).
pub(super) fn validate_score_vision(
    reqs: &[ScoreInput<'_>],
    encoder: Option<(usize, usize)>,
) -> Result<()> {
    let image_requests = reqs.iter().filter(|req| !req.images.is_empty()).count();
    ensure!(
        image_requests == 0 || encoder.is_some(),
        "score: this model has no vision tower"
    );
    ensure!(
        reqs.iter()
            .all(|req| req.images.is_empty() || req.pack.is_none()),
        "score: images cannot be combined with a knowledge pack"
    );
    // Every image of the pass is encoded in one batch into the tower's output.
    if let Some((capacity, merge)) = encoder {
        let rows: usize = reqs
            .iter()
            .flat_map(|req| req.images.iter())
            .map(|item| item.pad_count(merge))
            .sum();
        ensure!(
            rows <= capacity,
            "score: the pass's images take {rows} vision rows, over the tower's {capacity}"
        );
    }
    Ok(())
}

/// The request each stream of `plan` belongs to (heads by their branches).
fn request_of_streams(plan: &ScorePlan) -> Vec<usize> {
    let mut of = vec![0; plan.tokens.len()];
    let mut stream = plan.heads;
    for (request, &count) in plan.branch_counts.iter().enumerate() {
        for branch in stream..stream + count {
            of[branch] = request;
            if let Some(head) = plan.head_of[branch] {
                of[head] = request;
            }
        }
        stream += count;
    }
    of
}

impl TransformerModel {
    /// Encode every image of the pass in one tower batch and point each
    /// stream of an image request at its request's slice of the output.
    pub(super) fn prepare_score_vision(
        &self,
        reqs: &[ScoreInput<'_>],
        plan: &ScorePlan,
        seqs: &mut [SequenceState],
        stream: u64,
    ) -> Result<()> {
        if reqs.iter().all(|req| req.images.is_empty()) {
            return Ok(());
        }
        let per_request: Vec<Vec<crate::VisionItem>> =
            reqs.iter().map(|req| req.images.to_vec()).collect();
        let slices = self.prepare_vision_embed_batched_on_stream(&per_request, stream)?;
        ensure!(
            slices.len() == reqs.len(),
            "score: vision encode returned {} slices for {} requests",
            slices.len(),
            reqs.len()
        );
        // The splice's window is the whole shared output; each request's
        // slice inside it is bounded by (row_base, owned, slice rows).
        let resident: usize = slices.iter().map(|slice| slice.3).sum();
        for ((i, seq), request) in seqs.iter_mut().enumerate().zip(request_of_streams(plan)) {
            let images = reqs[request].images;
            if images.is_empty() {
                continue;
            }
            let (row_base, grid_base, owned, rows) = slices[request];
            self.stage_vision_grids_dispatch(seq, images);
            seq.vision.items = images.to_vec();
            seq.vision.rows_encoded = resident;
            seq.vision.rows_base = 0;
            seq.vision.skip_rows = 0;
            seq.vision.computed_from = plan.starts[i];
            seq.vision.deferred = false;
            Self::set_vision_slice_base_entry(seq, row_base, grid_base, owned, rows);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_map_to_their_requests() {
        let p1: Vec<u32> = (0..40).collect();
        let p2: Vec<u32> = vec![1, 2, 3];
        let p3: Vec<u32> = (0..20).collect();
        let (b1, b2, b3) = ([vec![9], vec![8]], [vec![7]], [vec![6], vec![5], vec![4]]);
        let plan = ScorePlan::new(
            &[
                (&p1, &b1, false, None, None),
                (&p2, &b2, false, None, None),
                (&p3, &b3, false, None, None),
            ],
            16,
        );
        // Heads of requests 0 and 2, then 2 + 1 + 3 branches.
        assert_eq!(request_of_streams(&plan), vec![0, 2, 0, 0, 1, 2, 2, 2]);
    }
}
