// SPDX-License-Identifier: AGPL-3.0-only

//! Which compressed positions a DeepSeek-V4.1 query attends to, given its
//! index scores: the reference's rules (`Indexer.forward` and
//! `select_candidate_blocks` in `inference/model.py`), on the host.
//!
//! Index scores are BF16, so equal scores are common: on the recorded prompt
//! three queries in four have several positions tied where the top-k cuts.
//! The reference's `topk` then keeps the earliest, and so does this.

use std::cmp::Ordering;

/// Best first; among equal scores, earliest first.
fn better(scores: &[f32], a: u32, b: u32) -> Ordering {
    scores[b as usize]
        .partial_cmp(&scores[a as usize])
        .unwrap_or(Ordering::Equal)
        .then(a.cmp(&b))
}

/// The `k` best of `scores`, as positions in ascending order. Every position
/// when there are no more than `k`.
pub fn top_positions(scores: &[f32], k: usize) -> Vec<i32> {
    let mut order: Vec<u32> = (0..scores.len() as u32).collect();
    if k < order.len() {
        if k > 0 {
            order.select_nth_unstable_by(k - 1, |&a, &b| better(scores, a, b));
        }
        order.truncate(k);
        order.sort_unstable();
    }
    order.into_iter().map(|p| p as i32).collect()
}

/// Level one of the two-level top-k: one flag per position, set for the
/// positions of the `topk_blocks` best blocks of `block_size`. A block scores
/// what its best position scores. The block that holds the newest position
/// is always kept: it is only partly filled and holds the most recent
/// tokens. A block none of whose positions can be reached (all `-inf`) is
/// not kept.
pub fn candidate_blocks(scores: &[f32], topk_blocks: usize, block_size: usize) -> Vec<bool> {
    let mut best: Vec<f32> = scores
        .chunks(block_size)
        .map(|block| block.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect();
    if let Some(newest) = best.last_mut() {
        *newest = f32::INFINITY;
    }
    let mut kept = vec![false; scores.len()];
    for block in top_positions(&best, topk_blocks) {
        let block = block as usize;
        if best[block] > f32::NEG_INFINITY {
            let end = ((block + 1) * block_size).min(scores.len());
            kept[block * block_size..end].fill(true);
        }
    }
    kept
}

/// Level two: a position outside the candidate blocks cannot be chosen.
pub fn keep_candidates(scores: &mut [f32], candidates: &[bool]) {
    for (score, &kept) in scores.iter_mut().zip(candidates) {
        if !kept {
            *score = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
#[path = "index_recipe_tests.rs"]
mod tests;
