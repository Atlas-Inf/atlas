// SPDX-License-Identifier: AGPL-3.0-only

//! Decode block selection: the host reference (`host_select`), the device
//! arm's parity check, and the pure ordering core shared by both. Split out
//! of `qsa.rs` for the 500-LoC cap.

use super::*;

/// Widest `complete` the device kernel handles (its shared-memory flag array);
/// wider selections take the host arm.
pub(super) const QSA_SELECT_MAX_BLOCKS: usize = 4096;

/// Shape of the decode selection for one query, from the position alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SelectGeometry {
    /// Complete `ratio`-token blocks in the visible prefix — the scored set.
    pub complete: usize,
    /// First token of the incomplete tail (`complete * ratio`).
    pub tail_start: usize,
    /// Tokens gathered: `block_topk` whole blocks plus the tail. Depends on
    /// the position only, never on the scores.
    pub n_sel: u32,
}

/// The selection geometry for the token at 0-based `pos` (`pos + 1` visible),
/// or `None` while the selection is INERT: with at most `block_topk` complete
/// blocks every block is selected, so dense attention is exact.
///
/// Pure, so the inert/active boundary and the per-row shape of a multi-row
/// step can be table-tested without a device.
pub(super) fn select_geometry(
    pos: usize,
    ratio: usize,
    block_topk: usize,
) -> Option<SelectGeometry> {
    let visible = pos + 1;
    let complete = visible / ratio;
    if complete <= block_topk {
        return None;
    }
    let tail_start = complete * ratio;
    Some(SelectGeometry {
        complete,
        tail_start,
        n_sel: (block_topk * ratio + (visible - tail_start)) as u32,
    })
}

/// The `block_topk` largest scores, ties to the LOWER index (torch.topk),
/// returned in ascending block order.
pub(super) fn select_blocks(scores: &[f32], block_topk: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..scores.len() as u32).collect();
    order.sort_by(|&a, &b| rank_cmp(scores, a, b));
    let mut blocks: Vec<u32> = order[..block_topk.min(order.len())].to_vec();
    blocks.sort_unstable();
    blocks
}

/// Expand selected blocks to token ids and append the partial-block tail —
/// exactly the layout `qsa_gather` reads from `sel_dev`.
pub(super) fn expand_selection(
    blocks: &[u32],
    ratio: usize,
    tail_start: usize,
    visible: usize,
) -> Vec<i32> {
    let mut sel: Vec<i32> =
        Vec::with_capacity(blocks.len() * ratio + visible.saturating_sub(tail_start));
    for b in blocks {
        let base = *b as i32 * ratio as i32;
        for r in 0..ratio as i32 {
            sel.push(base + r);
        }
    }
    for t in tail_start..visible {
        sel.push(t as i32);
    }
    sel
}

/// Ranking key for a block score.
///
/// Block scores are `sum_h relu(q_h . k_b) / sqrt(hd)`: `>= 0`, possibly
/// `+inf`, and NEVER NaN — the scoring kernels take the relu as
/// `fmaxf(dot, 0)`, which returns the non-NaN operand, so a NaN query or key
/// contributes 0 (pinned on the CPU reference by
/// `nan_activations_score_zero_not_nan`; GB10 kernels build without
/// fast-math). The key makes the ORDER total regardless, so that invariant is
/// not load-bearing here: NaN ranks below every real score, and `-0.0` folds
/// into `+0.0`, which the device kernel's `>` / `==` treat as equal and
/// `total_cmp` alone would not.
pub(super) fn rank_key(s: f32) -> f32 {
    if s.is_nan() {
        f32::NEG_INFINITY
    } else if s == 0.0 {
        0.0
    } else {
        s
    }
}

/// The order BOTH top-k arms rank blocks by: larger score first, lower index
/// on ties. Total for every input, so `sort_by` can neither panic on an
/// inconsistent comparator nor return an unspecified order.
pub(super) fn rank_cmp(scores: &[f32], a: u32, b: u32) -> std::cmp::Ordering {
    rank_key(scores[b as usize])
        .total_cmp(&rank_key(scores[a as usize]))
        .then(a.cmp(&b))
}

impl QsaIndexer {
    /// Host arm: D2H the block scores, select, expand. Also the reference the
    /// device arm is checked against under `ATLAS_QSA_TOPK_VERIFY=1`.
    pub(super) fn host_select(
        &self,
        gpu: &dyn GpuBackend,
        complete: usize,
        visible: usize,
        stream: u64,
    ) -> Result<Vec<i32>> {
        let mut raw = vec![0u8; complete * 4];
        gpu.copy_d2h_on_stream(self.scores_dev, &mut raw, stream)?;
        let scores: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let blocks = select_blocks(&scores, self.block_topk as usize);
        let ratio = self.ratio as usize;
        Ok(expand_selection(&blocks, ratio, complete * ratio, visible))
    }

    /// Parity check for the device arm: read back `sel_dev` and compare with
    /// the host reference; the first mismatch is an error (validation runs).
    pub(super) fn verify_device_selection(
        &self,
        gpu: &dyn GpuBackend,
        complete: usize,
        visible: usize,
        pos: usize,
        stream: u64,
    ) -> Result<()> {
        let host = self.host_select(gpu, complete, visible, stream)?;
        let mut raw = vec![0u8; host.len() * 4];
        gpu.copy_d2h_on_stream(self.sel_dev, &mut raw, stream)?;
        let dev: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if let Some((i, (h, d))) = host
            .iter()
            .zip(dev.iter())
            .enumerate()
            .find(|(_, (h, d))| h != d)
        {
            tracing::error!(
                "QSA device top-k mismatch at pos {pos}: index {i} host {h} device {d} (complete={complete}, n_sel={})",
                host.len()
            );
            anyhow::bail!("QSA device top-k mismatch at pos {pos} index {i}: host {h} device {d}");
        }
        if pos.is_multiple_of(256) {
            tracing::debug!(
                "QSA device top-k parity ok at pos {pos} ({} ids)",
                host.len()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{SelectGeometry, expand_selection, select_blocks, select_geometry};

    /// The published card: budget 2048, ratio 4 -> block_topk 512, inert
    /// bound 2051. Positions are 0-based, so 2050 is the LAST inert token
    /// (2051 visible) and 2051 the first active one.
    #[test]
    fn geometry_flips_exactly_at_the_inert_bound() {
        let (ratio, topk) = (4usize, 512usize);
        let bound = topk * ratio + ratio - 1; // QsaIndexer::inert_bound
        for pos in 0..bound {
            assert_eq!(
                select_geometry(pos, ratio, topk),
                None,
                "pos {pos} must be inert"
            );
        }
        assert_eq!(
            select_geometry(bound, ratio, topk),
            Some(SelectGeometry {
                complete: 513,
                tail_start: 2052,
                n_sel: 2048
            }),
        );
    }

    /// Consecutive rows of one multi-row step each get their OWN shape: the
    /// tail grows by one per row and folds into a new scored block every
    /// `ratio` rows. A step that reused row 0's geometry for the others would
    /// be wrong on every row but the first.
    #[test]
    fn consecutive_rows_walk_the_tail_and_close_blocks() {
        let (ratio, topk) = (4usize, 512usize);
        let got: Vec<(usize, u32)> = (2051..2060)
            .map(|pos| {
                let g = select_geometry(pos, ratio, topk).expect("active");
                assert_eq!(g.tail_start, g.complete * ratio);
                (g.complete, g.n_sel)
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (513, 2048),
                (513, 2049),
                (513, 2050),
                (513, 2051), // 2052..=2055 visible
                (514, 2048),
                (514, 2049),
                (514, 2050),
                (514, 2051), // block 513 closed
                (515, 2048),
            ]
        );
    }

    /// `n_sel` is what `select_blocks` + `expand_selection` actually produce,
    /// so the gather never reads past what the host arm wrote.
    #[test]
    fn geometry_agrees_with_the_host_selection_length() {
        let (ratio, topk) = (4usize, 8usize);
        for pos in 0..80 {
            let Some(g) = select_geometry(pos, ratio, topk) else {
                continue;
            };
            let scores: Vec<f32> = (0..g.complete).map(|b| ((b * 7) % 11) as f32).collect();
            let blocks = select_blocks(&scores, topk);
            let sel = expand_selection(&blocks, ratio, g.tail_start, pos + 1);
            assert_eq!(sel.len() as u32, g.n_sel, "pos {pos}");
            assert!(
                sel.iter().all(|t| (*t as usize) <= pos),
                "pos {pos}: a future token was selected"
            );
        }
    }

    #[test]
    fn ties_go_to_the_lower_index_and_output_is_ascending() {
        // blocks 1 and 3 tie at the top; 0 is next; 2 is lowest.
        let scores = [0.5, 0.9, 0.1, 0.9];
        assert_eq!(select_blocks(&scores, 2), vec![1, 3]);
        assert_eq!(select_blocks(&scores, 3), vec![0, 1, 3]);
    }

    #[test]
    fn expansion_matches_the_gather_layout() {
        // ratio 2, blocks [0, 3] → tokens 0,1,6,7, then the tail 8..10.
        assert_eq!(expand_selection(&[0, 3], 2, 8, 10), vec![0, 1, 6, 7, 8, 9]);
    }
}

#[cfg(test)]
mod rank_tests {
    use super::{expand_selection, select_blocks};

    /// The DEVICE kernel's selection rule (`qsa_select_topk`), transcribed
    /// with its OWN comparisons — NaN staged as -inf, `rank(b) = #{j : s_j >
    /// s_b || (s_j == s_b && j < b)}`, selected iff `rank < topk` — and NOT
    /// through `rank_key`, so the host arm is pinned against the device rule
    /// rather than against a helper they would share.
    fn device_rule(scores: &[f32], topk: usize) -> Vec<u32> {
        let sc: Vec<f32> = scores
            .iter()
            .map(|v| if v.is_nan() { f32::NEG_INFINITY } else { *v })
            .collect();
        (0..sc.len())
            .filter(|&b| {
                let rank = (0..sc.len())
                    .filter(|&j| sc[j] > sc[b] || (sc[j] == sc[b] && j < b))
                    .count();
                rank < topk
            })
            .map(|b| b as u32)
            .collect()
    }

    /// SplitMix64 — reproducible from the seed, no dev-dependency.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Scores as the kernels produce them: `>= 0`, heavy with exact ties,
    /// both zeros, and the odd `+inf`.
    fn reachable_scores(state: &mut u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| match next(state) % 16 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::INFINITY,
                v => (v % 5) as f32 * 0.25, // few levels => many ties
            })
            .collect()
    }

    #[test]
    fn host_selection_equals_the_device_rule_on_reachable_scores() {
        let mut state = 0xA71A5u64;
        for _ in 0..4000 {
            let n = 1 + (next(&mut state) % 96) as usize;
            let topk = 1 + (next(&mut state) % n as u64) as usize;
            let scores = reachable_scores(&mut state, n);
            let got = select_blocks(&scores, topk);
            assert_eq!(
                got,
                device_rule(&scores, topk),
                "scores {scores:?} topk {topk}"
            );
            assert_eq!(got.len(), topk.min(n));
        }
    }

    #[test]
    fn signed_zeros_tie_and_go_to_the_lower_index() {
        // `total_cmp` alone would rank +0.0 above -0.0 and pick block 1.
        assert_eq!(select_blocks(&[-0.0, 0.0, -0.0], 1), vec![0]);
        assert_eq!(select_blocks(&[0.0, -0.0, 0.5], 2), vec![0, 2]);
    }

    /// NaN cannot reach here today (see `rank_key`), but the order must stay
    /// total if it ever does: no panic, exactly `topk` blocks, real scores
    /// first, NaN blocks last and by index — so the expanded selection still
    /// has the length the gather reads.
    #[test]
    fn nan_scores_rank_last_and_never_change_the_count() {
        let nan = f32::NAN;
        assert_eq!(select_blocks(&[nan, 0.5, nan, 0.25], 2), vec![1, 3]);
        assert_eq!(select_blocks(&[nan, 0.5, nan, 0.25], 3), vec![0, 1, 3]);
        assert_eq!(select_blocks(&[nan; 6], 4), vec![0, 1, 2, 3]);
        let mut state = 0xBADF00Du64;
        for _ in 0..2000 {
            let n = 1 + (next(&mut state) % 64) as usize;
            let topk = 1 + (next(&mut state) % n as u64) as usize;
            let mut scores = reachable_scores(&mut state, n);
            for s in scores.iter_mut() {
                if next(&mut state).is_multiple_of(3) {
                    *s = nan;
                }
            }
            let blocks = select_blocks(&scores, topk);
            assert_eq!(blocks.len(), topk.min(n));
            let sel = expand_selection(&blocks, 4, n * 4, n * 4 + 2);
            assert_eq!(sel.len(), topk.min(n) * 4 + 2);
        }
    }
}
