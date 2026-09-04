// SPDX-License-Identifier: AGPL-3.0-only

//! Decode block selection: the host reference (`host_select`), the device
//! arm's parity check, and the pure ordering core shared by both. Split out
//! of `qsa.rs` for the 500-LoC cap.

use super::*;

/// Widest `complete` the device kernel handles (its shared-memory flag array);
/// wider selections take the host arm.
pub(super) const QSA_SELECT_MAX_BLOCKS: usize = 4096;

/// The `block_topk` largest scores, ties to the LOWER index (torch.topk),
/// returned in ascending block order.
pub(super) fn select_blocks(scores: &[f32], block_topk: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..scores.len() as u32).collect();
    order.sort_by(|&a, &b| {
        scores[b as usize]
            .partial_cmp(&scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
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
    use super::{expand_selection, select_blocks};

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
