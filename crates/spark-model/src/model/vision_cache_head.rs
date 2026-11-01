// SPDX-License-Identifier: AGPL-3.0-only

//! The head's per-chunk encode (`vision_cache`), and a diagnostic that checks
//! every item's rows in the window against the item encoded alone.

use std::sync::OnceLock;

use anyhow::{Result, ensure};

use super::super::types::TransformerModel;
use crate::VisionItem;
use crate::traits::SequenceState;

/// `ATLAS_VISION_CHECK_ROWS=1`: after every head encode, re-encode each item
/// of the window alone and compare its rows byte for byte (a request whose
/// rows differ fails). Diagnostic, slow: one extra encode per item. Read once.
fn row_check_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_VISION_CHECK_ROWS").as_deref() == Ok("1"))
}

impl TransformerModel {
    /// The head's encode of `items` into `buf_out` row 0, checked against
    /// the grids staged at admission (the key and the worker used those).
    pub(super) fn vision_encode_on_head(
        &self,
        seq: &SequenceState,
        items: &[usize],
        rows: usize,
    ) -> Result<()> {
        let ve = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("vision encode without an encoder"))?;
        let grids = &seq.vision.grids;
        ensure!(
            grids.len() == seq.vision.items.len(),
            "vision items and grids disagree"
        );
        let refs: Vec<&VisionItem> = items.iter().map(|&i| &seq.vision.items[i]).collect();
        let per_group = ve.forward_items(&refs, self.gpu.as_ref(), self.gpu.default_stream())?;
        // The grids were derived before the encode (the key and the worker
        // needed them); the encoder's own geometry must agree.
        let mut group = 0usize;
        for &i in items {
            let item = &seq.vision.items[i];
            let &(eh, ew, _) = per_group
                .get(group)
                .ok_or_else(|| anyhow::anyhow!("vision encoder returned {group} groups"))?;
            ensure!(
                (item.t_len(), eh, ew) == grids[i],
                "deferred vision grid {:?} disagrees with the encoder's {:?}",
                grids[i],
                (item.t_len(), eh, ew)
            );
            group += item.t_len();
        }
        let encoded: usize = per_group.iter().map(|&(_, _, m)| m).sum();
        ensure!(
            encoded == rows,
            "deferred vision encode produced {encoded} rows, the grids expect {rows}"
        );
        if row_check_on() {
            self.vision_check_rows_alone(seq, items)?;
        }
        Ok(())
    }

    /// [`row_check_on`]: the window `buf_out` holds for `items`, compared
    /// item by item with each item encoded alone, then restored.
    fn vision_check_rows_alone(&self, seq: &SequenceState, items: &[usize]) -> Result<()> {
        let ve = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("vision encode without an encoder"))?;
        let (gpu, stream) = (self.gpu.as_ref(), self.gpu.default_stream());
        let row = ve.out_hidden_size() * 2;
        let count = |i: usize| {
            let (t, h, w) = seq.vision.grids[i];
            t * h * w
        };
        gpu.synchronize(stream)?;
        let mut window = vec![0u8; items.iter().map(|&i| count(i)).sum::<usize>() * row];
        gpu.copy_d2h(ve.out_row(), &mut window)?;
        let (mut at, mut differ) = (0usize, 0usize);
        for &i in items {
            ve.forward_items(&[&seq.vision.items[i]], gpu, stream)?;
            gpu.synchronize(stream)?;
            let mut alone = vec![0u8; count(i) * row];
            gpu.copy_d2h(ve.out_row(), &mut alone)?;
            let got = &window[at * row..(at + count(i)) * row];
            if got != alone.as_slice() {
                let bytes = got.iter().zip(&alone).filter(|(a, b)| a != b).count();
                tracing::error!(
                    "vision row check: item {i} differs from its alone encode in {bytes} bytes"
                );
                differ += 1;
            }
            at += count(i);
        }
        gpu.copy_h2d(&window, ve.out_row())?;
        tracing::info!(
            "vision row check: {} items of the window, {differ} differ from their alone encodes",
            items.len()
        );
        ensure!(
            differ == 0,
            "vision row check: {differ} items differ from their alone encodes"
        );
        Ok(())
    }
}
