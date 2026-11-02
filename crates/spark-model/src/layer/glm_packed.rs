// SPDX-License-Identifier: AGPL-3.0-only

//! Packed multi-prompt prefill (`ATLAS_GLM_PACKED_PREFILL=1`, GLM-5): several
//! new short prompts in one target traversal.
//!
//! Prompt `p`'s rows sit at arena rows `[row0, row0 + rows)`, prompts packed
//! back to back. Row-local work (mHC, norms, projections, the FFN, the TP
//! reduce) runs once over every row, so each weight is read once for all of
//! them; each prompt's recurrence and attention advance only its own state,
//! over its own rows, from position 0. A prompt's rows go through the same
//! kernels whatever it is packed with (`model::glm_packed_prefill` pins the
//! row-count dependent choices), so its output does not depend on its
//! company.

use super::{AttnMetadataDev, LayerState};

/// One prompt of a packed traversal, for one layer.
pub struct GlmPackedOwner<'a> {
    pub state: &'a mut (dyn LayerState + 'static),
    /// First arena row of this prompt.
    pub row0: usize,
    /// Its rows (the whole prompt; positions `0..rows`).
    pub rows: usize,
    /// Its own chunk metadata: positions and slots of exactly its rows, its
    /// sequence lengths (one per attention piece) and block table.
    pub meta: AttnMetadataDev,
}

/// Rows of every owner, which must tile `[0, total)` in order.
pub fn tiled_rows(owners: &[GlmPackedOwner<'_>]) -> Option<usize> {
    let mut next = 0usize;
    for o in owners {
        if o.row0 != next || o.rows == 0 {
            return None;
        }
        next += o.rows;
    }
    (!owners.is_empty()).then_some(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::EmptyLayerState;
    use spark_runtime::gpu::DevicePtr;

    fn owner(state: &mut EmptyLayerState, row0: usize, rows: usize) -> GlmPackedOwner<'_> {
        GlmPackedOwner {
            state,
            row0,
            rows,
            meta: AttnMetadataDev {
                positions: DevicePtr::NULL,
                positions_h: DevicePtr::NULL,
                positions_w: DevicePtr::NULL,
                slot: DevicePtr::NULL,
                seq_len: DevicePtr::NULL,
                block_table: DevicePtr::NULL,
                max_blocks_per_seq: 0,
                num_seqs: 1,
                seq_slot: DevicePtr::NULL,
                moe_row_adapter: DevicePtr::NULL,
            },
        }
    }

    #[test]
    fn owners_must_tile_the_rows_in_order() {
        let (mut a, mut b, mut c) = (EmptyLayerState, EmptyLayerState, EmptyLayerState);
        assert_eq!(tiled_rows(&[owner(&mut a, 0, 50)]), Some(50));
        assert_eq!(
            tiled_rows(&[
                owner(&mut a, 0, 50),
                owner(&mut b, 50, 7),
                owner(&mut c, 57, 3)
            ]),
            Some(60)
        );
        assert_eq!(
            tiled_rows(&[owner(&mut a, 0, 50), owner(&mut b, 51, 7)]),
            None
        );
        assert_eq!(tiled_rows(&[owner(&mut a, 1, 50)]), None);
        assert_eq!(tiled_rows(&[owner(&mut a, 0, 0)]), None);
        assert_eq!(tiled_rows(&[]), None);
    }
}
