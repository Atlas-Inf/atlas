// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `paged_glm` dense-selection exactness.

use super::dense_selection_is_exact;
use super::owner::{GlmChunkOwner, GlmOwnerProjections, any_owner_selects};
use crate::layer::AttnMetadataDev;
use spark_runtime::gpu::DevicePtr;

fn owner(seq_len_start: usize, rows: usize) -> GlmChunkOwner {
    let p = DevicePtr::NULL;
    GlmChunkOwner {
        row0: 0,
        rows,
        seq_len_start,
        meta: AttnMetadataDev {
            positions: p,
            positions_h: p,
            positions_w: p,
            slot: p,
            seq_len: p,
            block_table: p,
            max_blocks_per_seq: 1,
            num_seqs: 1,
            seq_slot: p,
            moe_row_adapter: p,
        },
    }
}

#[test]
fn dense_reference_stops_at_the_semantic_topk_boundary() {
    assert!(dense_selection_is_exact(2048, 2048));
    assert!(!dense_selection_is_exact(2049, 2048));
    assert!(!dense_selection_is_exact(1, 0));
}

#[test]
fn owner_is_dense_while_its_sequence_ends_within_topk() {
    assert!(owner(2040, 8).dense_is_exact(2048));
    assert!(!owner(2041, 8).dense_is_exact(2048));
    assert!(!owner(0, 1).dense_is_exact(0));
    assert!(!owner(usize::MAX, 1).dense_is_exact(2048));
}

/// Index queries and weights are read only by an owner's sparse selection, so
/// the owner-batched verify projects them only when some owner selects.
#[test]
fn owner_batch_needs_index_queries_only_when_an_owner_selects() {
    let short = [owner(30, 8), owner(500, 8), owner(2040, 8), owner(0, 5)];
    assert!(!any_owner_selects(&short, 2048));
    let mut mixed = short;
    mixed[1] = owner(2041, 8);
    assert!(any_owner_selects(&mixed, 2048));
    assert!(any_owner_selects(
        &[owner(61_000, 8), owner(40_000, 8)],
        2048
    ));
    assert!(any_owner_selects(&short, 0));
}

/// A batch that skipped the index projections hands a selecting owner an
/// error, never the unwritten scratch.
#[test]
fn owner_batch_without_index_queries_refuses_a_selecting_owner() {
    let mut batch = GlmOwnerProjections {
        keys: DevicePtr::NULL,
        gates: DevicePtr::NULL,
        index: None,
        q_absorbed: DevicePtr::NULL,
        key_row: 256,
        query_row: 8192,
        weight_row: 64,
    };
    assert!(batch.index_rows(8).is_err());
    batch.index = Some((DevicePtr(0x10_0000), DevicePtr(0x20_0000)));
    let (query, weights) = batch.index_rows(8).unwrap();
    assert_eq!(query.0, 0x10_0000 + 8 * 8192);
    assert_eq!(weights.0, 0x20_0000 + 8 * 64);
}
