// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side validation for the rank-to-rank vision prefill payload.
//!
//! The encoder output itself stays on device and is sent directly through the
//! communicator. This module owns the bounded wire metadata and the prompt
//! indexing rule so the head and worker cannot silently accept different
//! shapes.

use anyhow::{Result, ensure};

/// Command sent immediately before an EP/TP prefill command. It carries the
/// current vision metadata and, when present, the encoded BF16 rows.
pub(crate) const EP_CMD_VISION_STATE: u32 = 0xFFFF_FFF7;

/// Six u32 words precede the optional grid triples and BF16 payload:
/// `rows, grid_count, row_base, grid_base, owned_images, slice_rows`.
pub(crate) const VISION_HEADER_WORDS: usize = 6;

/// A malformed or accidentally unbounded request must not turn the worker's
/// scratch broadcast into an arbitrary allocation/collective. Normal prompts
/// use very few items; this is a protocol limit, not a model capacity claim.
pub(crate) const VISION_MAX_GRID_ITEMS: usize = 1024;
const VISION_MAX_GRID_CELLS: usize = 1 << 24;
const VISION_BF16_BYTES: usize = 2;
const VISION_MAX_PAYLOAD_BYTES: usize = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VisionWireState {
    pub rows: usize,
    pub grid_count: usize,
    pub row_base: usize,
    pub grid_base: usize,
    pub owned_images: usize,
    pub slice_rows: usize,
}

/// Parse and validate the fixed-width state header received over the u32 wire.
pub(crate) fn parse_state_header(words: &[u32], max_rows: usize) -> Result<VisionWireState> {
    ensure!(
        words.len() == VISION_HEADER_WORDS,
        "vision state header must contain {} words, got {}",
        VISION_HEADER_WORDS,
        words.len()
    );
    let state = VisionWireState {
        rows: words[0] as usize,
        grid_count: words[1] as usize,
        row_base: words[2] as usize,
        grid_base: words[3] as usize,
        owned_images: words[4] as usize,
        slice_rows: words[5] as usize,
    };
    validate_state(&state, max_rows)
}

/// Validate a state assembled by the rank-0 scheduler before it is put on the
/// wire. `rows == 0` is the explicit text request representation.
pub(crate) fn validate_state(state: &VisionWireState, max_rows: usize) -> Result<VisionWireState> {
    ensure!(
        state.grid_count <= VISION_MAX_GRID_ITEMS,
        "vision grid count {} exceeds protocol limit {}",
        state.grid_count,
        VISION_MAX_GRID_ITEMS
    );
    if state.rows == 0 {
        ensure!(
            state.grid_count == 0
                && state.row_base == 0
                && state.grid_base == 0
                && state.owned_images == 0
                && state.slice_rows == 0,
            "empty vision state must clear rows, grids, and slice bases"
        );
        return Ok(*state);
    }

    ensure!(
        max_rows > 0,
        "vision state received but this rank has no vision encoder capacity"
    );

    ensure!(
        state.rows <= max_rows,
        "vision row count {} exceeds encoder capacity {}",
        state.rows,
        max_rows
    );
    ensure!(
        state.grid_count > 0,
        "non-empty vision state has no grid metadata"
    );
    ensure!(
        state.row_base <= state.rows,
        "vision row base exceeds row count"
    );
    ensure!(
        state.slice_rows > 0 && state.slice_rows <= state.rows - state.row_base,
        "vision slice rows {} exceed rows {} at base {}",
        state.slice_rows,
        state.rows,
        state.row_base
    );
    if state.owned_images == 0 {
        ensure!(
            state.row_base == 0 && state.grid_base == 0 && state.slice_rows == state.rows,
            "legacy vision state must address the complete row/grid range"
        );
    } else {
        ensure!(
            state.grid_base <= state.grid_count,
            "vision grid base exceeds grid count"
        );
        ensure!(
            state.owned_images <= state.grid_count - state.grid_base,
            "vision owned image count exceeds grid range"
        );
    }
    Ok(*state)
}

/// Decode `(t_len, grid_h, grid_w)` triples and reject malformed geometry
/// before the worker stores it in the MRoPE state.
pub(crate) fn parse_grid_words(
    words: &[u32],
    state: &VisionWireState,
) -> Result<Vec<(usize, usize, usize)>> {
    let expected = state
        .grid_count
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("vision grid metadata length overflow"))?;
    ensure!(
        words.len() == expected,
        "vision grid metadata must contain {expected} words, got {}",
        words.len()
    );
    let mut grids = Vec::with_capacity(state.grid_count);
    for triple in words.chunks_exact(3) {
        let t_len = triple[0] as usize;
        let grid_h = triple[1] as usize;
        let grid_w = triple[2] as usize;
        ensure!(
            t_len > 0 && grid_h > 0 && grid_w > 0,
            "vision grid dimensions must be non-zero"
        );
        let cells = t_len
            .checked_mul(grid_h)
            .and_then(|v| v.checked_mul(grid_w))
            .ok_or_else(|| anyhow::anyhow!("vision grid geometry overflow"))?;
        ensure!(
            cells <= VISION_MAX_GRID_CELLS,
            "vision grid geometry {cells} exceeds protocol limit {VISION_MAX_GRID_CELLS}"
        );
        grids.push((t_len, grid_h, grid_w));
    }
    let rows = grids
        .iter()
        .try_fold(0usize, |total, &(t_len, grid_h, grid_w)| {
            let cells = t_len
                .checked_mul(grid_h)
                .and_then(|v| v.checked_mul(grid_w))
                .ok_or_else(|| anyhow::anyhow!("vision grid row count overflow"))?;
            total
                .checked_add(cells)
                .ok_or_else(|| anyhow::anyhow!("vision grid row count overflow"))
        })?;
    ensure!(
        rows == state.rows,
        "vision grid rows {rows} do not match encoded rows {}",
        state.rows
    );
    Ok(grids)
}

/// Compute the direct device broadcast size for the fixed BF16 vision output.
pub(crate) fn payload_bytes(
    state: &VisionWireState,
    hidden_size: usize,
    max_rows: usize,
) -> Result<usize> {
    validate_state(state, max_rows)?;
    ensure!(
        hidden_size > 0 && hidden_size <= 65_536,
        "vision hidden size {} is outside the BF16 wire limit",
        hidden_size
    );
    let bytes = state
        .rows
        .checked_mul(hidden_size)
        .and_then(|v| v.checked_mul(VISION_BF16_BYTES))
        .ok_or_else(|| anyhow::anyhow!("vision BF16 payload size overflow"))?;
    ensure!(
        bytes <= VISION_MAX_PAYLOAD_BYTES,
        "vision BF16 payload {bytes} bytes exceeds protocol limit {VISION_MAX_PAYLOAD_BYTES}"
    );
    Ok(bytes)
}

/// Count image/video pad rows emitted before a chunk. The encoder output rows
/// are consumed in exactly this order by the pad-token splice, so resetting the
/// source index to zero for chunk N aliases the first image into later chunks.
pub(crate) fn pad_rows_before_chunk(
    tokens: &[u32],
    chunk_start: usize,
    image_pad: u32,
    video_pad: u32,
) -> usize {
    tokens
        .get(..chunk_start.min(tokens.len()))
        .unwrap_or_default()
        .iter()
        .filter(|&&token| token == image_pad || token == video_pad)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy(rows: u32, grids: u32) -> VisionWireState {
        VisionWireState {
            rows: rows as usize,
            grid_count: grids as usize,
            row_base: 0,
            grid_base: 0,
            owned_images: 0,
            slice_rows: rows as usize,
        }
    }

    #[test]
    fn accepts_legacy_and_co_dispatched_ranges() {
        assert_eq!(
            parse_state_header(&[8, 1, 0, 0, 0, 8], 16).unwrap(),
            legacy(8, 1)
        );
        let packed = VisionWireState {
            rows: 12,
            grid_count: 3,
            row_base: 5,
            grid_base: 1,
            owned_images: 1,
            slice_rows: 4,
        };
        assert_eq!(validate_state(&packed, 16).unwrap(), packed);
    }

    #[test]
    fn rejects_unbounded_or_inconsistent_state() {
        assert!(parse_state_header(&[17, 1, 0, 0, 0, 17], 16).is_err());
        assert!(parse_state_header(&[0, 1, 0, 0, 0, 0], 16).is_err());
        assert!(
            validate_state(
                &VisionWireState {
                    rows: 8,
                    grid_count: 3,
                    row_base: 7,
                    grid_base: 0,
                    owned_images: 1,
                    slice_rows: 2,
                },
                16,
            )
            .is_err()
        );
        let huge = VisionWireState {
            rows: 8,
            grid_count: 1,
            row_base: 0,
            grid_base: 0,
            owned_images: 0,
            slice_rows: 8,
        };
        assert!(parse_grid_words(&[1, u32::MAX, 1], &huge).is_err());
        assert!(parse_grid_words(&[1, 2, 2], &legacy(8, 1)).is_err());
    }

    #[test]
    fn validates_grid_triples_and_payload_size() {
        let state = legacy(12, 2);
        assert_eq!(
            parse_grid_words(&[1, 2, 2, 2, 2, 2], &state).unwrap(),
            vec![(1, 2, 2), (2, 2, 2)]
        );
        assert_eq!(payload_bytes(&state, 2048, 16).unwrap(), 12 * 2048 * 2);
        assert!(payload_bytes(&state, 0, 16).is_err());
    }

    #[test]
    fn counts_image_and_video_rows_before_chunk() {
        let tokens = [9, 11, 11, 9, 12, 12, 9];
        assert_eq!(pad_rows_before_chunk(&tokens, 4, 11, 12), 2);
        assert_eq!(pad_rows_before_chunk(&tokens, 6, 11, 12), 4);
        assert_eq!(pad_rows_before_chunk(&tokens, 99, 11, 12), 4);
    }
}
