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

/// Seven u32 words precede the optional grid records and BF16 payload:
/// `rows, grid_count, row_base, grid_base, owned_images, slice_rows,
/// deferred`. `deferred == 1` means the rows are not on the wire yet: they
/// follow inside the prefill's prefix lookup, once both ranks agree which
/// items the cache already covers (`vision_cache`).
pub(crate) const VISION_HEADER_WORDS: usize = 7;

/// Words per item record: `t_len, grid_h, grid_w` and the item's 128-bit
/// content hash as four little-endian words. The hash keys the worker's
/// prefix cache exactly like the head's.
pub(crate) const VISION_GRID_WORDS: usize = 7;

/// A malformed or accidentally unbounded request must not turn the worker's
/// scratch broadcast into an arbitrary allocation/collective. The limit also
/// has to admit a Qwen video split into one temporal group per item: the
/// reference hour profile is 7,200 frames at `temporal_patch_size = 2`, or
/// 3,600 grid records. This is a protocol limit, not a model capacity claim.
pub(crate) const VISION_MAX_GRID_ITEMS: usize = 8192;
const VISION_MAX_GRID_CELLS: usize = 1 << 24;
const VISION_BF16_BYTES: usize = 2;
const VISION_MAX_PAYLOAD_BYTES: usize = 1 << 30;

/// The row bounds a rank applies to a received state. `window` is the
/// encoder output buffer (`VisionEncoder::output_rows`): rows that arrive in
/// one payload land there. `request` bounds a deferred state, whose rows only
/// describe the grids — each chunk's encode later sends at most `window` of
/// them — so it is the context (`max_seq_len`), not the buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VisionRowBounds {
    pub window: usize,
    pub request: usize,
}

/// A request whose images need more encoder rows than one encode holds.
/// The client resending the same history cannot succeed, so the server
/// answers it with HTTP 400 rather than a retryable 500 (#180).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionCapacityError {
    pub rows: usize,
    pub capacity: usize,
    /// Image/video grids in the request (an image is one grid).
    pub grids: usize,
}

impl std::fmt::Display for VisionCapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "vision row count {} ({} image grids) exceeds encoder capacity {}; \
             drop or downscale images from the request",
            self.rows, self.grids, self.capacity
        )
    }
}

impl std::error::Error for VisionCapacityError {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VisionWireState {
    pub rows: usize,
    pub grid_count: usize,
    pub row_base: usize,
    pub grid_base: usize,
    pub owned_images: usize,
    pub slice_rows: usize,
    pub deferred: bool,
}

/// Parse and validate the fixed-width state header received over the u32 wire.
pub(crate) fn parse_state_header(
    words: &[u32],
    bounds: VisionRowBounds,
) -> Result<VisionWireState> {
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
        deferred: match words[6] {
            0 => false,
            1 => true,
            other => anyhow::bail!("vision state deferred flag must be 0 or 1, got {other}"),
        },
    };
    validate_state(&state, bounds)
}

/// Validate a state assembled by the rank-0 scheduler before it is put on the
/// wire. `rows == 0` is the explicit text request representation.
pub(crate) fn validate_state(
    state: &VisionWireState,
    bounds: VisionRowBounds,
) -> Result<VisionWireState> {
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
                && state.slice_rows == 0
                && !state.deferred,
            "empty vision state must clear rows, grids, and slice bases"
        );
        return Ok(*state);
    }

    ensure!(
        bounds.window > 0,
        "vision state received but this rank has no vision encoder capacity"
    );

    // A deferred state's rows only describe the grids (each chunk's encode
    // later sends at most the window); rows in one payload land in the window.
    let capacity = if state.deferred {
        bounds.request
    } else {
        bounds.window
    };
    if state.rows > capacity {
        return Err(VisionCapacityError {
            rows: state.rows,
            capacity,
            grids: state.grid_count,
        }
        .into());
    }
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
            !state.deferred,
            "a co-dispatched vision slice is always encoded before its prefill"
        );
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

/// Encode per-item `(t_len, grid_h, grid_w)` and content hashes as
/// [`VISION_GRID_WORDS`]-word records.
pub(crate) fn grid_words(grids: &[(usize, usize, usize)], hashes: &[u128]) -> Result<Vec<u32>> {
    ensure!(
        grids.len() == hashes.len(),
        "vision grids ({}) and content hashes ({}) disagree",
        grids.len(),
        hashes.len()
    );
    let word = |v: usize| {
        u32::try_from(v).map_err(|_| anyhow::anyhow!("vision geometry exceeds u32 wire type"))
    };
    let mut out = Vec::with_capacity(grids.len() * VISION_GRID_WORDS);
    for (&(t_len, grid_h, grid_w), &hash) in grids.iter().zip(hashes) {
        out.extend([word(t_len)?, word(grid_h)?, word(grid_w)?]);
        out.extend((0..4).map(|i| (hash >> (32 * i)) as u32));
    }
    Ok(out)
}

/// Decode the item records and reject malformed geometry before the worker
/// stores it in the MRoPE state. Returns the grids and the content hashes.
pub(crate) fn parse_grid_words(
    words: &[u32],
    state: &VisionWireState,
) -> Result<(Vec<(usize, usize, usize)>, Vec<u128>)> {
    let expected = state
        .grid_count
        .checked_mul(VISION_GRID_WORDS)
        .ok_or_else(|| anyhow::anyhow!("vision grid metadata length overflow"))?;
    ensure!(
        words.len() == expected,
        "vision grid metadata must contain {expected} words, got {}",
        words.len()
    );
    let mut grids = Vec::with_capacity(state.grid_count);
    let mut hashes = Vec::with_capacity(state.grid_count);
    for record in words.chunks_exact(VISION_GRID_WORDS) {
        let t_len = record[0] as usize;
        let grid_h = record[1] as usize;
        let grid_w = record[2] as usize;
        hashes.push(
            record[3..]
                .iter()
                .enumerate()
                .fold(0u128, |h, (i, &w)| h | (w as u128) << (32 * i)),
        );
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
    Ok((grids, hashes))
}

/// Compute the direct device broadcast size for the fixed BF16 vision output.
pub(crate) fn payload_bytes(
    state: &VisionWireState,
    hidden_size: usize,
    bounds: VisionRowBounds,
) -> Result<usize> {
    validate_state(state, bounds)?;
    ensure!(
        !state.deferred,
        "a deferred vision state carries no payload"
    );
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

    const B16: VisionRowBounds = VisionRowBounds {
        window: 16,
        request: 16,
    };

    fn legacy(rows: u32, grids: u32) -> VisionWireState {
        VisionWireState {
            rows: rows as usize,
            grid_count: grids as usize,
            row_base: 0,
            grid_base: 0,
            owned_images: 0,
            slice_rows: rows as usize,
            deferred: false,
        }
    }

    fn records(grids: &[(usize, usize, usize)]) -> Vec<u32> {
        let hashes: Vec<u128> = (0..grids.len() as u128).collect();
        grid_words(grids, &hashes).unwrap()
    }

    #[test]
    fn accepts_legacy_and_co_dispatched_ranges() {
        assert_eq!(
            parse_state_header(&[8, 1, 0, 0, 0, 8, 0], B16).unwrap(),
            legacy(8, 1)
        );
        let packed = VisionWireState {
            rows: 12,
            grid_count: 3,
            row_base: 5,
            grid_base: 1,
            owned_images: 1,
            slice_rows: 4,
            deferred: false,
        };
        assert_eq!(validate_state(&packed, B16).unwrap(), packed);
        let deferred = parse_state_header(&[8, 1, 0, 0, 0, 8, 1], B16).unwrap();
        assert!(deferred.deferred);
        assert!(
            validate_state(
                &VisionWireState {
                    deferred: true,
                    ..packed
                },
                B16
            )
            .is_err(),
            "a co-dispatched slice is never deferred"
        );
    }

    #[test]
    fn over_capacity_is_a_typed_error() {
        let e = parse_state_header(&[17, 2, 0, 0, 0, 17, 0], B16).unwrap_err();
        assert_eq!(
            e.downcast_ref::<VisionCapacityError>(),
            Some(&VisionCapacityError {
                rows: 17,
                capacity: 16,
                grids: 2,
            })
        );
        // Typed through context, not once flattened to a string.
        assert!(!anyhow::anyhow!("{e:#}").is::<VisionCapacityError>());
        assert!(e.context("prefill").is::<VisionCapacityError>());
    }

    #[test]
    fn rejects_unbounded_or_inconsistent_state() {
        assert!(parse_state_header(&[17, 1, 0, 0, 0, 17, 0], B16).is_err());
        assert!(parse_state_header(&[0, 1, 0, 0, 0, 0, 0], B16).is_err());
        assert!(parse_state_header(&[0, 0, 0, 0, 0, 0, 1], B16).is_err());
        assert!(parse_state_header(&[8, 1, 0, 0, 0, 8, 2], B16).is_err());
        assert!(
            validate_state(
                &VisionWireState {
                    rows: 8,
                    grid_count: 3,
                    row_base: 7,
                    grid_base: 0,
                    owned_images: 1,
                    slice_rows: 2,
                    deferred: false,
                },
                B16,
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
            deferred: false,
        };
        assert!(parse_grid_words(&records(&[(1, u32::MAX as usize, 1)]), &huge).is_err());
        assert!(parse_grid_words(&records(&[(1, 2, 2)]), &legacy(8, 1)).is_err());
        assert!(
            parse_grid_words(&[1, 2, 2], &legacy(4, 1)).is_err(),
            "hashes are required"
        );
    }

    #[test]
    fn validates_grid_triples_and_payload_size() {
        let state = legacy(12, 2);
        let hashes = [u128::MAX - 1, 0x0123_4567_89ab_cdef_0011_2233_4455_6677];
        let words = grid_words(&[(1, 2, 2), (2, 2, 2)], &hashes).unwrap();
        assert_eq!(words.len(), 2 * VISION_GRID_WORDS);
        assert_eq!(
            parse_grid_words(&words, &state).unwrap(),
            (vec![(1, 2, 2), (2, 2, 2)], hashes.to_vec()),
            "the worker decodes the head's grids and content hashes exactly"
        );
        assert!(grid_words(&[(1, 2, 2)], &[]).is_err());
        assert_eq!(payload_bytes(&state, 2048, B16).unwrap(), 12 * 2048 * 2);
        assert!(payload_bytes(&state, 0, B16).is_err());
    }

    #[test]
    fn counts_image_and_video_rows_before_chunk() {
        let tokens = [9, 11, 11, 9, 12, 12, 9];
        assert_eq!(pad_rows_before_chunk(&tokens, 4, 11, 12), 2);
        assert_eq!(pad_rows_before_chunk(&tokens, 6, 11, 12), 4);
        assert_eq!(pad_rows_before_chunk(&tokens, 99, 11, 12), 4);
    }

    #[test]
    fn a_deferred_state_is_bounded_by_the_request_not_the_window() {
        // Issue #180: a deferred header carries the rows its grids describe;
        // each chunk's encode later sends at most the window. 16680 rows
        // against a 16384-row window must pass on both ranks.
        let bounds = VisionRowBounds {
            window: 16_384,
            request: 262_144,
        };
        let deferred = parse_state_header(&[16_680, 5, 0, 0, 0, 16_680, 1], bounds).unwrap();
        assert!(deferred.deferred && deferred.rows == 16_680);
        assert!(
            payload_bytes(&deferred, 2048, bounds).is_err(),
            "a deferred state has no payload to size"
        );
        // Rows that arrive in one payload still land in the window.
        assert!(parse_state_header(&[16_680, 5, 0, 0, 0, 16_680, 0], bounds).is_err());
        assert!(parse_state_header(&[16_384, 4, 0, 0, 0, 16_384, 0], bounds).is_ok());
        // The context still bounds a deferred request.
        assert!(parse_state_header(&[262_145, 1, 0, 0, 0, 262_145, 1], bounds).is_err());
        // No encoder: nothing is accepted, deferred or not.
        let none = VisionRowBounds {
            window: 0,
            request: 262_144,
        };
        assert!(parse_state_header(&[8, 1, 0, 0, 0, 8, 1], none).is_err());
    }
}
