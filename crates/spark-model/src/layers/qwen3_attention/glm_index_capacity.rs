// SPDX-License-Identifier: AGPL-3.0-only
//! Capacity checks for the existing tiled GLM semantic selector.
use anyhow::{Context, Result, ensure};

pub(super) fn tile_rows(rows: usize, stride: usize, capacity_bytes: usize) -> Result<usize> {
    ensure!(
        rows > 0 && stride > 0,
        "GLM index rows and stride must be positive"
    );
    let row_bytes = stride
        .checked_mul(std::mem::size_of::<f32>())
        .context("GLM index row size overflow")?;
    ensure!(
        row_bytes <= capacity_bytes,
        "GLM index logits require {row_bytes} bytes for one row, but the allocated arena has {capacity_bytes} bytes"
    );
    Ok(rows.min(capacity_bytes / row_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_short_chunk_uses_allocated_arena_at_large_context() {
        let arena = 4096 * 8 * 2048 * 2;
        for context in [262144, 524288] {
            for rows in [1, 2, 7, 15] {
                assert_eq!(tile_rows(rows, context / 4, arena).unwrap(), rows);
            }
            assert_eq!(
                tile_rows(4096, context / 4, arena).unwrap(),
                arena / context
            );
        }
    }
    #[test]
    fn exact_row_fits_and_short_or_invalid_arena_fails() {
        assert_eq!(tile_rows(1, 65536, 262144).unwrap(), 1);
        assert!(tile_rows(1, 65536, 262143).is_err());
        assert!(tile_rows(0, 1, 4).is_err());
        assert!(tile_rows(1, 0, 4).is_err());
        assert!(tile_rows(1, usize::MAX, usize::MAX).is_err());
    }
}
