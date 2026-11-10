// SPDX-License-Identifier: AGPL-3.0-only

//! `chunks` and row-eligibility unit tests.

use super::{CTA_FIXED, LOWP_ROWS, ModelLevers, ROWS, WS_FIXED, WS_TILE_ROWS, chunks, rows_ok};

#[test]
fn score_passes_take_these_kernels_at_every_row_count() {
    // Chat: decode / verify batches below 32 rows keep their kernels.
    let chat = ModelLevers::defaults();
    assert!(!rows_ok(31, &chat) && rows_ok(32, &chat));
    // A score pass: one row or a thousand, the same kernels.
    let score = ModelLevers {
        dense_prefill_any_rows: true,
        ..chat
    };
    assert!(rows_ok(1, &score) && rows_ok(31, &score) && !rows_ok(0, &score));
}

#[test]
fn chunks_stay_within_the_kernel_tile_and_fill_short_grids() {
    for (rows, slots, fixed) in [
        (ROWS, 96, CTA_FIXED),
        (LOWP_ROWS, 48, CTA_FIXED),
        (WS_TILE_ROWS, 48, WS_FIXED),
    ] {
        for m in 1..5000 {
            for tiles in [20, 64, 136] {
                let c = chunks(m, tiles, slots, rows, fixed);
                assert!(
                    c >= m.div_ceil(rows) && m.div_ceil(c) <= rows,
                    "m={m} tiles={tiles} c={c}"
                );
                // The kernel's balanced split never leaves an empty chunk.
                assert!((c - 1) * m.div_ceil(c) < m, "m={m} tiles={tiles} c={c}");
            }
        }
    }
    // down / out_proj at 288 rows: not 5 x 20 = 100 CTAs on 96 slots.
    let c = chunks(288, 20, 96, ROWS, CTA_FIXED);
    assert!(c * 20 <= 96 || c * 20 > 150);
    // Full waves already: keep 64-row chunks.
    assert_eq!(chunks(4128, 136, 96, ROWS, CTA_FIXED), 65);
    // Warp-specialized: 288 rows as two 144-row tiles, 40 CTAs on 48 SMs.
    assert_eq!(chunks(288, 20, 48, WS_TILE_ROWS, WS_FIXED), 2);
}
