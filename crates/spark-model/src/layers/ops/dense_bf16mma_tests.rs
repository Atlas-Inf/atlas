// SPDX-License-Identifier: AGPL-3.0-only

//! `chunks` unit tests.

use super::{CTA_FIXED, LOWP_ROWS, ROWS, WS_FIXED, WS_TILE_ROWS, chunks};

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
