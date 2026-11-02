// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn max_tokens_defaults_to_512_and_stays_below_flash_kda() {
    assert_eq!(parse_max_tokens(None), 512);
    assert_eq!(parse_max_tokens(Some("256")), 256);
    assert_eq!(parse_max_tokens(Some(" 1024 ")), 1024);
    assert_eq!(parse_max_tokens(Some("x")), 512);
    assert_eq!(parse_max_tokens(Some("0")), 2);
    assert_eq!(parse_max_tokens(Some("4096")), MAX_TOKENS_CAP);
    const { assert!(MAX_TOKENS_CAP < 2048) };
}

#[test]
fn prompts_sit_back_to_back() {
    assert_eq!(row_offsets(&[50]), vec![0]);
    assert_eq!(row_offsets(&[50, 7, 3, 512]), vec![0, 50, 57, 60]);
    assert!(row_offsets(&[]).is_empty());
}

#[test]
fn metadata_blocks_hold_positions_then_slots() {
    // 50 rows: positions 200 B (-> 200, 8-aligned), slots 400 B, 256-aligned.
    assert_eq!(meta_block_bytes(50, false), 768);
    // MRoPE: three position streams first.
    assert_eq!(
        meta_block_bytes(50, true),
        (600 + 400usize).div_ceil(256) * 256
    );
    assert_eq!(meta_block_bytes(1, false) % 256, 0);
}

#[test]
fn scratch_layout_keeps_the_moe_staging_then_joint_then_owners() {
    let lens = [50usize, 7, 3];
    let (joint, owners, end) = scratch_layout(&lens, 8, false);
    // MoE top-k staging of 60 rows x 8 x 8 bytes, 256-aligned.
    assert_eq!(joint, (60 * 8 * 8usize).div_ceil(256) * 256);
    assert_eq!(owners[0], joint + meta_block_bytes(60, false));
    assert_eq!(owners[1], owners[0] + meta_block_bytes(50, false));
    assert_eq!(owners[2], owners[1] + meta_block_bytes(7, false));
    assert_eq!(end, owners[2] + meta_block_bytes(3, false));
    for o in &owners {
        assert_eq!(o % 256, 0);
    }
}
