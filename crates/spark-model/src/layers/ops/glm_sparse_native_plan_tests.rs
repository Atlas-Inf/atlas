// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn storage() -> [Span; 10] {
    let capacities = [
        134348800,
        8196 * 16 * 512 * 2,
        268959744,
        2048 * 4,
        268697600,
        134479872,
        201523200,
        134348800,
        134348800,
        134348800,
    ];
    std::array::from_fn(|i| Span {
        ptr: (i as u64 + 1) * 0x100000000,
        bytes: capacities[i],
    })
}

#[test]
fn admission_excludes_first_chunk_short_rows_verify_and_capture() {
    for rows in [2048, 3515, 4096, 4100] {
        assert_eq!(admit(rows, 4100, false, false, false), Some(rows + 4100));
    }
    for rows in [0, 1, 3, 12, 2047, 4101] {
        assert_eq!(admit(rows, 4100, false, false, false), None);
    }
    assert_eq!(admit(4096, 0, false, false, false), None);
    assert_eq!(admit(4096, 2047, false, false, false), None);
    assert_eq!(admit(4096, 2048, false, false, false), Some(6144));
    assert_eq!(admit(4096, 28672, false, false, false), Some(32768));
    assert_eq!(admit(4096, 28673, false, false, false), None);
    assert_eq!(admit(4096, usize::MAX, false, false, false), None);
    for flags in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        assert_eq!(admit(4096, 4100, flags.0, flags.1, flags.2), None);
    }
}

#[test]
fn maximum_layout_has_three_lengths_and_three_lse_arrays() {
    let p = Plan::new(4100, 28668, 8196, 2048, storage(), 19).unwrap();
    let (required, metadata) = required_bytes(4100, 32768, 8196, 2048).unwrap();
    assert_eq!(
        metadata.offsets,
        [
            0, 33587200, 67174400, 67191040, 67207680, 67224320, 67749120, 68273920
        ]
    );
    assert_eq!(metadata.bytes, 68798720);
    assert_eq!(
        required,
        [
            134348800, 134283264, 33636400, 8192, 151142400, 21495808, 68798720, 134348800,
            134348800, 134348800
        ]
    );
    assert_eq!(p.operands.rows, 4100);
    assert_eq!(p.operands.seq_start, 28668);
    assert_eq!(p.operands.physical_blocks, 8196);
    assert_eq!(p.operands.block_table_count, 2048);
    assert_eq!(p.operands.metadata_bytes, 68798720);
    assert_eq!(p.operands.stream, 19);
}

#[test]
fn every_short_owner_and_live_operand_overlap_is_rejected() {
    let (required, _) = required_bytes(4100, 32768, 8196, 2048).unwrap();
    for i in 0..10 {
        let mut s = storage();
        s[i].bytes = required[i] - 1;
        assert!(
            Plan::new(4100, 28668, 8196, 2048, s, 0).is_err(),
            "owner {i}"
        );
        let mut s = storage();
        s[i].ptr = 0;
        assert!(Plan::new(4100, 28668, 8196, 2048, s, 0).is_err());
        let mut s = storage();
        s[i].ptr = u64::MAX - 255;
        assert!(Plan::new(4100, 28668, 8196, 2048, s, 0).is_err());
        for j in 0..i {
            let mut s = storage();
            s[i].ptr = s[j].ptr;
            assert!(Plan::new(4100, 28668, 8196, 2048, s, 0).is_err());
        }
    }
}

#[test]
fn packing_rounds_only_logical_history_and_requires_its_block_table() {
    let (required, _) = required_bytes(3515, 15807, 8196, 2048).unwrap();
    assert_eq!(required[5], 15808 * 656);
    assert!(Plan::new(3515, 12292, 8196, 987, storage(), 7).is_err());
    assert!(Plan::new(3515, 12292, 0, 2048, storage(), 7).is_err());
    let mut s = storage();
    s[4].ptr += 16;
    assert!(Plan::new(3515, 12292, 8196, 2048, s, 7).is_err());
}
