// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
const ROW_COUNTS: [usize; 3] = [2, 3, MAX_ROWS];
#[test]
fn staging_preserves_index_prefix_and_all_three_rows() {
    // The repaired K3 lane keeps its original byte partition.
    assert_eq!((scratch_bytes(3), output_bytes(3)), (49664, 24576));
    let p = StagePlan::new(3, DevicePtr(0x1000), 49664, DevicePtr(0x20000), 24576, &[]).unwrap();
    assert_eq!(p.row(0).unwrap(), DevicePtr(0x1200));
    assert_eq!(p.row(1).unwrap(), DevicePtr(0x5200));
    assert_eq!(p.row(2).unwrap(), DevicePtr(0x9200));
    assert!(p.row(3).is_err());
    StagePlan::new(3, DevicePtr(0x1000), 49664, DevicePtr(0x20002), 24576, &[]).unwrap();
    assert_eq!(p.row(2).unwrap().0 + 16384, 0x1000 + 49664);
}
#[test]
fn staging_scales_rows_and_rejects_unsupported_counts() {
    for rows in ROW_COUNTS {
        let (bytes, out) = (scratch_bytes(rows), output_bytes(rows));
        let p =
            StagePlan::new(rows, DevicePtr(0x1000), bytes, DevicePtr(0x40000), out, &[]).unwrap();
        for i in 0..rows {
            assert_eq!(p.row(i).unwrap().0, 0x1200 + (i * ROW) as u64);
        }
        assert!(p.row(rows).is_err());
        assert_eq!(
            p.row(rows - 1).unwrap().0 + ROW as u64,
            0x1000 + bytes as u64
        );
        for (capacity, output_capacity) in [(bytes - 1, out), (bytes, out - 1)] {
            assert!(
                StagePlan::new(
                    rows,
                    DevicePtr(0x1000),
                    capacity,
                    DevicePtr(0x40000),
                    output_capacity,
                    &[]
                )
                .is_err()
            );
        }
        // The last retained row aliases a live range; the index prefix may not.
        let last = DevicePtr(0x1200 + ((rows - 1) * ROW) as u64);
        assert!(
            StagePlan::new(
                rows,
                DevicePtr(0x1000),
                bytes,
                DevicePtr(0x40000),
                out,
                &[(last, 16)]
            )
            .is_err()
        );
        StagePlan::new(
            rows,
            DevicePtr(0x1000),
            bytes,
            DevicePtr(0x40000),
            out,
            &[(DevicePtr(0x1000), 512)],
        )
        .unwrap();
        // Output directly after the whole scratch span is adjacent, not aliased.
        let adjacent = DevicePtr(0x1000 + bytes as u64);
        StagePlan::new(rows, DevicePtr(0x1000), bytes, adjacent, out, &[]).unwrap();
    }
    let (bytes, out) = (scratch_bytes(MAX_ROWS + 1), output_bytes(MAX_ROWS + 1));
    for rows in [0, 1, MAX_ROWS + 1] {
        assert!(
            StagePlan::new(rows, DevicePtr(0x1000), bytes, DevicePtr(0x40000), out, &[]).is_err()
        );
    }
}
#[test]
fn staging_rejects_short_alias_unaligned_and_overflow_ranges() {
    for (ptr, bytes, out, outbytes) in [
        (0x1000, 49663, 0x20000, 24576),
        (0x1002, 49664, 0x20000, 24576),
        (0x1000, 49664, 0x20001, 24576),
        (0x1000, 49664, 0x20000, 24575),
        (0x1000, 49664, 0x1200, 24576),
        (u64::MAX - 15, 49664, 0x20000, 24576),
        (0x1000, 49664, u64::MAX - 1, 24576),
        (0, 49664, 0x20000, 24576),
    ] {
        assert!(StagePlan::new(3, DevicePtr(ptr), bytes, DevicePtr(out), outbytes, &[]).is_err());
    }
    assert!(
        StagePlan::new(
            3,
            DevicePtr(0x1000),
            49664,
            DevicePtr(0x20000),
            24576,
            &[(DevicePtr(0x9200), 16)]
        )
        .is_err()
    );
    // Exactly adjacent index scratch and output boundaries do not overlap.
    StagePlan::new(
        3,
        DevicePtr(0x1000),
        49664,
        DevicePtr(0xd200),
        24576,
        &[(DevicePtr(0x1000), 512)],
    )
    .unwrap();
}
#[test]
fn strict_flag_and_exact_local_weight_geometry() {
    assert!(!parse_flag(None).unwrap());
    assert!(!parse_flag(Some("0")).unwrap());
    assert!(parse_flag(Some("1")).unwrap());
    for s in ["", "true", "2", " 1"] {
        assert!(parse_flag(Some(s)).is_err());
    }
    validate_weight(DevicePtr(0x1000), 4096, 8192).unwrap();
    for (p, n, k) in [
        (0, 4096, 8192),
        (0x1002, 4096, 8192),
        (0x1000, 4096, 16384),
        (0x1000, 2048, 8192),
    ] {
        assert!(validate_weight(DevicePtr(p), n, k).is_err());
    }
}
