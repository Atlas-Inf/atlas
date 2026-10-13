// SPDX-License-Identifier: AGPL-3.0-only
use super::super::{MAX_ROWS, StagePlan, scratch_bytes};
use super::*;
const OUTPUT: usize = 3 * 4096 * 2;
#[test]
fn compare_is_strict_and_requires_the_staged_route() {
    assert!(!parse_compare(None, false).unwrap());
    assert!(!parse_compare(Some("0"), false).unwrap());
    assert!(parse_compare(Some("1"), true).unwrap());
    assert!(parse_compare(Some("1"), false).is_err());
    for value in ["", "2", "true", " 1"] {
        assert!(parse_compare(Some(value), true).is_err());
    }
}
#[test]
fn comparison_output_requires_capacity_and_independent_storage() {
    assert_eq!(output_bytes(3), OUTPUT);
    let plan =
        StagePlan::new(3, DevicePtr(0x1000), 49664, DevicePtr(0x20000), OUTPUT, &[]).unwrap();
    plan.with_compare(DevicePtr(0x30000), OUTPUT, &[]).unwrap();
    for (ptr, size) in [
        (0, OUTPUT),
        (0x30001, OUTPUT),
        (0x30000, OUTPUT - 1),
        (0x1200, OUTPUT),
        (0x20000, OUTPUT),
        (u64::MAX - 1, OUTPUT),
    ] {
        assert!(plan.with_compare(DevicePtr(ptr), size, &[]).is_err());
    }
    assert!(
        plan.with_compare(DevicePtr(0x30000), OUTPUT, &[(DevicePtr(0x30000), 16)])
            .is_err()
    );
}
#[test]
fn comparison_output_scales_with_rows() {
    for rows in [2, 3, MAX_ROWS] {
        let out = output_bytes(rows);
        let scratch = scratch_bytes(rows);
        let plan = StagePlan::new(
            rows,
            DevicePtr(0x1000),
            scratch,
            DevicePtr(0x40000),
            out,
            &[],
        )
        .unwrap();
        plan.with_compare(DevicePtr(0x60000), out, &[]).unwrap();
        assert!(plan.with_compare(DevicePtr(0x60000), out - 1, &[]).is_err());
        // The last retained row and the last reference row are both guarded.
        let last_row = 0x1000 + scratch as u64 - 16;
        assert!(plan.with_compare(DevicePtr(last_row), out, &[]).is_err());
        let last_ref = 0x40000 + out as u64 - 16;
        assert!(plan.with_compare(DevicePtr(last_ref), out, &[]).is_err());

        let reference = vec![0; out];
        let mut candidate = reference.clone();
        compare_bytes(&reference, &candidate, rows).unwrap();
        candidate[out - 2..].copy_from_slice(&0x3f80_u16.to_le_bytes());
        let error = compare_bytes(&reference, &candidate, rows)
            .unwrap_err()
            .to_string();
        let last = out / 2 - 1;
        assert!(
            error.contains(&format!("first_index={last} row={} column=4095", rows - 1)),
            "{error}"
        );
        // A full buffer for a different row count is rejected.
        assert!(compare_bytes(&reference, &reference, rows + 1).is_err());
    }
}
#[test]
fn full_bf16_comparison_rejects_late_and_nonfinite_bit_differences() {
    let reference = vec![0; OUTPUT];
    let mut candidate = reference.clone();
    compare_bytes(&reference, &candidate, 3).unwrap();
    for bits in [0x7fc0_u16, 0x7f80_u16, 0xff80_u16] {
        let mut invalid = reference.clone();
        invalid[OUTPUT - 2..].copy_from_slice(&bits.to_le_bytes());
        let error = compare_bytes(&invalid, &invalid, 3)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("nonfinite") && error.contains("first_index=12287"),
            "{error}"
        );
    }

    candidate[14..16].copy_from_slice(&0x3f80_u16.to_le_bytes());
    candidate[OUTPUT - 2..].copy_from_slice(&0x4000_u16.to_le_bytes());
    let error = compare_bytes(&reference, &candidate, 3)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("mismatches=2")
            && error.contains("first_index=7")
            && error.contains("max_abs=2"),
        "{error}"
    );
    candidate.fill(0);
    candidate[OUTPUT - 2..].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    assert!(compare_bytes(&reference, &candidate, 3).is_err());
    assert!(compare_bytes(&reference[..OUTPUT - 2], &reference, 3).is_err());
    // Identical signed-zero bits pass; a sign-bit change is still a mismatch.
    candidate.fill(0);
    candidate[0..2].copy_from_slice(&0x8000_u16.to_le_bytes());
    assert!(compare_bytes(&reference, &candidate, 3).is_err());
}
