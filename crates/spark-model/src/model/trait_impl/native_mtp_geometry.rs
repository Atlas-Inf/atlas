// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded row geometry for request-owned native MTP context.

pub(super) fn final_row_offset(
    rows: usize,
    capacity: usize,
    hidden: usize,
) -> Result<usize, &'static str> {
    if rows == 0 || rows > capacity {
        return Err("native MTP commit rows exceed arena");
    }
    hidden
        .checked_mul(2)
        .and_then(|b| b.checked_mul(rows - 1))
        .ok_or("native MTP hidden offset overflow")
}

pub(super) fn ready_context(
    target: usize,
    private: usize,
    retained_end: usize,
) -> Result<(), &'static str> {
    if target == 0 || retained_end != target || private != target - 1 {
        return Err("native MTP context does not cover the committed target");
    }
    Ok(())
}

/// Number of source rows already represented by an adopted prefix snapshot.
pub(super) fn covered_rows(
    start: usize,
    rows: usize,
    retained_end: Option<usize>,
) -> Result<usize, &'static str> {
    let end = start
        .checked_add(rows)
        .ok_or("native MTP target span overflow")?;
    let retained = retained_end.unwrap_or(0);
    if retained < start || retained > end {
        return Err("native MTP target chunk is not contiguous with its request state");
    }
    Ok(retained - start)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_mtp_context_is_incremental_and_request_local() {
        for target in [1, 128, 8192, 1_048_576] {
            assert!(ready_context(target, target - 1, target).is_ok());
            assert!(ready_context(target, target, target).is_err());
            assert!(ready_context(target, target - 1, target + 1).is_err());
        }
        assert!(ready_context(0, 0, 0).is_err());
    }
    #[test]
    fn native_mtp_prefix_overlap_consumes_only_new_rows() {
        assert_eq!(covered_rows(0, 128, None).unwrap(), 0);
        assert_eq!(covered_rows(128, 128, Some(128)).unwrap(), 0);
        assert_eq!(covered_rows(127, 1, Some(128)).unwrap(), 1);
        assert_eq!(covered_rows(127, 129, Some(128)).unwrap(), 1);
        assert!(covered_rows(129, 1, Some(128)).is_err());
        assert!(covered_rows(127, 1, Some(129)).is_err());
        assert!(covered_rows(128, 128, None).is_err());
        assert!(covered_rows(usize::MAX, 1, None).is_err());
    }
    #[test]
    fn native_mtp_final_hidden_uses_only_the_committed_row() {
        assert_eq!(final_row_offset(1, 128, 4096).unwrap(), 0);
        assert_eq!(final_row_offset(7, 128, 4096).unwrap(), 6 * 8192);
        assert!(final_row_offset(0, 128, 4096).is_err());
        assert!(final_row_offset(129, 128, 4096).is_err());
        assert!(final_row_offset(2, 128, usize::MAX).is_err());
    }
}
