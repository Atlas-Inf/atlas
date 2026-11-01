// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-width projection selection for genuinely independent decode rows.

use anyhow::{Result, ensure};
use spark_runtime::gpu::KernelHandle;

/// `independent` comes from the shared policy plus actual indexed state view,
/// never from row count alone. Returning None preserves temporal/ordinary paths.
pub(super) fn select(
    rows: usize,
    independent: bool,
    handles: [KernelHandle; 7],
) -> Result<Option<KernelHandle>> {
    if !independent {
        return Ok(None);
    }
    ensure!(
        (2..=8).contains(&rows),
        "independent MLA requires rows 2..8"
    );
    let kernel = handles[rows - 2];
    ensure!(
        kernel.0 != 0,
        "independent MLA row{rows} kernel is unavailable"
    );
    Ok(Some(kernel))
}

#[cfg(test)]
#[path = "mla_independent_abi_tests.rs"]
mod abi_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_independent_drain_width_selects_its_exact_export() {
        let handles = std::array::from_fn(|i| KernelHandle((i + 2) as u64));
        for rows in 2..=8 {
            assert_eq!(select(rows, true, handles).unwrap().unwrap().0, rows as u64);
        }
    }

    #[test]
    fn temporal_five_and_off_keep_old_selection_and_invalid_rows_refuse() {
        let handles = [KernelHandle(0); 7];
        for rows in 0..=9 {
            assert!(select(rows, false, handles).unwrap().is_none());
        }
        for rows in [0, 1, 9, usize::MAX] {
            assert!(select(rows, true, [KernelHandle(1); 7]).is_err());
        }
        for rows in 2..=8 {
            assert!(select(rows, true, handles).is_err());
        }
    }
}
