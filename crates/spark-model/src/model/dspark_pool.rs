// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, anyhow};

pub(crate) fn dflash_verify_rows(active: bool, num_drafts: usize) -> Result<usize> {
    if !active {
        return Ok(0);
    }
    num_drafts
        .checked_add(1)
        .ok_or_else(|| anyhow!("DFlash/DSpark verify-row count overflow for K={num_drafts}"))
}
