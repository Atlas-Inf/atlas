// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, anyhow};

pub(crate) fn next_dspark_generation(counter: &AtomicU64) -> Result<u64> {
    let previous = counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
            generation.checked_add(1)
        })
        .map_err(|_| anyhow!("DSpark sequence generation exhausted"))?;
    Ok(previous + 1)
}
