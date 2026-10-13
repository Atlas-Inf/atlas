// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-index guarded slot claims and guard-owner identity for
//! [`SsmStatePool`] / [`SlotGuard`].

use std::sync::Arc;

use anyhow::Result;

use super::{SlotGuard, SsmStatePool};

#[cfg(test)]
#[path = "aligned_claim_tests.rs"]
mod aligned_claim_tests;

impl SsmStatePool {
    /// Guard exactly the index removed by the existing specific-claim path.
    /// Refusal never consumes another available slot; generic claims stay LIFO.
    pub(in crate::model) fn claim_specific_guarded(
        self: &Arc<Self>,
        slot: usize,
    ) -> Result<SlotGuard> {
        anyhow::ensure!(
            slot < self.max_slots && self.claim_specific(slot),
            "SSM target slot {slot} unavailable or out of range"
        );
        Ok(SlotGuard {
            pool: Arc::clone(self),
            idx: Some(slot),
        })
    }
}

impl SlotGuard {
    /// Read-only owner identity for selected checked request-state consumers.
    pub(crate) fn belongs_to(&self, pool: &Arc<SsmStatePool>) -> bool {
        Arc::ptr_eq(&self.pool, pool)
    }
}
