// SPDX-License-Identifier: AGPL-3.0-only

use super::MockGpuBackend;

impl MockGpuBackend {
    /// Fail one allocation after `successful_allocations` successful calls.
    pub fn fail_alloc_after(&self, successful_allocations: usize) {
        *self.alloc_failure_countdown.lock() = Some(successful_allocations);
    }

    pub(super) fn check_alloc_failure(&self) -> anyhow::Result<()> {
        {
            let mut countdown = self.alloc_failure_countdown.lock();
            if let Some(remaining) = *countdown {
                *countdown = remaining.checked_sub(1);
                if remaining == 0 {
                    anyhow::bail!("mock allocation failed (injected)");
                }
            }
        }
        Ok(())
    }
}
