// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit command-boundary validation and shared synchronous broadcast execution.

pub(crate) fn validate_idle_receiver(
    rank: usize,
    world_size: usize,
    ptr: u64,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        world_size > 1 && rank > 0 && rank < world_size,
        "idle command receive requires a non-root communicator rank"
    );
    anyhow::ensure!(
        ptr != 0 && ptr.is_multiple_of(4),
        "idle command receive requires an aligned non-null word"
    );
    Ok(())
}

#[cfg(any(feature = "nccl", test))]
mod execution {
    use anyhow::Result;
    use std::time::Duration;

    /// Post-completion duration threshold, not an interrupting watchdog.
    pub(crate) const COLLECTIVE_TIMEOUT_SECS: u64 = 30;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Classification {
        TimedPayload,
        IdleCommand,
    }

    /// Real NCCL operations implement this narrow I/O boundary; CPU tests record
    /// the same runner without calling CUDA or creating a communicator.
    pub(crate) trait Operations {
        type Stamp;
        fn start(&mut self) -> Self::Stamp;
        fn launch(&mut self) -> Result<()>;
        fn synchronize(&mut self) -> Result<()>;
        fn elapsed(&mut self, start: Self::Stamp) -> Duration;
        fn mark_slow(&mut self, elapsed: Duration);
        fn check_async_error(&mut self);
    }

    pub(crate) fn run(classification: Classification, ops: &mut impl Operations) -> Result<()> {
        let start = ops.start();
        ops.launch()?;
        ops.synchronize()?;
        let elapsed = ops.elapsed(start);
        if classification == Classification::TimedPayload
            && elapsed.as_secs() >= COLLECTIVE_TIMEOUT_SECS
        {
            ops.mark_slow(elapsed);
        }
        ops.check_async_error();
        Ok(())
    }
}

#[cfg(any(feature = "nccl", test))]
pub(crate) use execution::{COLLECTIVE_TIMEOUT_SECS, Classification, Operations, run};

#[cfg(test)]
#[path = "broadcast_tests.rs"]
mod tests;
