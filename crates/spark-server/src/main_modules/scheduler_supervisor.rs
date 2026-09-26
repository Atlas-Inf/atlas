// SPDX-License-Identifier: AGPL-3.0-only

//! Panic supervision for the scheduler thread.
//!
//! `scheduler::run` owns the only thread that can serve requests. If it
//! panics, the HTTP layer otherwise stays "ready": the receiver thread keeps
//! accepting requests into a channel nobody drains, and every request hangs
//! until the client times out. Latching the process fault lets
//! `gpu_fault_middleware` answer 503 and `main` exit `EXIT_GPU_FAULT` instead.

use std::panic::{AssertUnwindSafe, catch_unwind};

/// Runs the scheduler body. If it panics, latch the process fault so the HTTP
/// layer answers 503 (gpu_fault_middleware) and `main` exits EXIT_GPU_FAULT,
/// instead of the API staying "ready" while every request waits on a
/// scheduler thread that no longer exists. Returns true iff `body` panicked.
pub(crate) fn run_supervised(latch: &atlas_core::fault::FaultLatch, body: impl FnOnce()) -> bool {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(()) => false,
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                *s
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.as_str()
            } else {
                "non-string panic payload"
            };
            let reason = format!(
                "scheduler thread panicked ({msg}); no request can be served — restart the server"
            );
            if latch.latch(reason.clone()) {
                tracing::error!(target: "atlas::fault", "{reason}");
            }
            true
        }
    }
}

#[cfg(test)]
#[path = "scheduler_supervisor_tests.rs"]
mod tests;
