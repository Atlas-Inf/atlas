// SPDX-License-Identifier: AGPL-3.0-only
//! Real NCCL adapter for the shared synchronous broadcast runner.
use super::{COLLECTIVE_TIMEOUT_SECS, NcclBackend};
use crate::broadcast::{Classification, Operations, run};
use crate::nccl::{self, NcclComm, NcclDataType};
use anyhow::Result;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

struct NcclBroadcast<'a> {
    backend: &'a NcclBackend,
    ptr: u64,
    bytes: usize,
    root: usize,
    comm: Option<NcclComm>,
}
impl Operations for NcclBroadcast<'_> {
    type Stamp = Instant;
    fn start(&mut self) -> Instant {
        Instant::now()
    }
    fn launch(&mut self) -> Result<()> {
        let comm = *self.backend.comm.lock();
        self.comm = Some(comm);
        // SAFETY: Same owned buffer, communicator and legacy stream contract as
        // NcclBackend::broadcast; this adapter changes no FFI arguments.
        let result = unsafe {
            nccl::ncclBroadcast(
                self.ptr as *const _,
                self.ptr as *mut _,
                self.bytes,
                NcclDataType::Uint8,
                self.root as i32,
                comm,
                self.backend.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclBroadcast")
    }
    fn synchronize(&mut self) -> Result<()> {
        nccl::sync_stream(self.backend.legacy_stream)
    }
    fn elapsed(&mut self, start: Instant) -> Duration {
        start.elapsed()
    }
    fn mark_slow(&mut self, elapsed: Duration) {
        tracing::error!(
            "NCCL broadcast took {:.1}s (threshold: {}s) — marking communicator unhealthy",
            elapsed.as_secs_f64(),
            COLLECTIVE_TIMEOUT_SECS,
        );
        self.backend.unhealthy.store(true, Ordering::Release);
    }
    fn check_async_error(&mut self) {
        // run() reaches this only after the successful launch and stream sync.
        self.backend
            .check_async_error(self.comm.expect("broadcast launched before async check"));
    }
}

impl NcclBackend {
    pub(super) fn broadcast_classified(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        class: Classification,
    ) -> Result<()> {
        run(
            class,
            &mut NcclBroadcast {
                backend: self,
                ptr,
                bytes,
                root,
                comm: None,
            },
        )
    }
}
