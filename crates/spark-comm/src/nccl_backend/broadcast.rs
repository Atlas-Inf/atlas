// SPDX-License-Identifier: AGPL-3.0-only
//! Real NCCL adapter for the shared synchronous broadcast runner (on an RDMA
//! chain, broadcasts take its one-shot channel instead).
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
        // On a chain every broadcast rides the one-shot channel, an idle
        // command word's included: the route depends on nothing but the
        // size, so the root's send and a worker's idle receive always meet.
        // Like ncclBroadcast, it waits without a limit (`OneShot::broadcast`).
        if self.backend.try_chain_broadcast(
            self.ptr,
            self.bytes,
            self.root,
            self.backend.legacy_stream,
        )? {
            return Ok(());
        }
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
        // run() reaches this only after the successful launch and stream sync;
        // a broadcast over the chain took no NCCL communicator.
        if let Some(comm) = self.comm {
            self.backend.check_async_error(comm);
        }
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
