// SPDX-License-Identifier: AGPL-3.0-only

//! `impl CommBackend for NcclBackend` — collective operations.
//!
//! See `crate::nccl_backend` module-level docs for the global SAFETY
//! contract that applies to every `unsafe { nccl*(...) }` call here:
//! comm/buffers/sizes/streams come from valid prior allocations on this
//! rank's device, and the `extern "C"` ABI matches NCCL 2.28+.

#[cfg(test)]
use super::COLLECTIVE_TIMEOUT_SECS;
use super::{ALL_REDUCE_DTYPE_BYTES, NcclBackend};
use crate::CommBackend;
use crate::broadcast::{Classification, validate_idle_receiver};
use crate::nccl::{self, NcclDataType, NcclRedOp};
use anyhow::Result;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::Ordering;

impl CommBackend for NcclBackend {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        if self.try_oneshot(ptr, ptr, bytes, true, self.legacy_stream)? {
            return Ok(());
        }
        if self.world_size == 2 && self.add_kernel.load(Ordering::Relaxed) != 0 {
            if self.try_rdma_all_reduce(ptr, bytes, self.legacy_stream)? {
                return Ok(());
            }
            return self.all_reduce_2rank(ptr, bytes, self.legacy_stream);
        }
        // Fallback: ncclAllReduce reduces IN-PLACE on `ptr` and never touches
        // `recv_buffer`, so it is not subject to the receive-buffer bound.
        let count = bytes / ALL_REDUCE_DTYPE_BYTES;
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr as *const _,
                ptr as *mut _,
                count,
                NcclDataType::Bfloat16,
                NcclRedOp::Sum,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllReduce")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn all_reduce_async(&self, ptr: u64, bytes: usize, compute_stream: u64) -> Result<()> {
        if self.try_oneshot(ptr, ptr, bytes, true, compute_stream)? {
            return Ok(());
        }
        if self.world_size == 2 && self.add_kernel.load(Ordering::Relaxed) != 0 {
            // The RDMA pair waits on the compute stream itself: no comm-stream
            // event hop.
            if self.try_rdma_all_reduce(ptr, bytes, compute_stream)? {
                return Ok(());
            }
            // Use event-based async with 2-rank send/recv path.
            nccl::record_event(self.compute_done_event, compute_stream)?;
            nccl::stream_wait_event(self.comm_stream, self.compute_done_event)?;

            self.all_reduce_2rank(ptr, bytes, self.comm_stream)?;

            nccl::record_event(self.comm_done_event, self.comm_stream)?;
            nccl::stream_wait_event(compute_stream, self.comm_done_event)?;
            return Ok(());
        }

        // Fallback: in-place on `ptr`, does not use `recv_buffer`.
        let count = bytes / ALL_REDUCE_DTYPE_BYTES;
        let comm = *self.comm.lock();

        // 1. Record "MoE compute done" on compute stream
        nccl::record_event(self.compute_done_event, compute_stream)?;

        // 2. Comm stream waits for compute to finish
        nccl::stream_wait_event(self.comm_stream, self.compute_done_event)?;

        // 3. Launch all_reduce on dedicated comm stream
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr as *const _,
                ptr as *mut _,
                count,
                NcclDataType::Bfloat16,
                NcclRedOp::Sum,
                comm,
                self.comm_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllReduce (async)")?;

        // 4. Record "all_reduce done" on comm stream
        nccl::record_event(self.comm_done_event, self.comm_stream)?;

        // 5. Compute stream waits for all_reduce before residual_add
        nccl::stream_wait_event(compute_stream, self.comm_done_event)?;

        // Check for async errors (non-blocking, just flags unhealthy).
        self.check_async_error(comm);

        Ok(())
    }

    fn peer_exchange_async(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        bytes: usize,
        compute_stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.world_size == 2,
            "peer_exchange_async requires exactly two ranks (got {})",
            self.world_size
        );
        anyhow::ensure!(
            bytes.is_multiple_of(ALL_REDUCE_DTYPE_BYTES),
            "peer_exchange_async BF16 payload must be a whole number of elements ({bytes} bytes)"
        );
        if self.try_oneshot(send_ptr, recv_ptr, bytes, false, compute_stream)? {
            return Ok(());
        }

        // Preserve the all_reduce_async ordering contract: the send observes
        // all producer kernels on `compute_stream`, and subsequent compute on
        // that stream observes the peer payload. The caller owns and registers
        // `recv_ptr`; no shared backend scratch can be aliased by a fused user.
        nccl::record_event(self.compute_done_event, compute_stream)?;
        nccl::stream_wait_event(self.comm_stream, self.compute_done_event)?;

        let count = bytes / ALL_REDUCE_DTYPE_BYTES;
        let partner = (1 - self.rank) as i32;
        let comm = *self.comm.lock();
        let result = unsafe { nccl::ncclGroupStart() };
        nccl::check_nccl(result, "peer exchange ncclGroupStart")?;
        let result = unsafe {
            nccl::ncclSend(
                send_ptr as *const c_void,
                count,
                NcclDataType::Bfloat16,
                partner,
                comm,
                self.comm_stream,
            )
        };
        nccl::check_nccl(result, "peer exchange ncclSend")?;
        let result = unsafe {
            nccl::ncclRecv(
                recv_ptr as *mut c_void,
                count,
                NcclDataType::Bfloat16,
                partner,
                comm,
                self.comm_stream,
            )
        };
        nccl::check_nccl(result, "peer exchange ncclRecv")?;
        let result = unsafe { nccl::ncclGroupEnd() };
        nccl::check_nccl(result, "peer exchange ncclGroupEnd")?;

        nccl::record_event(self.comm_done_event, self.comm_stream)?;
        nccl::stream_wait_event(compute_stream, self.comm_done_event)?;
        self.check_async_error(comm);
        Ok(())
    }

    fn supports_peer_exchange_async(&self) -> bool {
        self.world_size == 2
    }

    fn exchange_async(
        &self,
        send: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        compute_stream: u64,
    ) -> Result<bool> {
        if self.try_oneshot(send, dst, bytes, add, compute_stream)? {
            return Ok(true);
        }
        self.try_rdma_exchange(send, dst, bytes, add, compute_stream)
    }

    fn all_reduce_capturable(&self, ptr: u64, bytes: usize, stream: u64) -> Result<bool> {
        self.try_oneshot(ptr, ptr, bytes, true, stream)
    }

    fn peer_exchange_capturable(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        self.try_oneshot(send_ptr, recv_ptr, bytes, false, stream)
    }

    fn capturable_all_reduce_max_bytes(&self) -> usize {
        self.oneshot_max_bytes()
    }

    fn set_oneshot_kernel(&self, handle: u64) {
        self.set_oneshot_kernel_handle(handle);
    }

    fn supports_exchange_async(&self, bytes: usize) -> bool {
        self.rdma_capacity()
            .is_some_and(|capacity| bytes <= capacity)
    }

    fn register_buffer(&self, ptr: u64, bytes: usize) -> Result<u64> {
        let mut handle: *mut c_void = ptr::null_mut();
        let comm = *self.comm.lock();
        let result =
            unsafe { nccl::ncclCommRegister(comm, ptr as *mut c_void, bytes, &mut handle) };
        nccl::check_nccl(result, "ncclCommRegister")?;
        self.registered_handles.lock().push(handle);
        Ok(handle as u64)
    }

    fn deregister_buffer(&self, handle: u64) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe { nccl::ncclCommDeregister(comm, handle as *mut c_void) };
        nccl::check_nccl(result, "ncclCommDeregister")
    }

    fn symmetric_alloc(&self, bytes: usize) -> Result<u64> {
        // Safety: matched by the caller's symmetric_free; both go through
        // ncclMemAlloc/ncclMemFree on the live communicator's allocator.
        let ptr = unsafe { nccl::nccl_mem_alloc(bytes) }?;
        Ok(ptr as u64)
    }

    fn symmetric_free(&self, ptr: u64) -> Result<()> {
        // Safety: caller guarantees `ptr` was returned by `symmetric_alloc`.
        unsafe { nccl::nccl_mem_free(ptr as *mut c_void) }
    }

    fn set_add_kernel(&self, handle: u64) {
        self.add_kernel.store(handle, Ordering::Relaxed);
        tracing::info!(
            "NCCL backend: bf16_add_inplace kernel set \
             (2-rank send/recv enabled)"
        );
    }

    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclAllGather(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllGather")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclReduceScatter(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                NcclRedOp::Sum,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclReduceScatter")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        self.broadcast_classified(ptr, bytes, root, Classification::TimedPayload)
    }

    fn receive_idle_command_word(&self, ptr: u64) -> Result<()> {
        validate_idle_receiver(self.rank, self.world_size, ptr)?;
        self.broadcast_classified(ptr, 4, 0, Classification::IdleCommand)
    }

    fn command_words_max(&self) -> usize {
        self.cmd_ring_max_words()
    }

    fn send_command_words(&self, words: &[u32]) -> Result<()> {
        self.cmd_ring_send(words)
    }

    fn recv_command_words(&self, words: &mut [u32]) -> Result<()> {
        self.cmd_ring_recv(words)
    }

    fn barrier(&self) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr::null(),
                ptr::null_mut(),
                0,
                NcclDataType::Float32,
                NcclRedOp::Sum,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "barrier (ncclAllReduce count=0)")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclSend(
                ptr as *const c_void,
                bytes,
                NcclDataType::Uint8,
                dest_rank as i32,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclSend (send_to)")
    }

    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclRecv(
                ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                src_rank as i32,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclRecv (recv_from)")
    }

    fn group_start(&self) -> Result<()> {
        let result = unsafe { nccl::ncclGroupStart() };
        nccl::check_nccl(result, "ncclGroupStart")
    }

    fn group_end(&self) -> Result<()> {
        let result = unsafe { nccl::ncclGroupEnd() };
        nccl::check_nccl(result, "ncclGroupEnd")
    }

    fn is_healthy(&self) -> bool {
        if self.unhealthy.load(Ordering::Acquire) {
            return false;
        }
        if let Some(why) = self.oneshot_poisoned() {
            tracing::error!("RDMA one-shot collective trapped at {why} — marking unhealthy");
            self.unhealthy.store(true, Ordering::Release);
            return false;
        }
        // Actively probe the communicator for async errors.
        let comm = *self.comm.lock();
        self.check_async_error(comm)
    }

    fn attempt_reconnect(&self) -> Result<()> {
        self.reconnect_inner()
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_collective_timeout_constant() {
        // Sanity: timeout should be a reasonable value (not 0, not too large).
        assert!(COLLECTIVE_TIMEOUT_SECS >= 10);
        assert!(COLLECTIVE_TIMEOUT_SECS <= 300);
    }

    // The receive-buffer capacity invariant — which replaced a
    // `test_recv_buffer_size_sufficient` that asserted a fixed 64 MiB constant
    // against a 4096-token prefill chunk while the shipped default was 8192 —
    // is tested next to the code that enforces it, in `recv_buffer.rs`.
}
