// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

//! Communication backend abstraction (SBIO IORouter for collective ops).
//!
//! All distributed communication flows through [`CommBackend`]. Business
//! logic never calls NCCL or MPI directly.
//!
//! - [`SingleGpuBackend`] — all ops are no-ops (single GPU).
//! - [`NcclBackend`] — real multi-GPU via NCCL (expert parallelism).

use anyhow::Result;

mod broadcast;
#[cfg(test)]
mod idle_command_tests;
pub mod peer_lifeline;

// NCCL FFI + the multi-GPU `NcclBackend` are gated on the `nccl`
// feature because they `#[link(name = "nccl")]`. `nccl` is separate
// from `cuda` so SCALE/AMD (gfx1151) builds can use the CUDA compute
// backend without an NCCL library. On metal builds (single Apple
// Silicon device) only `SingleGpuBackend` below is needed.
#[cfg(feature = "nccl")]
pub mod nccl;
#[cfg(feature = "nccl")]
pub mod nccl_backend;
#[cfg(feature = "nccl")]
pub use nccl_backend::NcclBackend;

/// Communication backend trait for distributed operations.
///
/// All collective operations take raw device pointer + byte count.
/// The pointer type is u64 (matching CUDA CUdeviceptr) to avoid
/// coupling this crate to spark-runtime's DevicePtr.
pub trait CommBackend: Send + Sync {
    /// All-reduce: sum across all ranks, result on all ranks.
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()>;

    /// All-gather: each rank contributes a chunk, all ranks get full buffer.
    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;

    /// Reduce-scatter: reduce + scatter (inverse of all-gather).
    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;

    /// Broadcast from root rank to all ranks.
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()>;

    /// Receive exactly the first four-byte word of an outer worker command
    /// from rank zero. Only non-root ranks of a multi-rank communicator may
    /// call this, with an aligned, live device word. This explicit boundary
    /// may wait for a future command: NCCL excludes that idle duration from
    /// its post-completion slow-broadcast latch, but keeps all error checks.
    /// Subsequent words and payloads must use ordinary [Self::broadcast].
    /// Other backends retain their ordinary broadcast behavior by default.
    fn receive_idle_command_word(&self, ptr: u64) -> Result<()> {
        broadcast::validate_idle_receiver(self.rank(), self.world_size(), ptr)?;
        self.broadcast(ptr, 4, 0)
    }

    /// Words one [`Self::send_command_words`] / [`Self::recv_command_words`]
    /// call may carry between the two ranks' hosts, or 0 when the backend has
    /// no host command channel. It depends only on configuration the ranks
    /// agree on at startup, so both take the same path for a given count.
    fn command_words_max(&self) -> usize {
        0
    }

    /// Send command words to the other rank's host, in order behind every
    /// earlier call and on no stream: the peer reads them with
    /// [`Self::recv_command_words`] without a device kernel, a stream sync or
    /// a copy back. At most [`Self::command_words_max`] words, at least one.
    fn send_command_words(&self, _words: &[u32]) -> Result<()> {
        anyhow::bail!("command words are not supported by this CommBackend")
    }

    /// Receive the peer's next `words.len()` command words, waiting for them
    /// (a worker waits here for its next command, however long that takes).
    fn recv_command_words(&self, _words: &mut [u32]) -> Result<()> {
        anyhow::bail!("command words are not supported by this CommBackend")
    }

    /// Barrier: block until all ranks reach this point.
    fn barrier(&self) -> Result<()>;

    /// Async all-reduce using GPU-side event synchronization.
    ///
    /// Replaces `gpu.synchronize(stream) + all_reduce(ptr, bytes)`.
    /// Uses a dedicated comm stream + CUDA events so the CPU never blocks.
    /// `compute_stream` is where MoE kernels ran and where residual_add will run.
    fn all_reduce_async(&self, ptr: u64, bytes: usize, compute_stream: u64) -> Result<()> {
        let _ = compute_stream;
        self.all_reduce(ptr, bytes)
    }

    /// Asynchronously exchange one BF16 payload with the other rank.
    ///
    /// Unlike an all-reduce, `recv_ptr` receives the peer's unmodified payload.
    /// This lets a consumer fuse Atlas's established local reduction arithmetic
    /// with its immediately-dependent kernel without materialising an
    /// intermediate reduced buffer. Only available on backends that explicitly
    /// support the two-rank event-ordered path.
    fn peer_exchange_async(
        &self,
        _send_ptr: u64,
        _recv_ptr: u64,
        _bytes: usize,
        _compute_stream: u64,
    ) -> Result<()> {
        anyhow::bail!("peer_exchange_async is not supported by this CommBackend")
    }

    /// Whether [`Self::peer_exchange_async`] is available for this communicator.
    fn supports_peer_exchange_async(&self) -> bool {
        false
    }

    /// Two-rank stream-ordered exchange: send `bytes` from `send` and land the
    /// peer's `bytes` in `dst`, added in place (BF16) when `add`, else copied.
    /// With equal halves this is a reduce-scatter (`add`) or an all-gather
    /// step. Both ranks must call it with the same `bytes`. Returns `false`
    /// (nothing enqueued) when unavailable for this call.
    fn exchange_async(
        &self,
        _send: u64,
        _dst: u64,
        _bytes: usize,
        _add: bool,
        _compute_stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Graph-capturable BF16 sum all-reduce in place on `stream`: only device
    /// work goes on the stream (no host sync, no per-call host state), into
    /// persistent registered buffers, with a device-resident sequence the
    /// kernel advances -- so a CUDA graph may capture it and replay it any
    /// number of times, interleaved with eager collectives. Eager calls may
    /// use it too. `Ok(false)` (nothing enqueued) when unavailable for this
    /// size; the answer depends only on `bytes` and configuration, so both
    /// ranks agree. Both ranks must issue capturable collectives in the same
    /// order and one at a time on the GPU: the sequence is shared by every
    /// stream, so calls on different streams (decode on the default stream,
    /// prefill on the scheduler's prefill stream) must be serialized by the
    /// caller with events or syncs, as for [`Self::all_reduce_async`] over
    /// the RDMA pair.
    fn all_reduce_capturable(&self, _ptr: u64, _bytes: usize, _stream: u64) -> Result<bool> {
        Ok(false)
    }

    /// Graph-capturable [`Self::peer_exchange_async`]: `recv_ptr` receives the
    /// peer's unmodified payload. Same contract as
    /// [`Self::all_reduce_capturable`].
    fn peer_exchange_capturable(
        &self,
        _send_ptr: u64,
        _recv_ptr: u64,
        _bytes: usize,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Largest payload [`Self::all_reduce_capturable`] and
    /// [`Self::peer_exchange_capturable`] accept (0 when unavailable). Where
    /// available, [`Self::exchange_async`] is capturable up to this size too.
    fn capturable_all_reduce_max_bytes(&self) -> usize {
        0
    }

    /// Provide the kernel handle for the capturable collectives
    /// (`rdma_oneshot_bf16`); loaded by the model layer like
    /// [`Self::set_add_kernel`].
    fn set_oneshot_kernel(&self, _handle: u64) {}

    /// Whether [`Self::exchange_async`] can serve payloads up to `bytes`.
    fn supports_exchange_async(&self, _bytes: usize) -> bool {
        false
    }

    /// Pre-register a GPU buffer with the communication backend.
    ///
    /// For NCCL over IB/RoCE, this caches the IB memory registration
    /// (`ibv_reg_mr`), avoiding per-call overhead in all_reduce.
    /// Returns an opaque handle for deregistration.
    fn register_buffer(&self, _ptr: u64, _bytes: usize) -> Result<u64> {
        Ok(0)
    }

    /// Deregister a previously registered buffer.
    fn deregister_buffer(&self, _handle: u64) -> Result<()> {
        Ok(())
    }

    /// Allocate a GPU buffer in NCCL's symmetric-memory window
    /// (NCCL ≥ 2.28 / `ncclMemAlloc`). Returns the device pointer as `u64`.
    ///
    /// Symmetric-memory allocations are the substrate for:
    ///   1. Copy-engine offload of NVLink collectives (frees SMs for compute).
    ///   2. Device-side communication API (kernels invoke collectives in-kernel),
    ///      which TokenWeave-style fused AR+RMSNorm+Residual builds on.
    ///
    /// On Atlas's 2-rank Spark over RoCE, the copy-engine offload itself does
    /// not apply (RoCE is not NVLink), but the symmetric windows are still
    /// required for future device-API fusions and to reduce per-call setup.
    /// Returns an error if the linked NCCL is < 2.28; backends that don't
    /// support symmetric memory return `Err` and callers must fall back.
    fn symmetric_alloc(&self, _bytes: usize) -> Result<u64> {
        anyhow::bail!("symmetric_alloc not supported by this CommBackend");
    }

    /// Free a buffer previously returned by `symmetric_alloc`.
    fn symmetric_free(&self, _ptr: u64) -> Result<()> {
        anyhow::bail!("symmetric_free not supported by this CommBackend");
    }

    /// Provide a kernel handle for the BF16 in-place addition kernel.
    ///
    /// Used by the 2-rank send/recv all-reduce path. The kernel is loaded
    /// by the model layer (which has access to AtlasRegistry) and passed
    /// to the comm backend at init time.
    fn set_add_kernel(&self, _handle: u64) {
        // Default: no-op (single GPU or backends that don't need it)
    }

    /// Send tokens to a specific rank (for EP token dispatch).
    ///
    /// Sends `bytes` from `ptr` on this rank to `dest_rank`.
    /// Must be paired with a matching `recv_from` on the destination rank.
    /// `stream` is the CUDA stream on which the operation is enqueued.
    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()>;

    /// Receive tokens from a specific rank (for EP token combine).
    ///
    /// Receives `bytes` into `ptr` on this rank from `src_rank`.
    /// Must be paired with a matching `send_to` on the source rank.
    /// `stream` is the CUDA stream on which the operation is enqueued.
    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()>;

    /// Begin a group of point-to-point operations (send_to/recv_from).
    ///
    /// All send_to/recv_from calls between group_start and group_end are
    /// batched into a single NCCL launch for efficiency.
    fn group_start(&self) -> Result<()> {
        Ok(())
    }

    /// End a group of point-to-point operations.
    fn group_end(&self) -> Result<()> {
        Ok(())
    }

    /// Check if the communicator is healthy (no async errors, no timeouts).
    ///
    /// Returns `true` if the communicator is operational. Implementations
    /// may actively probe the underlying transport (e.g., `ncclCommGetAsyncError`).
    fn is_healthy(&self) -> bool {
        true
    }

    /// Attempt to recover a degraded communicator.
    ///
    /// For NCCL, this aborts the dead communicator and re-initializes
    /// via TCP bootstrap. Both ranks must call this concurrently.
    /// Returns `Ok(())` on successful recovery, `Err` if recovery failed.
    fn attempt_reconnect(&self) -> Result<()> {
        Ok(())
    }

    /// This rank's index (0-based).
    fn rank(&self) -> usize;

    /// Total number of ranks.
    fn world_size(&self) -> usize;
}

/// Single-GPU backend: all collective ops are no-ops.
///
/// Used in Phase 1 where the entire model fits on one GPU.
pub struct SingleGpuBackend;

impl CommBackend for SingleGpuBackend {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn all_gather(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn reduce_scatter(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn broadcast(&self, _ptr: u64, _bytes: usize, _root: usize) -> Result<()> {
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        Ok(())
    }

    fn send_to(&self, _ptr: u64, _bytes: usize, _dest_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn recv_from(&self, _ptr: u64, _bytes: usize, _src_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gpu_noop() {
        let comm = SingleGpuBackend;
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 1);
        comm.all_reduce(0x1000, 1024).unwrap();
        comm.all_gather(0x1000, 0x2000, 512).unwrap();
        comm.reduce_scatter(0x1000, 0x2000, 512).unwrap();
        comm.broadcast(0x1000, 256, 0).unwrap();
        comm.barrier().unwrap();
        comm.send_to(0x1000, 256, 0, 0).unwrap();
        comm.recv_from(0x2000, 256, 0, 0).unwrap();
        comm.group_start().unwrap();
        comm.group_end().unwrap();
        assert!(comm.is_healthy());
        comm.attempt_reconnect().unwrap();
    }

    #[test]
    fn capturable_collectives_default_to_unavailable() {
        // The contract callers (graph capture policies) rely on: a backend
        // without them enqueues nothing and reports no capacity.
        let comm = SingleGpuBackend;
        assert!(!comm.all_reduce_capturable(0x1000, 8192, 0).unwrap());
        assert!(
            !comm
                .peer_exchange_capturable(0x1000, 0x2000, 8192, 0)
                .unwrap()
        );
        assert_eq!(comm.capturable_all_reduce_max_bytes(), 0);
        comm.set_oneshot_kernel(0x1234);
        assert_eq!(comm.capturable_all_reduce_max_bytes(), 0);
    }

    #[test]
    fn command_words_default_to_unavailable() {
        // The contract the model's command path relies on: a backend without
        // a host command channel reports no capacity and moves nothing, so
        // every word stays on the broadcast path.
        let comm = SingleGpuBackend;
        assert_eq!(comm.command_words_max(), 0);
        assert!(comm.send_command_words(&[1]).is_err());
        assert!(comm.recv_command_words(&mut [0]).is_err());
    }

    #[test]
    fn test_single_gpu_symmetric_alloc_unsupported() {
        // SingleGpuBackend doesn't override symmetric_alloc/free — it must
        // fall back to the trait default which returns an error. This is
        // the contract callers depend on for fallback paths.
        let comm = SingleGpuBackend;
        assert!(comm.symmetric_alloc(1024).is_err());
        assert!(comm.symmetric_free(0x1000).is_err());
    }
}
