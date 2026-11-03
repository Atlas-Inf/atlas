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
pub mod chain_channel;
mod chain_ops;
pub use chain_ops::{AllToAllPlane, RS_FUSED_MAX_PIECES, RsFused, RsFusedPiece, RsFusedPlan};
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

/// Whether `ATLAS_RDMA_ALLREDUCE=1`: collectives take the direct RDMA pair
/// (two ranks) or chain (three or four in a line) instead of NCCL.
pub fn rdma_requested() -> bool {
    std::env::var("ATLAS_RDMA_ALLREDUCE").as_deref() == Ok("1")
}

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
    /// (nothing enqueued) when unavailable for this call, and always with
    /// other than two ranks.
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

    /// Stream-ordered in-place all-gather over every rank: rank `r`'s `bytes`
    /// at `base + r * stride` (`stride` may be negative; the parts must not
    /// overlap) land at the same addresses on every rank, copied. Every rank
    /// must call it with the same `bytes` and `stride`. Returns `false`
    /// (nothing enqueued) when unavailable; [`Self::supports_gather_async`]
    /// says beforehand, from the size and configuration alone. The RDMA chain
    /// serves it.
    fn gather_async(&self, _base: u64, _stride: i64, _bytes: usize, _stream: u64) -> Result<bool> {
        Ok(false)
    }

    /// Independent collective channels this communicator has: the RDMA
    /// chain's one-shot channels (`ATLAS_RDMA_ONESHOT_CHANNELS`), selected
    /// per thread with [`chain_channel::enter`]; 1 elsewhere.
    fn collective_channels(&self) -> usize {
        1
    }

    /// Whether [`Self::gather_async`] can serve parts of `bytes`.
    fn supports_gather_async(&self, _bytes: usize) -> bool {
        false
    }

    /// Stream-ordered broadcast of rank `root`'s `ptr[..bytes]` into every
    /// rank's `ptr` (each rank's own address), copied. Every rank must call
    /// it with the same `bytes` and `root`. Returns `false` (nothing
    /// enqueued) when unavailable; it is available exactly when
    /// [`Self::gather_async`] is (the RDMA chain serves both).
    fn broadcast_async(
        &self,
        _ptr: u64,
        _bytes: usize,
        _root: usize,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Stream-ordered reduce-scatter over every rank of a four-rank chain:
    /// segment `d` (`bytes` at `seg0 + d * stride`) of every rank's BF16
    /// partial is summed, in the all-reduce's own order (so the words are
    /// the all-reduce's), into rank `d`'s segment; the other segments are
    /// left unspecified. Every rank must call it with the same `bytes` and
    /// `stride`. Returns `false` (nothing enqueued) when unavailable;
    /// [`Self::supports_reduce_scatter_async`] says beforehand.
    fn reduce_scatter_async(
        &self,
        _seg0: u64,
        _stride: i64,
        _bytes: usize,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Whether [`Self::reduce_scatter_async`] can serve segments of `bytes`
    /// (16-byte aligned).
    fn supports_reduce_scatter_async(&self, _bytes: usize) -> bool {
        false
    }

    /// [`Self::reduce_scatter_async`] with the pieces on the wire as E4M3
    /// values and one f32 scale per 128 (about half the bytes): each hop
    /// dequantizes, adds in f32 and requantizes only what it forwards, in
    /// the same (x0 + x1) + (x2 + x3) pairing, and the owner rounds the f32
    /// sum to BF16 once. Lossy, so opt-in, and needs `bytes` in whole 8 KiB.
    /// Returns `false` (nothing enqueued) when unavailable.
    fn reduce_scatter_fp8_async(
        &self,
        _seg0: u64,
        _stride: i64,
        _bytes: usize,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Begin a fused four-rank chain reduce-scatter of the same segments as
    /// [`Self::reduce_scatter_async`] (`bytes` at `seg0 + d * stride`): the
    /// plan a producer kernel takes, which writes every input row of the op
    /// itself, raw pieces straight into the op's staging, and publishes each
    /// piece as it finishes (`rdma_rs_fused.cuh`). Enqueues nothing; after
    /// `Some` the caller must launch the producer (stream-ordered after every
    /// earlier collective) and [`Self::reduce_scatter_fused_async`]. Same
    /// sums as the reduce-scatter. The producer writes whole `unit`-byte
    /// rows, so every piece must be whole rows. `None` when unavailable (off,
    /// too large, pieces that split a row, or right after another fused op):
    /// fall back to the plain one.
    fn rs_fused_begin(
        &self,
        _seg0: u64,
        _stride: i64,
        _bytes: usize,
        _unit: usize,
    ) -> Result<Option<RsFused>> {
        Ok(None)
    }

    /// The chain kernel of a [`Self::rs_fused_begin`] op on `stream`; it may
    /// run beside the producer (it waits for each piece the producer
    /// publishes).
    fn reduce_scatter_fused_async(&self, _fused: &RsFused, _stream: u64) -> Result<bool> {
        Ok(false)
    }

    /// Make `stream` wait until piece `piece` of this rank's segment of
    /// `fused` holds its final sum.
    fn rs_fused_wait(&self, _fused: &RsFused, _piece: usize, _stream: u64) -> Result<()> {
        anyhow::bail!("fused reduce-scatter unavailable")
    }

    /// Stream-ordered all-to-all over every rank of a four-rank chain: for
    /// each plane, rank `r`'s part for rank `d` (`rows` rows of `row_bytes`,
    /// contiguous at `src + d * src_stride`) lands on rank `d` as the `rows`
    /// rows at `dst + r * dst_stride`, `pitch` bytes apart; a rank's own part
    /// is copied the same way. Two planes carry, say, E4M3 rows and their
    /// scale tiles in one op. Every rank must call it with the same shapes.
    /// Returns `false` (nothing enqueued) when unavailable.
    fn all_to_all_async(&self, _planes: &[AllToAllPlane], _stream: u64) -> Result<bool> {
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

    /// Provide the kernel handle for the capturable collectives (module
    /// `rdma_oneshot`, function [`Self::oneshot_kernel`]); loaded by the
    /// model layer like [`Self::set_add_kernel`].
    fn set_oneshot_kernel(&self, _handle: u64) {}

    /// The function [`Self::set_oneshot_kernel`] expects: `rdma_oneshot_bf16`
    /// for two ranks, `rdma_chain_bf16` for a chain.
    fn oneshot_kernel(&self) -> &'static str {
        "rdma_oneshot_bf16"
    }

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
#[path = "lib_tests.rs"]
mod tests;
