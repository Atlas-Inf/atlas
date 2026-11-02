// SPDX-License-Identifier: AGPL-3.0-only
//! The communicator a piecewise verify capture records through.

use super::*;

impl CommBackend for Recorder<'_> {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        self.split(CommOp::AllReduce { ptr, bytes })
    }
    /// With `capture_collectives`, the capturable one-shot records the reduce
    /// INTO the open graph. It is the operation the eager `all_reduce_async`
    /// would have enqueued (both try the one-shot first, under the same
    /// size-only eligibility), so replay stays lossless and both ranks
    /// execute the same collective sequence whatever each rank captured. A
    /// refusal (size, channel off) splits as usual.
    fn all_reduce_async(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        if self.capture_collectives
            && stream == self.stream
            && self.inner.all_reduce_capturable(ptr, bytes, stream)?
        {
            return Ok(());
        }
        self.on_stream(stream, CommOp::AllReduceAsync { ptr, bytes, stream })
    }
    fn peer_exchange_async(&self, send: u64, recv: u64, bytes: usize, stream: u64) -> Result<()> {
        let op = CommOp::PeerExchangeAsync {
            send,
            recv,
            bytes,
            stream,
        };
        self.on_stream(stream, op)
    }
    fn supports_peer_exchange_async(&self) -> bool {
        self.inner.supports_peer_exchange_async()
    }
    /// Recorded when the communicator would serve it (its answer depends on
    /// the size alone), declined as the eager call would be otherwise.
    fn gather_async(&self, base: u64, stride: i64, bytes: usize, stream: u64) -> Result<bool> {
        if !self.inner.supports_gather_async(bytes) {
            return Ok(false);
        }
        let op = CommOp::GatherAsync {
            base,
            stride,
            bytes,
            stream,
        };
        self.on_stream(stream, op)?;
        Ok(true)
    }
    fn supports_gather_async(&self, bytes: usize) -> bool {
        self.inner.supports_gather_async(bytes)
    }
    fn supports_exchange_async(&self, bytes: usize) -> bool {
        self.inner.supports_exchange_async(bytes)
    }
    fn exchange_async(&self, _: u64, _: u64, _: usize, _: bool, _: u64) -> Result<bool> {
        unsupported("exchange_async")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        unsupported("all_gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        unsupported("reduce_scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        unsupported("broadcast")
    }
    fn receive_idle_command_word(&self, _: u64) -> Result<()> {
        unsupported("receive_idle_command_word")
    }
    fn barrier(&self) -> Result<()> {
        unsupported("barrier")
    }
    fn register_buffer(&self, _: u64, _: usize) -> Result<u64> {
        unsupported("register_buffer")
    }
    fn deregister_buffer(&self, _: u64) -> Result<()> {
        unsupported("deregister_buffer")
    }
    fn symmetric_alloc(&self, _: usize) -> Result<u64> {
        unsupported("symmetric_alloc")
    }
    fn symmetric_free(&self, _: u64) -> Result<()> {
        unsupported("symmetric_free")
    }
    fn set_add_kernel(&self, _: u64) {}
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        unsupported("send_to")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        unsupported("recv_from")
    }
    fn group_start(&self) -> Result<()> {
        unsupported("group_start")
    }
    fn group_end(&self) -> Result<()> {
        unsupported("group_end")
    }
    fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }
    fn attempt_reconnect(&self) -> Result<()> {
        unsupported("attempt_reconnect")
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
}
