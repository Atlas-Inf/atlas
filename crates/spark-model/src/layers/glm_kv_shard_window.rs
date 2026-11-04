// SPDX-License-Identifier: AGPL-3.0-only

//! The pair exchange's overlap window (`ATLAS_GLM_KV_SHARD_OVERLAP`).

use super::*;

/// The communicator inside an [`overlapped_exchange`] window: the pair's
/// rank and size, and an error from every operation that would use the pair.
/// The pair orders its sends by stream order and counts landings in one
/// place, so a second user beside the lane's exchange would corrupt or hang
/// both ranks without an error of its own.
pub struct WindowComm<'a>(&'a dyn spark_comm::CommBackend);

impl WindowComm<'_> {
    fn refuse<T>(&self, what: &str) -> Result<T> {
        bail!(
            "{what} inside a GLM KV shard overlap window on rank {}: the pair is carrying the lane's exchange",
            self.0.rank()
        )
    }
}

impl spark_comm::CommBackend for WindowComm<'_> {
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        self.refuse("all-reduce")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        self.refuse("all-gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        self.refuse("reduce-scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        self.refuse("broadcast")
    }
    fn barrier(&self) -> Result<()> {
        self.refuse("barrier")
    }
    fn peer_exchange_async(&self, _: u64, _: u64, _: usize, _: u64) -> Result<()> {
        self.refuse("peer exchange")
    }
    fn exchange_async(&self, _: u64, _: u64, _: usize, _: bool, _: u64) -> Result<bool> {
        self.refuse("exchange")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        self.refuse("send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        self.refuse("receive")
    }
    fn rank(&self) -> usize {
        self.0.rank()
    }
    fn world_size(&self) -> usize {
        self.0.world_size()
    }
}

/// [`pair_exchange`] on the shard's side stream, so the compute stream runs
/// `during` while the payload is in flight, and waits for it to land after.
///
/// `during` must not read `recv` or write `send`, and cannot use the pair:
/// only this fence orders the side stream against the compute stream, so
/// the communicator it is handed (give it to whatever `during` calls)
/// refuses every pair operation. The wait is enqueued even when `during`
/// fails. Both ranks run the same exchanges in the same order, so an overlap
/// only moves where each waits.
pub fn overlapped_exchange<T>(
    gpu: &dyn GpuBackend,
    comm: &dyn spark_comm::CommBackend,
    lane: ExchangeLane,
    (send, recv, bytes): (DevicePtr, DevicePtr, usize),
    stream: u64,
    during: impl FnOnce(&WindowComm) -> Result<T>,
) -> Result<T> {
    gpu.record_event(lane.begun, stream)?;
    gpu.stream_wait_event(lane.stream, lane.begun)?;
    pair_exchange(comm, send, recv, bytes, lane.stream)?;
    gpu.record_event(lane.landed, lane.stream)?;
    let out = during(&WindowComm(comm));
    gpu.stream_wait_event(stream, lane.landed)?;
    out
}
