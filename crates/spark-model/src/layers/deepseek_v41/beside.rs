// SPDX-License-Identifier: AGPL-3.0-only

//! A second stream for the collectives a batch's own work can run beside,
//! and the events that order the two; and an event for the host to read
//! what a stream has made while the stream goes on.
//!
//! A gather waits on the wire, and the stream it is on waits with it. Where
//! a batch has work that does not read what is being gathered, the gather
//! goes here and the batch's stream goes on: it stops only where it reads
//! what was brought ([`Beside::brought`]). The ranks' collectives keep one
//! order, for a gather here starts after everything the batch's stream had
//! queued when it was sent ([`Beside::send`]), collectives included, and
//! the batch's stream queues its next collective only after waiting for
//! the last one sent.

use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;

/// The sends one batch may have in flight or unread at a time.
pub const SENDS: usize = 2;

pub struct Beside {
    stream: u64,
    /// Recorded on the batch's stream when a send's bytes are ready.
    ready: [u64; SENDS],
    /// Recorded on this stream when a send's gathers are done.
    done: [u64; SENDS],
    /// Recorded on the batch's stream where the host reads what it has made
    /// so far ([`Beside::made`]).
    made: u64,
}

impl Beside {
    pub fn alloc(gpu: &dyn GpuBackend) -> Result<Self> {
        let events = || -> Result<[u64; SENDS]> {
            let mut events = [0u64; SENDS];
            for event in &mut events {
                *event = gpu.create_event()?;
            }
            Ok(events)
        };
        Ok(Self {
            // Ahead of the batch's stream: a gather's launches are short,
            // and behind a long launch's waiting blocks they start as it
            // ends.
            stream: gpu.create_stream_ahead()?,
            ready: events()?,
            done: events()?,
            made: gpu.create_event()?,
        })
    }

    /// Send `at`: what `stream` has queued so far leaves bytes to gather.
    /// `gathers` queues their gathers on the stream it is given, and they run
    /// beside what `stream` queues next.
    pub fn send(
        &self,
        gpu: &dyn GpuBackend,
        at: usize,
        stream: u64,
        gathers: impl FnOnce(u64) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            at < SENDS,
            "deepseek_v41: send {at} of {SENDS} beside a batch"
        );
        gpu.record_event(self.ready[at], stream)?;
        gpu.stream_wait_event(self.stream, self.ready[at])?;
        gathers(self.stream)?;
        gpu.record_event(self.done[at], self.stream)
    }

    /// The host is to read what `stream` has queued so far
    /// ([`Self::read`]), and not wait for what it queues next: nothing
    /// queued after this may write what the host reads or what it then
    /// writes.
    pub fn made(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        gpu.record_event(self.made, stream)
    }

    /// Wait, on the host, for what the stream had queued at the last
    /// [`Self::made`].
    pub fn read(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.event_synchronize(self.made)
    }

    /// What `stream` queues next reads what send `at` brought.
    pub fn brought(&self, gpu: &dyn GpuBackend, at: usize, stream: u64) -> Result<()> {
        ensure!(
            at < SENDS,
            "deepseek_v41: send {at} of {SENDS} beside a batch"
        );
        gpu.stream_wait_event(stream, self.done[at])
    }
}
