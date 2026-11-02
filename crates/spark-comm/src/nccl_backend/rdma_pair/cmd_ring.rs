// SPDX-License-Identifier: AGPL-3.0-only

//! Host command words over the RDMA pair (`ATLAS_GLM_CMD_RDMA=1`, default
//! off).
//!
//! The head's command words (slot, command, width, tokens, verdict) otherwise
//! each ride an NCCL broadcast: a device kernel on both ranks, a stream sync
//! and a copy back on the receiver, and a wake-up of NCCL's progress thread.
//! In the 2026-09-30 step profile that was 130-570 us per word at the
//! receiver, five words a verify step: the worker started each step about
//! 0.75 ms behind the head, which the head then waited out at its first
//! collective.
//!
//! Here a word is host memory end to end. Each rank has an `out` ring it
//! fills and an `in` ring the peer's NIC fills, in the pair's pinned region.
//! [`CmdRing::send`] appends words to `out` and publishes the count; the
//! proxy WRITEs the new span into the peer's `in` ring and, once that
//! completed, the count into the peer's `tail` word. [`CmdRing::recv`] spins
//! on `tail` and reads the words: in order, exactly once, with no stream, no
//! kernel and no sleeping thread on the way.
//!
//! Flow control: a sender never has more than one ring of words
//! unacknowledged, so it cannot overwrite what the peer has not read. The
//! receiving rank's proxy WRITEs its consumed count back (`ack`) once a
//! quarter ring is outstanding; a sender that would overrun waits for it. A
//! blocked sender has at least `RING_WORDS - MAX_WORDS` words out, more than
//! that quarter, so the acknowledgement it waits for is always on its way
//! once the peer reads.
//!
//! Both ranks must agree on the switch (the layout of the region depends on
//! it): the bootstrap head carries [`wire`].

use anyhow::{Result, bail, ensure};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Words per ring.
pub(in crate::nccl_backend) const RING_WORDS: usize = 1024;
/// Largest message: a width-32 verify's tokens with room to spare.
pub(in crate::nccl_backend) const MAX_WORDS: usize = 64;
/// The receiving rank acknowledges once this many words are unacknowledged.
const ACK_EVERY: u64 = (RING_WORDS / 4) as u64;
const RING_BYTES: usize = RING_WORDS * 4;
/// Region layout: `out`, `in`, then one 64-byte line per word: `tail` and
/// `ack` (the peer's NIC writes them) and the two local source words the
/// proxy WRITEs from.
const OUT: usize = 0;
const IN: usize = RING_BYTES;
const TAIL: usize = 2 * RING_BYTES;
const ACK: usize = TAIL + 64;
const TAIL_SRC: usize = TAIL + 128;
const ACK_SRC: usize = TAIL + 192;
/// Bytes of pinned region the command ring appends.
pub(in crate::nccl_backend) const REGION_BYTES: usize = 2 * RING_BYTES + 4096;
/// How long a waiter spins before it naps between looks. A worker waits for
/// the next step's words through the head's drafter pass (6-16 ms); only a
/// rank that is idle between requests naps.
const SPIN: Duration = Duration::from_millis(50);
const NAP: Duration = Duration::from_micros(50);
/// A sender waits this long for the peer to read before it gives up.
const SEND_TIMEOUT: Duration = Duration::from_secs(60);

/// Whether `ATLAS_GLM_CMD_RDMA=1`.
pub(in crate::nccl_backend) fn requested() -> bool {
    parse(std::env::var("ATLAS_GLM_CMD_RDMA").ok().as_deref())
}

fn parse(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Bootstrap form: the ring size when on, else 0; both ranks must send the
/// same.
pub(in crate::nccl_backend) fn wire(on: bool) -> u64 {
    if on { RING_WORDS as u64 } else { 0 }
}

/// Delivers bytes of this rank's command region into the peer's: an RDMA
/// WRITE on rail 0, or a copy in tests.
pub(in crate::nccl_backend) trait Wire {
    /// WRITE `len` bytes at offset `src` of the local region to offset `dst`
    /// of the peer's; returns once they have landed.
    fn write(&mut self, src: usize, dst: usize, len: usize) -> Result<()>;
}

/// What the caller's threads and the proxy share.
#[derive(Default)]
struct Shared {
    /// Words appended to `out` so far.
    written: AtomicU64,
    /// Words read from `in` so far.
    consumed: AtomicU64,
    /// The proxy is gone: nothing will be sent or acknowledged again.
    closed: AtomicBool,
}

/// `(ring index, words)` spans covering words `from..to` of the stream.
fn spans(from: u64, to: u64) -> impl Iterator<Item = (usize, usize)> {
    let (at, n) = ((from % RING_WORDS as u64) as usize, (to - from) as usize);
    let first = n.min(RING_WORDS - at);
    [(at, first), (0, n - first)]
        .into_iter()
        .filter(|&(_, words)| words > 0)
}

/// Caller side, owned by the backend.
pub(in crate::nccl_backend) struct CmdRing {
    /// Host address of this rank's command region.
    host: usize,
    shared: Arc<Shared>,
    /// Serializes senders; `written` only advances under it.
    sending: Mutex<()>,
    /// Words read so far; serializes receivers.
    read: Mutex<u64>,
}

/// The ring and its proxy half over the zeroed, pinned `REGION_BYTES` at
/// `host`, which must outlive both.
pub(in crate::nccl_backend) fn channel(host: usize) -> (CmdRing, CmdProxy) {
    let shared = Arc::new(Shared::default());
    let ring = CmdRing {
        host,
        shared: shared.clone(),
        sending: Mutex::new(()),
        read: Mutex::new(0),
    };
    let proxy = CmdProxy {
        host,
        shared,
        sent: 0,
        acked: 0,
    };
    (ring, proxy)
}

impl CmdRing {
    /// A word of the region the peer's NIC writes.
    fn peer_word(&self, off: usize) -> &AtomicU64 {
        // SAFETY: the region is pinned, 64-byte aligned and lives as long as
        // the pair; this word is written only by the peer's WRITEs.
        unsafe { &*((self.host + off) as *const AtomicU64) }
    }

    /// Wait until `ready`, spinning first. Fails once the proxy is gone, or
    /// after `limit`.
    fn wait(&self, limit: Option<Duration>, what: &str, ready: impl Fn() -> bool) -> Result<()> {
        let mut since: Option<Instant> = None;
        loop {
            if ready() {
                return Ok(());
            }
            ensure!(
                !self.shared.closed.load(Ordering::Acquire),
                "RDMA pair command ring: {what}: the proxy stopped (see its error)"
            );
            let waited = since.get_or_insert_with(Instant::now).elapsed();
            if limit.is_some_and(|limit| waited > limit) {
                bail!("RDMA pair command ring: {what}: timed out after {waited:?}");
            }
            if waited > SPIN {
                std::thread::sleep(NAP);
            } else {
                std::hint::spin_loop();
            }
        }
    }

    /// Append `words` to the stream to the peer. Returns once they are
    /// published to the proxy, which sends them at once; waits only while
    /// the peer has a whole ring still to read.
    pub(in crate::nccl_backend) fn send(&self, words: &[u32]) -> Result<()> {
        ensure!(
            !words.is_empty() && words.len() <= MAX_WORDS,
            "RDMA pair command ring: a message is 1..={MAX_WORDS} words, got {}",
            words.len()
        );
        let _sending = self.sending.lock();
        let at = self.shared.written.load(Ordering::Relaxed);
        let end = at + words.len() as u64;
        let ack = self.peer_word(ACK);
        self.wait(Some(SEND_TIMEOUT), "the peer is not reading", || {
            end - ack.load(Ordering::Acquire) <= RING_WORDS as u64
        })?;
        let out = (self.host + OUT) as *mut u32;
        for (i, &word) in words.iter().enumerate() {
            let slot = ((at + i as u64) % RING_WORDS as u64) as usize;
            // SAFETY: inside `out`. The slot is past `written`, so the proxy
            // is not sending it, and the peer has read its previous word (the
            // wait above), so that WRITE completed.
            unsafe { out.add(slot).write_volatile(word) };
        }
        self.shared.written.store(end, Ordering::Release);
        Ok(())
    }

    /// Take the next `words.len()` words of the stream from the peer,
    /// waiting for them however long the peer takes to send.
    pub(in crate::nccl_backend) fn recv(&self, words: &mut [u32]) -> Result<()> {
        ensure!(
            !words.is_empty() && words.len() <= MAX_WORDS,
            "RDMA pair command ring: a message is 1..={MAX_WORDS} words, got {}",
            words.len()
        );
        let mut read = self.read.lock();
        let end = *read + words.len() as u64;
        let tail = self.peer_word(TAIL);
        self.wait(None, "receive", || tail.load(Ordering::Acquire) >= end)?;
        let ring = (self.host + IN) as *const u32;
        for (i, word) in words.iter_mut().enumerate() {
            let slot = ((*read + i as u64) % RING_WORDS as u64) as usize;
            // SAFETY: inside `in`; the peer published `tail` after this
            // word's WRITE completed and rewrites the slot only once we
            // acknowledge it.
            *word = unsafe { ring.add(slot).read_volatile() };
        }
        *read = end;
        self.shared.consumed.store(end, Ordering::Release);
        Ok(())
    }
}

/// Proxy side: sends what the caller published and acknowledges what it read.
pub(in crate::nccl_backend) struct CmdProxy {
    host: usize,
    shared: Arc<Shared>,
    /// Words delivered to the peer.
    sent: u64,
    /// The consumed count last sent to the peer.
    acked: u64,
}

impl CmdProxy {
    fn publish(&self, wire: &mut impl Wire, src: usize, dst: usize, count: u64) -> Result<()> {
        // SAFETY: our source word; its previous WRITE completed before the
        // `write` that sent it returned.
        unsafe { ((self.host + src) as *mut u64).write_volatile(count) };
        wire.write(src, dst, 8)
    }

    /// Deliver newly published words (data, then `tail`) and a due
    /// acknowledgement. `false` when there was nothing to do.
    pub(in crate::nccl_backend) fn serve(&mut self, wire: &mut impl Wire) -> Result<bool> {
        let mut worked = false;
        let written = self.shared.written.load(Ordering::Acquire);
        if written > self.sent {
            for (at, words) in spans(self.sent, written) {
                wire.write(OUT + at * 4, IN + at * 4, words * 4)?;
            }
            self.publish(wire, TAIL_SRC, TAIL, written)?;
            self.sent = written;
            worked = true;
        }
        let consumed = self.shared.consumed.load(Ordering::Acquire);
        if consumed - self.acked >= ACK_EVERY {
            self.publish(wire, ACK_SRC, ACK, consumed)?;
            self.acked = consumed;
            worked = true;
        }
        Ok(worked)
    }

    /// The proxy is going away: fail the waiters rather than leave them
    /// spinning on words that will never come.
    pub(in crate::nccl_backend) fn close(&self) {
        self.shared.closed.store(true, Ordering::Release);
    }
}

#[cfg(test)]
#[path = "cmd_ring_tests.rs"]
mod tests;
