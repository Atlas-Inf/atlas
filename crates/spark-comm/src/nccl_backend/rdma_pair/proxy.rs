// SPDX-License-Identifier: AGPL-3.0-only

//! The RDMA pair proxy thread. One event loop serves every channel -- the
//! one-shot `stage` word, the host command ring and the head legacy job --
//! without blocking on any: the host queues legacy jobs milliseconds ahead of
//! the GPU, so a one-shot op or a command word can precede the head job.

use super::cmd_ring::{CmdProxy, Wire};
use super::oneshot::{Channel, stripes};
use super::{ARRIVED, FLAG_SRC, Job, Peer, READY, SPLIT_MIN, flag_off, recv_off, send_off};
use anyhow::Result;
use atlas_rdma::Verbs;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Whether `ATLAS_RDMA_PAIR_CHAIN=1`; checked equal on both ranks at
/// bootstrap.
pub(super) fn chain_requested() -> bool {
    std::env::var("ATLAS_RDMA_PAIR_CHAIN").as_deref() == Ok("1")
}

/// The command ring's proxy half with the addresses of this rank's command
/// region and the peer's.
pub(super) struct CmdLink {
    pub(super) proxy: CmdProxy,
    pub(super) local: usize,
    pub(super) peer: u64,
}

/// The command ring's WRITEs: rail 0, each reaped before the next.
struct CmdWire<'a> {
    rail: &'a mut Verbs,
    lkey: u32,
    rkey: u32,
    local: usize,
    peer: u64,
}

impl Wire for CmdWire<'_> {
    fn write(&mut self, src: usize, dst: usize, len: usize) -> Result<()> {
        // SAFETY: both spans lie in the command regions, inside the
        // registered pinned regions; the ring leaves the local span
        // unmodified until this WRITE is reaped just below.
        unsafe {
            self.rail.post_write(
                (self.local + src) as *mut c_void,
                self.lkey,
                self.peer + dst as u64,
                self.rkey,
                u32::try_from(len)?,
                0,
            )
        }?;
        self.rail.poll().map(drop)
    }
}

/// Run the proxy until `stop`. If it fails, no one-shot send would ever
/// drain, so it stops that channel ([`Channel::fail`]) rather than leave
/// streams waiting on it. However it ends, no command word moves again, so
/// the ring's waiters are failed ([`CmdProxy::close`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn proxy_loop(
    rails: Vec<Verbs>,
    lkeys: &[u32],
    peer: &Peer,
    host: usize,
    capacity: usize,
    jobs: &Mutex<VecDeque<Job>>,
    stop: &AtomicBool,
    mut oneshot: Option<Channel>,
    mut cmd: Option<CmdLink>,
) -> Result<()> {
    let end = serve(
        rails,
        lkeys,
        peer,
        host,
        capacity,
        jobs,
        stop,
        &mut oneshot,
        &mut cmd,
    );
    if let (Err(_), Some(ch)) = (&end, &oneshot) {
        ch.fail();
    }
    if let Some(cmd) = &cmd {
        cmd.proxy.close();
    }
    end
}

#[allow(clippy::too_many_arguments)]
fn serve(
    mut rails: Vec<Verbs>,
    lkeys: &[u32],
    peer: &Peer,
    host: usize,
    capacity: usize,
    jobs: &Mutex<VecDeque<Job>>,
    stop: &AtomicBool,
    oneshot: &mut Option<Channel>,
    cmd: &mut Option<CmdLink>,
) -> Result<()> {
    // SAFETY: the flag page lives in the pinned region for the pair's lifetime;
    // `ready` is written by the GPU (stream memop) and read only here.
    let ready = unsafe { &*((host + flag_off(capacity) + READY) as *const AtomicU64) };
    let chain = chain_requested();
    let mut idle = Idle::from_env(oneshot.is_some(), cmd.is_some());
    let mut stats = Stats::from_env();
    // The head legacy job, its next segment, and when that segment became due.
    let mut head: Option<(Job, usize, Instant)> = None;
    loop {
        if let Some(ch) = oneshot.as_mut()
            && ch.serve(&mut rails, lkeys, &peer.rkeys, chain)?
        {
            idle.reset();
            continue;
        }
        if let Some(cmd) = cmd.as_mut() {
            let mut wire = CmdWire {
                rail: &mut rails[0],
                lkey: lkeys[0],
                rkey: peer.rkeys[0],
                local: cmd.local,
                peer: cmd.peer,
            };
            if cmd.proxy.serve(&mut wire)? {
                idle.reset();
                continue;
            }
        }
        if head.is_none() {
            head = jobs.lock().pop_front().map(|job| (job, 0, Instant::now()));
        }
        let Some((job, i, due)) = head.as_mut() else {
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
            idle.wait();
            continue;
        };
        idle.reset();
        if ready.load(Ordering::Acquire) < job.first + *i as u64 + 1 {
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
            std::hint::spin_loop();
            continue;
        }
        let t = send_segment(&mut rails, lkeys, peer, host, capacity, job, *i, chain)?;
        stats.record(job.parts.iter().map(|p| p.1).sum(), *due, t);
        *i += 1;
        *due = Instant::now();
        if *i == job.parts.len() {
            head = None;
        }
    }
}

/// WRITE segment `i` of `job` into the peer's receive slot (split across
/// rails when large), then the segment count into the peer's `arrived`.
/// Returns when the data was posted and when its completions were reaped
/// (chained: when it was posted).
#[allow(clippy::too_many_arguments)]
fn send_segment(
    rails: &mut [Verbs],
    lkeys: &[u32],
    peer: &Peer,
    host: usize,
    capacity: usize,
    job: &Job,
    i: usize,
    chain: bool,
) -> Result<[Instant; 2]> {
    let (seg_off, seg_len) = job.parts[i];
    let count = job.first + i as u64 + 1;
    let slot = (job.seq & 1) as usize;
    let src = host + send_off(capacity, slot) + seg_off;
    let dst = peer.base + (recv_off(capacity, slot) + seg_off) as u64;
    let parts = stripes(seg_len, rails.len(), SPLIT_MIN);
    let src_flag = host + flag_off(capacity) + FLAG_SRC + slot * 8;
    let dst_flag = peer.base + (flag_off(capacity) + ARRIVED) as u64;
    // SAFETY: the flag source word is ours; the previous WRITE from this
    // word completed before that segment's poll returned.
    unsafe { (src_flag as *mut u64).write_volatile(count) };
    // A single-rail segment may chain its flag behind the data on the same
    // QP; a split one needs every rail's data complete before the flag.
    let chained = chain && parts.len() == 1;
    for (r, (rail, &(off, len))) in rails.iter_mut().zip(&parts).enumerate() {
        // SAFETY: src..+len lies in the registered region (send slot);
        // the slot is not rewritten until this seq's flag is consumed.
        unsafe {
            let (local, remote, len) = (
                (src + off) as *mut c_void,
                dst + off as u64,
                u32::try_from(len)?,
            );
            if chained {
                rail.post_write_flag(
                    local,
                    lkeys[r],
                    remote,
                    peer.rkeys[r],
                    len,
                    src_flag as *mut c_void,
                    dst_flag,
                    count,
                )
            } else {
                rail.post_write(local, lkeys[r], remote, peer.rkeys[r], len, count)
            }
        }?;
    }
    let t1 = Instant::now();
    if !chained {
        for rail in rails.iter_mut().take(parts.len()) {
            rail.poll()?;
        }
    }
    let t2 = Instant::now();
    if !chained {
        // SAFETY: as above.
        unsafe {
            rails[0].post_write(
                src_flag as *mut c_void,
                lkeys[0],
                dst_flag,
                peer.rkeys[0],
                8,
                count,
            )
        }?;
    }
    rails[0].poll()?;
    Ok([t1, t2])
}

/// What the proxy does with nothing to serve. Legacy (one-shot off and no
/// `ATLAS_RDMA_PAIR_SPIN_US`): 20,000 spins, then 50 us sleeps. Time-based:
/// spin `ATLAS_RDMA_PAIR_SPIN_US` (10,000 with one-shot) after the last work,
/// then nap `ATLAS_RDMA_PAIR_NAP_US` (20) with 1 us timer slack -- a graph
/// replay gives the proxy no host-side notice, so the first stage after a
/// pause waits at most about one nap. With the command ring alone the spin
/// is 50,000 us and the naps stay the legacy ones: the head's next step
/// words come a drafter pass (6-16 ms) after its last collective and must
/// not meet a napping proxy, while a nap between requests costs nothing.
struct Idle {
    spin: Option<Duration>,
    nap: Duration,
    since: Option<Instant>,
    spins: u32,
}

impl Idle {
    fn from_env(oneshot: bool, cmd: bool) -> Self {
        let us = |key: &str| std::env::var(key).ok().and_then(|v| v.parse::<u64>().ok());
        let fine = us("ATLAS_RDMA_PAIR_SPIN_US").or(oneshot.then_some(10_000));
        let spin = fine.or(cmd.then_some(50_000)).map(Duration::from_micros);
        let nap = us("ATLAS_RDMA_PAIR_NAP_US").unwrap_or(if fine.is_some() { 20 } else { 50 });
        #[cfg(target_os = "linux")]
        if fine.is_some() {
            // SAFETY: plain prctl on the calling (proxy) thread.
            unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, 1000u64) };
        }
        Self {
            spin,
            nap: Duration::from_micros(nap),
            since: None,
            spins: 0,
        }
    }

    fn reset(&mut self) {
        self.since = None;
        self.spins = 0;
    }

    fn wait(&mut self) {
        let napping = match self.spin {
            None => {
                self.spins = self.spins.saturating_add(1);
                self.spins > 20_000
            }
            Some(spin) => self.since.get_or_insert_with(Instant::now).elapsed() > spin,
        };
        if napping {
            std::thread::sleep(self.nap);
        } else {
            std::hint::spin_loop();
        }
    }
}

/// `ATLAS_RDMA_PAIR_STATS=1`: mean legacy ready-wait / data / flag
/// microseconds (data covers the posts; chained, the flag covers both).
struct Stats {
    on: bool,
    acc: [f64; 3],
    small: u64,
    large: u64,
}

impl Stats {
    fn from_env() -> Self {
        Self {
            on: std::env::var("ATLAS_RDMA_PAIR_STATS").as_deref() == Ok("1"),
            acc: [0.0; 3],
            small: 0,
            large: 0,
        }
    }

    fn record(&mut self, bytes: usize, due: Instant, [t1, t2]: [Instant; 2]) {
        if !self.on {
            return;
        }
        let t3 = Instant::now();
        self.acc[0] += (t1 - due).as_secs_f64() * 1e6;
        self.acc[1] += (t2 - t1).as_secs_f64() * 1e6;
        self.acc[2] += (t3 - t2).as_secs_f64() * 1e6;
        if bytes >= SPLIT_MIN {
            self.large += 1;
        } else {
            self.small += 1;
        }
        let n = self.small + self.large;
        if n.is_multiple_of(4096) {
            tracing::info!(
                "RDMA pair stats: {n} jobs ({} small, {} large); mean us: ready-wait {:.1} data {:.1} flag {:.1}",
                self.small,
                self.large,
                self.acc[0] / n as f64,
                self.acc[1] / n as f64,
                self.acc[2] / n as f64
            );
        }
    }
}
