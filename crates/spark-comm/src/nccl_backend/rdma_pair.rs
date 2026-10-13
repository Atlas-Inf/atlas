// SPDX-License-Identifier: AGPL-3.0-only

//! Two-rank BF16 all-reduce over direct RDMA WRITEs (`ATLAS_RDMA_ALLREDUCE=1`).
//!
//! Without GPUDirect, NCCL on GB10 stages every send through its proxy: about
//! 7 GB/s for prefill-sized payloads and ~100 us for a decode-sized one. A raw
//! RC WRITE between the two Sparks sustains ~14 GB/s per 200G rail and lands
//! 64 KB in ~8 us. Both GPUs use ATS, so they read and write pinned host
//! memory directly.
//!
//! Per all-reduce `seq`, on the caller's stream: copy the partial into the
//! pinned send slot `seq % 2`, write the segment count to the host `ready`
//! word, wait until the host `arrived` word reaches it, then add the peer's
//! receive slot in place. A proxy thread watches `ready`, RDMA-WRITEs the send
//! slot into the peer's receive slot (split across rails when large) and, once
//! those complete, WRITEs the count into the peer's `arrived` word.
//!
//! Large payloads (prefill chunks) go as `segment_count()` pieces: each is sent as
//! soon as its copy lands and added as soon as it arrives, so the staging
//! copies and the adds overlap the wire. The flag words count segments; both
//! ranks segment identically (by size). Staging stays on the copy engine: SM
//! stores to the pinned slot were occasionally read stale by the NIC.
//!
//! Two slots per direction suffice. A rank only signals `seq + 2` after its
//! add of `seq` (stream order), and the peer only sends `seq + 2` after its
//! own add of `seq + 1`, which waited on our `seq + 1` flag, which we only
//! post after our `seq` data completed. `ATLAS_RDMA_PAIR_CHAIN=1` posts a
//! single-rail segment's flag in the same post right behind its data (one
//! completion instead of two), relying on same-QP WRITEs landing in order.
//!
//! `ATLAS_RDMA_ONESHOT=1` adds a graph-capturable channel for decode-sized
//! payloads in its own part of the region ([`oneshot`]); larger payloads stay
//! here. `ATLAS_GLM_CMD_RDMA=1` adds the host command ring ([`cmd_ring`]),
//! last in the region. The proxy serves every channel from one loop.

use anyhow::{Context, Result, ensure};
use atlas_rdma::{Gid, Verbs};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub(super) mod bootstrap;
pub(super) mod cmd_ring;
mod oneshot;
mod proxy;
use cmd_ring::CmdRing;
use oneshot::OneShot;
use proxy::{CmdLink, proxy_loop};

unsafe extern "C" {
    fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, flags: u32) -> i32;
    fn cuMemHostGetDevicePointer_v2(pdptr: *mut u64, p: *mut c_void, flags: u32) -> i32;
    fn cuMemFreeHost(p: *mut c_void) -> i32;
    pub(super) fn cuMemcpyAsync(dst: u64, src: u64, bytes: usize, stream: u64) -> i32;
    fn cuStreamWriteValue64_v2(stream: u64, addr: u64, value: u64, flags: u32) -> i32;
    fn cuStreamWaitValue64_v2(stream: u64, addr: u64, value: u64, flags: u32) -> i32;
    fn cuStreamIsCapturing(stream: u64, status: *mut i32) -> i32;
}

const CU_MEMHOSTALLOC_PORTABLE_DEVICEMAP: u32 = 0x1 | 0x2;
const CU_STREAM_WAIT_VALUE_GEQ: u32 = 0x0;
/// Payloads below this go over one rail: splitting only adds a completion.
const SPLIT_MIN: usize = 1 << 20;
/// Payloads from this size are pipelined as `segment_count()` pieces.
const SEGMENT_MIN: usize = 8 << 20;

/// `ATLAS_RDMA_PAIR_SEGMENTS` (default 4; 1 = one copy/send/add per payload).
/// Read from the shared profile, so both ranks agree.
fn segment_count() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_RDMA_PAIR_SEGMENTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
            .clamp(1, 16)
    })
}
/// Flag page: `ready` (GPU -> proxy), `arrived` (peer -> GPU), and the two
/// local source words the proxy RDMA-WRITEs as the peer's `arrived`.
const READY: usize = 0;
const ARRIVED: usize = 64;
const FLAG_SRC: usize = 128;
const FLAG_PAGE: usize = 4096;

/// Bootstrap identity: the region base, then per rail QPN, PSN, GID and
/// region rkey.
const BASE_WIRE: usize = 8;
const RAIL_WIRE: usize = 4 + 4 + 16 + 4;

struct Job {
    seq: u64,
    /// Segment count before this job (flag words carry segment counts).
    first: u64,
    /// `(offset, len)` of each segment.
    parts: Vec<(usize, usize)>,
}

/// Segment layout of a `bytes` payload; identical on both ranks.
fn segments(bytes: usize) -> Vec<(usize, usize)> {
    let n = if bytes >= SEGMENT_MIN {
        segment_count()
    } else {
        1
    };
    let seg = bytes.div_ceil(n).next_multiple_of(64);
    (0..bytes.div_ceil(seg))
        .map(|i| (i * seg, seg.min(bytes - i * seg)))
        .collect()
}

pub(super) struct RdmaPair {
    /// Pinned, device-mapped region: send[2] | recv[2] | flag page.
    host: *mut u8,
    dev: u64,
    capacity: usize,
    seq: AtomicU64,
    /// Segments enqueued so far (mirrors the flag words).
    segs: AtomicU64,
    jobs: Arc<Mutex<VecDeque<Job>>>,
    stop: Arc<AtomicBool>,
    proxy: Option<std::thread::JoinHandle<()>>,
    oneshot: Option<OneShot>,
    cmd: Option<CmdRing>,
}

// SAFETY: `host` is a pinned allocation owned by this struct; the proxy thread
// reads the send slots and flag words through its own copy of the address, and
// the host thread only enqueues stream operations and jobs.
unsafe impl Send for RdmaPair {}
unsafe impl Sync for RdmaPair {}

fn region_bytes(capacity: usize) -> usize {
    4 * capacity + FLAG_PAGE
}
fn send_off(capacity: usize, slot: usize) -> usize {
    slot * capacity
}
fn recv_off(capacity: usize, slot: usize) -> usize {
    (2 + slot) * capacity
}
fn flag_off(capacity: usize) -> usize {
    4 * capacity
}

fn cu(status: i32, what: &str) -> Result<()> {
    ensure!(status == 0, "{what} failed: CUDA status {status}");
    Ok(())
}

impl RdmaPair {
    /// Whether `ATLAS_RDMA_ALLREDUCE=1`.
    pub(super) fn requested() -> bool {
        std::env::var("ATLAS_RDMA_ALLREDUCE").as_deref() == Ok("1")
    }

    /// Bring up one RC QP per rail (`ATLAS_RDMA_RAILS`, else the NCCL HCA
    /// list) against the peer, exchanging identities over TCP on `link`
    /// (see [`bootstrap`]). `capacity` is the largest payload in bytes.
    pub(super) fn connect(rank: usize, link: &bootstrap::Link, capacity: usize) -> Result<Self> {
        ensure!(
            capacity.is_multiple_of(64) && capacity > 0,
            "RDMA pair capacity must be 64-byte aligned"
        );
        let rails = rail_names(
            std::env::var("ATLAS_RDMA_RAILS").ok(),
            std::env::var("NCCL_IB_HCA").ok(),
        );
        ensure!(
            !rails.is_empty(),
            "RDMA pair: no rails (set ATLAS_RDMA_RAILS or NCCL_IB_HCA)"
        );
        let gid_override: Option<u32> = std::env::var("ATLAS_RDMA_GID")
            .ok()
            .and_then(|v| v.parse().ok());
        let os_cfg = oneshot::Config::from_env();
        let os_base = region_bytes(capacity);
        let cmd_on = cmd_ring::requested();
        let cmd_base = os_base + os_cfg.map_or(0, |c| oneshot::region_bytes(c.max));
        let bytes = cmd_base + if cmd_on { cmd_ring::REGION_BYTES } else { 0 };
        let mut host: *mut c_void = std::ptr::null_mut();
        cu(
            unsafe { cuMemHostAlloc(&mut host, bytes, CU_MEMHOSTALLOC_PORTABLE_DEVICEMAP) },
            "cuMemHostAlloc(RDMA pair)",
        )?;
        unsafe { std::ptr::write_bytes(host as *mut u8, 0, bytes) };
        let mut dev = 0u64;
        cu(
            unsafe { cuMemHostGetDevicePointer_v2(&mut dev, host, 0) },
            "cuMemHostGetDevicePointer(RDMA pair)",
        )?;

        let psn = (0x5a5a00 + rank as u32 * 0x1111) & 0xff_ffff;
        let mut verbs = Vec::with_capacity(rails.len());
        let mut local = (host as u64).to_le_bytes().to_vec();
        let mut lkeys = Vec::with_capacity(rails.len());
        for name in &rails {
            let gid_idx =
                gid_override.map_or_else(|| atlas_rdma::gid::roce_v2_ipv4_index(name), Ok)?;
            let mut v = Verbs::create(name, gid_idx, psn)
                .with_context(|| format!("RDMA pair rail {name} (gid index {gid_idx})"))?;
            // SAFETY: the region outlives the Verbs (freed in Drop after the
            // proxy, which owns the Verbs, has joined).
            let keys = unsafe { v.reg_mr_rw(host, bytes) }?;
            local.extend_from_slice(&v.qpn().to_le_bytes());
            local.extend_from_slice(&v.psn().to_le_bytes());
            local.extend_from_slice(&v.gid());
            local.extend_from_slice(&keys.rkey.to_le_bytes());
            lkeys.push(keys.lkey);
            verbs.push(v);
        }

        let head = bootstrap::Head {
            rank,
            rails: rails.len(),
            chain: proxy::chain_requested(),
            segments: segment_count(),
            capacity,
            oneshot: oneshot::Config::wire(os_cfg),
            cmd: cmd_ring::wire(cmd_on),
        };
        let (mut stream, remote) = bootstrap::open(&head, &local, link)?;
        let peer_base = u64::from_le_bytes(remote[..BASE_WIRE].try_into()?);
        let mut peer_rkeys = Vec::with_capacity(rails.len());
        for (v, w) in verbs
            .iter_mut()
            .zip(remote[BASE_WIRE..].chunks_exact(RAIL_WIRE))
        {
            let qpn = u32::from_le_bytes(w[0..4].try_into()?);
            let rpsn = u32::from_le_bytes(w[4..8].try_into()?);
            let gid: Gid = w[8..24].try_into()?;
            peer_rkeys.push(u32::from_le_bytes(w[24..28].try_into()?));
            v.connect(qpn, rpsn, &gid)?;
        }
        // Barrier: both ends are RTS before either posts a WRITE.
        stream.write_all(&[1]).map_err(bootstrap::io("barrier"))?;
        stream
            .read_exact(&mut [0u8])
            .map_err(bootstrap::io("barrier"))?;

        let oneshot = os_cfg
            .map(|c| {
                OneShot::new(
                    c,
                    rails.len(),
                    host as usize + os_base,
                    dev + os_base as u64,
                )
            })
            .transpose()?;
        let channel = os_cfg
            .map(|c| oneshot::Channel::new(c, host as usize + os_base, peer_base + os_base as u64));
        let (cmd, cmd_link) = if cmd_on {
            let (ring, proxy) = cmd_ring::channel(host as usize + cmd_base);
            let link = CmdLink {
                proxy,
                local: host as usize + cmd_base,
                peer: peer_base + cmd_base as u64,
            };
            (Some(ring), Some(link))
        } else {
            (None, None)
        };
        let jobs = Arc::new(Mutex::new(VecDeque::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let proxy = {
            let (jobs, stop) = (jobs.clone(), stop.clone());
            let host = host as usize;
            std::thread::Builder::new()
                .name("atlas-rdma-pair".into())
                .spawn(move || {
                    let peer = Peer {
                        base: peer_base,
                        rkeys: peer_rkeys,
                    };
                    if let Err(e) = proxy_loop(
                        verbs, &lkeys, &peer, host, capacity, &jobs, &stop, channel, cmd_link,
                    ) {
                        tracing::error!("RDMA pair proxy failed: {e:#}");
                    }
                })?
        };
        tracing::info!(
            "RDMA pair all-reduce ready: rank {rank}, rails {rails:?}, capacity {} MB, one-shot {os_cfg:?}",
            capacity >> 20
        );
        if cmd_on {
            tracing::info!(
                "ATLAS_GLM_CMD_RDMA: command words ride the RDMA pair (ring of {} words)",
                cmd_ring::RING_WORDS
            );
        }
        Ok(Self {
            host: host as *mut u8,
            dev,
            capacity,
            seq: AtomicU64::new(0),
            segs: AtomicU64::new(0),
            jobs,
            stop,
            proxy: Some(proxy),
            oneshot,
            cmd,
        })
    }

    /// Exchange on `stream`: copy each segment of `src[..bytes]` into the send
    /// slot, then `land(dst, recv, len)` the peer's matching segment into
    /// `dst` as it arrives (an in-place add for an all-reduce with
    /// `src == dst`, or a copy for an all-gather step).
    /// Returns `false` (nothing enqueued) when the payload exceeds the
    /// capacity or `stream` is capturing — both ranks see the same answer.
    pub(super) fn exchange(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        stream: u64,
        land: impl Fn(u64, u64, usize) -> Result<()>,
    ) -> Result<bool> {
        if bytes > self.capacity {
            return Ok(false);
        }
        let mut capturing = 0i32;
        cu(
            unsafe { cuStreamIsCapturing(stream, &mut capturing) },
            "cuStreamIsCapturing",
        )?;
        if capturing != 0 {
            return Ok(false);
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let slot = (seq & 1) as usize;
        let flags = self.dev + flag_off(self.capacity) as u64;
        let send = self.dev + send_off(self.capacity, slot) as u64;
        let recv = self.dev + recv_off(self.capacity, slot) as u64;
        let parts = segments(bytes);
        let first = self.segs.fetch_add(parts.len() as u64, Ordering::Relaxed);
        for (i, &(off, len)) in parts.iter().enumerate() {
            cu(
                unsafe { cuMemcpyAsync(send + off as u64, src + off as u64, len, stream) },
                "cuMemcpyAsync(RDMA send slot)",
            )?;
            // Default flags fence prior writes (the copy) before the value lands.
            let ready = first + i as u64 + 1;
            cu(
                unsafe { cuStreamWriteValue64_v2(stream, flags + READY as u64, ready, 0) },
                "cuStreamWriteValue64(ready)",
            )?;
        }
        self.jobs.lock().push_back(Job {
            seq,
            first,
            parts: parts.clone(),
        });
        for (i, &(off, len)) in parts.iter().enumerate() {
            let arrived = first + i as u64 + 1;
            cu(
                unsafe {
                    cuStreamWaitValue64_v2(
                        stream,
                        flags + ARRIVED as u64,
                        arrived,
                        CU_STREAM_WAIT_VALUE_GEQ,
                    )
                },
                "cuStreamWaitValue64(arrived)",
            )?;
            land(dst + off as u64, recv + off as u64, len)?;
        }
        Ok(true)
    }
}

impl RdmaPair {
    /// Largest payload in bytes.
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// The graph-capturable channel, when `ATLAS_RDMA_ONESHOT=1`.
    pub(super) fn oneshot(&self) -> Option<&OneShot> {
        self.oneshot.as_ref()
    }

    /// The host command ring, when `ATLAS_GLM_CMD_RDMA=1`.
    pub(super) fn cmd(&self) -> Option<&CmdRing> {
        self.cmd.as_ref()
    }
}

/// Stream-ordered copy (e.g. a landed receive segment into its destination).
pub(super) fn copy_async(dst: u64, src: u64, bytes: usize, stream: u64) -> Result<()> {
    cu(
        unsafe { cuMemcpyAsync(dst, src, bytes, stream) },
        "cuMemcpyAsync(RDMA land)",
    )
}

impl Drop for RdmaPair {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(proxy) = self.proxy.take() {
            let _ = proxy.join();
        }
        unsafe { cuMemFreeHost(self.host as *mut c_void) };
    }
}

struct Peer {
    base: u64,
    rkeys: Vec<u32>,
}

/// Rail device names: `ATLAS_RDMA_RAILS`, else `NCCL_IB_HCA` without NCCL's
/// `^`/`=` match prefixes and `:port` suffixes, else the first GB10 port.
fn rail_names(rails: Option<String>, nccl: Option<String>) -> Vec<String> {
    let list = rails.or(nccl).unwrap_or_else(|| "rocep1s0f0".into());
    list.trim_start_matches(['^', '='])
        .split(',')
        .filter_map(|s| s.split(':').next())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod cmd_ring_nic_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rails_prefer_explicit_then_nccl_list() {
        let own = |v: &str| Some(v.to_owned());
        assert_eq!(rail_names(own("a,b"), own("c")), ["a", "b"]);
        assert_eq!(rail_names(None, own("=c:1,d:1")), ["c", "d"]);
        assert_eq!(rail_names(None, None), ["rocep1s0f0"]);
    }

    #[test]
    fn large_staged_payloads_split_into_aligned_segments() {
        // An 8K-row prefill chunk: four 16 MiB pieces covering the payload.
        let bytes = 8196 * 4096 * 2;
        let parts = segments(bytes);
        assert_eq!(parts.len(), segment_count());
        let mut next = 0;
        for &(off, len) in &parts {
            assert_eq!(off, next);
            assert!(off % 64 == 0 && len > 0);
            next += len;
        }
        assert_eq!(next, bytes);
        // Decode payloads stay whole.
        assert_eq!(segments(64 << 10), [(0, 64 << 10)]);
    }

    #[test]
    fn region_layout_is_disjoint_and_aligned() {
        let cap = 1 << 20;
        let spans = [
            send_off(cap, 0),
            send_off(cap, 1),
            recv_off(cap, 0),
            recv_off(cap, 1),
            flag_off(cap),
        ];
        for pair in spans.windows(2) {
            assert_eq!(pair[1] - pair[0], cap);
        }
        assert_eq!(region_bytes(cap), flag_off(cap) + FLAG_PAGE);
        const { assert!(FLAG_SRC + 16 <= FLAG_PAGE && ARRIVED.is_multiple_of(8)) };
    }
}
