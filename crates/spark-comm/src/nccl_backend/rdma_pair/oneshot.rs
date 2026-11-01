// SPDX-License-Identifier: AGPL-3.0-only

//! Graph-capturable one-shot exchange over the RDMA pair
//! (`ATLAS_RDMA_ONESHOT=1`, default off).
//!
//! The legacy pair keeps per-call state on the host (sequence numbers baked
//! into stream memops, a job queue), so it refuses CUDA-graph capture. Here a
//! call enqueues three stream operations whose arguments depend only on the
//! payload size, so eager calls and graph replays run the same protocol:
//!
//! 1. wait until the host `stage` word is 0 (the previous send has left `S`);
//! 2. copy the payload into the staging buffer `S` on the copy engine (SM
//!    stores to pinned memory were read stale by the NIC, see `super`);
//! 3. `rdma_oneshot_bf16`: block 0 publishes `stage = bytes` (the copy is
//!    complete by stream order), then all wait for the peer's flags and land
//!    the peer's payload (BF16 add, bit-identical to `bf16_add_inplace`, or a
//!    copy). `ATLAS_RDMA_ONESHOT_STAGE_FENCE=1` instead publishes `stage` with
//!    a fenced stream memop between 2 and 3, the legacy `ready` ordering
//!    (about 5-7 us slower per call on GB10).
//!
//! The proxy serves a non-zero `stage` as op `served + 1`: it WRITEs `S` into
//! the peer's receive slot `seq % 2` (striped over the rails from
//! `stripe_min`), then a flag `seq << 24 | bytes` per rail, and clears `stage`
//! once those complete. The kernel keeps `seq` in device memory and the proxy
//! counts stages, so both advance once per op whether it was launched eagerly
//! or replayed. Two receive slots suffice: we only stage `seq + 2` after our
//! kernel `seq + 1`, which waited on the peer's flag `seq + 1`, which the peer
//! only staged after its kernel `seq` finished reading slot `seq % 2`.
//!
//! A late rank never waits for its own send: the peer's data has landed and
//! its kernel exits; only its next stage waits for `S` to drain. Both ranks
//! must issue the same one-shot ops in the same order, and one at a time on
//! the GPU: the sequence and `S` are shared by every stream, so ops on
//! different streams (decode on the default stream, prefill on the
//! scheduler's prefill stream) must be serialized by the caller with events
//! or syncs, as they are for the legacy channel.
//!
//! Nothing hangs on a fault; the poison word names it and the kernel traps.
//! The kernel poisons on a size mismatch at a sequence number, a flag more
//! than one op ahead (the peer cannot stage `seq + 2` before our flag
//! `seq + 1`), or a peer that never arrives within
//! `ATLAS_RDMA_ONESHOT_TIMEOUT_MS`. The host poisons when it can no longer
//! serve: the proxy failed (it then clears `stage`, so a stream blocked in
//! step 1 reaches a kernel), or a launch failed after a fenced `stage` was
//! published. Every kernel checks the word at launch and while it waits.
//!
//! Prior art (ideas only, no code; docs/glm-prior-art.md): the protocol
//! follows b12x RoCEnante (<https://github.com/local-inference-lab/b12x>,
//! `b12x/comm/roce/`, `docs/rocenante.md`; RoCEnante by Jason Cook,
//! local-inference-lab/b12x#295; Apache-2.0): pinned parity slots, a host
//! proxy that WRITEs each stripe then a seq flag on the same QP, a
//! device-resident sequence for graph replay, and poison on timeout, for the
//! hazard in <https://github.com/local-inference-lab/b12x/issues/313> (which
//! reached us through jayleaton/glm53-tensorfold-spark `docs/SFXNZ-AUDIT.md`).
//! The kernel's block structure (block 0 polls the host flags and republishes
//! seq in device memory; the last block out stores seq) follows mmastrac's
//! arx one-shot all-reduce (<https://github.com/mmastrac/glm-5.3-flash-4x-gx10>,
//! `experimental/arx/arx_vllm.cu`, PR #4; unlicensed, so idea only).

use super::{SPLIT_MIN, cu, cuMemcpyAsync};
use crate::nccl_backend::{cuLaunchKernel, cuMemAlloc_v2, cuMemFree_v2};
use anyhow::{Result, ensure};
use atlas_rdma::Verbs;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

unsafe extern "C" {
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    fn cuStreamWaitValue32_v2(stream: u64, addr: u64, value: u32, flags: u32) -> i32;
    fn cuStreamWriteValue32_v2(stream: u64, addr: u64, value: u32, flags: u32) -> i32;
}

const CU_STREAM_WAIT_VALUE_EQ: u32 = 0x1;
/// Payload sizes ride in the low 24 bits of a flag word.
const MAX_LIMIT: usize = (1 << 24) - 64;
/// The capture contract promises at least this much (8 decode rows x 5 x
/// 4096 BF16 on the k5 route is 320 KiB).
const MIN_MAX: usize = 512 << 10;
const MAX_RAILS: usize = 8;
const THREADS: usize = 256;
/// Kernel state in device memory (see `rdma_oneshot.cu`).
const STATE_BYTES: usize = 32;
const MAX_BLOCKS: usize = 16;

/// Control page after `S` and the two receive slots, one 64-byte line per
/// word: `stage` (u32, GPU -> proxy), `poison` (kernel -> host), one flag per
/// rail (peer NIC -> kernel) and one flag source word per rail (proxy -> NIC).
const STAGE: usize = 0;
const POISON: usize = 64;
const FLAGS: usize = 128;
const FLAG_SRC: usize = FLAGS + 64 * MAX_RAILS;
const CTRL_PAGE: usize = 4096;
/// Poison reasons (high bits) over the op's sequence number; the kernel
/// writes the top three, the host the others.
const POISON_LAUNCH: u64 = 1 << 59;
const POISON_PROXY: u64 = 1 << 60;
const POISON_DESYNC: u64 = 1 << 61;
const POISON_TIMEOUT: u64 = 1 << 62;
const POISON_MISMATCH: u64 = 1 << 63;

/// One-shot settings; read from the shared profile, and checked equal on both
/// ranks at bootstrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Config {
    /// Largest one-shot payload in bytes (`ATLAS_RDMA_ONESHOT_MAX`, 1 MiB,
    /// at least 512 KiB).
    pub(super) max: usize,
    /// Payloads from this size stripe over every rail
    /// (`ATLAS_RDMA_ONESHOT_STRIPE_MIN`, default the legacy split size).
    pub(super) stripe_min: usize,
    /// Kernel wait limit (`ATLAS_RDMA_ONESHOT_TIMEOUT_MS`, 30 s; 0 = none).
    pub(super) timeout_ns: u64,
    /// Publish `stage` with a fenced stream memop rather than from the
    /// kernel (`ATLAS_RDMA_ONESHOT_STAGE_FENCE=1`); local only.
    pub(super) stage_fence: bool,
}

impl Config {
    /// `None` unless `ATLAS_RDMA_ONESHOT=1`.
    pub(super) fn from_env() -> Option<Self> {
        Self::parse(|key| std::env::var(key).ok())
    }

    fn parse(var: impl Fn(&str) -> Option<String>) -> Option<Self> {
        if var("ATLAS_RDMA_ONESHOT").as_deref() != Some("1") {
            return None;
        }
        let num = |key: &str, default: usize| {
            var(key)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(default)
        };
        Some(Self {
            max: num("ATLAS_RDMA_ONESHOT_MAX", 1 << 20)
                .clamp(MIN_MAX, MAX_LIMIT)
                .next_multiple_of(64),
            stripe_min: num("ATLAS_RDMA_ONESHOT_STRIPE_MIN", SPLIT_MIN),
            timeout_ns: (num("ATLAS_RDMA_ONESHOT_TIMEOUT_MS", 30_000) as u64)
                .saturating_mul(1_000_000),
            stage_fence: var("ATLAS_RDMA_ONESHOT_STAGE_FENCE").as_deref() == Some("1"),
        })
    }

    /// Whether a payload takes the one-shot path. Rank-invariant by
    /// construction: it looks at nothing but the size (never at addresses).
    pub(super) fn eligible(&self, bytes: usize) -> bool {
        bytes > 0 && bytes <= self.max && bytes.is_multiple_of(2)
    }

    /// Bootstrap form, `max` and `stripe_min`: both ranks must send the same.
    pub(super) fn wire(cfg: Option<Self>) -> [u64; 2] {
        cfg.map_or([0; 2], |c| [c.max as u64, c.stripe_min as u64])
    }
}

/// Bytes of pinned region the one-shot channel appends: `S`, two receive
/// slots and the control page.
pub(super) fn region_bytes(max: usize) -> usize {
    3 * max + CTRL_PAGE
}
fn recv_off(max: usize, slot: usize) -> usize {
    (1 + slot) * max
}
fn ctrl_off(max: usize) -> usize {
    3 * max
}

/// `(offset, len)` per rail: one rail below `stripe_min`, else split over all
/// `rails` in 64-byte-aligned parts. Identical on both ranks.
pub(super) fn stripes(bytes: usize, rails: usize, stripe_min: usize) -> Vec<(usize, usize)> {
    let used = if bytes >= stripe_min { rails } else { 1 };
    let part = bytes.div_ceil(used).next_multiple_of(64);
    (0..used)
        .map(|r| r * part)
        .take_while(|&off| off < bytes)
        .map(|off| (off, part.min(bytes - off)))
        .collect()
}

/// The flag the proxy writes for op `seq` of `bytes` (see the kernel).
fn flag_word(seq: u64, bytes: usize) -> u64 {
    (seq << 24) | bytes as u64
}

fn describe_poison(word: u64) -> String {
    let why = match word & !(POISON_LAUNCH - 1) {
        POISON_MISMATCH => "the peer sent a different size (ranks diverged)",
        POISON_DESYNC => "the peer's flag is more than one op ahead (sequences diverged)",
        POISON_PROXY => "the RDMA proxy failed (see its error)",
        POISON_LAUNCH => "a kernel launch failed after its stage was published",
        POISON_TIMEOUT => "the peer never arrived (ATLAS_RDMA_ONESHOT_TIMEOUT_MS)",
        _ => "unknown fault",
    };
    format!("op {}: {why}", word & (POISON_LAUNCH - 1))
}

/// Why the channel with control page `ctrl` (host address) stopped, if it did.
fn poison_at(ctrl: usize) -> Option<String> {
    // SAFETY: the control page lives in the pinned region for the pair's
    // lifetime; the kernel writes this word only before trapping.
    let word = unsafe { ((ctrl + POISON) as *const u64).read_volatile() };
    (word != 0).then(|| describe_poison(word))
}

/// Stop the channel from the host: every later kernel traps at launch. An
/// earlier reason is kept.
fn poison_host(ctrl: usize, word: u64) {
    // SAFETY: as in `poison_at`.
    let poison = unsafe { &*((ctrl + POISON) as *const AtomicU64) };
    let _ = poison.compare_exchange(0, word, Ordering::AcqRel, Ordering::Relaxed);
}

/// Enqueue side, owned by the backend.
pub(in crate::nccl_backend) struct OneShot {
    cfg: Config,
    rails: usize,
    /// Host and device addresses of the channel's part of the pinned region.
    host: usize,
    dev: u64,
    /// Device-resident kernel state: sequence (u64), finished-block counter
    /// (u32 at 8) and the sequence block 0 saw arrive (u64 at 16).
    state: u64,
    kernel: AtomicU64,
}

impl OneShot {
    pub(super) fn new(cfg: Config, rails: usize, host: usize, dev: u64) -> Result<Self> {
        ensure!(
            (1..=MAX_RAILS).contains(&rails),
            "RDMA one-shot supports 1..={MAX_RAILS} rails, got {rails}"
        );
        let (mut state, zero) = (0u64, [0u8; STATE_BYTES]);
        cu(
            unsafe { cuMemAlloc_v2(&mut state, STATE_BYTES) },
            "cuMemAlloc(one-shot state)",
        )?;
        cu(
            unsafe { cuMemcpyHtoD_v2(state, zero.as_ptr().cast(), STATE_BYTES) },
            "cuMemcpyHtoD(one-shot state)",
        )?;
        Ok(Self {
            cfg,
            rails,
            host,
            dev,
            state,
            kernel: AtomicU64::new(0),
        })
    }

    /// Provide `rdma_oneshot_bf16`; the channel is unavailable until then.
    pub(in crate::nccl_backend) fn set_kernel(&self, handle: u64) {
        self.kernel.store(handle, Ordering::Relaxed);
    }

    /// Largest payload [`Self::enqueue`] accepts (0 while unavailable).
    pub(in crate::nccl_backend) fn max_bytes(&self) -> usize {
        if self.kernel.load(Ordering::Relaxed) == 0 {
            0
        } else {
            self.cfg.max
        }
    }

    /// Why the channel stopped (the kernel trapped or will), if it did.
    pub(in crate::nccl_backend) fn poisoned(&self) -> Option<String> {
        poison_at(self.host + ctrl_off(self.cfg.max))
    }

    /// Send `src[..bytes]` to the peer and land its payload in `dst` (added
    /// in place when `add`, else copied), all on `stream` and capturable.
    /// `false` (nothing enqueued) when the size is ineligible or the kernel
    /// is missing; both ranks see the same answer.
    pub(in crate::nccl_backend) fn enqueue(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        let kernel = self.kernel.load(Ordering::Relaxed);
        if kernel == 0 || !self.cfg.eligible(bytes) {
            return Ok(false);
        }
        let ctrl = self.dev + ctrl_off(self.cfg.max) as u64;
        let stage = ctrl + STAGE as u64;
        cu(
            unsafe { cuStreamWaitValue32_v2(stream, stage, 0, CU_STREAM_WAIT_VALUE_EQ) },
            "cuStreamWaitValue32(one-shot stage)",
        )?;
        cu(
            unsafe { cuMemcpyAsync(self.dev, src, bytes, stream) },
            "cuMemcpyAsync(one-shot stage)",
        )?;
        if self.cfg.stage_fence {
            // Default flags fence the copy before the word lands.
            cu(
                unsafe { cuStreamWriteValue32_v2(stream, stage, bytes as u32, 0) },
                "cuStreamWriteValue32(one-shot stage)",
            )?;
        }
        let blocks = (bytes / 16).div_ceil(THREADS).clamp(1, MAX_BLOCKS) as u32;
        let mut args = (
            dst,
            self.dev + recv_off(self.cfg.max, 0) as u64,
            self.cfg.max as u64,
            bytes as u32,
            add as u32,
            ctrl + FLAGS as u64,
            stripes(bytes, self.rails, self.cfg.stripe_min).len() as u32,
            self.state,
            if self.cfg.stage_fence { 0 } else { stage },
            ctrl + POISON as u64,
            self.cfg.timeout_ns,
        );
        let mut params: [*mut c_void; 11] = [
            (&raw mut args.0).cast(),
            (&raw mut args.1).cast(),
            (&raw mut args.2).cast(),
            (&raw mut args.3).cast(),
            (&raw mut args.4).cast(),
            (&raw mut args.5).cast(),
            (&raw mut args.6).cast(),
            (&raw mut args.7).cast(),
            (&raw mut args.8).cast(),
            (&raw mut args.9).cast(),
            (&raw mut args.10).cast(),
        ];
        let status = unsafe {
            cuLaunchKernel(
                kernel,
                blocks,
                1,
                1,
                THREADS as u32,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != 0 && self.cfg.stage_fence {
            // `stage` is already on the stream: the proxy will serve an op no
            // kernel consumes, and its count would stay ahead of the device
            // sequence (stale receive slots). Stop the channel instead.
            poison_host(self.host + ctrl_off(self.cfg.max), POISON_LAUNCH);
        }
        cu(status, "cuLaunchKernel(rdma_oneshot_bf16)")?;
        Ok(true)
    }
}

impl Drop for OneShot {
    fn drop(&mut self) {
        unsafe { cuMemFree_v2(self.state) };
    }
}

/// Proxy side: sends each staged op.
pub(super) struct Channel {
    cfg: Config,
    host: usize,
    /// The peer's address of its one-shot region.
    peer: u64,
    served: u64,
}

impl Channel {
    pub(super) fn new(cfg: Config, host: usize, peer: u64) -> Self {
        Self {
            cfg,
            host,
            peer,
            served: 0,
        }
    }

    /// The proxy is going away and nothing will clear `stage` again: poison
    /// the channel, then release a stream blocked in the stage wait so it
    /// reaches a kernel, which traps on the poison.
    pub(super) fn fail(&self) {
        let ctrl = self.host + ctrl_off(self.cfg.max);
        poison_host(ctrl, POISON_PROXY | (self.served + 1));
        // SAFETY: as in `serve`.
        unsafe { &*((ctrl + STAGE) as *const AtomicU32) }.store(0, Ordering::Release);
    }

    /// Send the staged op, if any: data then flag per rail, chained in one
    /// post when `chain` (same-QP WRITEs land in order without relaxed
    /// ordering), else each flag after the data completions. `false` when
    /// nothing is staged.
    pub(super) fn serve(
        &mut self,
        rails: &mut [Verbs],
        lkeys: &[u32],
        rkeys: &[u32],
        chain: bool,
    ) -> Result<bool> {
        let ctrl = self.host + ctrl_off(self.cfg.max);
        // SAFETY: the control page lives in the pinned region for the pair's
        // lifetime; `stage` is written by the GPU (stream memop) and here.
        let stage = unsafe { &*((ctrl + STAGE) as *const AtomicU32) };
        let bytes = stage.load(Ordering::Acquire) as usize;
        if bytes == 0 {
            return Ok(false);
        }
        // Only our own stream writes `stage`, with an eligible size; refuse
        // anything else rather than WRITE past the peer's receive slot.
        ensure!(
            self.cfg.eligible(bytes),
            "RDMA one-shot: staged size {bytes} is not eligible"
        );
        let seq = self.served + 1;
        let parts = stripes(bytes, rails.len(), self.cfg.stripe_min);
        let recv = self.peer + recv_off(self.cfg.max, (seq & 1) as usize) as u64;
        let flag = |r: usize| {
            (
                ctrl + FLAG_SRC + 64 * r,
                self.peer + (ctrl_off(self.cfg.max) + FLAGS + 64 * r) as u64,
            )
        };
        for (r, &(off, len)) in parts.iter().enumerate() {
            let (src_flag, dst_flag) = flag(r);
            // SAFETY: our flag source word; its previous WRITE was reaped
            // before the previous serve returned.
            unsafe { (src_flag as *mut u64).write_volatile(flag_word(seq, bytes)) };
            let data = (self.host + off) as *mut c_void;
            let (dst, len) = (recv + off as u64, u32::try_from(len)?);
            // SAFETY: `S` and the flag word lie in the registered region and
            // stay unmodified until these WRITEs are reaped below (the GPU
            // only restages after `stage` is cleared).
            unsafe {
                if chain {
                    rails[r].post_write_flag(
                        data,
                        lkeys[r],
                        dst,
                        rkeys[r],
                        len,
                        src_flag as *mut c_void,
                        dst_flag,
                        seq,
                    )
                } else {
                    rails[r].post_write(data, lkeys[r], dst, rkeys[r], len, seq)
                }
            }?;
        }
        if !chain {
            for rail in rails.iter_mut().take(parts.len()) {
                rail.poll()?;
            }
            for (r, rail) in rails.iter_mut().enumerate().take(parts.len()) {
                let (src_flag, dst_flag) = flag(r);
                // SAFETY: as above.
                unsafe {
                    rail.post_write(
                        src_flag as *mut c_void,
                        lkeys[r],
                        dst_flag,
                        rkeys[r],
                        8,
                        seq,
                    )
                }?;
            }
        }
        for rail in rails.iter_mut().take(parts.len()) {
            rail.poll()?;
        }
        self.served = seq;
        stage.store(0, Ordering::Release);
        Ok(true)
    }
}

#[cfg(test)]
#[path = "oneshot_gpu_tests.rs"]
mod gpu_tests;

#[cfg(test)]
#[path = "oneshot_tests.rs"]
mod tests;
