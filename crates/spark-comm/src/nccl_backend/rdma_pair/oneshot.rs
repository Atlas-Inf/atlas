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
//! `ATLAS_RDMA_ONESHOT_SM_STAGE=1` makes it one stream operation: the kernel
//! waits for `stage` == 0 and stages the payload itself (write-through
//! stores, a system-scope fence, then `stage`), and is launched with
//! programmatic dependent launch (`ATLAS_RDMA_ONESHOT_PDL`, default on), so
//! the producer -> all-reduce -> consumer chain has no copy-engine hops or
//! launch gaps. The landing is the same code, so the result is bit-identical
//! to the copy-engine path whenever the NIC reads what the SMs stored; mode
//! `2` (plain stores) exists only as the control of the stress test for that
//! (`nic_loopback_sm_stage_stress`). Checked equal on both ranks.
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
//! A chain of up to [`MAX_RANKS`] ranks ([`Shape`], `rdma_chain_bf16`) runs
//! the same protocol with a receive slot pair and a flag line per piece
//! ([`pieces`]) for every other rank: each rank stages once and its proxy sends `S` to both
//! neighbours, which forward it on ([`relay`]), so every rank lands the same
//! sum of all payloads. The two-slot argument holds per origin: a rank only
//! stages `seq + 2` after its kernel `seq + 1`, which waited on every rank's
//! flag `seq + 1`, each staged after that rank's kernel `seq` and forwarded
//! by every rank in between after it arrived there.
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

pub(super) use super::bootstrap::{MAX_RAILS, MAX_RANKS};
use super::{SPLIT_MIN, cu, cuMemcpyAsync};
use crate::nccl_backend::{cuLaunchKernel, cuMemAlloc_v2, cuMemFree_v2};
use anyhow::{Context, Result, ensure};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// `CUlaunchAttribute` / `CUlaunchConfig` (driver API), as in atlas-core's
/// `registry/launch.rs`.
#[repr(C)]
struct CuLaunchAttribute {
    id: u32,
    pad: [u8; 4],
    value: [u8; 64],
}

#[repr(C)]
struct CuLaunchConfig {
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem_bytes: u32,
    stream: u64,
    attrs: *mut CuLaunchAttribute,
    num_attrs: u32,
}

const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 6;

unsafe extern "C" {
    fn cuLaunchKernelEx(
        config: *const CuLaunchConfig,
        f: u64,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
    fn cuStreamWaitValue32_v2(stream: u64, addr: u64, value: u32, flags: u32) -> i32;
    fn cuStreamWriteValue32_v2(stream: u64, addr: u64, value: u32, flags: u32) -> i32;
}

const CU_STREAM_WAIT_VALUE_EQ: u32 = 0x1;
const CU_STREAM_WAIT_VALUE_GEQ: u32 = 0x0;
/// Payload sizes ride in the low 24 bits of a flag word (on a chain, below
/// [`RS_BIT`]: [`Config::for_chain`]).
const MAX_LIMIT: usize = (1 << 24) - 64;
/// The capture contract promises at least this much (8 decode rows x 5 x
/// 4096 BF16 on the k5 route is 320 KiB).
const MIN_MAX: usize = 512 << 10;
/// A chain payload of this many bytes per piece is cut finer ([`pieces`]).
const PIECE: usize = 128 << 10;
const THREADS: usize = 256;
/// Kernel state in device memory (see `rdma_oneshot.cu`): seq, finished
/// blocks, go, staged blocks, then the reduce-scatter's per-piece writer
/// counts.
const STATE_BYTES: usize = 1024;
/// The fused reduce-scatter's producer counters (u32 a piece) and published
/// pieces (u64 a piece) in the state (`ONESHOT_STATE_PRODUCED`, `_OWN`).
const STATE_PRODUCED: usize = 256;
const STATE_OWN: usize = 512;
const _: () = assert!(
    STATE_PRODUCED + 4 * MAX_PIECES <= STATE_OWN && STATE_OWN + 8 * MAX_PIECES <= STATE_BYTES
);
const MAX_BLOCKS: usize = 16;

/// The reduce-scatter's and the all-to-all's grid (the four-rank chain's
/// only; local, as the split of an op's pieces over blocks changes no
/// value). Their kernels stage and land every piece through the pinned
/// buffers: at 16 blocks the SP seam's 64 MB reduce-scatter took 4009 us
/// (FP8 3953), at 48 3857 us (FP8 3495; four nodes, 2026-10-18). The
/// four-rank profile (`4x1m-tp4`) serves at 24, which leaves SMs to the
/// unpermute producing the next seam block beside it. The NIC loopback
/// tests run 16: their four ranks (and two channels) share one GPU, and
/// every block of an op waits on blocks of the others, so all must be
/// resident at once (three blocks an SM, `rdma_chain_bf16`).
const RS_BLOCKS: usize = if cfg!(test) { 16 } else { 24 };

/// Pieces a chain op may cut its payload into, one flag line each per
/// origin (`ONESHOT_ORIGIN_LINES` in the kernel); a pair's rails take the
/// first [`MAX_RAILS`] lines.
pub(super) const MAX_PIECES: usize = 32;

/// Control page after `S` and the receive slots, one 64-byte line per word:
/// `stage` (u32, GPU -> proxy), `poison` (kernel -> host), per origin one
/// flag per piece (neighbour NIC -> kernel), one flag source word per piece
/// for our own sends and, on a chain, per origin one per piece for the sends
/// that forward it (proxy -> NIC).
const STAGE: usize = 0;
const POISON: usize = 64;
const FLAGS: usize = 128;
const ORIGIN_LINES: usize = 64 * MAX_PIECES;
const FLAG_SRC: usize = FLAGS + ORIGIN_LINES * (MAX_RANKS - 1);
const FWD_SRC: usize = FLAG_SRC + 64 * MAX_PIECES;
/// The pipelined ops' ready words, one u64 per piece (kernel -> proxy: the
/// op whose staged or computed piece is in `S`); `ONESHOT_READY` in
/// `rdma_oneshot.cu`.
const READY: usize = FWD_SRC + ORIGIN_LINES * (MAX_RANKS - 1);
const CTRL_PAGE: usize = 16384;
/// A pair's control page: one origin of [`MAX_RAILS`] flag lines, then their
/// source words.
const PAIR_FLAG_SRC: usize = FLAGS + 64 * MAX_RAILS;
const PAIR_CTRL_PAGE: usize = 4096;
const _: () = assert!(READY == 14464 && READY + 8 * MAX_PIECES <= CTRL_PAGE);
/// Poison reasons (high bits) over the op's sequence number; the kernel
/// writes the top three, the host the others.
const POISON_LAUNCH: u64 = 1 << 59;
const POISON_PROXY: u64 = 1 << 60;
const POISON_DESYNC: u64 = 1 << 61;
const POISON_TIMEOUT: u64 = 1 << 62;
const POISON_MISMATCH: u64 = 1 << 63;

/// Where a rank sits: one of two (a pair), or on a chain where every
/// payload reaches every rank through the ranks in between.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    pub(super) ranks: usize,
    pub(super) rank: usize,
    /// Rails of the link to the rank below and to the one above; 0 where
    /// there is none. A pair's one link is both.
    pub(super) rails: [usize; 2],
}

impl Shape {
    pub(super) fn pair(rank: usize, rails: usize) -> Self {
        Self {
            ranks: 2,
            rank,
            rails: [rails; 2],
        }
    }

    /// Receive slot pairs: one per other rank.
    pub(super) fn origins(&self) -> usize {
        self.ranks - 1
    }

    pub(super) fn check(&self) -> Result<()> {
        ensure!(
            (2..=MAX_RANKS).contains(&self.ranks) && self.rank < self.ranks,
            "RDMA one-shot: rank {} of {} (supported: 2..={MAX_RANKS} ranks)",
            self.rank,
            self.ranks
        );
        let used = [self.rank > 0, self.rank + 1 < self.ranks];
        for (side, (&rails, used)) in ["below", "above"].iter().zip(self.rails.iter().zip(used)) {
            ensure!(
                !used || (1..=MAX_RAILS).contains(&rails),
                "RDMA one-shot supports 1..={MAX_RAILS} rails, got {rails} to the rank {side}"
            );
        }
        Ok(())
    }
}

/// Origin slot of rank `origin`'s payload at rank `at`: ranks in order,
/// skipping `at`. A payload keeps its slot index along its way: it moves
/// away from its origin, so every rank it passes is on the same side of it.
pub(super) fn origin_slot(origin: usize, at: usize) -> usize {
    origin - usize::from(origin > at)
}

/// Bytes of pinned region the one-shot channel appends: `S`, two receive
/// slots per origin and the control page.
pub(super) fn region_bytes(max: usize, origins: usize) -> usize {
    let page = if origins == 1 {
        PAIR_CTRL_PAGE
    } else {
        CTRL_PAGE
    };
    ctrl_off(max, origins) + page
}
fn recv_off(max: usize, origin: usize, slot: usize) -> usize {
    (1 + 2 * origin + slot) * max
}
fn ctrl_off(max: usize, origins: usize) -> usize {
    (1 + 2 * origins) * max
}
/// The flag line of `origin`'s rail `rail` in the control page.
fn flag_line(origin: usize, rail: usize) -> usize {
    FLAGS + ORIGIN_LINES * origin + 64 * rail
}

/// Multi-rank completion words match the payload's two parity slots. Ring
/// flood and specialized collectives can deliver one origin over different
/// QPs, so seq+1 may arrive before seq: it must not publish seq's completion
/// or overwrite its flag. The pair retains its original single-word ABI.
fn chain_flag_line(origin: usize, piece: usize, seq: u64) -> usize {
    flag_line(origin, piece) + 8 * (seq as usize & 1)
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

/// A chain's `(offset, len)` pieces: one below `stripe_min`, else two (one
/// per rail of a two-rail link), and payloads of several [`PIECE`]s one per
/// `PIECE`, at most one per flag line; 64-byte aligned. Piece `p` goes on rail
/// `p % rails` of every link, so the layout does not depend on the rails and
/// a rank forwards each piece as soon as it lands: the hops of a large
/// payload overlap.
pub(super) fn pieces(bytes: usize, stripe_min: usize) -> Vec<(usize, usize)> {
    let n = if bytes >= stripe_min {
        (bytes / PIECE).clamp(2, MAX_PIECES)
    } else {
        1
    };
    stripes(bytes, n, 0)
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

/// A proxy that is going away, with `served` ops sent, leaves nothing to
/// clear `stage` again: poison the channel with control page `ctrl`, then
/// release a stream blocked in the stage wait so it reaches a kernel, which
/// traps on the poison.
fn fail_at(ctrl: usize, served: u64) {
    poison_host(ctrl, POISON_PROXY | (served + 1));
    // SAFETY: the control page lives in the pinned region for the channel's
    // lifetime; `stage` is written by the GPU (stream memop) and the proxy.
    unsafe { &*((ctrl + STAGE) as *const AtomicU32) }.store(0, Ordering::Release);
}

/// Enqueue side, owned by the backend.
pub(in crate::nccl_backend) struct OneShot {
    cfg: Config,
    shape: Shape,
    /// Host and device addresses of the channel's part of the pinned region.
    host: usize,
    dev: u64,
    /// Device-resident kernel state: sequence (u64), finished-block counter
    /// (u32 at 8), the sequence block 0 saw arrive (u64 at 16) and the
    /// staged-block counter (u32 at 24).
    state: u64,
    kernel: AtomicU64,
    /// Channel 0 of a four-rank chain (`config::RS_FUSED_SEG`): the fused
    /// region and its completion words.
    fused: Option<FusedRegion>,
    /// Fused ops begun (each op's `epoch`), and whether the channel has
    /// run no op since the last that every rank hears from its neighbours
    /// (`fused` module doc).
    epoch: AtomicU32,
    fused_last: AtomicBool,
}

/// This rank's fused region (`fused` module): host and device addresses,
/// the segment capacity, and the u32 completion word of each piece of its
/// own segment (device memory).
#[derive(Clone, Copy, Debug)]
pub(super) struct FusedRegion {
    pub(super) dev: u64,
    pub(super) cap: usize,
    pub(super) done: u64,
}

impl OneShot {
    pub(super) fn new(cfg: Config, shape: Shape, host: usize, dev: u64) -> Result<Self> {
        shape.check()?;
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
            shape,
            host,
            dev,
            state,
            kernel: AtomicU64::new(0),
            fused: None,
            epoch: AtomicU32::new(0),
            fused_last: AtomicBool::new(false),
        })
    }

    /// Give channel 0 of a four-rank chain the fused region at device
    /// address `dev` (`fused::entries` slots of `cap` bytes).
    pub(super) fn with_fused(mut self, dev: u64, cap: usize) -> Result<Self> {
        let mut done = 0u64;
        let bytes = 4 * MAX_PIECES;
        cu(
            unsafe { cuMemAlloc_v2(&mut done, bytes) },
            "cuMemAlloc(fused done)",
        )?;
        let zero = [0u8; 4 * MAX_PIECES];
        cu(
            unsafe { cuMemcpyHtoD_v2(done, zero.as_ptr().cast(), bytes) },
            "cuMemcpyHtoD(fused done)",
        )?;
        self.fused = Some(FusedRegion { dev, cap, done });
        Ok(self)
    }

    /// Provide the kernel (`oneshot_kernel_name`); the channel is
    /// unavailable until then.
    pub(in crate::nccl_backend) fn set_kernel(&self, handle: u64) {
        self.kernel.store(handle, Ordering::Relaxed);
    }

    /// Ranks of the channel.
    pub(in crate::nccl_backend) fn ranks(&self) -> usize {
        self.shape.ranks
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
        poison_at(self.ctrl())
    }

    /// Host address of the control page.
    fn ctrl(&self) -> usize {
        self.host + ctrl_off(self.cfg.max, self.shape.origins())
    }
}

impl Drop for OneShot {
    fn drop(&mut self) {
        unsafe { cuMemFree_v2(self.state) };
        if let Some(f) = self.fused {
            unsafe { cuMemFree_v2(f.done) };
        }
    }
}

mod channel;
mod collectives;
mod config;
pub(super) mod fused;
#[path = "oneshot_launch.rs"]
mod launch;
mod ops;
pub(super) mod relay;
pub(super) use channel::Channel;
pub(super) use config::Config;
#[cfg(test)]
pub(super) use config::RS_FUSED_SEG;
pub(super) use ops::RS_BIT;
use ops::*;
pub(in crate::nccl_backend) use ops::{A2a, A2aPlane};

#[cfg(test)]
#[path = "oneshot_gpu_tests.rs"]
mod gpu_tests;

#[cfg(test)]
#[path = "oneshot_tests.rs"]
mod tests;
