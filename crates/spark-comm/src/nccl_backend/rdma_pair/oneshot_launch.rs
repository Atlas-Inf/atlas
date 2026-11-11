// SPDX-License-Identifier: AGPL-3.0-only

//! Stream ordering and CUDA launch for one one-shot exchange: the pair's
//! `rdma_oneshot_bf16` ([`OneShot::enqueue`]) and the chain's `rdma_chain_bf16`
//! ([`OneShot::launch`]). The two share their staging and their launch call;
//! the chain's choices stay off the pair's path.

use super::*;

/// What an op lands on each rank.
#[derive(Clone, Copy)]
pub(super) enum Land {
    /// The sum of every rank's payload.
    Sum,
    /// Rank `root`'s payload (a chain broadcast).
    Root(usize),
    /// Each other rank's payload `stride` bytes per rank from rank 0's (a
    /// chain gather).
    Gather(i64),
    /// The four-rank chain's reduce-scatter of segments `stride` bytes
    /// apart (`oneshot_rs4`).
    ReduceScatter(i64),
    /// The four-rank chain's all-to-all (`oneshot_a2a4`).
    AllToAll(A2a),
    /// The fused reduce-scatter (`oneshot_rs4_fused`) of segments `stride`
    /// bytes apart, the op of `epoch`.
    Fused(i64, u32),
}

/// The fused reduce-scatter's kernel argument (`OneshotRsFused`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FusedArg {
    done: u64,
    epoch: u32,
    pad: u32,
}

impl OneShot {
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
        let ctrl = self.dev + ctrl_off(self.cfg.max, 1) as u64;
        let stage = ctrl + STAGE as u64;
        let sm = self.cfg.sm_stage;
        let fence = self.stage_on_stream(stage, src, bytes, sm, stream)?;
        let blocks = (bytes / 16).div_ceil(THREADS).clamp(1, MAX_BLOCKS) as u32;
        let mut args = (
            dst,
            self.dev + recv_off(self.cfg.max, 0, 0) as u64,
            self.cfg.max as u64,
            bytes as u32,
            add as u32,
            ctrl + FLAGS as u64,
            stripes(bytes, self.shape.rails[1], self.cfg.stripe_min).len() as u32,
            self.state,
            if fence { 0 } else { stage },
            ctrl + POISON as u64,
            self.cfg.timeout_ns,
            src,
            self.dev,
            sm,
        );
        let mut params: [*mut c_void; 14] = [
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
            (&raw mut args.11).cast(),
            (&raw mut args.12).cast(),
            (&raw mut args.13).cast(),
        ];
        let pdl = sm != 0 && self.cfg.pdl;
        let status = launch_kernel(kernel, blocks, params.as_mut_ptr(), pdl, stream);
        self.launched(status, fence, "cuLaunchKernel(rdma_oneshot_bf16)")
    }

    /// The stream operations before the kernel: under copy-engine staging
    /// (`sm` == 0) wait until the previous send left `S` and copy the payload
    /// in, and with the stage fence publish `stage`. Whether it did.
    #[inline(always)]
    fn stage_on_stream(
        &self,
        stage: u64,
        src: u64,
        bytes: usize,
        sm: u32,
        stream: u64,
    ) -> Result<bool> {
        let fence = self.cfg.stage_fence && sm == 0;
        if sm == 0 {
            cu(
                unsafe { cuStreamWaitValue32_v2(stream, stage, 0, CU_STREAM_WAIT_VALUE_EQ) },
                "cuStreamWaitValue32(one-shot stage)",
            )?;
            cu(
                unsafe { cuMemcpyAsync(self.dev, src, bytes, stream) },
                "cuMemcpyAsync(one-shot stage)",
            )?;
        }
        if fence {
            // Default flags fence the copy before the word lands.
            cu(
                unsafe { cuStreamWriteValue32_v2(stream, stage, bytes as u32, 0) },
                "cuStreamWriteValue32(one-shot stage)",
            )?;
        }
        Ok(fence)
    }

    /// The launch's `status`, stopping the channel when a fenced `stage`
    /// went without its kernel.
    #[inline(always)]
    fn launched(&self, status: i32, fence: bool, what: &str) -> Result<bool> {
        if status != 0 && fence {
            // `stage` is already on the stream: the proxy will serve an op no
            // kernel consumes, and its count would stay ahead of the device
            // sequence (stale receive slots). Stop the channel instead.
            poison_host(self.ctrl(), POISON_LAUNCH);
        }
        cu(status, what)?;
        Ok(true)
    }

    /// One op: stage `src`, then land what `land` says in `dst`, waiting up
    /// to `timeout_ns` (0: no limit).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        land: Land,
        timeout_ns: u64,
        stream: u64,
    ) -> Result<bool> {
        let ranks = self.shape.ranks;
        // The kernel's `root`: < ranks a broadcast, == ranks the sum,
        // ranks + 1 a gather `stride` bytes apart, and so on.
        let (root, stride) = match land {
            Land::Sum => (ranks, 0),
            Land::Root(root) => (root, 0),
            Land::Gather(stride) => (ranks + 1, stride),
            Land::ReduceScatter(stride) => (ranks + 2, stride),
            Land::AllToAll(_) => (ranks + 3, 0),
            Land::Fused(stride, _) => (ranks + 4, stride),
        };
        let mut a2a = match land {
            Land::AllToAll(a2a) => a2a,
            _ => A2a::default(),
        };
        let (mut fused, region) = match (land, self.fused) {
            (Land::Fused(_, epoch), Some(f)) => (
                FusedArg {
                    done: f.done,
                    epoch,
                    pad: 0,
                },
                Some(f),
            ),
            (Land::Fused(..), None) => return Ok(false),
            _ => (FusedArg::default(), None),
        };
        // The reduce-scatter's and the all-to-all's kernels always stage
        // (pipelined, piece by piece), and run every block.
        let rs = matches!(
            land,
            Land::ReduceScatter(_) | Land::AllToAll(_) | Land::Fused(..)
        );
        let kernel = self.kernel.load(Ordering::Relaxed);
        let fits = match region {
            Some(f) => bytes <= MAX_RANKS * f.cap,
            None => self.cfg.eligible(bytes),
        };
        if kernel == 0 || !fits {
            return Ok(false);
        }
        // Whether a fused op may follow (`fused` module doc): not right
        // after one, until an op every rank hears from its neighbours.
        match land {
            Land::Fused(..) => self.fused_last.store(true, Ordering::Relaxed),
            Land::Root(_) => {}
            _ => self.fused_last.store(false, Ordering::Relaxed),
        }
        let ctrl = self.dev + ctrl_off(self.cfg.max, self.shape.origins()) as u64;
        let stage = ctrl + STAGE as u64;
        let sm = if rs {
            self.cfg.sm_stage.max(1)
        } else {
            self.cfg.sm_stage
        };
        let fence = self.stage_on_stream(stage, src, bytes, sm, stream)?;
        let blocks = if rs {
            RS_BLOCKS
        } else {
            (bytes / 16).div_ceil(THREADS).clamp(1, MAX_BLOCKS)
        } as u32;
        // The kernel's pieces: a reduce-scatter's (`rs_pieces`), else the
        // flood's.
        let rails = if rs {
            (MAX_RANKS * rs_per_segment(bytes)) as u32
        } else {
            pieces(bytes, self.cfg.stripe_min).len() as u32
        };
        // A fused op lands in its region, slots `cap` bytes apart.
        let (recv, slot_stride) = match region {
            Some(f) => (f.dev, f.cap as u64),
            None => (
                self.dev + recv_off(self.cfg.max, 0, 0) as u64,
                self.cfg.max as u64,
            ),
        };
        let mut args = (
            dst,
            recv,
            slot_stride,
            bytes as u32,
            add as u32,
            ctrl + FLAGS as u64,
            rails,
            self.state,
            if fence { 0 } else { stage },
            ctrl + POISON as u64,
            timeout_ns,
            src,
            self.dev,
            sm,
            self.shape.ranks as u32,
            self.shape.rank as u32,
            root as u32,
            stride,
        );
        let mut params: [*mut c_void; 20] = [
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
            (&raw mut args.11).cast(),
            (&raw mut args.12).cast(),
            (&raw mut args.13).cast(),
            (&raw mut args.14).cast(),
            (&raw mut args.15).cast(),
            (&raw mut args.16).cast(),
            (&raw mut args.17).cast(),
            (&raw mut a2a).cast(),
            (&raw mut fused).cast(),
        ];
        // A forced kernel stage (the reduce-scatter under a copy-engine
        // configuration) launches as the copy-engine path does: no PDL.
        let pdl = sm != 0 && self.cfg.sm_stage != 0 && self.cfg.pdl;
        let status = launch_kernel(kernel, blocks, params.as_mut_ptr(), pdl, stream);
        self.launched(status, fence, "cuLaunchKernel(rdma_chain_bf16)")
    }
}

/// Launch `kernel` on `stream`, with programmatic dependent launch when
/// `pdl`: the kernel then enters with `atlas_pdl_enter()` (or its serial
/// twin) before it reads anything its predecessor wrote.
#[inline(always)]
fn launch_kernel(
    kernel: u64,
    blocks: u32,
    params: *mut *mut c_void,
    pdl: bool,
    stream: u64,
) -> i32 {
    if pdl {
        let mut value = [0u8; 64];
        value[..4].copy_from_slice(&1i32.to_ne_bytes());
        let mut attr = CuLaunchAttribute {
            id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            value,
        };
        let config = CuLaunchConfig {
            grid: [blocks, 1, 1],
            block: [THREADS as u32, 1, 1],
            shared_mem_bytes: 0,
            stream,
            attrs: &mut attr,
            num_attrs: 1,
        };
        unsafe { cuLaunchKernelEx(&config, kernel, params, std::ptr::null_mut()) }
    } else {
        unsafe {
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
                params,
                std::ptr::null_mut(),
            )
        }
    }
}
