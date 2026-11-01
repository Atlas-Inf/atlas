// SPDX-License-Identifier: AGPL-3.0-only

//! Two-rank fast-path helpers on [`NcclBackend`]: the BF16 add launch, the
//! RDMA pair connect/dispatch, and the peer lifeline accessor.

#[cfg(atlas_rdma_verbs)]
use super::{ALL_REDUCE_DTYPE_BYTES, ensure_payload_fits, rdma_pair};
use super::{NcclBackend, cuLaunchKernel};
use crate::peer_lifeline::PeerLifeline;
use anyhow::Result;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::Ordering;

impl NcclBackend {
    /// `dst[i] += src[i]` over `count` BF16 values on `stream`.
    pub(super) fn launch_add(&self, dst: u64, src: u64, count: usize, stream: u64) -> Result<()> {
        let kernel = self.add_kernel.load(Ordering::Relaxed);
        if kernel == 0 {
            anyhow::bail!("bf16_add_inplace kernel not set — call set_add_kernel() first");
        }
        let threads: u32 = 256;
        let blocks: u32 = (count as u32).div_ceil(threads);
        let mut p_dst = dst;
        let mut p_src = src;
        let mut p_n = count as i32;
        let mut params: [*mut c_void; 3] = [
            &mut p_dst as *mut u64 as *mut c_void,
            &mut p_src as *mut u64 as *mut c_void,
            &mut p_n as *mut i32 as *mut c_void,
        ];
        let status = unsafe {
            cuLaunchKernel(
                kernel,
                blocks,
                1,
                1,
                threads,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        if status != 0 {
            anyhow::bail!("cuLaunchKernel (bf16_add_inplace) failed: status {status}");
        }
        Ok(())
    }

    /// 2-rank all-reduce over the RDMA pair on `stream` when it is up and the
    /// payload qualifies; `false` means the caller takes the NCCL path.
    pub(super) fn try_rdma_all_reduce(&self, ptr: u64, bytes: usize, stream: u64) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(rdma) = &self.rdma {
            ensure_payload_fits(bytes, self.recv_capacity, self.rank, self.world_size)?;
            if bytes == 0 {
                return Ok(true);
            }
            if rdma.exchange(ptr, ptr, bytes, stream, |dst, src, len| {
                self.launch_add(dst, src, len / ALL_REDUCE_DTYPE_BYTES, stream)
            })? {
                return Ok(true);
            }
        }
        let _ = (ptr, bytes, stream);
        Ok(false)
    }

    /// Reduce-scatter/all-gather step over the RDMA pair (see
    /// `CommBackend::exchange_async`); `false` when the pair is not up.
    pub(super) fn try_rdma_exchange(
        &self,
        send: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(rdma) = &self.rdma
            && bytes > 0
            && (!add || self.add_kernel.load(Ordering::Relaxed) != 0)
        {
            return rdma.exchange(send, dst, bytes, stream, |dst, src, len| {
                if add {
                    self.launch_add(dst, src, len / ALL_REDUCE_DTYPE_BYTES, stream)
                } else {
                    rdma_pair::copy_async(dst, src, len, stream)
                }
            });
        }
        let _ = (send, dst, bytes, add, stream);
        Ok(false)
    }

    /// One-shot exchange over the RDMA pair (`rdma_pair::oneshot`): on
    /// `stream`, graph-capturable, sharing one device sequence between eager
    /// calls and replays. `false` (nothing enqueued) when `ATLAS_RDMA_ONESHOT`
    /// is off, the kernel is missing or the size is ineligible.
    pub(super) fn try_oneshot(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(oneshot) = self.rdma.as_ref().and_then(|r| r.oneshot()) {
            return oneshot.enqueue(src, dst, bytes, add, stream);
        }
        let _ = (src, dst, bytes, add, stream);
        Ok(false)
    }

    /// Largest one-shot payload (0 when the channel is unavailable).
    pub(super) fn oneshot_max_bytes(&self) -> usize {
        #[cfg(atlas_rdma_verbs)]
        if let Some(oneshot) = self.rdma.as_ref().and_then(|r| r.oneshot()) {
            return oneshot.max_bytes();
        }
        0
    }

    /// Provide `rdma_oneshot_bf16` to the one-shot channel, if it is up.
    pub(super) fn set_oneshot_kernel_handle(&self, handle: u64) {
        #[cfg(atlas_rdma_verbs)]
        if let Some(oneshot) = self.rdma.as_ref().and_then(|r| r.oneshot()) {
            oneshot.set_kernel(handle);
            tracing::info!("RDMA one-shot kernel set (graph-capturable collectives enabled)");
        }
        let _ = handle;
    }

    /// Why the one-shot channel stopped (the peer never arrived, the ranks
    /// diverged, or the host could no longer serve it), if it did.
    pub fn oneshot_poisoned(&self) -> Option<String> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(oneshot) = self.rdma.as_ref().and_then(|r| r.oneshot()) {
            return oneshot.poisoned();
        }
        None
    }

    /// The RDMA pair's host command ring, when `ATLAS_GLM_CMD_RDMA=1` (see
    /// `rdma_pair::cmd_ring`): the words a message may carry, else 0.
    pub(super) fn cmd_ring_max_words(&self) -> usize {
        #[cfg(atlas_rdma_verbs)]
        if self.rdma.as_ref().and_then(|r| r.cmd()).is_some() {
            return rdma_pair::cmd_ring::MAX_WORDS;
        }
        0
    }

    /// Send `words` to the peer over the command ring.
    pub(super) fn cmd_ring_send(&self, words: &[u32]) -> Result<()> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(ring) = self.rdma.as_ref().and_then(|r| r.cmd()) {
            return ring.send(words);
        }
        anyhow::bail!(
            "no command ring for {} command words (ATLAS_GLM_CMD_RDMA)",
            words.len()
        )
    }

    /// Receive the peer's next `words.len()` words from the command ring.
    pub(super) fn cmd_ring_recv(&self, words: &mut [u32]) -> Result<()> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(ring) = self.rdma.as_ref().and_then(|r| r.cmd()) {
            return ring.recv(words);
        }
        anyhow::bail!(
            "no command ring for {} command words (ATLAS_GLM_CMD_RDMA)",
            words.len()
        )
    }

    /// Connections to this rank's peers from the initial bootstrap; watch them
    /// to learn when a peer process exits.
    pub fn peer_lifeline(&self) -> &PeerLifeline {
        &self.peer_lifeline
    }

    /// RDMA pair payload capacity, when the pair is up.
    pub(super) fn rdma_capacity(&self) -> Option<usize> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(rdma) = &self.rdma {
            return Some(rdma.capacity());
        }
        None
    }

    /// The RDMA pair for a 2-rank world when `ATLAS_RDMA_ALLREDUCE=1`. Its
    /// bootstrap stays on the link of the NCCL bootstrap connection
    /// (`peers`; see `bootstrap::Link::of`).
    #[cfg(atlas_rdma_verbs)]
    pub(super) fn connect_rdma(
        rank: usize,
        world_size: usize,
        peers: &[std::net::TcpStream],
        master_addr: &str,
        master_port: u16,
        recv_capacity: usize,
    ) -> Result<Option<rdma_pair::RdmaPair>> {
        // Port +1 is the reconnect bootstrap; +2 carries the RDMA identities.
        let rdma = if world_size == 2 && rdma_pair::RdmaPair::requested() {
            let port = master_port.wrapping_add(2);
            let link = rdma_pair::bootstrap::Link::of(rank, &peers[0], port, master_addr)?;
            Some(rdma_pair::RdmaPair::connect(
                rank,
                &link,
                recv_capacity.next_multiple_of(64),
            )?)
        } else {
            if rdma_pair::cmd_ring::requested() {
                tracing::warn!(
                    "ATLAS_GLM_CMD_RDMA=1 needs the RDMA pair (two ranks, ATLAS_RDMA_ALLREDUCE=1): \
                     command words stay on NCCL broadcasts"
                );
            }
            None
        };
        Ok(rdma)
    }
}
