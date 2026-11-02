// SPDX-License-Identifier: AGPL-3.0-only

//! The RDMA chain's side of [`NcclBackend`]: connecting it and reaching its
//! collectives (`rdma_pair::chain`). Off the chain everything here is `false`
//! or empty.

use super::NcclBackend;
#[cfg(atlas_rdma_verbs)]
use super::rdma_pair;
use anyhow::Result;

impl NcclBackend {
    /// One capturable all-reduce op over the chain; `false` off the chain or
    /// when the size is ineligible.
    pub(super) fn try_chain_sum(&self, ptr: u64, bytes: usize, stream: u64) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.oneshot().sum(ptr, ptr, bytes, true, stream);
        }
        let _ = (ptr, bytes, stream);
        Ok(false)
    }

    /// A chain all-reduce of any size on `stream`: one one-shot op when it
    /// fits, else consecutive ones of at most the maximum, or a reduce-scatter
    /// and an all-gather. `false` off the chain or while its kernel is missing.
    pub(super) fn try_chain_all_reduce(&self, ptr: u64, bytes: usize, stream: u64) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            let one = chain.oneshot().sum(ptr, ptr, bytes, true, stream)?;
            return Ok(one || chain.all_reduce(ptr, bytes, stream)?);
        }
        let _ = (ptr, bytes, stream);
        Ok(false)
    }

    /// A chain all-gather of every rank's part (`CommBackend::gather_async`)
    /// on `stream`; `false` off the chain or while its kernel is missing.
    pub(super) fn try_chain_gather(
        &self,
        base: u64,
        stride: i64,
        bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.gather(base, stride, bytes, stream);
        }
        let _ = (base, stride, bytes, stream);
        Ok(false)
    }

    /// A chain broadcast of any size (`CommBackend::broadcast_async`) on
    /// `stream`; `false` off the chain or while its kernel is missing.
    pub(super) fn try_chain_broadcast_parts(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.broadcast(ptr, bytes, root, stream);
        }
        let _ = (ptr, bytes, root, stream);
        Ok(false)
    }

    /// A four-rank chain reduce-scatter (`CommBackend::reduce_scatter_async`,
    /// or with `fp8` `reduce_scatter_fp8_async`) on `stream`; `false` off the
    /// chain or when it does not apply.
    pub(super) fn try_chain_reduce_scatter(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
        fp8: bool,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.reduce_scatter(seg0, stride, bytes, fp8, stream);
        }
        let _ = (seg0, stride, bytes, fp8, stream);
        Ok(false)
    }

    /// The chain's fused reduce-scatter (`CommBackend::rs_fused_begin` and
    /// the rest); `None` / `false` off the chain.
    pub(super) fn try_chain_rs_fused_begin(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
        unit: usize,
    ) -> Result<Option<crate::RsFused>> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.rs_fused_begin(seg0, stride, bytes, unit);
        }
        let _ = (seg0, stride, bytes, unit);
        Ok(None)
    }

    pub(super) fn try_chain_reduce_scatter_fused(
        &self,
        fused: &crate::RsFused,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.reduce_scatter_fused(fused, stream);
        }
        let _ = (fused, stream);
        Ok(false)
    }

    pub(super) fn try_chain_rs_fused_wait(
        &self,
        fused: &crate::RsFused,
        piece: usize,
        stream: u64,
    ) -> Result<()> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.rs_fused_wait(fused, piece, stream);
        }
        let _ = (fused, piece, stream);
        anyhow::bail!("fused reduce-scatter: no RDMA chain")
    }

    /// A chain all-to-all (`CommBackend::all_to_all_async`) on `stream`;
    /// `false` off the chain or while its kernel is missing.
    pub(super) fn try_chain_all_to_all(
        &self,
        planes: &[crate::AllToAllPlane],
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.all_to_all(planes, stream);
        }
        let _ = (planes, stream);
        Ok(false)
    }

    /// Whether the chain's reduce-scatter can run (its kernel is set and a
    /// slice fits the one-shot maximum).
    pub(super) fn chain_rs_ready(&self) -> bool {
        #[cfg(atlas_rdma_verbs)]
        if self.chain.is_some() {
            return rdma_pair::chain::RdmaChain::rs_slice(self.chain_oneshot_max_bytes()) > 0;
        }
        false
    }

    /// The chain's one-shot maximum (0 off the chain or while its kernel is
    /// missing).
    pub(super) fn chain_oneshot_max_bytes(&self) -> usize {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.oneshot().max_bytes();
        }
        0
    }

    /// A chain broadcast of rank `root`'s `ptr[..bytes]` on `stream` over the
    /// one-shot channel; `false` off the chain or when ineligible.
    pub(super) fn try_chain_broadcast(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        stream: u64,
    ) -> Result<bool> {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.oneshot().broadcast(ptr, bytes, root, stream);
        }
        let _ = (ptr, bytes, root, stream);
        Ok(false)
    }

    /// One-shot channels the RDMA chain opened (1 off a chain).
    pub(super) fn oneshot_channels(&self) -> usize {
        #[cfg(atlas_rdma_verbs)]
        if let Some(chain) = &self.chain {
            return chain.channels().len();
        }
        1
    }

    /// The RDMA chain for three or four ranks when `ATLAS_RDMA_ALLREDUCE=1`,
    /// bootstrapping through rank 0 on the NCCL bootstrap connections
    /// (`peers`) at the pair's port.
    #[cfg(atlas_rdma_verbs)]
    pub(super) fn connect_chain(
        rank: usize,
        world_size: usize,
        peers: &[std::net::TcpStream],
        master_port: u16,
    ) -> Result<Option<rdma_pair::chain::RdmaChain>> {
        if world_size <= 2 || !rdma_pair::RdmaPair::requested() {
            return Ok(None);
        }
        let socket = std::env::var("NCCL_NET").is_ok_and(|v| v.eq_ignore_ascii_case("socket"))
            || std::env::var("NCCL_IB_DISABLE").as_deref() == Ok("1");
        if !socket {
            tracing::warn!(
                "RDMA chain: NCCL still carries startup and rare collectives between ranks that \
                 share no RDMA link; set NCCL_IB_DISABLE=1 (or NCCL_NET=Socket) and \
                 NCCL_SOCKET_IFNAME to the routed IP interface on every rank"
            );
        }
        let at =
            rdma_pair::bootstrap::chain::Rendezvous::of(rank, peers, master_port.wrapping_add(2))?;
        rdma_pair::chain::RdmaChain::connect(rank, world_size, &at).map(Some)
    }
}
