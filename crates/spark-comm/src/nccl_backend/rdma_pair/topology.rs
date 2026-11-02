// SPDX-License-Identifier: AGPL-3.0-only

//! Physical switchless topology. A ring changes the path of copied payloads,
//! never the reduction kernel or its arithmetic order. The specialized
//! four-rank reduce-scatter/all-to-all retain their linear schedules.

use anyhow::{Result, bail, ensure};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::nccl_backend) enum Topology {
    #[default]
    Chain,
    Ring,
}

impl Topology {
    pub(in crate::nccl_backend) fn parse(value: Option<&str>, ranks: usize) -> Result<Self> {
        let topology = match value {
            None | Some("chain") => Self::Chain,
            Some("ring") => Self::Ring,
            Some(value) => bail!("ATLAS_RDMA_TOPOLOGY must be chain or ring, got {value:?}"),
        };
        topology.check(ranks)?;
        Ok(topology)
    }

    pub(in crate::nccl_backend) fn check(self, ranks: usize) -> Result<()> {
        ensure!(
            self != Self::Ring || ranks == 4,
            "RDMA ring requires four ranks, got {ranks}"
        );
        Ok(())
    }

    pub(in crate::nccl_backend) fn wire(self) -> u64 {
        match self {
            Self::Chain => 0,
            Self::Ring => 1,
        }
    }

    pub(in crate::nccl_backend) fn from_wire(value: u64) -> Result<Self> {
        match value {
            0 => Ok(Self::Chain),
            1 => Ok(Self::Ring),
            _ => bail!("RDMA topology wire value {value} is unsupported"),
        }
    }

    pub(in crate::nccl_backend) fn name(self) -> &'static str {
        match self {
            Self::Chain => "chain",
            Self::Ring => "ring",
        }
    }

    /// Below/left (0), above/right (1), with ring endpoints wrapping.
    pub(in crate::nccl_backend) fn neighbor(
        self,
        rank: usize,
        ranks: usize,
        side: usize,
    ) -> Option<usize> {
        match (self, side) {
            (Self::Ring, 0) => Some((rank + ranks - 1) % ranks),
            (Self::Ring, _) => Some((rank + 1) % ranks),
            (Self::Chain, 0) => rank.checked_sub(1),
            (Self::Chain, _) => (rank + 1 < ranks).then_some(rank + 1),
        }
    }

    /// A flood sends to both neighbors. In a four-rank ring only the
    /// clockwise neighbor forwards, once, to the opposite rank. Each peer
    /// receives exactly once in at most two hops, with equal load per link.
    /// A chain forwards away from the origin until it reaches an endpoint.
    pub(in crate::nccl_backend) fn flood_forward(
        self,
        rank: usize,
        origin: usize,
        ranks: usize,
    ) -> Option<usize> {
        match self {
            Self::Ring => (rank == (origin + 1) % ranks).then_some(1),
            Self::Chain => {
                let side = usize::from(origin < rank);
                self.neighbor(rank, ranks, side).map(|_| side)
            }
        }
    }
}

#[cfg(test)]
#[path = "topology_tests.rs"]
mod tests;
