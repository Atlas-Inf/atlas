// SPDX-License-Identifier: AGPL-3.0-only
//! The RoCE v2 GID pick for one rail: this rank's own bootstrap-link address
//! wins over the lowest-index rule when the netdev carries several IPv4
//! addresses. `ATLAS_RDMA_GID` still takes precedence in the caller.

use anyhow::Result;
use std::net::Ipv4Addr;

/// Index of `dev`'s RoCE v2 GID to open the QP on. `prefer` is this rank's
/// own IPv4 address on the pair's bootstrap link: a netdev can carry a LAN
/// address and the point-to-point fabric address at once, and the lowest RoCE
/// v2 GID can sit on the wrong subnet for the peer.
pub(super) fn pick(dev: &str, prefer: Option<Ipv4Addr>) -> Result<u32> {
    let index = atlas_rdma::gid::roce_v2_ipv4_index_for(dev, prefer)?;
    if let Some(want) = prefer {
        let gids = atlas_rdma::gid::roce_v2_ipv4_gids(dev).unwrap_or_default();
        let lowest = gids.iter().map(|(i, _)| *i).min();
        if lowest != Some(index) {
            tracing::info!(
                "RDMA pair rail {dev}: RoCE v2 GID {index} preferred on link address \
                 {want} (lowest index {lowest:?} is another subnet)"
            );
        } else if gids.len() >= 2 && gids.iter().all(|(_, addr)| *addr != want) {
            let chosen = gids.iter().find(|(i, _)| *i == index).map(|(_, a)| *a);
            tracing::warn!(
                "RDMA pair rail {dev}: {} RoCE v2 IPv4 GIDs, none on link address {want}; \
                 using index {index} ({chosen:?}) — set ATLAS_RDMA_GID to override",
                gids.len()
            );
        }
    }
    Ok(index)
}
