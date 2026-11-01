// SPDX-License-Identifier: AGPL-3.0-only
//
// RoCE GID-index discovery. The kernel builds each port's GID table from the
// netdev's addresses and rebuilds entries whenever an address is re-added (a
// link flap, a directly cabled peer rebooting). A freed slot that live QPs
// still reference is not reused, so the RoCE v2 IPv4 entry can move from its
// usual index 3 to 4 or beyond. Look it up instead of assuming an index,
// as NCCL does when NCCL_IB_GID_INDEX is unset.
//
// A netdev can carry several IPv4 addresses — e.g. a LAN address and the
// point-to-point fabric address — each getting a RoCE v2 GID. The lowest
// index is not necessarily the subnet the peer is on, so callers that know
// their own link address can pass it as `prefer`.

use anyhow::{Context, Result, bail};
use std::net::Ipv4Addr;
use std::path::Path;

const SYSFS_INFINIBAND: &str = "/sys/class/infiniband";

/// Port the shim opens (`rs_create` always uses port 1).
const PORT: u32 = 1;

/// Index of the RoCE v2 GID carrying an IPv4-mapped address on `dev` port 1.
/// With several IPv4 addresses on the netdev, the lowest index wins.
pub fn roce_v2_ipv4_index(dev: &str) -> Result<u32> {
    roce_v2_ipv4_index_for(dev, None)
}

/// [`roce_v2_ipv4_index`] against a sysfs-shaped `root` (tests use fixtures).
pub fn roce_v2_ipv4_index_in(root: &Path, dev: &str) -> Result<u32> {
    roce_v2_ipv4_index_for_in(root, dev, None)
}

/// [`roce_v2_ipv4_index`], but preferring the GID whose IPv4 address equals
/// `prefer`: when one or more RoCE v2 IPv4 GIDs carry it, the lowest such
/// index wins. Otherwise (no preference, or none carries it) the lowest RoCE
/// v2 IPv4 index wins — the no-preference behaviour.
pub fn roce_v2_ipv4_index_for(dev: &str, prefer: Option<Ipv4Addr>) -> Result<u32> {
    roce_v2_ipv4_index_for_in(Path::new(SYSFS_INFINIBAND), dev, prefer)
}

/// [`roce_v2_ipv4_index_for`] against a sysfs-shaped `root` (tests use
/// fixtures).
pub fn roce_v2_ipv4_index_for_in(root: &Path, dev: &str, prefer: Option<Ipv4Addr>) -> Result<u32> {
    let gids = roce_v2_ipv4_gids_in(root, dev)?;
    let lowest = gids.iter().map(|(index, _)| *index).min();
    if let Some(want) = prefer
        && let Some(index) = gids
            .iter()
            .filter(|(_, addr)| *addr == want)
            .map(|(index, _)| *index)
            .min()
    {
        return Ok(index);
    }
    match lowest {
        Some(index) => Ok(index),
        None => bail!(
            "RDMA device {dev} port {PORT}: no RoCE v2 GID with an IPv4 address \
             (is the rail's netdev up with an IPv4 address?); set ATLAS_RDMA_GID to override"
        ),
    }
}

/// Every RoCE v2 GID carrying an IPv4-mapped address on `dev` port 1, as
/// `(index, address)` ascending by index. Empty when the netdev has no IPv4
/// address; the error is only a sysfs failure.
pub fn roce_v2_ipv4_gids(dev: &str) -> Result<Vec<(u32, Ipv4Addr)>> {
    roce_v2_ipv4_gids_in(Path::new(SYSFS_INFINIBAND), dev)
}

/// [`roce_v2_ipv4_gids`] against a sysfs-shaped `root` (tests use fixtures).
pub fn roce_v2_ipv4_gids_in(root: &Path, dev: &str) -> Result<Vec<(u32, Ipv4Addr)>> {
    let port = root.join(dev).join("ports").join(PORT.to_string());
    let gids = port.join("gids");
    let entries = std::fs::read_dir(&gids)
        .with_context(|| format!("RDMA device {dev}: cannot read {}", gids.display()))?;
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry?;
        let Some(index) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<u32>().ok())
        else {
            continue;
        };
        // Unused slots fail the type read (EINVAL); they are never a match.
        let Ok(kind) =
            std::fs::read_to_string(port.join("gid_attrs/types").join(index.to_string()))
        else {
            continue;
        };
        let gid = std::fs::read_to_string(entry.path()).unwrap_or_default();
        if kind.trim() == "RoCE v2"
            && let Some(addr) = ipv4_mapped_addr(gid.trim())
        {
            out.push((index, addr));
        }
    }
    out.sort_by_key(|(index, _)| *index);
    Ok(out)
}

/// `::ffff:a.b.c.d` in sysfs form (`0000:…:0000:ffff:xxxx:xxxx`) → `a.b.c.d`.
fn ipv4_mapped_addr(gid: &str) -> Option<Ipv4Addr> {
    let groups: Vec<&str> = gid.split(':').collect();
    if groups.len() != 8
        || !groups[..5].iter().all(|g| *g == "0000")
        || !groups[5].eq_ignore_ascii_case("ffff")
    {
        return None;
    }
    let hi = u16::from_str_radix(groups[6], 16).ok()?;
    let lo = u16::from_str_radix(groups[7], 16).ok()?;
    Some(Ipv4Addr::new(
        (hi >> 8) as u8,
        hi as u8,
        (lo >> 8) as u8,
        lo as u8,
    ))
}
