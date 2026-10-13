// SPDX-License-Identifier: AGPL-3.0-only
//
// RoCE GID-index discovery. The kernel builds each port's GID table from the
// netdev's addresses and rebuilds entries whenever an address is re-added (a
// link flap, a directly cabled peer rebooting). A freed slot that live QPs
// still reference is not reused, so the RoCE v2 IPv4 entry can move from its
// usual index 3 to 4 or beyond. Look it up instead of assuming an index,
// as NCCL does when NCCL_IB_GID_INDEX is unset.

use anyhow::{Context, Result, bail};
use std::path::Path;

const SYSFS_INFINIBAND: &str = "/sys/class/infiniband";

/// Port the shim opens (`rs_create` always uses port 1).
const PORT: u32 = 1;

/// Index of the RoCE v2 GID carrying an IPv4-mapped address on `dev` port 1.
pub fn roce_v2_ipv4_index(dev: &str) -> Result<u32> {
    roce_v2_ipv4_index_in(Path::new(SYSFS_INFINIBAND), dev)
}

/// [`roce_v2_ipv4_index`] against a sysfs-shaped `root` (tests use fixtures).
/// With several IPv4 addresses on the netdev, the lowest index wins.
pub fn roce_v2_ipv4_index_in(root: &Path, dev: &str) -> Result<u32> {
    let port = root.join(dev).join("ports").join(PORT.to_string());
    let gids = port.join("gids");
    let entries = std::fs::read_dir(&gids)
        .with_context(|| format!("RDMA device '{dev}': cannot read {}", gids.display()))?;
    let mut best: Option<u32> = None;
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
        if kind.trim() == "RoCE v2" && is_ipv4_mapped(gid.trim()) {
            best = Some(best.map_or(index, |b| b.min(index)));
        }
    }
    match best {
        Some(index) => Ok(index),
        None => bail!(
            "RDMA device '{dev}' port {PORT}: no RoCE v2 GID with an IPv4 address \
             (is the rail's netdev up with an IPv4 address?); set ATLAS_RDMA_GID to override"
        ),
    }
}

/// `::ffff:a.b.c.d` in sysfs form (`0000:…:0000:ffff:xxxx:xxxx`).
fn is_ipv4_mapped(gid: &str) -> bool {
    let groups: Vec<&str> = gid.split(':').collect();
    groups.len() == 8
        && groups[..5].iter().all(|g| *g == "0000")
        && groups[5].eq_ignore_ascii_case("ffff")
}
