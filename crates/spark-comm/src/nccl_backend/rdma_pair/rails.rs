// SPDX-License-Identifier: AGPL-3.0-only

//! Bringing a link's rails up, shared by the pair and the chain: the pinned
//! region every rail registers, the rail device names, one QP per rail and
//! this end's identity for the peer, and connecting to the peer's.

use super::bootstrap::{BASE_WIRE, RAIL_WIRE};
use super::{
    CU_MEMHOSTALLOC_PORTABLE_DEVICEMAP, cu, cuMemHostAlloc, cuMemHostGetDevicePointer_v2, gid,
};
use anyhow::{Context, Result, ensure};
use atlas_rdma::{Gid, Verbs};
use std::ffi::c_void;
use std::net::Ipv4Addr;

/// A zeroed, pinned, device-mapped host region of `bytes` for `what`: its
/// host and device addresses. Freed with `cuMemFreeHost`.
pub(super) fn pinned(bytes: usize, what: &str) -> Result<(*mut c_void, u64)> {
    let mut host: *mut c_void = std::ptr::null_mut();
    cu(
        unsafe { cuMemHostAlloc(&mut host, bytes, CU_MEMHOSTALLOC_PORTABLE_DEVICEMAP) },
        &format!("cuMemHostAlloc({what})"),
    )?;
    unsafe { std::ptr::write_bytes(host as *mut u8, 0, bytes) };
    let mut dev = 0u64;
    cu(
        unsafe { cuMemHostGetDevicePointer_v2(&mut dev, host, 0) },
        &format!("cuMemHostGetDevicePointer({what})"),
    )?;
    Ok((host, dev))
}

/// One link's rails: a QP per rail device, each with the whole region
/// registered, and this end's identity for the peer (see [`bootstrap`]).
pub(super) struct Rails {
    pub(super) verbs: Vec<Verbs>,
    pub(super) lkeys: Vec<u32>,
    pub(super) ident: Vec<u8>,
}

impl Rails {
    /// Open `names` (GID index found per device, on `prefer` when that is
    /// this end's address on the link, or `gid_override`) for `rank` over the
    /// region at `host` of `bytes`.
    pub(super) fn open(
        names: &[String],
        gid_override: Option<u32>,
        prefer: Option<Ipv4Addr>,
        rank: usize,
        host: *mut c_void,
        bytes: usize,
    ) -> Result<Self> {
        let psn = (0x5a5a00 + rank as u32 * 0x1111) & 0xff_ffff;
        let mut verbs = Vec::with_capacity(names.len());
        let mut ident = (host as u64).to_le_bytes().to_vec();
        let mut lkeys = Vec::with_capacity(names.len());
        for name in names {
            let gid_idx = gid_override.map_or_else(|| gid::pick(name, prefer), Ok)?;
            let mut v = Verbs::create(name, gid_idx, psn)
                .with_context(|| format!("RDMA rail {name} (gid index {gid_idx})"))?;
            // SAFETY: the region outlives the Verbs (freed in Drop after the
            // proxy, which owns the Verbs, has joined).
            let keys = unsafe { v.reg_mr_rw(host, bytes) }?;
            ident.extend_from_slice(&v.qpn().to_le_bytes());
            ident.extend_from_slice(&v.psn().to_le_bytes());
            ident.extend_from_slice(&v.gid());
            ident.extend_from_slice(&keys.rkey.to_le_bytes());
            lkeys.push(keys.lkey);
            verbs.push(v);
        }
        Ok(Self {
            verbs,
            lkeys,
            ident,
        })
    }
}

/// Bring each rail to RTS against the peer's identity `remote`; returns the
/// peer's region base and its rkey per rail.
pub(super) fn connect_rails(verbs: &mut [Verbs], remote: &[u8]) -> Result<(u64, Vec<u32>)> {
    ensure!(
        remote.len() == BASE_WIRE + verbs.len() * RAIL_WIRE,
        "RDMA: the peer's identity is {} bytes for {} rails",
        remote.len(),
        verbs.len()
    );
    let base = u64::from_le_bytes(remote[..BASE_WIRE].try_into()?);
    let mut rkeys = Vec::with_capacity(verbs.len());
    for (v, w) in verbs
        .iter_mut()
        .zip(remote[BASE_WIRE..].chunks_exact(RAIL_WIRE))
    {
        let qpn = u32::from_le_bytes(w[0..4].try_into()?);
        let rpsn = u32::from_le_bytes(w[4..8].try_into()?);
        let gid: Gid = w[8..24].try_into()?;
        rkeys.push(u32::from_le_bytes(w[24..28].try_into()?));
        v.connect(qpn, rpsn, &gid)?;
    }
    Ok((base, rkeys))
}

/// Rail device names: `ATLAS_RDMA_RAILS`, else `NCCL_IB_HCA` without NCCL's
/// `^`/`=` match prefixes and `:port` suffixes, else the first GB10 port.
pub(super) fn rail_names(rails: Option<String>, nccl: Option<String>) -> Vec<String> {
    let list = rails.or(nccl).unwrap_or_else(|| "rocep1s0f0".into());
    list.trim_start_matches(['^', '='])
        .split(',')
        .filter_map(|s| s.split(':').next())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

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
}
