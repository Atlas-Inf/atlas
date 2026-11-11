// SPDX-License-Identifier: AGPL-3.0-only

//! Peer-death lifeline for multi-rank serving.
//!
//! NCCL waits block in `cuStreamSynchronize` with no deadline, so a rank whose
//! peer process has died (crash, OOM kill, container stop) blocks in its next
//! collective forever while its HTTP readiness still says ready. The TCP
//! connections that carried the NCCL bootstrap ID are kept open for the
//! process lifetime instead: the kernel closes them when either process exits,
//! for any reason, so their end is a liveness signal independent of NCCL and
//! of the RDMA pair transport. TCP keepalive covers a peer host that stops
//! answering altogether and never sends a FIN.
//!
//! Nothing is written after the bootstrap ID, so a watcher's blocking read
//! returns only when the connection ends.

use anyhow::{Context, Result};
use std::io::Read;
use std::net::TcpStream;
use std::sync::Arc;

#[cfg(test)]
#[path = "peer_lifeline_tests.rs"]
mod tests;

/// The bootstrap connections to this rank's peers (every worker on rank 0,
/// rank 0 on a worker).
///
/// Dropping one closes it, which a watching peer reads as this rank's death,
/// so the owner holds it for the process lifetime whether or not it watches.
pub struct PeerLifeline {
    peers: Vec<TcpStream>,
}

impl PeerLifeline {
    #[cfg(any(feature = "nccl", test))]
    pub(crate) fn new(peers: Vec<TcpStream>) -> Self {
        Self { peers }
    }

    /// Start one watcher thread per peer connection. `on_lost` runs on the
    /// watcher thread when that connection ends, with a description of how.
    pub fn watch(&self, on_lost: impl Fn(String) + Send + Sync + 'static) -> Result<()> {
        let on_lost = Arc::new(on_lost);
        for peer in &self.peers {
            let mut peer = peer
                .try_clone()
                .context("peer lifeline: failed to clone bootstrap connection")?;
            keepalive::enable(&peer)?;
            let on_lost = Arc::clone(&on_lost);
            std::thread::Builder::new()
                .name("ep-peer-lifeline".into())
                .spawn(move || on_lost(wait_for_close(&mut peer)))
                .context("peer lifeline: failed to spawn watcher")?;
        }
        Ok(())
    }
}

fn wait_for_close(peer: &mut TcpStream) -> String {
    let addr = peer
        .peer_addr()
        .map_or_else(|_| "peer".to_owned(), |addr| addr.to_string());
    let mut buf = [0u8; 64];
    loop {
        match peer.read(&mut buf) {
            Ok(0) => return format!("bootstrap connection to {addr} closed"),
            // Nothing is sent after bootstrap; stray bytes are not a death.
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return format!("bootstrap connection to {addr} failed: {e}"),
        }
    }
}

#[cfg(target_os = "linux")]
mod keepalive {
    use anyhow::{Result, bail};
    use std::net::TcpStream;
    use std::os::fd::AsRawFd;

    /// Idle seconds before the first probe, seconds between probes, and
    /// unanswered probes before the connection fails: a silent peer host is
    /// declared dead ~30 s after it stops answering.
    const IDLE_SECS: libc::c_int = 10;
    const INTERVAL_SECS: libc::c_int = 5;
    const PROBES: libc::c_int = 4;

    pub(super) fn enable(stream: &TcpStream) -> Result<()> {
        let fd = stream.as_raw_fd();
        for (level, name, value, label) in [
            (libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1, "SO_KEEPALIVE"),
            (
                libc::IPPROTO_TCP,
                libc::TCP_KEEPIDLE,
                IDLE_SECS,
                "TCP_KEEPIDLE",
            ),
            (
                libc::IPPROTO_TCP,
                libc::TCP_KEEPINTVL,
                INTERVAL_SECS,
                "TCP_KEEPINTVL",
            ),
            (libc::IPPROTO_TCP, libc::TCP_KEEPCNT, PROBES, "TCP_KEEPCNT"),
        ] {
            // SAFETY: `fd` is a live socket owned by `stream` for this call;
            // `value` is a c_int whose size is passed as the option length.
            let status = unsafe {
                libc::setsockopt(
                    fd,
                    level,
                    name,
                    (&value as *const libc::c_int).cast(),
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                )
            };
            if status != 0 {
                bail!(
                    "peer lifeline: setsockopt {label} failed: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
        Ok(())
    }
}

/// Keepalive tuning is Linux-only; elsewhere a peer exit still closes the
/// connection, only a silent peer host goes undetected.
#[cfg(not(target_os = "linux"))]
mod keepalive {
    pub(super) fn enable(_stream: &std::net::TcpStream) -> anyhow::Result<()> {
        Ok(())
    }
}
