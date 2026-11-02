// SPDX-License-Identifier: AGPL-3.0-only

//! TCP bootstrap of the RDMA chain, relayed through rank 0. Every rank
//! already holds a connection to rank 0 from the NCCL bootstrap, and only
//! rank 0 is known to all, so no rank needs another rank's address: each
//! worker dials rank 0 where it reached it for NCCL (the master port + 2, as
//! the pair) and rank 0 admits only the addresses of those connections.
//!
//! 1. Each worker sends its [`Head`]. Rank 0 collects one per rank, checks
//!    that all agree with its own and that both ends of every link have the
//!    same rail count, and answers with its head, which the worker checks.
//! 2. Each worker sends its identity for the link below and the one above
//!    (the pair's identity format); rank 0 returns each its neighbours'.
//! 3. Each rank connects its QPs, then reports; once all have, rank 0
//!    releases them, so every QP is RTS before any rank posts a WRITE.
//!
//! A rank that disagrees is named in rank 0's error; rank 0 then drops every
//! connection, and the workers' errors send the operator to its log. The
//! window and its variable are the pair's (`ATLAS_RDMA_PAIR_BOOT_TIMEOUT_S`).

use super::super::topology::Topology;
use super::{
    BASE_WIRE, Bounded, MAX_RAILS, MAX_RANKS, POLL, RAIL_WIRE, RETRY, bind, boot_timeout, gone,
    io_of,
};
use anyhow::{Context, Result, bail, ensure};
use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::time::{Duration, Instant};

/// Names this wire format; bump the digit when the head or layout changes.
const MAGIC: [u8; 8] = *b"ATLCHAN5";
const WHO: &str = "RDMA chain";
/// What every rank must agree on, after the rank and its two rail counts.
const FIELDS: [(&str, &str); 8] = [
    (
        "rank count",
        "every rank must run with the same --world-size",
    ),
    ("one-shot max", "check ATLAS_RDMA_ONESHOT_MAX on every node"),
    (
        "one-shot stripe min",
        "check ATLAS_RDMA_ONESHOT_STRIPE_MIN on every node",
    ),
    (
        "one-shot SM staging",
        "check ATLAS_RDMA_ONESHOT_SM_STAGE on every node",
    ),
    ("command ring", "check ATLAS_GLM_CMD_RDMA on every node"),
    (
        "all-reduce split",
        "check ATLAS_RDMA_CHAIN_AR_SPLIT_MIN on every node",
    ),
    (
        "fused reduce-scatter segment",
        "check ATLAS_RDMA_RS_FUSED_SEG on every node",
    ),
    ("topology", "check ATLAS_RDMA_TOPOLOGY on every node"),
];
const WORDS: usize = 3 + FIELDS.len();
const HEAD_WIRE: usize = 8 * (1 + WORDS);

/// What a rank's use of the chain depends on, besides its identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::nccl_backend) struct Head {
    pub(in crate::nccl_backend) rank: usize,
    /// Rails to the rank below and to the one above (0 where none).
    pub(in crate::nccl_backend) rails: [usize; 2],
    pub(in crate::nccl_backend) ranks: usize,
    /// One-shot `max`, `stripe_min` and `sm_stage`.
    pub(in crate::nccl_backend) oneshot: [u64; 3],
    /// Command ring words, zero when off.
    pub(in crate::nccl_backend) cmd: u64,
    /// Smallest all-reduce taken as reduce-scatter + all-gather.
    pub(in crate::nccl_backend) ar_split: u64,
    /// The fused reduce-scatter's segment capacity (0: no fused region).
    pub(in crate::nccl_backend) rs_fused: u64,
    pub(in crate::nccl_backend) topology: Topology,
}

impl Head {
    fn words(&self) -> [u64; WORDS] {
        [
            self.rank as u64,
            self.rails[0] as u64,
            self.rails[1] as u64,
            self.ranks as u64,
            self.oneshot[0],
            self.oneshot[1],
            self.oneshot[2],
            self.cmd,
            self.ar_split,
            self.rs_fused,
            self.topology.wire(),
        ]
    }

    fn wire(&self) -> [u8; HEAD_WIRE] {
        let mut w = [0u8; HEAD_WIRE];
        w[..8].copy_from_slice(&MAGIC);
        for (at, word) in w[8..].chunks_exact_mut(8).zip(self.words()) {
            at.copy_from_slice(&word.to_le_bytes());
        }
        w
    }

    fn read(stream: &mut impl Read) -> Result<Self> {
        let mut w = [0u8; HEAD_WIRE];
        stream
            .read_exact(&mut w[..8])
            .map_err(io_of(WHO, "peer head"))?;
        ensure!(
            w[..8] == MAGIC,
            "{WHO}: the peer runs a different build (bootstrap magic {:02x?})",
            &w[..8]
        );
        stream
            .read_exact(&mut w[8..])
            .map_err(io_of(WHO, "peer head"))?;
        let word = |i: usize| u64::from_le_bytes(w[8 + 8 * i..16 + 8 * i].try_into().unwrap());
        let head = Self {
            rank: word(0) as usize,
            rails: [word(1) as usize, word(2) as usize],
            ranks: word(3) as usize,
            oneshot: [word(4), word(5), word(6)],
            cmd: word(7),
            ar_split: word(8),
            rs_fused: word(9),
            topology: Topology::from_wire(word(10))?,
        };
        head.check()?;
        Ok(head)
    }

    /// A shape a chain can have: rank below 3..=`MAX_RANKS` ranks, and
    /// 1..=`MAX_RAILS` rails on exactly the sides where it has a neighbour.
    /// Checked before a peer's rail counts size anything read from it.
    fn check(&self) -> Result<()> {
        ensure!(
            (3..=MAX_RANKS).contains(&self.ranks) && self.rank < self.ranks,
            "{WHO}: rank {} of {} (supported: 3..={MAX_RANKS} ranks)",
            self.rank,
            self.ranks
        );
        self.topology.check(self.ranks)?;
        let linked = [0, 1].map(|side| {
            self.topology
                .neighbor(self.rank, self.ranks, side)
                .is_some()
        });
        for ((side, &rails), linked) in ["below", "above"].iter().zip(&self.rails).zip(linked) {
            let fits = if linked {
                (1..=MAX_RAILS).contains(&rails)
            } else {
                rails == 0
            };
            ensure!(
                fits,
                "{WHO}: rank {} of {} has {rails} rails to the rank {side} (supported: \
                 1..={MAX_RAILS} where there is one, else none)",
                self.rank,
                self.ranks
            );
        }
        Ok(())
    }

    /// Whether `peer` agrees on everything every rank must.
    fn agree(&self, peer: &Self) -> Result<()> {
        let (ours, theirs) = (self.words(), peer.words());
        for ((name, hint), (here, there)) in FIELDS.iter().zip(ours[3..].iter().zip(&theirs[3..])) {
            ensure!(
                here == there,
                "{WHO}: {name}: rank {} has {here}, rank {} {there} - {hint}",
                self.rank,
                peer.rank
            );
        }
        Ok(())
    }
}

/// Every rank's head, by rank: the two ends of each link have the same rails.
fn check_links(heads: &[Head]) -> Result<()> {
    for (k, head) in heads.iter().enumerate() {
        let Some(next) = head.topology.neighbor(k, heads.len(), 1) else {
            continue;
        };
        ensure!(
            head.rails[1] == heads[next].rails[0],
            "{WHO}: the link between ranks {k} and {next}: rank {k} has {} rails \
             (ATLAS_RDMA_RIGHT_RAILS), rank {next} has {} (ATLAS_RDMA_LEFT_RAILS)",
            head.rails[1],
            heads[next].rails[0]
        );
    }
    Ok(())
}

/// Where the chain bootstraps: rank 0's address as its workers reached it,
/// the addresses it admits, and the NCCL bootstrap connections, whose end
/// tells that the peer process has exited.
pub(in crate::nccl_backend) struct Rendezvous<'a> {
    pub(in crate::nccl_backend) at: SocketAddr,
    pub(in crate::nccl_backend) workers: Vec<IpAddr>,
    pub(in crate::nccl_backend) lifelines: Vec<&'a TcpStream>,
}

impl<'a> Rendezvous<'a> {
    /// The rendezvous of `conns`, this rank's NCCL bootstrap connections
    /// (rank 0: one per worker; a worker: the one to rank 0), at `port`.
    pub(in crate::nccl_backend) fn of(
        rank: usize,
        conns: &'a [TcpStream],
        port: u16,
    ) -> Result<Self> {
        let head = |c: &TcpStream| {
            if rank == 0 {
                c.local_addr()
            } else {
                c.peer_addr()
            }
        };
        let mut heads = conns.iter().map(|c| head(c).map(|a| a.ip()));
        let ip = heads
            .next()
            .context("RDMA chain: no NCCL bootstrap connection")??;
        for other in heads {
            let other = other?;
            ensure!(
                other.to_canonical() == ip.to_canonical(),
                "RDMA chain: workers reached rank 0 at {ip} and at {other}; \
                 give every rank the same --master-addr"
            );
        }
        let workers = if rank == 0 {
            conns
                .iter()
                .map(|c| c.peer_addr().map(|a| a.ip()))
                .collect::<std::io::Result<_>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            at: SocketAddr::new(ip, port),
            workers,
            lifelines: conns.iter().collect(),
        })
    }

    fn lost(&self) -> std::io::Result<bool> {
        for lifeline in &self.lifelines {
            if gone(lifeline)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Bytes of the identity a rank sends for a link of `rails` (none: 0).
pub(in crate::nccl_backend) fn ident_len(rails: usize) -> usize {
    if rails == 0 {
        0
    } else {
        BASE_WIRE + rails * RAIL_WIRE
    }
}

/// Run the bootstrap for this rank's `head` and identities (below, above),
/// calling `connect` with its neighbours' identities (below, above; empty
/// where none) before the closing barrier.
pub(in crate::nccl_backend) fn rendezvous(
    head: &Head,
    idents: [Vec<u8>; 2],
    at: &Rendezvous,
    connect: impl FnOnce([Vec<u8>; 2]) -> Result<()>,
) -> Result<()> {
    head.check()?;
    let timeout = boot_timeout(std::env::var("ATLAS_RDMA_PAIR_BOOT_TIMEOUT_S").ok());
    let deadline = Instant::now() + timeout;
    if head.rank == 0 {
        lead(head, idents, at, deadline, connect)
    } else {
        join(head, idents, at, deadline, connect)
    }
}

/// Rank 0: admit one connection per worker, then relay.
fn lead(
    head: &Head,
    idents: [Vec<u8>; 2],
    at: &Rendezvous,
    deadline: Instant,
    connect: impl FnOnce([Vec<u8>; 2]) -> Result<()>,
) -> Result<()> {
    let listener = bind(at.at)?;
    tracing::info!(
        "{WHO}: rank 0 listening on {}, admitting {} workers from {:?}",
        at.at,
        head.ranks - 1,
        at.workers
    );
    let mut workers: Vec<Option<(Bounded, Head)>> = (0..head.ranks).map(|_| None).collect();
    let mut last = "no connection".to_owned();
    while workers.iter().skip(1).any(Option::is_none) {
        ensure!(
            Instant::now() < deadline,
            "{WHO}: not every worker bootstrapped on {} in time (last: {last})",
            at.at
        );
        let (stream, from) = match listener.accept() {
            Ok(conn) => conn,
            Err(e) => {
                if e.kind() != ErrorKind::WouldBlock {
                    last = format!("accept: {e}");
                }
                ensure!(
                    !at.lost()?,
                    "{WHO}: a worker exited before the bootstrap (last: {last})"
                );
                std::thread::sleep(POLL);
                continue;
            }
        };
        let admitted = (|| -> Result<_> {
            ensure!(
                at.workers
                    .iter()
                    .any(|w| w.to_canonical() == from.ip().to_canonical()),
                "not a worker of the NCCL bootstrap"
            );
            let until = deadline.min(Instant::now() + Duration::from_secs(10));
            let mut stream = Bounded::new(stream, until)?;
            let peer = Head::read(&mut stream)?;
            ensure!(
                (1..head.ranks).contains(&peer.rank),
                "rank {} of {}",
                peer.rank,
                head.ranks
            );
            ensure!(
                workers[peer.rank].is_none(),
                "a second rank {} (two nodes run the same --rank)",
                peer.rank
            );
            head.agree(&peer)?;
            stream.deadline = deadline;
            Ok((stream, peer))
        })();
        match admitted {
            Ok((stream, peer)) => {
                let rank = peer.rank;
                workers[rank] = Some((stream, peer));
            }
            Err(e) => {
                tracing::warn!("{WHO}: rejected a bootstrap connection from {from}: {e:#}");
                last = format!("rejected {from}: {e:#}");
            }
        }
    }
    let mut heads = vec![head.clone()];
    let mut streams = Vec::new();
    for (stream, peer) in workers.into_iter().flatten() {
        heads.push(peer);
        streams.push(stream);
    }
    check_links(&heads)?;
    let mut all = vec![idents];
    for (stream, peer) in streams.iter_mut().zip(&heads[1..]) {
        stream
            .write_all(&head.wire())
            .map_err(io_of(WHO, "send head"))?;
        let mut theirs = peer.rails.map(|r| vec![0u8; ident_len(r)]);
        for ident in &mut theirs {
            stream
                .read_exact(ident)
                .map_err(io_of(WHO, "worker identity"))?;
        }
        all.push(theirs);
    }
    let neighbours = |k: usize| -> [Vec<u8>; 2] {
        [0, 1].map(|side| {
            head.topology
                .neighbor(k, head.ranks, side)
                .map_or_else(Vec::new, |peer| all[peer][1 - side].clone())
        })
    };
    for (k, stream) in (1..).zip(streams.iter_mut()) {
        for ident in neighbours(k) {
            stream
                .write_all(&ident)
                .map_err(io_of(WHO, "send identities"))?;
        }
    }
    connect(neighbours(0))?;
    for stream in &mut streams {
        stream
            .read_exact(&mut [0u8])
            .map_err(io_of(WHO, "barrier"))?;
    }
    for stream in &mut streams {
        stream.write_all(&[1]).map_err(io_of(WHO, "barrier"))?;
    }
    Ok(())
}

/// A worker: reach rank 0, which may not be listening yet, then follow.
fn join(
    head: &Head,
    idents: [Vec<u8>; 2],
    at: &Rendezvous,
    deadline: Instant,
    connect: impl FnOnce([Vec<u8>; 2]) -> Result<()>,
) -> Result<()> {
    let stream = loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match TcpStream::connect_timeout(&at.at, left.max(RETRY)) {
            Ok(s) => break s,
            Err(e) => {
                ensure!(
                    !at.lost()?,
                    "{WHO}: rank 0 exited before listening on {} (its log says why)",
                    at.at
                );
                ensure!(left > RETRY, "{WHO}: connect {}: {e}", at.at);
                std::thread::sleep(RETRY);
            }
        }
    };
    let mut stream = Bounded::new(stream, deadline)?;
    stream
        .write_all(&head.wire())
        .map_err(io_of(WHO, "send head"))?;
    let lead = Head::read(&mut stream)?;
    if lead.rank != 0 {
        bail!("{WHO}: rank 0's listener answered as rank {}", lead.rank);
    }
    head.agree(&lead)?;
    for ident in &idents {
        stream
            .write_all(ident)
            .map_err(io_of(WHO, "send identities"))?;
    }
    let mut theirs = head.rails.map(|r| vec![0u8; ident_len(r)]);
    for ident in &mut theirs {
        stream
            .read_exact(ident)
            .map_err(io_of(WHO, "neighbour identities"))?;
    }
    connect(theirs)?;
    stream.write_all(&[1]).map_err(io_of(WHO, "barrier"))?;
    stream
        .read_exact(&mut [0u8])
        .map_err(io_of(WHO, "barrier"))?;
    Ok(())
}

#[cfg(test)]
#[path = "bootstrap_tests.rs"]
mod tests;
