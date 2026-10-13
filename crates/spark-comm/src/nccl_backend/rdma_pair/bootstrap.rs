// SPDX-License-Identifier: AGPL-3.0-only

//! TCP bootstrap of the RDMA pair (rank 0 listens). Both ranks address the
//! peer's region with their own layout, so each first sends a fixed-size
//! [`Head`] with everything the layout and the flag protocol depend on. Only
//! a peer whose head agrees is sent the identity: the region base, then per
//! rail the QPN, PSN, GID and region rkey, which together grant remote writes.
//!
//! The ranks get here seconds apart: the NCCL bootstrap and
//! `ncclCommInitRank` before it have already absorbed any difference in model
//! load time, and what remains is each rank pinning and registering its
//! region. The boot fails after `ATLAS_RDMA_PAIR_BOOT_TIMEOUT_S` (default
//! 120) rather than hang, and at once when the peer process has exited.

use anyhow::{Context, Result, anyhow, bail, ensure};
use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

/// Names this wire format; bump the digit when the head or layout changes.
const MAGIC: [u8; 8] = *b"ATLPAIR4";
/// What both ranks must agree on, in wire order, and where to look when
/// they do not.
const FIELDS: [(&str, &str); 7] = [
    ("rail count", "check ATLAS_RDMA_RAILS on both nodes"),
    ("PAIR_CHAIN", "check ATLAS_RDMA_PAIR_CHAIN on both nodes"),
    (
        "segment count",
        "check ATLAS_RDMA_PAIR_SEGMENTS on both nodes",
    ),
    ("capacity", "max_batch_tokens and the model must match"),
    (
        "one-shot max",
        "check ATLAS_RDMA_ONESHOT and ATLAS_RDMA_ONESHOT_MAX on both nodes",
    ),
    (
        "one-shot stripe min",
        "check ATLAS_RDMA_ONESHOT_STRIPE_MIN on both nodes",
    ),
    ("command ring", "check ATLAS_GLM_CMD_RDMA on both nodes"),
];
/// The rank, then the [`FIELDS`].
const WORDS: usize = 1 + FIELDS.len();
/// Magic, then the words as little-endian u64.
const HEAD_WIRE: usize = 8 * (1 + WORDS);
/// Between rank 0's accept polls, and between the worker's connect attempts.
const POLL: Duration = Duration::from_millis(10);
const RETRY: Duration = Duration::from_millis(500);

/// What a rank's use of the pair depends on, besides its own identity.
pub(super) struct Head {
    pub(super) rank: usize,
    pub(super) rails: usize,
    pub(super) chain: bool,
    pub(super) segments: usize,
    pub(super) capacity: usize,
    /// One-shot `max` and `stripe_min`, zero when off.
    pub(super) oneshot: [u64; 2],
    /// Command ring words, zero when off.
    pub(super) cmd: u64,
}

impl Head {
    fn words(&self) -> [u64; WORDS] {
        [
            self.rank as u64,
            self.rails as u64,
            self.chain as u64,
            self.segments as u64,
            self.capacity as u64,
            self.oneshot[0],
            self.oneshot[1],
            self.cmd,
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
}

/// Names a failed bootstrap read or write; a timeout reads as one, and a
/// hang-up as the rejection it is (the wire has no error frame): an EOF, or
/// a reset when the peer dropped the connection with bytes unread.
pub(super) fn io(what: &str) -> impl Fn(std::io::Error) -> anyhow::Error {
    move |e| {
        let why = match e.kind() {
            ErrorKind::WouldBlock | ErrorKind::TimedOut => "timed out".into(),
            ErrorKind::UnexpectedEof
            | ErrorKind::ConnectionReset
            | ErrorKind::ConnectionAborted
            | ErrorKind::BrokenPipe => {
                "the peer closed the connection (its log says why it rejected this rank)".into()
            }
            _ => e.to_string(),
        };
        anyhow!("RDMA pair: {what}: {why}")
    }
}

/// The peer's head words. The magic is read first and alone: a peer on a
/// build with another format fails here on both sides -- it reads a foreign
/// or missing magic too -- rather than being misparsed.
fn read_head(stream: &mut impl Read) -> Result<[u64; WORDS]> {
    let mut magic = [0u8; 8];
    stream.read_exact(&mut magic).map_err(io("peer head"))?;
    ensure!(
        magic == MAGIC,
        "RDMA pair: the peer runs a different build (bootstrap magic {magic:02x?})"
    );
    let mut rest = [0u8; HEAD_WIRE - 8];
    stream.read_exact(&mut rest).map_err(io("peer head"))?;
    Ok(std::array::from_fn(|i| {
        u64::from_le_bytes(rest[8 * i..8 * i + 8].try_into().unwrap())
    }))
}

/// Agree on the heads, then send `ident` and return the peer's identity (as
/// many bytes). Rank 0 listens, so it stays silent until the connecting side
/// has shown this format, and neither rank writes its identity to a peer
/// whose head differs or is its own reflected.
pub(super) fn exchange(
    stream: &mut (impl Read + Write),
    head: &Head,
    ident: &[u8],
) -> Result<Vec<u8>> {
    let (ours, wire) = (head.words(), head.wire());
    if head.rank != 0 {
        stream.write_all(&wire).map_err(io("send head"))?;
    }
    let theirs = read_head(stream)?;
    if head.rank == 0 {
        stream.write_all(&wire).map_err(io("send head"))?;
    }
    ensure!(
        theirs[0] == ours[0] ^ 1,
        "RDMA pair: rank: here {}, peer {} - a reflected head, or both nodes run the same --rank",
        ours[0],
        theirs[0]
    );
    for ((name, hint), (here, peer)) in FIELDS.iter().zip(ours.iter().zip(&theirs).skip(1)) {
        ensure!(
            here == peer,
            "RDMA pair: {name}: here {here}, peer {peer} - {hint}"
        );
    }
    stream.write_all(ident).map_err(io("send identity"))?;
    let mut remote = vec![0u8; ident.len()];
    stream
        .read_exact(&mut remote)
        .map_err(io("peer identity"))?;
    Ok(remote)
}

/// Where the pair bootstraps.
pub(in crate::nccl_backend) struct Link<'a> {
    /// Rank 0's address on the link between the ranks: rank 0 listens there
    /// only, and the worker dials it.
    pub(in crate::nccl_backend) at: SocketAddr,
    /// The only address rank 0 admits, when known.
    pub(in crate::nccl_backend) worker: Option<IpAddr>,
    /// A connection that ends when the peer process does, after which no
    /// valid peer can come.
    pub(in crate::nccl_backend) lifeline: Option<&'a TcpStream>,
}

impl<'a> Link<'a> {
    /// The link of `conn`, this rank's NCCL bootstrap connection: rank 0
    /// listens on `port` of the address the worker reached it at and admits
    /// only the worker's address. Rank 0 does not bind its own
    /// `--master-addr` (`master_addr`): it never read it, and it may be a
    /// hostname or left at its default. One that names another address is
    /// said, since the listener then is not where the operator pointed.
    pub(in crate::nccl_backend) fn of(
        rank: usize,
        conn: &'a TcpStream,
        port: u16,
        master_addr: &str,
    ) -> Result<Self> {
        let (head, worker) = ends(rank, conn.local_addr()?.ip(), conn.peer_addr()?.ip());
        if rank == 0 && elsewhere(master_addr, head) {
            tracing::warn!(
                "RDMA pair: the worker reached rank 0 at {head}, not at its --master-addr \
                 {master_addr}; the bootstrap listens on {head}"
            );
        }
        Ok(Self {
            at: SocketAddr::new(head, port),
            worker: Some(worker),
            lifeline: Some(conn),
        })
    }
}

/// Rank 0's address and the worker's, from `rank`'s own and its peer's.
fn ends(rank: usize, own: IpAddr, peer: IpAddr) -> (IpAddr, IpAddr) {
    if rank == 0 { (own, peer) } else { (peer, own) }
}

/// Whether `master_addr` names an address other than `bound`; a hostname,
/// the loopback default and the wildcard name none.
fn elsewhere(master_addr: &str, bound: IpAddr) -> bool {
    master_addr.parse::<IpAddr>().is_ok_and(|ip| {
        !ip.is_loopback() && !ip.is_unspecified() && ip.to_canonical() != bound.to_canonical()
    })
}

/// Whether the peer process has exited. Nothing is sent on its lifeline
/// after the NCCL ID, so only its end is readable. The connection is left
/// blocking: its watcher reads it for the life of the process.
fn gone(lifeline: &TcpStream) -> std::io::Result<bool> {
    lifeline.set_nonblocking(true)?;
    let peeked = lifeline.peek(&mut [0u8]);
    lifeline.set_nonblocking(false)?;
    Ok(match peeked {
        Ok(n) => n == 0,
        Err(e) => !matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted),
    })
}

/// [`gone`], of the peer on `link` when it has a lifeline.
fn lost(link: &Link) -> std::io::Result<bool> {
    link.lifeline.map_or(Ok(false), gone)
}

/// A bootstrap connection whose reads and writes all end by one deadline,
/// however slowly the peer feeds them.
pub(super) struct Bounded {
    stream: TcpStream,
    deadline: Instant,
}

impl Bounded {
    /// Blocking (an accepted socket can inherit the listener's non-blocking
    /// mode) and unbuffered.
    fn new(stream: TcpStream, deadline: Instant) -> std::io::Result<Self> {
        stream.set_nonblocking(false)?;
        stream.set_nodelay(true)?;
        Ok(Self { stream, deadline })
    }

    /// Give the next read or write the time that is left.
    fn arm(&self) -> std::io::Result<()> {
        let left = self.deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(ErrorKind::TimedOut.into());
        }
        self.stream.set_read_timeout(Some(left))?;
        self.stream.set_write_timeout(Some(left))
    }
}

impl Read for Bounded {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.arm()?;
        self.stream.read(buf)
    }
}

impl Write for Bounded {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.arm()?;
        self.stream.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `ATLAS_RDMA_PAIR_BOOT_TIMEOUT_S`, from a second to a day. Zero does not
/// mean "no deadline" (the boot must end); it and a value that is not a
/// number of seconds are said.
fn boot_timeout(secs: Option<String>) -> Duration {
    const NAME: &str = "ATLAS_RDMA_PAIR_BOOT_TIMEOUT_S";
    let secs = match secs.as_deref().map(str::parse::<u64>) {
        None => 120,
        Some(Ok(n)) => {
            let within = n.clamp(1, 86_400);
            if within != n {
                tracing::warn!("{NAME}={n} is outside 1..=86400 s; using {within}");
            }
            within
        }
        Some(Err(_)) => {
            tracing::warn!("{NAME} is not a number of seconds; using 120");
            120
        }
    };
    Duration::from_secs(secs)
}

/// The bootstrap stream to the peer, exchanged, with the peer's identity.
/// What is then read or written on it (the barrier) ends with the boot's
/// window too.
pub(super) fn open(head: &Head, ident: &[u8], link: &Link) -> Result<(Bounded, Vec<u8>)> {
    let timeout = boot_timeout(std::env::var("ATLAS_RDMA_PAIR_BOOT_TIMEOUT_S").ok());
    if head.rank == 0 {
        admit(&bind(link.at)?, head, ident, link, timeout)
    } else {
        dial(head, ident, link, timeout)
    }
}

/// Rank 0's listener, on `at` only: a failure is the error, never a wider
/// bind.
fn bind(at: SocketAddr) -> Result<TcpListener> {
    let listener = TcpListener::bind(at).with_context(|| format!("RDMA pair: bind {at}"))?;
    // std has no accept timeout: poll.
    listener.set_nonblocking(true)?;
    Ok(listener)
}

/// Rank 0: take connections until one completes the exchange. One that does
/// not -- a scanner, a stale or misconfigured rank, anyone but
/// `link.worker` when that is known -- is dropped and logged; it cannot end
/// the boot. `timeout` does, or the worker's exit (`link.lifeline`), with
/// the last rejection. Each connection gets a twelfth of `timeout` for the
/// whole exchange, so neither a silent nor a dripping one holds the listener
/// for the window.
fn admit(
    listener: &TcpListener,
    head: &Head,
    ident: &[u8],
    link: &Link,
    timeout: Duration,
) -> Result<(Bounded, Vec<u8>)> {
    let (at, deadline) = (listener.local_addr()?, Instant::now() + timeout);
    let who = link
        .worker
        .map_or_else(|| "any address".to_owned(), |w| w.to_string());
    tracing::info!("RDMA pair: rank 0 listening on {at}, admitting {who}");
    let mut last = "no connection".to_owned();
    while Instant::now() < deadline {
        let (stream, from) = match listener.accept() {
            Ok(conn) => conn,
            // Nothing pending, or a connection that already went away.
            Err(e) => {
                if e.kind() != ErrorKind::WouldBlock {
                    last = format!("accept: {e}");
                }
                ensure!(
                    !lost(link)?,
                    "RDMA pair: the worker exited before a valid bootstrap on {at} (last: {last})"
                );
                std::thread::sleep(POLL);
                continue;
            }
        };
        let admitted = (|| -> Result<_> {
            ensure!(
                link.worker
                    .is_none_or(|w| w.to_canonical() == from.ip().to_canonical()),
                "not the worker of the NCCL bootstrap (expected {who})"
            );
            let until = deadline.min(Instant::now() + timeout / 12);
            let mut stream = Bounded::new(stream, until)?;
            let remote = exchange(&mut stream, head, ident)?;
            // Admitted: bringing the QPs up and the barrier get the rest.
            stream.deadline = deadline;
            Ok((stream, remote))
        })();
        match admitted {
            Ok(pair) => return Ok(pair),
            Err(e) => {
                tracing::warn!("RDMA pair: rejected a bootstrap connection from {from}: {e:#}");
                last = format!("rejected {from}: {e:#}");
            }
        }
    }
    bail!("RDMA pair: no valid peer on {at} within {timeout:?} (last: {last})")
}

/// The worker: reach rank 0, which may not be listening yet, and exchange,
/// all within `timeout`. It stops retrying once rank 0 has exited.
fn dial(head: &Head, ident: &[u8], link: &Link, timeout: Duration) -> Result<(Bounded, Vec<u8>)> {
    let (at, deadline) = (link.at, Instant::now() + timeout);
    let stream = loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match TcpStream::connect_timeout(&at, left.max(RETRY)) {
            Ok(s) => break s,
            Err(e) => {
                ensure!(
                    !lost(link)?,
                    "RDMA pair: rank 0 exited before listening on {at} (its log says why)"
                );
                ensure!(left > RETRY, "RDMA pair: connect {at}: {e}");
                tracing::debug!("RDMA pair connect {at}: {e}; retrying");
                std::thread::sleep(RETRY);
            }
        }
    };
    let mut stream = Bounded::new(stream, deadline)?;
    let remote = exchange(&mut stream, head, ident)?;
    Ok((stream, remote))
}

#[cfg(test)]
#[path = "bootstrap_tests.rs"]
mod tests;
