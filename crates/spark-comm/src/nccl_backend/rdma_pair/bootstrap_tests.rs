// SPDX-License-Identifier: AGPL-3.0-only

//! Bootstrap tests: the exchange over a scripted stream, then rank 0's
//! listener and the worker's dial over loopback TCP.

use super::*;
use std::net::Ipv4Addr;

const LOCAL: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

/// One end of the bootstrap connection: `peer` is what the other rank
/// sent, `sent` what we wrote.
struct Wire {
    peer: std::io::Cursor<Vec<u8>>,
    sent: Vec<u8>,
}

impl Read for Wire {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.peer.read(buf)
    }
}

impl Write for Wire {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.sent.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn head(rank: usize) -> Head {
    Head {
        rank,
        rails: 2,
        chain: false,
        segments: 4,
        capacity: 1 << 20,
        oneshot: [1 << 20, 1 << 20],
        cmd: 1024,
    }
}

/// A region base and two rails, recognisable per rank.
fn ident(rank: usize) -> Vec<u8> {
    vec![0xa0 + rank as u8; 8 + 2 * 28]
}

/// Everything `h`'s rank writes when the exchange succeeds.
fn script(h: &Head) -> Vec<u8> {
    [&h.wire()[..], &ident(h.rank)].concat()
}

/// `local`'s side against a peer that sent `peer`: the result and what we
/// wrote.
fn run(local: &Head, peer: Vec<u8>) -> (Result<Vec<u8>>, Vec<u8>) {
    let mut wire = Wire {
        peer: std::io::Cursor::new(peer),
        sent: Vec::new(),
    };
    let got = exchange(&mut wire, local, &ident(local.rank));
    (got, wire.sent)
}

fn why<T>(got: Result<T>) -> String {
    format!("{:#}", got.err().expect("the bootstrap must fail"))
}

#[test]
fn matching_ranks_swap_identities() {
    for rank in 0..2 {
        let (local, peer) = (head(rank), head(1 - rank));
        let (got, sent) = run(&local, script(&peer));
        assert_eq!(got.unwrap(), ident(1 - rank));
        assert_eq!(sent, script(&local));
    }
}

#[test]
fn each_differing_field_is_named_on_both_sides() {
    type Edit = fn(&mut Head);
    let cases: [(Edit, &str, &str); 7] = [
        (
            |h| h.rails = 1,
            "rail count: here 2, peer 1 - check ATLAS_RDMA_RAILS on both nodes",
            "rail count: here 1, peer 2 - check ATLAS_RDMA_RAILS on both nodes",
        ),
        (
            |h| h.chain = true,
            "PAIR_CHAIN: here 0, peer 1 - check ATLAS_RDMA_PAIR_CHAIN",
            "PAIR_CHAIN: here 1, peer 0",
        ),
        (
            |h| h.segments = 1,
            "segment count: here 4, peer 1 - check ATLAS_RDMA_PAIR_SEGMENTS",
            "segment count: here 1, peer 4",
        ),
        (
            |h| h.capacity = 2 << 20,
            "capacity: here 1048576, peer 2097152 - max_batch_tokens and the model must match",
            "capacity: here 2097152, peer 1048576",
        ),
        (
            |h| h.oneshot[0] = 0,
            "one-shot max: here 1048576, peer 0 - check ATLAS_RDMA_ONESHOT",
            "one-shot max: here 0, peer 1048576",
        ),
        (
            |h| h.oneshot[1] = 0,
            "one-shot stripe min: here 1048576, peer 0 - check ATLAS_RDMA_ONESHOT_STRIPE_MIN",
            "one-shot stripe min: here 0, peer 1048576",
        ),
        (
            |h| h.cmd = 0,
            "command ring: here 1024, peer 0 - check ATLAS_GLM_CMD_RDMA",
            "command ring: here 0, peer 1024",
        ),
    ];
    for (edit, here, there) in cases {
        for rank in 0..2 {
            let (good, mut odd) = (head(rank), head(1 - rank));
            edit(&mut odd);
            // Each side names the field and has sent its head, nothing more.
            for (local, peer, want) in [(&good, &odd, here), (&odd, &good, there)] {
                let (got, sent) = run(local, script(peer));
                let said = why(got);
                assert!(said.contains(want), "{said}");
                assert_eq!(sent, local.wire());
            }
        }
    }
}

#[test]
fn a_bad_or_reflected_head_gets_no_identity() {
    // Not this format (a scanner, the previous build): rank 0 says nothing,
    // the worker has sent its head only.
    for junk in [b"GET / HTTP/1.1\r\n".repeat(8), b"ATLPAIR3".repeat(16)] {
        let (got, sent) = run(&head(0), junk.clone());
        assert!(why(got).contains("different build"));
        assert!(sent.is_empty());
        let (got, sent) = run(&head(1), junk);
        assert!(why(got).contains("different build"));
        assert_eq!(sent, head(1).wire());
    }
    // A rank's own head played back to it.
    for rank in 0..2 {
        let local = head(rank);
        let (got, sent) = run(&local, script(&local));
        let said = why(got);
        assert!(
            said.contains(&format!("rank: here {rank}, peer {rank}")),
            "{said}"
        );
        assert_eq!(sent, local.wire());
    }
    // A peer that hangs up mid-head.
    let (got, sent) = run(&head(0), head(1).wire()[..20].to_vec());
    assert!(why(got).contains("closed the connection"));
    assert!(sent.is_empty());
}

#[test]
fn the_boot_timeout_is_two_minutes_unless_set_within_a_second_and_a_day() {
    let secs = |v: &str| boot_timeout(Some(v.to_owned())).as_secs();
    assert_eq!(boot_timeout(None).as_secs(), 120);
    assert_eq!(secs("30"), 30);
    // Zero is not "no deadline", and "forever" must not overflow an Instant.
    assert_eq!((secs("0"), secs(&u64::MAX.to_string())), (1, 86_400));
    let _ = Instant::now() + boot_timeout(Some(u64::MAX.to_string()));
    // Not a number of seconds: the default rather than a guess.
    assert_eq!((secs("30s"), secs("soon"), secs("-5")), (120, 120, 120));
}

#[test]
fn a_hang_up_reads_as_a_rejection_however_it_arrives() {
    // Dropped with bytes unread, a connection ends in a reset, not an EOF.
    for kind in [
        ErrorKind::UnexpectedEof,
        ErrorKind::ConnectionReset,
        ErrorKind::ConnectionAborted,
        ErrorKind::BrokenPipe,
    ] {
        let said = io("peer head")(kind.into()).to_string();
        let want = "peer head: the peer closed the connection";
        assert!(said.contains(want), "{said}");
    }
    let said = io("barrier")(ErrorKind::TimedOut.into()).to_string();
    assert!(said.ends_with("barrier: timed out"), "{said}");
}

#[test]
fn the_link_is_that_of_the_nccl_bootstrap_connection() {
    let own: IpAddr = Ipv4Addr::new(192, 0, 2, 2).into();
    let peer: IpAddr = Ipv4Addr::new(192, 0, 2, 1).into();
    // (rank 0, the worker): rank 0 listens on its own end and admits the
    // other; the worker dials the other end.
    assert_eq!(ends(0, own, peer), (own, peer));
    assert_eq!(ends(1, own, peer), (peer, own));
    // Only an address on another link is worth a warning on rank 0.
    assert!(elsewhere("198.51.100.2", own));
    for quiet in [
        "192.0.2.2",
        "::ffff:192.0.2.2",
        "127.0.0.1",
        "0.0.0.0",
        "head.example",
    ] {
        assert!(!elsewhere(quiet, own), "{quiet}");
    }
}

/// Nothing listens on the TCP multiplexer port.
const NOBODY: SocketAddr = SocketAddr::new(LOCAL, 1);

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// A loopback connection as the NCCL bootstrap leaves it: rank 0's end,
/// then the worker's.
fn nccl_conn() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind((LOCAL, 0)).unwrap();
    let worker = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    (listener.accept().unwrap().0, worker)
}

/// The worker's link to rank 0 at `at`.
fn to(at: SocketAddr) -> Link<'static> {
    Link {
        at,
        worker: None,
        lifeline: None,
    }
}

/// Rank 0's bootstrap, running.
type Rank0 = std::thread::JoinHandle<Result<(Bounded, Vec<u8>)>>;

/// Rank 0 on a loopback port of its own, listening before this returns:
/// where, and what it ends with.
fn rank0(
    worker: Option<IpAddr>,
    lifeline: Option<TcpStream>,
    timeout: Duration,
) -> (SocketAddr, Rank0) {
    let listener = bind((LOCAL, 0).into()).unwrap();
    let at = listener.local_addr().unwrap();
    let run = std::thread::spawn(move || {
        let link = Link {
            at,
            worker,
            lifeline: lifeline.as_ref(),
        };
        admit(&listener, &head(0), &ident(0), &link, timeout)
    });
    (at, run)
}

/// What a connection that writes `say` is sent before rank 0 drops it.
fn probe(at: SocketAddr, say: &[u8]) -> Vec<u8> {
    let mut stream = TcpStream::connect(at).unwrap();
    // Dropped with our bytes unread, the connection ends in a reset.
    let _ = stream.write_all(say);
    let mut got = Vec::new();
    let _ = stream.read_to_end(&mut got);
    got
}

/// Write `say` to `stream` a byte every `gap`, until the peer hangs up.
fn drip(mut stream: TcpStream, say: Vec<u8>, gap: Duration) {
    for byte in say {
        if stream.write_all(&[byte]).is_err() {
            break;
        }
        std::thread::sleep(gap);
    }
}

#[test]
fn a_pair_boots_over_tcp() {
    // `open` binds the port itself, so this one test has to name a port
    // that was free a moment ago.
    let port = (TcpListener::bind((LOCAL, 0)).unwrap().local_addr().unwrap()).port();
    let (conn0, conn1) = nccl_conn();
    let rank = |r: usize, conn: TcpStream| {
        std::thread::spawn(move || {
            let link = Link::of(r, &conn, port, "127.0.0.1").unwrap();
            open(&head(r), &ident(r), &link).unwrap()
        })
    };
    let (zero, one) = (rank(0, conn0), rank(1, conn1));
    let ((mut s0, got0), (mut s1, got1)) = (zero.join().unwrap(), one.join().unwrap());
    assert_eq!((got0, got1), (ident(1), ident(0)));
    // The stream then carries the barrier.
    let mut byte = [0u8];
    s0.write_all(&[1]).unwrap();
    s1.read_exact(&mut byte).unwrap();
    assert_eq!(byte, [1]);
}

#[test]
fn rejected_connections_do_not_stop_a_later_valid_peer() {
    let timeout = Duration::from_secs(12);
    let (at, zero) = rank0(None, None, timeout);
    // A scanner and a connection that never speaks are sent nothing; a
    // worker with one rail is sent the head that tells it so.
    assert!(probe(at, &[0x55; HEAD_WIRE]).is_empty());
    assert!(probe(at, &[]).is_empty());
    let mut odd = head(1);
    odd.rails = 1;
    assert_eq!(probe(at, &odd.wire()), head(0).wire());
    let (_, got1) = dial(&head(1), &ident(1), &to(at), timeout).unwrap();
    assert_eq!(
        (zero.join().unwrap().unwrap().1, got1),
        (ident(1), ident(0))
    );
}

#[test]
fn a_rail_mismatch_ends_both_ranks_at_once_with_its_name() {
    let (conn0, conn1) = nccl_conn();
    let (at, zero) = rank0(Some(LOCAL), Some(conn0), Duration::from_secs(60));
    let began = Instant::now();
    let mut odd = head(1);
    odd.rails = 1;
    let said = why(dial(&odd, &ident(1), &to(at), Duration::from_secs(60)));
    let want = "rail count: here 1, peer 2 - check ATLAS_RDMA_RAILS on both nodes";
    assert!(said.contains(want), "{said}");
    // The worker exits on that, which ends its NCCL bootstrap connection:
    // rank 0 does not wait out its window for a peer that cannot come.
    drop(conn1);
    let said = why(zero.join().unwrap());
    let want = "rail count: here 2, peer 1 - check ATLAS_RDMA_RAILS on both nodes";
    assert!(
        said.contains("the worker exited") && said.contains(want),
        "{said}"
    );
    assert!(began.elapsed() < Duration::from_secs(30));
}

#[test]
fn the_worker_stops_when_rank0_has_exited() {
    let (conn0, conn1) = nccl_conn();
    drop(conn0);
    let link = Link {
        lifeline: Some(&conn1),
        ..to(NOBODY)
    };
    let began = Instant::now();
    let said = why(dial(&head(1), &ident(1), &link, Duration::from_secs(60)));
    assert!(said.contains("rank 0 exited"), "{said}");
    assert!(began.elapsed() < Duration::from_secs(30));
}

#[test]
fn a_lifeline_check_leaves_the_connection_blocking() {
    let (conn0, mut conn1) = nccl_conn();
    assert!(!gone(&conn0).unwrap());
    // Its watcher's read must still wait for the end, not return at once.
    conn0.set_read_timeout(Some(ms(80))).unwrap();
    let began = Instant::now();
    assert!((&conn0).read(&mut [0u8]).is_err());
    assert!(began.elapsed() >= ms(60));
    // A stray byte is not the end; the peer closing is.
    conn1.write_all(&[7]).unwrap();
    assert_eq!(conn0.peek(&mut [0u8]).unwrap(), 1);
    assert!(!gone(&conn0).unwrap());
    (&conn0).read_exact(&mut [0u8]).unwrap();
    drop(conn1);
    while !gone(&conn0).unwrap() {
        assert!(began.elapsed() < Duration::from_secs(10));
        std::thread::sleep(ms(5));
    }
}

#[test]
fn a_silent_peer_times_out() {
    // The worker, against a listener that never answers.
    let quiet = TcpListener::bind((LOCAL, 0)).unwrap();
    let at = quiet.local_addr().unwrap();
    let got = dial(&head(1), &ident(1), &to(at), ms(200));
    assert!(why(got).contains("timed out"));
    // Rank 0, against a connection that never speaks.
    let (at, zero) = rank0(None, None, Duration::from_secs(3));
    let held = TcpStream::connect(at).unwrap();
    let said = why(zero.join().unwrap());
    assert!(
        said.contains("no valid peer") && said.contains("timed out"),
        "{said}"
    );
    drop(held);
}

#[test]
fn a_dripping_peer_cannot_outlast_the_deadline() {
    // Rank 0 gives a connection half a second of its six in all, however
    // often a byte arrives: the worker queued behind a 13 s drip is served.
    let (at, zero) = rank0(None, None, Duration::from_secs(6));
    let slow = TcpStream::connect(at).unwrap();
    let slow = std::thread::spawn(move || drip(slow, head(1).wire().to_vec(), ms(200)));
    let began = Instant::now();
    let (_, got1) = dial(&head(1), &ident(1), &to(at), Duration::from_secs(6)).unwrap();
    assert_eq!(
        (zero.join().unwrap().unwrap().1, got1),
        (ident(1), ident(0))
    );
    assert!(began.elapsed() < Duration::from_secs(4));
    slow.join().unwrap();
    // The worker's deadline covers the whole exchange, not each read.
    let fake = TcpListener::bind((LOCAL, 0)).unwrap();
    let at = fake.local_addr().unwrap();
    let slow = std::thread::spawn(move || {
        drip(fake.accept().unwrap().0, head(0).wire().to_vec(), ms(100))
    });
    let began = Instant::now();
    let said = why(dial(&head(1), &ident(1), &to(at), ms(500)));
    assert!(said.contains("timed out"), "{said}");
    assert!(began.elapsed() < Duration::from_secs(3));
    slow.join().unwrap();
}

#[test]
fn an_admitted_peer_has_the_rest_of_the_window_for_the_barrier() {
    // The ranks bring their QPs up between the exchange and the barrier,
    // which a candidate's half second here must not cut short.
    let timeout = Duration::from_secs(6);
    let (at, zero) = rank0(None, None, timeout);
    let (mut s1, _) = dial(&head(1), &ident(1), &to(at), timeout).unwrap();
    let (mut s0, _) = zero.join().unwrap().unwrap();
    std::thread::sleep(ms(700));
    s1.write_all(&[1]).unwrap();
    s0.read_exact(&mut [0u8]).unwrap();
}

#[test]
fn only_the_worker_of_the_nccl_bootstrap_is_admitted() {
    let other: IpAddr = Ipv4Addr::new(127, 0, 0, 2).into();
    let (conn0, conn1) = nccl_conn();
    let (at, zero) = rank0(Some(other), Some(conn0), Duration::from_secs(60));
    let began = Instant::now();
    assert!(probe(at, &head(1).wire()).is_empty());
    // The worker proper, turned away like this, exits: so does rank 0.
    drop(conn1);
    let said = why(zero.join().unwrap());
    let want = "not the worker of the NCCL bootstrap (expected 127.0.0.2)";
    assert!(
        said.contains("the worker exited") && said.contains(want),
        "{said}"
    );
    assert!(began.elapsed() < Duration::from_secs(30));
}

#[test]
fn rank0_listens_on_the_given_address_or_not_at_all() {
    let here = bind((LOCAL, 0).into()).unwrap();
    assert_eq!(here.local_addr().unwrap().ip(), LOCAL);
    // TEST-NET-1 is nobody's interface. There is no fallback to every
    // interface: the bind fails or, where non-local binds are allowed,
    // holds that very address.
    let at = SocketAddr::new(Ipv4Addr::new(192, 0, 2, 1).into(), 0);
    match bind(at) {
        Ok(listener) => assert_eq!(listener.local_addr().unwrap().ip(), at.ip()),
        Err(e) => assert!(format!("{e:#}").contains("bind 192.0.2.1")),
    }
}
