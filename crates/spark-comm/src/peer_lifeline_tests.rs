// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Duration;

/// A connected (watched, peer) pair, like the NCCL bootstrap leaves behind.
fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (watched, _) = listener.accept().unwrap();
    (watched, peer)
}

fn watch(watched: TcpStream) -> (PeerLifeline, mpsc::Receiver<String>) {
    let (tx, rx) = mpsc::channel();
    let lifeline = PeerLifeline::new(vec![watched]);
    lifeline
        .watch(move |how| {
            let _ = tx.send(how);
        })
        .unwrap();
    (lifeline, rx)
}

#[test]
fn peer_exit_is_reported() {
    let (watched, peer) = pair();
    let (_lifeline, rx) = watch(watched);
    drop(peer);
    let how = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(how.contains("closed"), "{how}");
}

#[test]
fn live_peer_is_not_reported() {
    let (watched, _peer) = pair();
    let (_lifeline, rx) = watch(watched);
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(300)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
}

#[test]
fn stray_bytes_from_peer_are_not_loss() {
    let (watched, mut peer) = pair();
    let (_lifeline, rx) = watch(watched);
    peer.write_all(b"not a close").unwrap();
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(300)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
}

/// A rank that watches (or opts out) must keep its end open: the peer reads
/// any close as this rank's death.
#[test]
fn held_connection_stays_open_for_the_peer() {
    let (watched, mut peer) = pair();
    let (lifeline, _rx) = watch(watched);
    peer.set_read_timeout(Some(Duration::from_millis(300)))
        .unwrap();
    let err = peer.read(&mut [0u8; 1]).unwrap_err();
    assert!(
        matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut),
        "{err}"
    );
    drop(lifeline);
    let err = peer.read(&mut [0u8; 1]).unwrap_err();
    assert!(
        matches!(err.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut),
        "watcher threads keep the connection open after the handle drops: {err}"
    );
}
