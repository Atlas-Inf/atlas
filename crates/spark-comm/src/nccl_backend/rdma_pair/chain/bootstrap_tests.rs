// SPDX-License-Identifier: AGPL-3.0-only

//! Chain bootstrap tests: four ranks over loopback TCP, as the NCCL
//! bootstrap leaves them (every worker connected to rank 0).

use super::*;
use std::net::{Ipv4Addr, TcpListener};
use std::sync::mpsc;

const LOCAL: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

fn head(rank: usize, ranks: usize) -> Head {
    Head {
        rank,
        rails: [
            if rank > 0 { 2 } else { 0 },
            if rank + 1 < ranks { 2 } else { 0 },
        ],
        ranks,
        oneshot: [1 << 20, 1 << 20, 1],
        cmd: 1024,
        ar_split: 4 << 20,
        rs_fused: 0,
        topology: Topology::Chain,
    }
}

/// Rank `rank`'s identity for the link below (0) or above (1): its length
/// for the rails, filled with a byte naming rank and side.
fn ident(h: &Head, side: usize) -> Vec<u8> {
    vec![0x10 * h.rank as u8 + side as u8 + 1; ident_len(h.rails[side])]
}

/// The NCCL bootstrap's connections: rank 0's, one per worker, then each
/// worker's to rank 0, by rank.
fn nccl_conns(workers: usize) -> (Vec<TcpStream>, Vec<TcpStream>) {
    let listener = TcpListener::bind((LOCAL, 0)).unwrap();
    let mut ours = Vec::new();
    let mut theirs = Vec::new();
    for _ in 0..workers {
        theirs.push(TcpStream::connect(listener.local_addr().unwrap()).unwrap());
        ours.push(listener.accept().unwrap().0);
    }
    (ours, theirs)
}

/// A port that was free a moment ago (`rendezvous` binds it itself).
fn free_port() -> u16 {
    TcpListener::bind((LOCAL, 0))
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A rank's bootstrap result and the identities its `connect` was handed.
type Booted = (Result<()>, Option<[Vec<u8>; 2]>);

/// Run every rank's bootstrap (`heads[r]` for rank r) on its own thread.
fn boot(heads: Vec<Head>) -> Vec<Booted> {
    let (zero, workers) = nccl_conns(heads.len() - 1);
    let port = free_port();
    let mut conns: Vec<Vec<TcpStream>> = vec![zero];
    conns.extend(workers.into_iter().map(|c| vec![c]));
    let runs: Vec<_> = heads
        .into_iter()
        .zip(conns)
        .map(|(h, conns)| {
            std::thread::spawn(move || {
                let at = Rendezvous::of(h.rank, &conns, port).unwrap();
                let (tx, rx) = mpsc::channel();
                let idents = [ident(&h, 0), ident(&h, 1)];
                let end = rendezvous(&h, idents, &at, |theirs| {
                    tx.send(theirs).unwrap();
                    Ok(())
                });
                // A failed rank exits, which ends its NCCL connections.
                drop(conns);
                (end, rx.try_recv().ok())
            })
        })
        .collect();
    runs.into_iter().map(|r| r.join().unwrap()).collect()
}

#[test]
fn four_ranks_get_their_neighbours_identities() {
    let heads: Vec<Head> = (0..4).map(|r| head(r, 4)).collect();
    let got = boot(heads.clone());
    for (r, (end, theirs)) in got.into_iter().enumerate() {
        end.unwrap();
        let theirs = theirs.expect("connect ran");
        let below = if r > 0 {
            ident(&heads[r - 1], 1)
        } else {
            Vec::new()
        };
        let above = heads.get(r + 1).map_or_else(Vec::new, |h| ident(h, 0));
        assert_eq!(theirs, [below, above], "rank {r}");
    }
}

fn ring_heads() -> Vec<Head> {
    (0..4)
        .map(|rank| Head {
            rails: [2, 2],
            topology: Topology::Ring,
            ..head(rank, 4)
        })
        .collect()
}

#[test]
fn ring_bootstraps_the_wraparound_link() {
    let heads = ring_heads();
    for (rank, (end, theirs)) in boot(heads.clone()).into_iter().enumerate() {
        end.unwrap();
        assert_eq!(
            theirs.unwrap(),
            [
                ident(&heads[(rank + 3) % 4], 1),
                ident(&heads[(rank + 1) % 4], 0)
            ]
        );
    }
}

#[test]
fn ring_rejects_a_wraparound_rail_mismatch() {
    let mut heads = ring_heads();
    heads[3].rails[1] = 1;
    let got = boot(heads);
    let said = format!("{:#}", got[0].0.as_ref().unwrap_err());
    assert!(said.contains("link between ranks 3 and 0"), "{said}");
    assert!(got.iter().all(|(_, theirs)| theirs.is_none()));
}

#[test]
fn mixed_topologies_fail_before_any_qp_connects() {
    let mut heads = ring_heads();
    heads[2].topology = Topology::Chain;
    let got = boot(heads);
    let said = format!("{:#}", got[0].0.as_ref().unwrap_err());
    assert!(
        said.contains("topology:") && said.contains("ATLAS_RDMA_TOPOLOGY"),
        "{said}"
    );
    assert!(got.iter().all(|(_, theirs)| theirs.is_none()));
}

#[test]
fn a_rail_mismatch_names_the_link_on_rank0_and_ends_every_rank() {
    let mut heads: Vec<Head> = (0..4).map(|r| head(r, 4)).collect();
    heads[2].rails[0] = 1;
    let got = boot(heads);
    let said = format!("{:#}", got[0].0.as_ref().unwrap_err());
    assert!(
        said.contains("the link between ranks 1 and 2: rank 1 has 2 rails")
            && said.contains("rank 2 has 1 (ATLAS_RDMA_LEFT_RAILS)"),
        "{said}"
    );
    for (r, (end, theirs)) in got.iter().enumerate().skip(1) {
        let said = format!("{:#}", end.as_ref().unwrap_err());
        assert!(said.contains("its log says why"), "rank {r}: {said}");
        assert!(theirs.is_none(), "rank {r} connected");
    }
}

#[test]
fn a_rank_that_disagrees_is_named_and_nobody_connects() {
    let mut heads: Vec<Head> = (0..4).map(|r| head(r, 4)).collect();
    heads[3].cmd = 0;
    let got = boot(heads);
    let said = format!("{:#}", got[0].0.as_ref().unwrap_err());
    assert!(
        said.contains("a worker exited")
            && said.contains("command ring: rank 0 has 1024, rank 3 0 - check ATLAS_GLM_CMD_RDMA"),
        "{said}"
    );
    assert!(got.iter().all(|(_, theirs)| theirs.is_none()));
}

#[test]
fn three_ranks_boot_and_two_workers_from_one_address_are_told_apart() {
    // Every worker here dials from 127.0.0.1: ranks are told apart by head.
    let heads: Vec<Head> = (0..3).map(|r| head(r, 3)).collect();
    for (r, (end, theirs)) in boot(heads).into_iter().enumerate() {
        end.unwrap();
        assert_eq!(theirs.unwrap().iter().filter(|i| !i.is_empty()).count(), {
            if r == 1 { 2 } else { 1 }
        });
    }
}

#[test]
fn heads_round_trip_and_refuse_another_format() {
    let h = head(2, 4);
    let mut wire = std::io::Cursor::new(h.wire().to_vec());
    assert_eq!(Head::read(&mut wire).unwrap(), h);
    let mut pair = std::io::Cursor::new(b"ATLPAIR5".repeat(9));
    let said = format!("{:#}", Head::read(&mut pair).unwrap_err());
    assert!(said.contains("different build"), "{said}");
    assert_eq!(ident_len(0), 0);
    assert_eq!(ident_len(2), 8 + 2 * 28);
}

#[test]
fn a_head_with_rails_where_no_link_is_or_too_many_is_refused() {
    let read = |h: &Head| Head::read(&mut std::io::Cursor::new(h.wire().to_vec()));
    let mut h = head(1, 4);
    h.rails[1] = usize::MAX / 8;
    let said = format!("{:#}", read(&h).unwrap_err());
    assert!(said.contains("rails to the rank above"), "{said}");
    let mut end = head(0, 4);
    end.rails[0] = 2;
    assert!(read(&end).is_err());
    let mut end = head(3, 4);
    end.rails[1] = 1;
    assert!(read(&end).is_err());
    let mut middle = head(2, 4);
    middle.rails[0] = 0;
    assert!(read(&middle).is_err());
    for (rank, ranks) in [(4, 4), (0, 2), (0, 5)] {
        let mut h = head(rank.min(1), 4);
        (h.rank, h.ranks) = (rank, ranks);
        assert!(read(&h).is_err(), "rank {rank} of {ranks}");
    }
    assert!(read(&head(3, 4)).is_ok() && read(&head(1, 3)).is_ok());
}

#[test]
fn a_rank_whose_own_shape_is_off_never_dials() {
    let mut heads: Vec<Head> = (0..4).map(|r| head(r, 4)).collect();
    heads[3].rails[1] = 2;
    let got = boot(heads);
    let said = format!("{:#}", got[3].0.as_ref().unwrap_err());
    assert!(
        said.contains("rank 3 of 4 has 2 rails to the rank above"),
        "{said}"
    );
    assert!(got.iter().all(|(_, theirs)| theirs.is_none()));
}
