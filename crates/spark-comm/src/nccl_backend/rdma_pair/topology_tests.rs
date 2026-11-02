// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn topology_is_explicit_and_ring_requires_four_ranks() {
    assert_eq!(Topology::parse(None, 4).unwrap(), Topology::Chain);
    assert_eq!(Topology::parse(Some("chain"), 3).unwrap(), Topology::Chain);
    assert_eq!(Topology::parse(Some("ring"), 4).unwrap(), Topology::Ring);
    for value in ["", "mesh", "RING"] {
        assert!(Topology::parse(Some(value), 4).is_err(), "{value:?}");
    }
    assert!(Topology::parse(Some("ring"), 3).is_err());
    assert!(Topology::parse(Some("ring"), 2).is_err());
    assert!(Topology::from_wire(2).is_err());
}

#[test]
fn ring_connects_wrapping_neighbors_and_chain_keeps_its_ends() {
    assert_eq!(Topology::Ring.neighbor(0, 4, 0), Some(3));
    assert_eq!(Topology::Ring.neighbor(3, 4, 1), Some(0));
    assert_eq!(Topology::Chain.neighbor(0, 4, 0), None);
    assert_eq!(Topology::Chain.neighbor(3, 4, 1), None);
    for rank in 0..4 {
        for side in 0..2 {
            let peer = Topology::Ring.neighbor(rank, 4, side).unwrap();
            assert_eq!(Topology::Ring.neighbor(peer, 4, 1 - side), Some(rank));
        }
    }
}

#[test]
fn every_flood_origin_reaches_every_peer_exactly_once() {
    for topology in [Topology::Chain, Topology::Ring] {
        for origin in 0..4 {
            let mut arrivals = [0; 4];
            let mut pending: Vec<_> = (0..2)
                .filter_map(|side| topology.neighbor(origin, 4, side).map(|r| (r, 1)))
                .collect();
            let mut max_hops = 0;
            while let Some((rank, hops)) = pending.pop() {
                arrivals[rank] += 1;
                assert!(
                    arrivals[rank] <= 1,
                    "duplicate/loop: {topology:?} {origin}->{rank}"
                );
                max_hops = max_hops.max(hops);
                if let Some(side) = topology.flood_forward(rank, origin, 4) {
                    pending.push((topology.neighbor(rank, 4, side).unwrap(), hops + 1));
                }
            }
            for (rank, count) in arrivals.into_iter().enumerate() {
                assert_eq!(count, usize::from(rank != origin));
            }
            if topology == Topology::Ring {
                assert_eq!(max_hops, 2);
            }
        }
    }
}

#[test]
fn ring_floods_use_all_four_links() {
    let mut traversals = [[0; 4]; 4];
    for origin in 0..4 {
        for side in 0..2 {
            let rank = Topology::Ring.neighbor(origin, 4, side).unwrap();
            traversals[origin][rank] += 1;
            if let Some(side) = Topology::Ring.flood_forward(rank, origin, 4) {
                traversals[rank][Topology::Ring.neighbor(rank, 4, side).unwrap()] += 1;
            }
        }
    }
    assert!(traversals[0][3] > 0 && traversals[3][0] > 0);
    for rank in 0..4 {
        assert_eq!(traversals[rank].iter().sum::<usize>(), 3);
    }
}
