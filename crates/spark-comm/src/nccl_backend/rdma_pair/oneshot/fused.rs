// SPDX-License-Identifier: AGPL-3.0-only

//! The fused four-rank reduce-scatter's layout (`oneshot_rs4_fused`,
//! `rdma_rs_fused.cuh`): one op of a whole MoE seam, its input rows written
//! and published piece by piece by the producer kernel that computes them.
//!
//! The chain's pinned region ends with one fused region (channel 0's ops
//! only), single-buffered: a rank's raw segments' staging ([`raw_index`]),
//! then one segment per `(origin, segment)` it receives ([`recv_index`]), in
//! [`entries`] slots of the agreed segment capacity. The routes are
//! `relay::rs_role`'s; a middle rank sums its pair in place over the near
//! payload and sends it on from there, so its staging holds only its one raw
//! segment. At 16 MiB segments: 80 MiB at the ends, 96 MiB in the middle.
//!
//! A neighbour rewrites a slot only with its next fused op, and a fused op
//! is refused until an op every rank hears from every neighbour (anything
//! but a broadcast) ran on the channel since the last one: the neighbour's
//! that op waited for this rank's, which this rank's stream issued after the
//! kernel that read the slot. A middle rank's proxy clears `stage` only once
//! its forwards of the op completed too.

/// Raw segments a rank stages: the ends three, the middle ranks one.
pub(in crate::nccl_backend::rdma_pair) fn nraw(rank: usize) -> usize {
    if rank == 0 || rank == 3 { 3 } else { 1 }
}

/// The staging slot of rank `rank`'s raw segment `seg`.
pub(in crate::nccl_backend::rdma_pair) fn raw_index(rank: usize, seg: usize) -> usize {
    match rank {
        0 => seg - 1,
        3 => seg,
        _ => 0,
    }
}

/// The slot (after the staging) of origin `origin`'s segment `seg` at rank
/// `rank` (`oneshot_fused_recv`): the ends hear the raw near segment and the
/// far pair of their own; rank 1 x0's segments 1..3 and b's 0..1, rank 2
/// x3's 0..2 and a's 2..3.
pub(in crate::nccl_backend::rdma_pair) fn recv_index(
    rank: usize,
    origin: usize,
    seg: usize,
) -> usize {
    match rank {
        0 => usize::from(origin != 1),
        1 => {
            if origin == 0 {
                seg - 1
            } else {
                3 + seg
            }
        }
        2 => {
            if origin == 3 {
                seg
            } else {
                3 + seg - 2
            }
        }
        _ => usize::from(origin != 2),
    }
}

/// Slots of rank `rank`'s fused region.
pub(in crate::nccl_backend::rdma_pair) fn entries(rank: usize) -> usize {
    nraw(rank) + if rank == 0 || rank == 3 { 2 } else { 5 }
}

/// The origin whose pieces a middle rank forwards in a fused op (b down at
/// rank 1, a up at rank 2), and the segment.
pub(in crate::nccl_backend::rdma_pair) fn forwarded(rank: usize) -> Option<(usize, usize)> {
    match rank {
        1 => Some((2, 0)),
        2 => Some((1, 3)),
        _ => None,
    }
}

/// A fused op's size field: `RS_BIT`, `FUSED_BIT` and the payload in
/// 512-byte units from bit 3.
pub(in crate::nccl_backend::rdma_pair) fn code(bytes: usize) -> usize {
    super::RS_BIT | super::FUSED_BIT | (bytes / 512) << 3
}
