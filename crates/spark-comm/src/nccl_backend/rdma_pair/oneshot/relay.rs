// SPDX-License-Identifier: AGPL-3.0-only

//! The chain proxy's side of the one-shot channel. It sends each staged op
//! to both neighbours, and forwards every payload that arrives from one
//! neighbour to the other, so every rank receives every other rank's payload
//! (rank 3's reaches rank 0 through ranks 2 and 1, proxy to proxy, without
//! waiting for a GPU on the way). Data and flag go chained per rail
//! (`ATLAS_RDMA_PAIR_CHAIN=1`): a forward needs no completion before the
//! next hop sees it, and nothing waits for a completion while other work is
//! due; `stage` is cleared once our own sends completed. A payload goes in
//! pieces (`pieces`, spread over the rails), each forwarded as soon as it
//! lands, so the hops of a large payload overlap.
//!
//! A forward of origin `o`'s op `seq` reads `o`'s receive slot `seq % 2`
//! here, which `o` rewrites only with `seq + 2`: after every rank's flag
//! `seq + 1`, including those of the ranks beyond this one, whose payloads
//! `seq + 1` they staged after their kernels `seq`, which waited for this
//! forward. By the same argument `o`'s flag here is at most one op ahead of
//! the forward: when it already shows `seq + 1`, the size of `seq` is our
//! own op `seq`'s (every rank issues the same ops), which we have staged
//! since (`o`'s kernel `seq` waited for it).
//!
//! A pipelined op (an all-gather, or a reduce-scatter under SM staging) has
//! the PIPE bit in `stage`, which its kernel publishes before staging: the
//! proxy then sends each of its pieces once the kernel shows it staged in its
//! `READY` word, so the copy into `S` overlaps the sends.
//!
//! A four-rank reduce-scatter op (`OneShot::reduce_scatter`, its sizes
//! carrying `RS_BIT`) is not flooded: each rank sends the pieces
//! [`rs_role`] names, raw from `S` once staged and, at the middle ranks,
//! computed into `S` by the kernel piece by piece (each published in its
//! `READY` word), and the middle ranks forward one segment of the other
//! middle rank's computed pieces. Every rank's kernel `seq` still waits,
//! through those pieces, for every other rank's kernel `seq - 1` and stage
//! `seq`, so the two-slot argument above holds across both kinds of op. An
//! origin whose reduce-scatter op sends nothing here is passed when its next
//! op's flags arrive, from this rank's own record of the op.

#[path = "relay/forward.rs"]
mod forward;

use super::super::{chain::link::Links, topology::Topology};
use super::{
    Config, FLAG_SRC, FWD_SRC, MAX_PIECES, MAX_RANKS, ORIGIN_LINES, OpKind, PIPE_BIT, READY, STAGE,
    Shape, chain_flag_line, ctrl_off, decode_size, fail_at, flag_word, fused, is_fused,
    origin_slot, pieces, recv_off, rs_per_segment, rs_wire_pieces,
};
use anyhow::{Result, ensure};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Our recent op sizes kept for forwards: we are at most one op past the
/// forward (module doc), so ours and the one before.
const SIZES: usize = 2;

/// What rank `rank` of four sends in a reduce-scatter op, as `(link side
/// (0 below, 1 above), segments)`: its raw segments, the segments its kernel
/// computes, and the origin whose segments it forwards (`oneshot_rs4`), each
/// segment as its [`rs_per_segment`] pieces ([`rs_segment_pieces`]). Raw
/// segments go in the order the far side needs them first.
pub(in crate::nccl_backend::rdma_pair) struct RsRole {
    pub raw: (usize, &'static [usize]),
    pub computed: Option<(usize, &'static [usize])>,
    pub forward: Option<(usize, &'static [usize])>,
}

/// Whether origin `o`'s reduce-scatter pieces reach rank `rank` (sent or
/// forwarded to it): the ends hear their neighbour and the far middle rank,
/// the middle ranks both neighbours.
fn rs_hears(rank: usize, o: usize) -> bool {
    match rank {
        0 => matches!(o, 1 | 2),
        1 => matches!(o, 0 | 2),
        2 => matches!(o, 1 | 3),
        _ => matches!(o, 1 | 2),
    }
}

pub(in crate::nccl_backend::rdma_pair) fn rs_role(rank: usize) -> RsRole {
    match rank {
        0 => RsRole {
            raw: (1, &[2, 3, 1]),
            computed: None,
            forward: None,
        },
        1 => RsRole {
            raw: (0, &[0]),
            computed: Some((1, &[2, 3])),
            forward: Some((2, &[0])),
        },
        2 => RsRole {
            raw: (1, &[3]),
            computed: Some((0, &[0, 1])),
            forward: Some((1, &[3])),
        },
        _ => RsRole {
            raw: (0, &[0, 1, 2]),
            computed: None,
            forward: None,
        },
    }
}

/// The pieces of `segments`, in order, at `k` pieces a segment.
pub(in crate::nccl_backend::rdma_pair) fn rs_segment_pieces(
    segments: &[usize],
    k: usize,
) -> impl Iterator<Item = usize> + '_ {
    segments.iter().flat_map(move |&s| s * k..(s + 1) * k)
}

pub(in crate::nccl_backend::rdma_pair) struct Relay {
    cfg: Config,
    shape: Shape,
    topology: Topology,
    /// Host address of this channel's one-shot region.
    host: usize,
    /// This channel's offset in every rank's chain region (the neighbour's
    /// region base, `Link::base`, plus this).
    chan_off: usize,
    served: u64,
    sizes: [usize; SIZES],
    /// Our op's sends in flight: (link, rail, ticket).
    sending: Vec<(usize, usize, u64)>,
    /// Per origin slot: the last op forwarded, and its sends' tickets per
    /// piece (their flag source words are rewritten only once those are
    /// done).
    forwarded: [u64; MAX_RANKS - 1],
    fwd_tickets: [[u64; MAX_PIECES]; MAX_RANKS - 1],
    /// The side each ticket belongs to: ring flood and linear specialized
    /// ops can forward the same origin on different QPs.
    fwd_sides: [[usize; MAX_PIECES]; MAX_RANKS - 1],
    /// Per origin slot: the pieces of the next op forwarded so far.
    fwd_sent: [u64; MAX_RANKS - 1],
    /// Our reduce-scatter op's computed pieces not yet sent: (side, piece).
    computed: Vec<(usize, usize)>,
    /// Channel 0's fused region, when the chain has one.
    fused: Option<RelayFused>,
}

/// This rank's fused region (`oneshot::fused`): its host address, its offset
/// in every rank's chain region, and the segment capacity.
#[derive(Clone, Copy, Debug)]
pub(in crate::nccl_backend::rdma_pair) struct RelayFused {
    pub host: usize,
    pub at: usize,
    pub cap: usize,
}

impl RelayFused {
    /// Host address of piece `j` (of `len` bytes) of slot `entry` here.
    fn local(&self, entry: usize, j: usize, len: usize) -> usize {
        self.host + entry * self.cap + j * len
    }

    /// Offset in rank `to`'s chain region of piece `j` of origin `o`'s
    /// segment `seg`.
    fn remote(&self, to: usize, o: usize, seg: usize, j: usize, len: usize) -> u64 {
        (self.at + (fused::nraw(to) + fused::recv_index(to, o, seg)) * self.cap + j * len) as u64
    }
}

/// The near origin a middle rank sums its own partial with.
fn near(rank: usize) -> usize {
    if rank == 1 { 0 } else { 3 }
}

impl Relay {
    pub(in crate::nccl_backend::rdma_pair) fn new(
        cfg: Config,
        shape: Shape,
        host: usize,
        chan_off: usize,
    ) -> Self {
        Self {
            cfg,
            shape,
            topology: Topology::Chain,
            host,
            chan_off,
            served: 0,
            sizes: [0; SIZES],
            sending: Vec::new(),
            forwarded: [0; MAX_RANKS - 1],
            fwd_tickets: [[0; MAX_PIECES]; MAX_RANKS - 1],
            fwd_sides: [[0; MAX_PIECES]; MAX_RANKS - 1],
            fwd_sent: [0; MAX_RANKS - 1],
            computed: Vec::new(),
            fused: None,
        }
    }

    pub(in crate::nccl_backend::rdma_pair) fn with_topology(mut self, topology: Topology) -> Self {
        self.topology = topology;
        self
    }

    /// Give channel 0's relay the chain's fused region.
    pub(in crate::nccl_backend::rdma_pair) fn with_fused(mut self, fused: RelayFused) -> Self {
        self.fused = Some(fused);
        self
    }

    /// Whether an op of size field `code` fits this channel's slots (a fused
    /// one its region).
    fn fits(&self, code: usize) -> bool {
        let bytes = decode_size(code).0;
        match self.fused {
            _ if !is_fused(code) => self.cfg.eligible(bytes),
            Some(f) => bytes > 0 && bytes <= MAX_RANKS * f.cap,
            None => false,
        }
    }

    /// For our fused op `served`: whether this middle rank's forwards of it
    /// all completed (`stage` stays set until then: `oneshot::fused` doc).
    fn fused_forwards_done(&self, links: &Links) -> bool {
        let code = self.sizes[self.served as usize % SIZES];
        let rank = self.shape.rank;
        let Some((o, seg)) = fused::forwarded(rank).filter(|_| is_fused(code)) else {
            return true;
        };
        let slot = origin_slot(o, rank);
        if self.forwarded[slot] < self.served {
            return false;
        }
        let Some(link) = links[usize::from(o < rank)].as_ref() else {
            return true;
        };
        let k = rs_per_segment(decode_size(code).0);
        (seg * k..(seg + 1) * k)
            .all(|p| link.qps[p % link.qps.len()].done(self.fwd_tickets[slot][p]))
    }

    fn ctrl(&self) -> usize {
        self.host + ctrl_off(self.cfg.max, self.shape.origins())
    }

    /// A control word of this rank's page.
    fn word<T>(&self, off: usize) -> &T {
        // SAFETY: the control page lives in the pinned region for the chain's
        // lifetime, 64-byte aligned per word; `T` is an atomic of its width.
        unsafe { &*((self.ctrl() + off) as *const T) }
    }

    /// Whether our own op is still being sent (`stage` not yet cleared).
    pub(in crate::nccl_backend::rdma_pair) fn busy(&self) -> bool {
        !self.sending.is_empty() || !self.computed.is_empty()
    }

    /// See [`super::Channel::fail`].
    pub(in crate::nccl_backend::rdma_pair) fn fail(&self) {
        fail_at(self.ctrl(), self.served);
    }

    /// One pass over the channel: clear `stage` once our op is sent, send a
    /// newly staged op, forward what arrived. `links` were reaped just
    /// before. `false` when there was nothing to do.
    pub(in crate::nccl_backend::rdma_pair) fn serve(&mut self, links: &mut Links) -> Result<bool> {
        let mut worked = false;
        let Shape { ranks, rank, .. } = self.shape;
        for o in (0..ranks).filter(|&o| o != rank) {
            worked |= self.forward(links, o)?;
        }
        if self.busy() {
            worked |= self.send_computed(links)?;
            if self.computed.is_empty()
                && self
                    .sending
                    .iter()
                    .all(|&(l, r, t)| links[l].as_ref().is_some_and(|k| k.qps[r].done(t)))
                && self.fused_forwards_done(links)
            {
                self.sending.clear();
                self.word::<AtomicU32>(STAGE).store(0, Ordering::Release);
                worked = true;
            }
            return Ok(worked);
        }
        Ok(self.send(links)? || worked)
    }

    /// Send piece `p` (`off`, `len`) of our op `seq` from `S` to the
    /// neighbour on `side`, flagged `code`.
    #[allow(clippy::too_many_arguments)]
    fn post_own(
        &mut self,
        links: &mut Links,
        side: usize,
        p: usize,
        (off, len): (usize, usize),
        seq: u64,
    ) -> Result<()> {
        let rank = self.shape.rank;
        let link = links[side].as_mut().ok_or_else(|| {
            anyhow::anyhow!("RDMA one-shot: rank {rank} has no link on side {side}")
        })?;
        let neighbour = self
            .topology
            .neighbor(rank, self.shape.ranks, side)
            .expect("a connected link has a neighbor");
        let slot = origin_slot(rank, neighbour);
        let (remote, rails) = (link.base + self.chan_off as u64, link.qps.len());
        let ctrl = ctrl_off(self.cfg.max, self.shape.origins());
        let parity = (seq & 1) as usize;
        let code = self.sizes[seq as usize % SIZES];
        // A fused op's raw piece leaves its staging slot, a computed one the
        // near payload it was summed over, for the neighbour's slot of our
        // segment (`oneshot::fused`).
        let (from, to) = match self.fused.filter(|_| is_fused(code)) {
            Some(f) => {
                let k = rs_per_segment(decode_size(code).0);
                let (seg, j) = (p / k, p % k);
                let entry = if rs_role(rank).raw.1.contains(&seg) {
                    fused::raw_index(rank, seg)
                } else {
                    fused::nraw(rank) + fused::recv_index(rank, near(rank), seg)
                };
                (
                    f.local(entry, j, len),
                    link.base + f.remote(neighbour, rank, seg, j, len),
                )
            }
            None => (
                self.host + off,
                remote + (recv_off(self.cfg.max, slot, parity) + off) as u64,
            ),
        };
        // SAFETY: as in `send` (a fused op's slots: `oneshot::fused`).
        let ticket = unsafe {
            link.qps[p % rails].write_flag(
                from,
                to,
                len,
                self.ctrl() + FLAG_SRC + 64 * p,
                remote + (ctrl + chain_flag_line(slot, p, seq)) as u64,
            )
        }?;
        self.sending.push((side, p % rails, ticket));
        Ok(())
    }

    /// Send the pieces of our op whose `READY` word shows them: a
    /// reduce-scatter's computed pieces, or a pipelined gather's pieces as
    /// they are staged.
    fn send_computed(&mut self, links: &mut Links) -> Result<bool> {
        if self.computed.is_empty() {
            return Ok(false);
        }
        let seq = self.served;
        let code = self.sizes[seq as usize % SIZES];
        let (bytes, kind) = decode_size(code);
        let parts = match kind {
            OpKind::Flood => pieces(bytes, self.cfg.stripe_min),
            OpKind::ReduceScatter | OpKind::AllToAll => rs_wire_pieces(code),
        };
        let mut worked = false;
        let mut i = 0;
        while i < self.computed.len() {
            let (side, p) = self.computed[i];
            if self
                .word::<AtomicU64>(READY + 8 * p)
                .load(Ordering::Acquire)
                != seq
            {
                i += 1;
                continue;
            }
            self.post_own(links, side, p, parts[p], seq)?;
            self.computed.swap_remove(i);
            worked = true;
        }
        Ok(worked)
    }

    /// Send a newly staged op to both neighbours.
    fn send(&mut self, links: &mut Links) -> Result<bool> {
        let staged = self.word::<AtomicU32>(STAGE).load(Ordering::Acquire) as usize;
        if staged == 0 {
            return Ok(false);
        }
        // A pipelined op publishes its pieces in `READY` as they are staged.
        let pipelined = staged & PIPE_BIT != 0;
        let code = staged & !PIPE_BIT;
        let (bytes, kind) = decode_size(code);
        let rs = kind == OpKind::ReduceScatter;
        // Only our own stream writes `stage`, with an eligible size; refuse
        // anything else rather than WRITE past a neighbour's receive slot.
        ensure!(
            self.fits(code)
                && (kind == OpKind::Flood || self.shape.ranks == MAX_RANKS)
                && (kind != OpKind::AllToAll || pipelined)
                && (!is_fused(code) || pipelined),
            "RDMA one-shot: staged size {staged:#x} is not eligible"
        );
        let (seq, rank) = (self.served + 1, self.shape.rank);
        let parity = (seq & 1) as usize;
        for r in 0..MAX_PIECES {
            // SAFETY: our flag source words; the previous op's WRITEs of
            // them completed before its `stage` was cleared.
            unsafe { ((self.ctrl() + FLAG_SRC + 64 * r) as *mut u64).write(flag_word(seq, code)) };
        }
        self.sizes[seq as usize % SIZES] = code;
        self.served = seq;
        if rs {
            let role = rs_role(rank);
            let parts = rs_wire_pieces(code);
            let k = rs_per_segment(bytes);
            let (side, raw) = role.raw;
            if pipelined {
                // Raw pieces too go as the kernel publishes them.
                self.computed = rs_segment_pieces(raw, k).map(|p| (side, p)).collect();
            } else {
                for p in rs_segment_pieces(raw, k) {
                    self.post_own(links, side, p, parts[p], seq)?;
                }
            }
            if let Some((side, computed)) = role.computed {
                self.computed
                    .extend(rs_segment_pieces(computed, k).map(|p| (side, p)));
            }
            self.send_computed(links)?;
            return Ok(true);
        }
        if kind == OpKind::AllToAll {
            // Segment `d` toward rank `d`, each piece as it is staged.
            let k = rs_per_segment(bytes);
            self.computed = (0..MAX_RANKS)
                .filter(|&d| d != rank)
                .flat_map(|d| (d * k..(d + 1) * k).map(move |p| (usize::from(d > rank), p)))
                .collect();
            self.send_computed(links)?;
            return Ok(true);
        }
        if pipelined {
            let n = pieces(bytes, self.cfg.stripe_min).len();
            self.computed = (0..links.len())
                .filter(|&side| links[side].is_some())
                .flat_map(|side| (0..n).map(move |p| (side, p)))
                .collect();
            self.send_computed(links)?;
            return Ok(true);
        }
        for (side, link) in links.iter_mut().enumerate() {
            let Some(link) = link else { continue };
            let neighbour = self
                .topology
                .neighbor(rank, self.shape.ranks, side)
                .expect("a connected link has a neighbor");
            let slot = origin_slot(rank, neighbour);
            let remote = link.base + self.chan_off as u64;
            let ctrl = ctrl_off(self.cfg.max, self.shape.origins());
            let rails = link.qps.len();
            for (p, &(off, len)) in pieces(bytes, self.cfg.stripe_min).iter().enumerate() {
                // SAFETY: `S` and the flag source word lie in the registered
                // region and stay unmodified until the ticket is done (the
                // GPU restages only after `stage` is cleared).
                let ticket = unsafe {
                    link.qps[p % rails].write_flag(
                        self.host + off,
                        remote + (recv_off(self.cfg.max, slot, parity) + off) as u64,
                        len,
                        self.ctrl() + FLAG_SRC + 64 * p,
                        remote + (ctrl + chain_flag_line(slot, p, seq)) as u64,
                    )
                }?;
                self.sending.push((side, p % rails, ticket));
            }
        }
        Ok(true)
    }
}
