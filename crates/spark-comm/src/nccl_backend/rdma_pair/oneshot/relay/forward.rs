// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl Relay {
    /// Forward origin `o`'s next op on, each piece as soon as it landed.
    pub(super) fn forward(&mut self, links: &mut Links, o: usize) -> Result<bool> {
        let rank = self.shape.rank;
        if self.topology == Topology::Chain
            && self
                .topology
                .flood_forward(rank, o, self.shape.ranks)
                .is_none()
        {
            return Ok(false);
        }
        let slot = origin_slot(o, rank);
        let seq = self.forwarded[slot] + 1;
        // Our own reduce-scatter op `seq` that `o` sends nothing of here:
        // nothing to wait for (it would only show by `o`'s next op).
        if seq <= self.served
            && decode_size(self.sizes[seq as usize % SIZES]).1 == OpKind::ReduceScatter
            && !rs_hears(rank, o)
        {
            self.forwarded[slot] = seq;
            return Ok(true);
        }
        let here = self.ctrl();
        // SAFETY: as in `word`; the line is written by the neighbour's NIC.
        let flag = |p: usize| {
            unsafe { &*((here + chain_flag_line(slot, p, seq)) as *const AtomicU64) }
                .load(Ordering::Acquire)
        };
        // Every piece's flag names the op and its size.
        let Some(head) = (0..MAX_PIECES).map(flag).find(|f| f >> 24 >= seq) else {
            return Ok(false);
        };
        let code = if head >> 24 == seq {
            (head & 0xff_ffff) as usize
        } else {
            ensure!(
                self.served >= seq && self.served < seq + SIZES as u64,
                "RDMA one-shot: rank {o}'s flag {} is ahead of op {seq}, which this rank has not issued (ranks diverged)",
                head >> 24
            );
            self.sizes[seq as usize % SIZES]
        };
        let (bytes, kind) = decode_size(code);
        ensure!(
            self.fits(code),
            "RDMA one-shot: rank {o}'s op {seq} has an ineligible size {bytes}"
        );
        let route = if kind == OpKind::Flood {
            self.topology
        } else {
            Topology::Chain
        };
        let Some(side) = route.flood_forward(rank, o, self.shape.ranks) else {
            self.forwarded[slot] = seq;
            self.fwd_sent[slot] = 0;
            return Ok(true);
        };
        let next = route.neighbor(rank, self.shape.ranks, side).unwrap();
        // A flood op: every piece on; a reduce-scatter: the pieces this rank
        // forwards for `o`, if any; an all-to-all: `o`'s segments for the
        // ranks beyond this one.
        let (parts, wanted) = if kind == OpKind::AllToAll {
            let k = rs_per_segment(bytes);
            let beyond = |d: usize| if o < rank { d > rank } else { d < rank };
            let wanted = (0..MAX_RANKS)
                .filter(|&d| beyond(d))
                .flat_map(|d| d * k..(d + 1) * k)
                .fold(0u64, |m, p| m | 1 << p);
            (rs_wire_pieces(code), wanted)
        } else if kind == OpKind::ReduceScatter {
            let k = rs_per_segment(bytes);
            let wanted = match rs_role(rank).forward {
                Some((origin, segs)) if origin == o => {
                    rs_segment_pieces(segs, k).fold(0u64, |m, p| m | 1 << p)
                }
                _ => 0,
            };
            (rs_wire_pieces(code), wanted)
        } else {
            let parts = pieces(bytes, self.cfg.stripe_min);
            // At most MAX_PIECES (32) pieces.
            let all = (1u64 << parts.len()) - 1;
            (parts, all)
        };
        if wanted == 0 {
            self.forwarded[slot] = seq;
            self.fwd_sent[slot] = 0;
            return Ok(true);
        }
        let parity = (seq & 1) as usize;
        let local = self.host + recv_off(self.cfg.max, slot, parity);
        let ctrl = ctrl_off(self.cfg.max, self.shape.origins());
        let remote_slot = origin_slot(o, next);
        // A fused op's piece goes from our slot of `o`'s segment to the next
        // rank's (`oneshot::fused`).
        let fused_at = self
            .fused
            .filter(|_| is_fused(code))
            .map(|f| (f, next, rs_per_segment(bytes)));
        let mut worked = false;
        for (p, &(off, len)) in parts.iter().enumerate() {
            if wanted & 1 << p == 0 || self.fwd_sent[slot] & 1 << p != 0 || flag(p) >> 24 < seq {
                continue;
            }
            // The old flag-source word may still be read by the previous
            // op's QP, which need not be the current route's QP in a ring.
            let ticket = self.fwd_tickets[slot][p];
            if ticket > 0 {
                let old = links[self.fwd_sides[slot][p]].as_mut().unwrap();
                let rail = p % old.qps.len();
                old.qps[rail].wait(ticket)?;
            }
            let link = links[side].as_mut().unwrap();
            let remote = link.base + self.chan_off as u64;
            let rails = link.qps.len();
            let qp = &mut link.qps[p % rails];
            let src = self.ctrl() + FWD_SRC + ORIGIN_LINES * slot + 64 * p;
            // SAFETY: our forward source word for this origin and piece,
            // whose previous WRITE completed just above.
            unsafe { (src as *mut u64).write(flag_word(seq, code)) };
            // SAFETY: the receive slot holds `o`'s op `seq` until `o` stages
            // `seq + 2`, which waits for this forward (module doc); the flag
            // source word stays as written until the ticket is done.
            let (from, to) = match fused_at {
                Some((f, next, k)) => {
                    let (seg, j) = (p / k, p % k);
                    let entry = fused::nraw(rank) + fused::recv_index(rank, o, seg);
                    (
                        f.local(entry, j, len),
                        link.base + f.remote(next, o, seg, j, len),
                    )
                }
                None => (
                    local + off,
                    remote + (recv_off(self.cfg.max, remote_slot, parity) + off) as u64,
                ),
            };
            self.fwd_tickets[slot][p] = unsafe {
                qp.write_flag(
                    from,
                    to,
                    len,
                    src,
                    remote + (ctrl + chain_flag_line(remote_slot, p, seq)) as u64,
                )
            }?;
            self.fwd_sides[slot][p] = side;
            self.fwd_sent[slot] |= 1 << p;
            worked = true;
        }
        if self.fwd_sent[slot] == wanted {
            self.forwarded[slot] = seq;
            self.fwd_sent[slot] = 0;
        }
        Ok(worked)
    }
}
