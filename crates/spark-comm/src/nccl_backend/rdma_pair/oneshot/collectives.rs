// SPDX-License-Identifier: AGPL-3.0-only

//! The chain's ops on the one-shot channel: each checks what it can serve and
//! launches one kernel ([`super::launch`]).

use super::launch::Land;
use super::*;

impl OneShot {
    /// A chain's in-place sum of every rank's `src[..bytes]`, the same on
    /// all, on `stream` and capturable: `false` (nothing enqueued) when the
    /// size or the call is ineligible or the kernel is missing (a chain only
    /// sums in place, `add` with `src == dst`); every rank sees the same
    /// answer.
    pub(in crate::nccl_backend) fn sum(
        &self,
        src: u64,
        dst: u64,
        bytes: usize,
        add: bool,
        stream: u64,
    ) -> Result<bool> {
        if !add || src != dst {
            return Ok(false);
        }
        self.launch(src, dst, bytes, add, Land::Sum, self.cfg.timeout_ns, stream)
    }

    /// A chain's all-gather of every rank's `base + rank * stride`
    /// `[..bytes]` into the same place on every rank, on `stream` and
    /// capturable; `false` as for [`Self::enqueue`], and on a pair.
    pub(in crate::nccl_backend) fn gather(
        &self,
        base: u64,
        stride: i64,
        bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks == 2 {
            return Ok(false);
        }
        let own = base.wrapping_add_signed(stride * self.shape.rank as i64);
        self.launch(
            own,
            base,
            bytes,
            false,
            Land::Gather(stride),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// The four-rank chain's reduce-scatter: segment `d` of every rank's
    /// partial at `seg0 + d * stride` (`bytes / 4` each), summed in the
    /// all-reduce's pairwise tree, lands in rank `d`'s own segment, on
    /// `stream` and capturable. `false` as for [`Self::enqueue`], and unless
    /// four ranks, 16-byte alignment and a payload of whole 64-byte pieces
    /// below [`RS_BIT`]. `fp8`: the pieces travel as E4M3 and f32 scales,
    /// summed in f32 (lossy, `oneshot_rs4_fp8`), and must be whole KiB.
    pub(in crate::nccl_backend) fn reduce_scatter(
        &self,
        seg0: u64,
        stride: i64,
        bytes: usize,
        fp8: bool,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks != MAX_RANKS
            || !bytes.is_multiple_of(RS_PIECES * 64)
            || bytes >= RS_BIT
            || !seg0.is_multiple_of(16)
            || stride % 16 != 0
            || fp8 && !bytes.is_multiple_of(MAX_RANKS * rs_per_segment(bytes) * 1024)
        {
            return Ok(false);
        }
        // The kernel takes `add` as the reduce-scatter's FP8 switch.
        self.launch(
            seg0,
            seg0,
            bytes,
            fp8,
            Land::ReduceScatter(stride),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// The four-rank chain's all-to-all of `a2a` (`CommBackend::
    /// all_to_all_async`, one op: its four segments within the one-shot
    /// maximum) on `stream`, capturable; `false` as for [`Self::enqueue`],
    /// and unless four ranks and a shape [`A2a::valid`] admits.
    pub(in crate::nccl_backend) fn all_to_all(&self, a2a: A2a, stream: u64) -> Result<bool> {
        let bytes = MAX_RANKS * a2a.segment();
        if self.shape.ranks != MAX_RANKS || !a2a.valid() || bytes >= RS_BIT {
            return Ok(false);
        }
        self.launch(
            0,
            0,
            bytes,
            false,
            Land::AllToAll(a2a),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// Begin a fused reduce-scatter of `seg`-byte segments at `seg0 + d *
    /// stride` (`CommBackend::rs_fused_begin`): its producer plan, its
    /// pieces in the order the producer finishes them (at the ends the raw
    /// segments as `relay::rs_role` sends them, farthest first; in the
    /// middle the segments it sums on, then its raw one; its own last). `None` off channel 0 of a four-rank
    /// chain with a fused region, for a segment beyond its capacity, of
    /// misaligned pieces or of pieces not whole `unit`-byte producer rows
    /// ([`fused::split`]), and right after another fused op.
    pub(in crate::nccl_backend) fn rs_fused_begin(
        &self,
        seg0: u64,
        stride: i64,
        seg: usize,
        unit: usize,
    ) -> Result<Option<crate::RsFused>> {
        let Some(f) = self.fused else {
            return Ok(None);
        };
        let Some((k, piece)) = fused::split(seg, f.cap, unit) else {
            return Ok(None);
        };
        let bytes = MAX_RANKS * seg;
        if self.kernel.load(Ordering::Relaxed) == 0
            || !seg0.is_multiple_of(16)
            || stride % 16 != 0
            || (stride.unsigned_abs() as usize) < seg
            || self.fused_last.load(Ordering::Relaxed)
        {
            return Ok(None);
        }
        let rank = self.shape.rank;
        let role = relay::rs_role(rank);
        let ctrl = self.dev + ctrl_off(self.cfg.max, self.shape.origins()) as u64;
        let mut plan = crate::RsFusedPlan {
            ready: ctrl + READY as u64,
            state: self.state,
            stage: ctrl + STAGE as u64,
            poison: ctrl + POISON as u64,
            timeout_ns: self.cfg.timeout_ns,
            code: fused::code(bytes) as u32,
            npieces: (MAX_RANKS * k) as u32,
            piece_bytes: piece as u32,
            mode: self.cfg.sm_stage.max(1),
            pad: 0,
            piece: [crate::RsFusedPiece::default(); crate::RS_FUSED_MAX_PIECES],
        };
        // A middle rank produces the segments it sums on first: their raw
        // inputs arrive first (the ends send those first), and their pairs
        // feed the far side's sums, so its links fill from the start; its
        // raw segment (one hop, for an end's own sum) follows.
        let computed = role.computed.map_or(&[][..], |(_, segs)| segs);
        let order = if computed.is_empty() {
            [(role.raw.1, true), (computed, false), (&[rank][..], false)]
        } else {
            [(computed, false), (role.raw.1, true), (&[rank][..], false)]
        };
        let mut at = plan.piece.iter_mut();
        for (segs, raw) in order {
            for &g in segs {
                for j in 0..k {
                    let slot = at.next().context("fused reduce-scatter: too many pieces")?;
                    *slot = crate::RsFusedPiece {
                        out: seg0.wrapping_add_signed(g as i64 * stride) + (j * piece) as u64,
                        s: if raw {
                            f.dev + (fused::raw_index(rank, g) * f.cap + j * piece) as u64
                        } else {
                            0
                        },
                        p: (g * k + j) as u32,
                        pad: 0,
                    };
                }
            }
        }
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(Some(crate::RsFused {
            plan,
            done: f.done,
            epoch,
            piece_bytes: piece,
            pieces: k,
            seg0,
            stride,
            bytes: seg,
        }))
    }

    /// The chain kernel of the fused op `fused` on `stream`, capturable;
    /// `false` as for [`Self::enqueue`].
    pub(in crate::nccl_backend) fn reduce_scatter_fused(
        &self,
        fused: &crate::RsFused,
        stream: u64,
    ) -> Result<bool> {
        self.launch(
            fused.seg0,
            fused.seg0,
            MAX_RANKS * fused.bytes,
            false,
            Land::Fused(fused.stride, fused.epoch),
            self.cfg.timeout_ns,
            stream,
        )
    }

    /// `stream` waits until piece `piece` of this rank's segment of `fused`
    /// is final.
    pub(in crate::nccl_backend) fn rs_fused_wait(
        &self,
        fused: &crate::RsFused,
        piece: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            piece < fused.pieces,
            "fused reduce-scatter: piece {piece} of {}",
            fused.pieces
        );
        cu(
            unsafe {
                cuStreamWaitValue32_v2(
                    stream,
                    fused.done + 4 * piece as u64,
                    fused.epoch,
                    CU_STREAM_WAIT_VALUE_GEQ,
                )
            },
            "cuStreamWaitValue32(fused done)",
        )
    }

    /// A chain's broadcast of rank `root`'s `ptr[..bytes]` into every rank's
    /// `ptr`, on `stream` and capturable; `false` as for [`Self::enqueue`],
    /// and on a pair (its broadcasts stay on NCCL). It waits for the root
    /// without a limit, as ncclBroadcast does: a worker idles here until the
    /// head's next command, and the host times a slow broadcast afterwards.
    pub(in crate::nccl_backend) fn broadcast(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.shape.ranks == 2 || root >= self.shape.ranks {
            return Ok(false);
        }
        self.launch(ptr, ptr, bytes, false, Land::Root(root), 0, stream)
    }
}
