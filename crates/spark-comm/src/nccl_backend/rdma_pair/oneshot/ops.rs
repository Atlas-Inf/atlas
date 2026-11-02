// SPDX-License-Identifier: AGPL-3.0-only

//! What a chain op's size field names: the flood's plain size, or the
//! reduce-scatter's and the all-to-all's codes, their pieces, and the
//! all-to-all's kernel argument.

use super::{MAX_RANKS, stripes};

/// A chain reduce-scatter op's flag and `stage` size carry this bit (its
/// payloads stay below it), so no proxy or kernel takes its pieces for a
/// flood op's ([`relay`]).
pub(in crate::nccl_backend::rdma_pair) const RS_BIT: usize = 1 << 23;
/// A `stage` word with this bit (above every size code): the kernel stages
/// the op piece by piece and publishes each in its `READY` word, and the
/// proxy sends each piece as it is published (the pipelined all-gather).
pub(super) const PIPE_BIT: usize = 1 << 31;
/// A reduce-scatter op's payload divides into pieces of every segment of
/// four ranks in 64-byte multiples at this granularity (two a segment).
pub(super) const RS_PIECES: usize = 8;
/// A reduce-scatter piece is cut no finer than this.
const RS_PIECE: usize = 128 << 10;

/// Pieces a segment of a `bytes` four-rank reduce-scatter op is cut into:
/// 8, 4 or 2, the most that leaves every piece whole 64-byte units of at
/// least [`RS_PIECE`] (2 below that), the same on every rank.
pub(super) fn rs_per_segment(bytes: usize) -> usize {
    let seg = bytes / MAX_RANKS;
    [8, 4]
        .into_iter()
        .find(|&k| seg.is_multiple_of(k * 64) && seg / k >= RS_PIECE)
        .unwrap_or(2)
}

/// A reduce-scatter op's pieces: [`rs_per_segment`] equal 64-byte
/// multiples of each segment, in segment order.
pub(super) fn rs_pieces(bytes: usize) -> Vec<(usize, usize)> {
    stripes(bytes, MAX_RANKS * rs_per_segment(bytes), 0)
}

/// An all-to-all op's size field: [`RS_BIT`] and this bit, which a
/// segmented payload (whole 512-byte units) never sets.
pub(super) const A2A_BIT: usize = 1;

/// A reduce-scatter op's size field carries this bit too when its pieces
/// travel as E4M3 with one f32 scale per 128 values (`oneshot_rs4_fp8`,
/// [`OneShot::reduce_scatter`] with `fp8`); the size stays the BF16 payload's.
pub(super) const FP8_BIT: usize = 2;

/// A fused reduce-scatter's size field carries this bit beside [`RS_BIT`],
/// and the payload in 512-byte units from bit 3 ([`fused::code`]): one op of
/// a whole seam is larger than the plain size field.
pub(super) const FUSED_BIT: usize = 4;

/// Whether size field `code` names a fused reduce-scatter.
pub(super) fn is_fused(code: usize) -> bool {
    code & RS_BIT != 0 && code & (A2A_BIT | FUSED_BIT) == FUSED_BIT
}

/// Wire bytes of `bytes` of BF16 (a multiple of 1024) as E4M3 values and
/// their f32 scales, one per 128 values.
pub(super) fn fp8_wire(bytes: usize) -> usize {
    bytes / 2 + bytes / 64
}

/// The `(offset, len)` pieces on the wire of the reduce-scatter or
/// all-to-all op whose size field is `code`: [`rs_pieces`], or under
/// [`FP8_BIT`] their [`fp8_wire`] images.
pub(super) fn rs_wire_pieces(code: usize) -> Vec<(usize, usize)> {
    let parts = rs_pieces(decode_size(code).0);
    if code & RS_BIT != 0 && code & (A2A_BIT | FP8_BIT) == FP8_BIT {
        parts
            .into_iter()
            .map(|(o, l)| (fp8_wire(o), fp8_wire(l)))
            .collect()
    } else {
        parts
    }
}

/// What a flag's or `stage`'s size field names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OpKind {
    /// Every rank's whole payload to every rank (all-reduce, gather,
    /// broadcast).
    Flood,
    /// The four-rank reduce-scatter (`relay::rs_role`).
    ReduceScatter,
    /// The four-rank all-to-all: segment `d` to rank `d`.
    AllToAll,
}

/// A flag's or `stage`'s size field: `(bytes, kind)`.
pub(super) fn decode_size(code: usize) -> (usize, OpKind) {
    match (code & RS_BIT != 0, code & A2A_BIT != 0) {
        (false, _) => (code, OpKind::Flood),
        (true, false) if code & FUSED_BIT != 0 => {
            (((code & (RS_BIT - 1)) >> 3) * 512, OpKind::ReduceScatter)
        }
        (true, false) => (code & !(RS_BIT | FP8_BIT), OpKind::ReduceScatter),
        (true, true) => (code & !(RS_BIT | A2A_BIT), OpKind::AllToAll),
    }
}

/// One plane of an all-to-all op, as the kernel takes it (`OneshotA2aPlane`
/// in `rdma_oneshot.cu`): this rank's part for rank `d` at `src + d *
/// src_stride`, `bytes` of `row_bytes` rows; rank `o`'s lands at `dst + o *
/// dst_stride`, rows `pitch` apart.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::nccl_backend) struct A2aPlane {
    pub src: u64,
    pub src_stride: i64,
    pub dst: u64,
    pub dst_stride: i64,
    pub row_bytes: u32,
    pub pitch: u32,
    pub bytes: u32,
    pub pad: u32,
}

/// An all-to-all op's two planes (an unused plane has no bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::nccl_backend) struct A2a {
    pub planes: [A2aPlane; 2],
}

impl A2a {
    /// Bytes of one segment (a rank's part for one rank, both planes).
    pub(in crate::nccl_backend) fn segment(&self) -> usize {
        self.planes.iter().map(|p| p.bytes as usize).sum()
    }

    /// Whether the kernel can take it: 16-byte rows, pitches, strides and
    /// addresses, whole rows, and a segment of whole 128-byte units.
    pub(in crate::nccl_backend) fn valid(&self) -> bool {
        let seg = self.segment();
        seg > 0
            && seg.is_multiple_of(128)
            && self.planes.iter().all(|p| {
                p.bytes == 0
                    || (p.row_bytes > 0
                        && p.row_bytes.is_multiple_of(16)
                        && p.pitch >= p.row_bytes
                        && p.pitch.is_multiple_of(16)
                        && p.bytes.is_multiple_of(p.row_bytes)
                        && [p.src, p.dst].iter().all(|a| a.is_multiple_of(16))
                        && [p.src_stride, p.dst_stride].iter().all(|s| s % 16 == 0))
            })
    }
}
