// SPDX-License-Identifier: AGPL-3.0-only

//! Arguments of the RDMA chain's four-rank collectives on
//! [`CommBackend`](crate::CommBackend): the all-to-all's planes and the fused
//! reduce-scatter's plan, which a producer kernel takes by value.

/// One plane of [`crate::CommBackend::all_to_all_async`]: device addresses, and
/// 16-byte multiples for `row_bytes`, `pitch` and the strides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllToAllPlane {
    /// This rank's part for rank `d` at `src + d * src_stride`.
    pub src: u64,
    pub src_stride: i64,
    /// Rank `o`'s part lands at `dst + o * dst_stride`, rows `pitch` apart.
    pub dst: u64,
    pub dst_stride: i64,
    pub rows: usize,
    pub row_bytes: usize,
    pub pitch: usize,
}

/// One piece of a fused reduce-scatter's producer walk (`RsFusedPiece` in
/// `rdma_rs_fused.cuh`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RsFusedPiece {
    /// The piece's first row in the reduce-scatter's input (device).
    pub out: u64,
    /// Where a raw piece is staged (pinned); 0 for a piece kept in `out`.
    pub s: u64,
    /// The op's piece index.
    pub p: u32,
    pub pad: u32,
}

/// Most pieces a fused reduce-scatter op has (`ONESHOT_RS_MAX_PIECES`).
pub const RS_FUSED_MAX_PIECES: usize = 32;

/// What a fused reduce-scatter's producer kernel takes by value
/// (`RsFusedPlan` in `rdma_rs_fused.cuh`): the op's buffers, and its pieces
/// in the order the producer must finish them.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RsFusedPlan {
    pub ready: u64,
    pub state: u64,
    pub stage: u64,
    pub poison: u64,
    pub timeout_ns: u64,
    pub code: u32,
    pub npieces: u32,
    pub piece_bytes: u32,
    pub mode: u32,
    pub pad: u64,
    pub piece: [RsFusedPiece; RS_FUSED_MAX_PIECES],
}

impl RsFusedPlan {
    /// The plan as 8-byte kernel-argument words.
    pub fn words(&self) -> &[u64] {
        const WORDS: usize = std::mem::size_of::<RsFusedPlan>() / 8;
        const _: () = assert!(std::mem::size_of::<RsFusedPlan>() == 64 + 24 * RS_FUSED_MAX_PIECES);
        // SAFETY: repr(C), 8-byte aligned, a whole number of u64 words, no
        // padding bytes beyond the explicit `pad` fields.
        unsafe { std::slice::from_raw_parts((self as *const Self).cast::<u64>(), WORDS) }
    }
}

/// A fused reduce-scatter begun by [`crate::CommBackend::rs_fused_begin`]: the
/// producer's plan, then [`crate::CommBackend::reduce_scatter_fused_async`]; the
/// consumer waits per piece of its own segment
/// ([`crate::CommBackend::rs_fused_wait`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RsFused {
    pub plan: RsFusedPlan,
    /// u32 per piece of this rank's segment: `epoch` once its sum is final.
    pub done: u64,
    pub epoch: u32,
    /// Bytes of a piece (`plan.piece_bytes`), and pieces of a segment.
    pub piece_bytes: usize,
    pub pieces: usize,
    /// The op, as begun: segment `d` at `seg0 + d * stride`, `bytes` each.
    pub seg0: u64,
    pub stride: i64,
    pub bytes: usize,
}
