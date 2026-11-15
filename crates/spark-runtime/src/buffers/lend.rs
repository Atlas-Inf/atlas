// SPDX-License-Identifier: AGPL-3.0-only

//! The arena's lendable span: its per-pass prefill scratch carved from one
//! allocation, so a consumer that runs between passes (the GLM vision tower
//! on the encoding rank) can borrow it whole instead of holding scratch of
//! its own for the life of the server.
//!
//! Contract with a borrower, enforced by [`BufferArena::lend`] and
//! [`BufferArena::return_lent`]:
//! - it runs between forward passes, on the stream it names, after every
//!   earlier pass on any stream has finished (the arena synchronizes the
//!   device before lending);
//! - nothing in the span outlives a pass, so no pass reads what a borrower
//!   wrote: every member is written before it is read within a pass;
//! - on return, what the borrower took of the members `zero_all` zeroes is
//!   zeroed on its stream and that stream waited for, so a pass enqueued
//!   afterwards finds every such member at least as clean as `zero_dirty`'s
//!   tracking assumes (the rows it counts as zero are zero, the others are
//!   now zero too); the leading members it never zeroes ([`UNZEROED`]) are
//!   written before they are read in a pass.

use super::BufferArena;
use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Result, ensure};

/// Alignment of every member inside the span: past the 256 B a `cuMemAlloc`
/// guarantees, so vector and TMA loads see the alignment they had as
/// separate allocations.
pub(super) const LEND_ALIGN: usize = 4096;

/// Leading members of the span that `zero_all` never zeroes (FP8 activation
/// bytes and scales, the FP32 router input and gate logits): a borrower's
/// bytes there need no zeroing on return.
pub(super) const UNZEROED: usize = 4;

/// Byte offsets of `sizes` packed at [`LEND_ALIGN`] and their total.
pub(super) fn layout(sizes: &[usize]) -> (Vec<usize>, usize) {
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut at = 0usize;
    for &bytes in sizes {
        offsets.push(at);
        at += bytes.next_multiple_of(LEND_ALIGN);
    }
    (offsets, at)
}

/// One allocation holding `sizes`, and each member's pointer (NULL for a
/// zero-byte member, as the separate allocations were).
pub(super) fn carve(
    gpu: &dyn GpuBackend,
    sizes: &[usize],
) -> Result<(DevicePtr, usize, Vec<DevicePtr>)> {
    let (offsets, total) = layout(sizes);
    if total == 0 {
        return Ok((DevicePtr::NULL, 0, vec![DevicePtr::NULL; sizes.len()]));
    }
    let base = gpu.alloc(total)?;
    let members = sizes
        .iter()
        .zip(&offsets)
        .map(|(&bytes, &at)| {
            if bytes == 0 {
                DevicePtr::NULL
            } else {
                base.offset(at)
            }
        })
        .collect();
    Ok((base, total, members))
}

impl super::BufferSizes {
    /// The lendable members' sizes, in span order: the [`UNZEROED`] ones
    /// first, so a borrower's prefix reaches the zeroed members as late as it
    /// can. The arena carves them in this order (`BufferArena::new`).
    pub(super) fn lent_sizes(&self) -> [usize; 15] {
        [
            self.fp8_act,
            self.fp8_act_scale,
            self.moe_router_in_f32,
            self.gate_logits_f32,
            self.qkv_output,
            self.attn_output,
            self.norm_output,
            self.moe_output,
            self.ssm_qkvz,
            self.ssm_ba,
            self.ssm_deinterleaved,
            self.ssm_conv_out_f32,
            self.expert_gate_out,
            self.expert_up_out,
            self.expert_down_out,
        ]
    }
}

impl BufferArena {
    /// Bytes a borrower can take from the lendable span.
    pub fn lendable_bytes(&self) -> usize {
        self.lend_bytes
    }

    /// The span's start: fixed for the arena's life, so a borrower can bind
    /// its buffers once and borrow around each use.
    pub fn lend_base(&self) -> DevicePtr {
        self.lend_base
    }

    /// Lend the whole span for work enqueued on `stream` after this returns.
    /// Every earlier pass has finished when it does.
    pub fn lend(&self, bytes: usize, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
        ensure!(
            bytes <= self.lend_bytes,
            "arena lends {} bytes, a borrower asked for {bytes}",
            self.lend_bytes
        );
        gpu.synchronize_device()?;
        Ok(self.lend_base)
    }

    /// Take back the first `bytes` of the span from work enqueued on
    /// `stream`: zero there what overlaps members `zero_all` zeroes, and wait
    /// for `stream`, so a pass enqueued afterwards on any stream finds them
    /// as `zero_dirty` assumes. Other streams' work (spill and checkpoint
    /// copies) never touches the span, so it is not waited for. The wait
    /// runs even when the memset fails; on any failure the arena forgets its
    /// dirty extent, so the next `zero_dirty` zeroes every member whole.
    pub fn return_lent(&self, bytes: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let end = bytes.min(self.lend_bytes);
        let zeroed = if end > self.lend_unzeroed {
            gpu.memset_zero_async(
                self.lend_base.offset(self.lend_unzeroed),
                end - self.lend_unzeroed,
                stream,
            )
        } else {
            Ok(())
        };
        let waited = gpu.synchronize(stream);
        if zeroed.is_err() || waited.is_err() {
            self.note_rows(usize::MAX);
        }
        zeroed.and(waited)
    }
}

#[cfg(test)]
mod tests {
    use super::{LEND_ALIGN, layout};

    #[test]
    fn members_are_aligned_and_disjoint() {
        let (offsets, total) = layout(&[1, 0, LEND_ALIGN, LEND_ALIGN + 1, 7]);
        assert_eq!(
            offsets,
            vec![0, LEND_ALIGN, LEND_ALIGN, 2 * LEND_ALIGN, 4 * LEND_ALIGN]
        );
        assert_eq!(total, 5 * LEND_ALIGN);
        assert!(offsets.iter().all(|o| o % LEND_ALIGN == 0));
    }

    #[test]
    fn an_empty_span_takes_nothing() {
        assert_eq!(layout(&[0, 0]), (vec![0, 0], 0));
    }
}
