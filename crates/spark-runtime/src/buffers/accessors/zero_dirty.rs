// SPDX-License-Identifier: AGPL-3.0-only

//! `zero_all` for an arena that is known to be zero past a prefix
//! (`ATLAS_GLM_ZERO_ROWS`, read by the model).
//!
//! A multi-rank prefill zeroes the whole arena before every pass, about
//! 3.5 GB at an 8K-row GLM-5 arena, whatever the pass computes. The large
//! buffers are row-major scratch a pass fills from the front, so after a
//! pass of a few dozen rows nearly all of each is still zero from the zero
//! before it.
//!
//! [`BufferArena::zero_dirty`] zeroes the small buffers whole and, of each
//! large one, the share of `dirty + floor` rows out of `max_batch_tokens`,
//! where `dirty` is the most rows a pass noted since the arena was last all
//! zero ([`BufferArena::note_rows`]; unknown, so everything, until the first
//! whole zero) and `floor` covers what writes without noting: decode and
//! verify steps, a row group of index logits over the context, per-expert
//! padding.
//!
//! `hidden_states` is the exception: a chunk embeds all its rows there, not
//! only the rows its pass computes and notes, so it is always zeroed whole.
//!
//! That is the same arena `zero_all` leaves only if no pass wrote past the
//! share. Nothing in the kernels enforces it (the buffers are shared scratch
//! and some layouts do not follow the row count), so the bound is measured,
//! not derived: [`BufferArena::stale_past_dirty`] reads what `zero_dirty`
//! would have left and reports any nonzero byte. A model runs that check
//! with `zero_all` after it (so it serves as without the switch) to qualify a
//! workload before it trims. What a clean check qualifies is that workload at
//! that arena size: which buffers are large enough to trim, and how far a
//! pass reaches into them, both change with `max_batch_tokens`.

use std::sync::atomic::Ordering;

use super::BufferArena;
use crate::gpu::{DevicePtr, GpuBackend};

/// Buffers under this size are zeroed whole by `zero_dirty`.
const TRIM_MIN_BYTES: usize = 128 << 20;

/// Bytes of a `bytes`-long buffer that `rows` of `capacity` rows span,
/// rounded up to a page; all of it under `min_bytes` or when the rows fill
/// the capacity.
pub(in crate::buffers) fn dirty_prefix(
    bytes: usize,
    rows: usize,
    capacity: usize,
    min_bytes: usize,
) -> usize {
    if bytes < min_bytes || rows >= capacity {
        return bytes;
    }
    let share = (bytes as u128 * rows as u128).div_ceil(capacity.max(1) as u128) as usize;
    share.next_multiple_of(4096).min(bytes)
}

impl BufferArena {
    /// The buffers `zero_all` zeroes, in its order: name, pointer, bytes.
    pub(in crate::buffers) fn zeroed(&self) -> [(&'static str, DevicePtr, usize); 18] {
        let s = &self.sizes;
        [
            ("hidden_states", self.hidden_states, s.hidden_states),
            ("residual", self.residual, s.residual),
            ("norm_output", self.norm_output, s.norm_output),
            ("qkv_output", self.qkv_output, s.qkv_output),
            ("attn_output", self.attn_output, s.attn_output),
            ("gate_logits", self.gate_logits, s.gate_logits),
            ("moe_output", self.moe_output, s.moe_output),
            ("ssm_qkvz", self.ssm_qkvz, s.ssm_qkvz),
            ("ssm_ba", self.ssm_ba, s.ssm_ba),
            (
                "ssm_deinterleaved",
                self.ssm_deinterleaved,
                s.ssm_deinterleaved,
            ),
            ("ssm_gates", self.ssm_gates, s.ssm_gates),
            (
                "ssm_conv_out_f32",
                self.ssm_conv_out_f32,
                s.ssm_conv_out_f32,
            ),
            (
                "splitk_workspace",
                self.splitk_workspace,
                s.splitk_workspace,
            ),
            ("expert_gate_out", self.expert_gate_out, s.expert_gate_out),
            ("expert_up_out", self.expert_up_out, s.expert_up_out),
            ("expert_down_out", self.expert_down_out, s.expert_down_out),
            ("logits", self.logits, s.logits),
            ("scratch", self.scratch, s.scratch),
        ]
    }

    /// A pass wrote up to `rows` rows of the arena (`usize::MAX`: unknown).
    pub fn note_rows(&self, rows: usize) {
        self.dirty_rows.fetch_max(rows, Ordering::Relaxed);
    }

    /// Each zeroed buffer with the bytes `zero_dirty` covers now.
    fn dirty_prefixes(
        &self,
        floor_rows: usize,
        min_bytes: usize,
    ) -> impl Iterator<Item = (&'static str, DevicePtr, usize, usize)> {
        let rows = self
            .dirty_rows
            .load(Ordering::Relaxed)
            .saturating_add(floor_rows);
        let (capacity, embedded) = (self.max_batch_tokens, self.hidden_states);
        self.zeroed().into_iter().map(move |(name, ptr, bytes)| {
            let prefix = if ptr == embedded {
                bytes
            } else {
                dirty_prefix(bytes, rows, capacity, min_bytes)
            };
            (name, ptr, bytes, prefix)
        })
    }

    /// Zero what may be dirty (see the module docs); the arena then counts
    /// as all zero.
    pub fn zero_dirty(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        floor_rows: usize,
    ) -> anyhow::Result<()> {
        self.zero_dirty_over(gpu, stream, floor_rows, TRIM_MIN_BYTES)
    }

    /// [`Self::zero_dirty`] trimming every buffer of `min_bytes` or more.
    pub(in crate::buffers) fn zero_dirty_over(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        floor_rows: usize,
        min_bytes: usize,
    ) -> anyhow::Result<()> {
        for (_, ptr, _, prefix) in self.dirty_prefixes(floor_rows, min_bytes) {
            gpu.memset_zero_async(ptr, prefix, stream)?;
        }
        self.dirty_rows.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// What `zero_dirty` would leave behind now: for each buffer with a
    /// nonzero byte past its prefix, its name, the offset of the first such
    /// byte and the prefix. Reads the device after draining `stream`; for
    /// qualification, not for serving.
    pub fn stale_past_dirty(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        floor_rows: usize,
    ) -> anyhow::Result<Vec<(&'static str, usize, usize)>> {
        self.stale_past_dirty_over(gpu, stream, floor_rows, TRIM_MIN_BYTES)
    }

    /// [`Self::stale_past_dirty`] for the prefixes of `zero_dirty_over`.
    pub(in crate::buffers) fn stale_past_dirty_over(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        floor_rows: usize,
        min_bytes: usize,
    ) -> anyhow::Result<Vec<(&'static str, usize, usize)>> {
        const PIECE: usize = 16 << 20;
        gpu.synchronize(stream)?;
        let mut host = Vec::new();
        let mut stale = Vec::new();
        for (name, ptr, bytes, prefix) in self.dirty_prefixes(floor_rows, min_bytes) {
            let mut at = prefix;
            while at < bytes {
                host.resize(PIECE.min(bytes - at), 0u8);
                gpu.copy_d2h(ptr.offset(at), &mut host)?;
                if let Some(i) = host.iter().position(|&b| b != 0) {
                    stale.push((name, at + i, prefix));
                    break;
                }
                at += host.len();
            }
        }
        Ok(stale)
    }
}
