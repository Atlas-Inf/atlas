// SPDX-License-Identifier: AGPL-3.0-only

//! Buffers every Inkling 2 visit shares within one forward of up to
//! `max_rows` tokens of one sequence (a decode step or a prefill chunk): the
//! block stack, the prelude's saved output, the previous pass's message
//! states, per-row scratch, and the rows' metadata visit 0 uploads (positions,
//! slots, lengths, block table) for the other visits. The model runs prefill
//! layer-major, so every per-token state here is per row.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Model dimensions the buffers are cut for.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dims {
    pub hidden: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub heads: usize,
    pub rel_dim: usize,
    pub dyn_dim: usize,
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    pub max_blocks_per_seq: usize,
    /// Blocks the stack can hold (schedule max).
    pub stack_blocks: usize,
    /// Rows one forward may carry (the prefill chunk).
    pub max_rows: usize,
}

impl Dims {
    /// Elements between two blocks of the stack (block-major `[block, row, H]`).
    pub(crate) fn block_stride(&self) -> usize {
        self.max_rows * self.hidden
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Shared {
    // forward state, BF16; the stacks are block-major [block, max_rows, H]
    pub blocks: DevicePtr,
    pub prelude_h: DevicePtr,
    pub msg_blocks: DevicePtr,
    pub msg_h: DevicePtr,
    // fp32 scratch
    pub res3: DevicePtr,
    pub msg4: DevicePtr,
    pub resid: DevicePtr,
    pub mlp_in: DevicePtr,
    pub logits: DevicePtr,
    pub weights: DevicePtr,
    pub final_out: DevicePtr,
    // BF16 scratch
    pub normed3: DevicePtr,
    pub q: DevicePtr,
    pub k_lin: DevicePtr,
    pub v_lin: DevicePtr,
    pub r: DevicePtr,
    pub gate: DevicePtr,
    pub sink: DevicePtr,
    pub dyn_k: DevicePtr,
    pub dyn_v: DevicePtr,
    pub kc: DevicePtr,
    pub vc: DevicePtr,
    pub qn: DevicePtr,
    pub kn: DevicePtr,
    pub vn: DevicePtr,
    pub rn: DevicePtr,
    pub attn: DevicePtr,
    pub gated: DevicePtr,
    pub proj: DevicePtr,
    pub conv_out: DevicePtr,
    pub mlp_normed: DevicePtr,
    pub act: DevicePtr,
    pub y: DevicePtr,
    pub moe_out: DevicePtr,
    pub mlp_conv: DevicePtr,
    pub ids: DevicePtr,
    // row metadata (visit 0 uploads)
    pub positions: DevicePtr,
    pub slots: DevicePtr,
    pub seq_lens: DevicePtr,
    pub block_table: DevicePtr,
    /// conv bookkeeping for one sequence: row_seq all 0, seq_start [0, rows], slot 0
    pub row_seq: DevicePtr,
    pub seq_start: DevicePtr,
    pub conv_slot: DevicePtr,
}

/// Carves one allocation into 256-byte aligned pieces.
struct Arena {
    base: DevicePtr,
    used: usize,
}

impl Arena {
    /// The next piece; NULL while only sizing (a NULL base).
    fn take(&mut self, bytes: usize) -> DevicePtr {
        let p = if self.base.is_null() {
            DevicePtr::NULL
        } else {
            self.base.offset(self.used)
        };
        self.used += bytes.div_ceil(256) * 256;
        p
    }
}

impl Shared {
    fn carve(a: &mut Arena, d: &Dims) -> Self {
        let (h, r, f, b) = (d.hidden, d.max_rows, 4usize, 2usize);
        Self {
            blocks: a.take(d.stack_blocks * r * h * b),
            prelude_h: a.take(r * h * b),
            msg_blocks: a.take(d.stack_blocks * r * h * b),
            msg_h: a.take(r * h * b),
            res3: a.take(3 * r * h * f),
            msg4: a.take(4 * r * h * f),
            resid: a.take(r * h * f),
            mlp_in: a.take(r * h * f),
            logits: a.take(r * d.experts * f),
            weights: a.take(r * d.top_k * f),
            final_out: a.take(r * h * f),
            normed3: a.take(3 * r * h * b),
            q: a.take(r * d.q_dim * b),
            k_lin: a.take(r * d.kv_dim * b),
            v_lin: a.take(r * d.kv_dim * b),
            r: a.take(r * d.rel_dim * b),
            gate: a.take(r * d.q_dim * b),
            sink: a.take(r * d.heads * b),
            dyn_k: a.take(r * d.dyn_dim * b),
            dyn_v: a.take(r * d.dyn_dim * b),
            kc: a.take(r * d.kv_dim * b),
            vc: a.take(r * d.kv_dim * b),
            qn: a.take(r * d.q_dim * b),
            kn: a.take(r * d.kv_dim * b),
            vn: a.take(r * d.kv_dim * b),
            rn: a.take(r * d.rel_dim * b),
            attn: a.take(r * d.q_dim * b),
            gated: a.take(r * d.q_dim * b),
            proj: a.take(r * h * b),
            conv_out: a.take(r * h * b),
            mlp_normed: a.take(r * h * b),
            act: a.take(r * d.top_k * d.inter * b),
            y: a.take(r * d.top_k * h * b),
            moe_out: a.take(r * h * b),
            mlp_conv: a.take(r * h * b),
            ids: a.take(r * d.top_k * 4),
            positions: a.take(r * 4),
            slots: a.take(r * 8),
            seq_lens: a.take(r * 4),
            block_table: a.take(d.max_blocks_per_seq * 4),
            row_seq: a.take(r * 4),
            seq_start: a.take(8),
            conv_slot: a.take(4),
        }
    }

    /// All NULL: a layer before `attach_layer_buffers`.
    pub(crate) fn unattached(d: &Dims) -> Self {
        Self::carve(
            &mut Arena {
                base: DevicePtr::NULL,
                used: 0,
            },
            d,
        )
    }

    pub(crate) fn is_attached(&self) -> bool {
        !self.blocks.is_null()
    }

    /// Zeroed buffers (row_seq and the conv slot stay 0: one sequence, slot 0).
    pub(crate) fn alloc(gpu: &dyn GpuBackend, d: &Dims) -> Result<Self> {
        let mut probe = Arena {
            base: DevicePtr::NULL,
            used: 0,
        };
        Self::carve(&mut probe, d);
        let base = gpu
            .alloc(probe.used)
            .context("allocate inkling2 shared buffers")?;
        gpu.copy_h2d(&vec![0u8; probe.used], base)?;
        tracing::info!(
            "Inkling2 shared buffers: {:.1} MB for {} rows",
            probe.used as f64 / 1e6,
            d.max_rows
        );
        Ok(Self::carve(&mut Arena { base, used: 0 }, d))
    }
}
