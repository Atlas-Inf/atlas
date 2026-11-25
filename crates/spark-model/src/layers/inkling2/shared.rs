// SPDX-License-Identifier: AGPL-3.0-only

//! Buffers every Inkling 2 visit shares within one forward of up to
//! `max_rows` rows (a prefill chunk of one sequence, or one decode token each
//! of up to `max_seqs` sequences): the block stack, the prelude's saved
//! output, the previous pass's message states, per-row scratch, the rows'
//! metadata visit 0 uploads for the other visits (positions, slots, lengths,
//! block tables, row-to-sequence map), and each visit's table of per-sequence
//! conv-state pointers. The model runs layer-major, so every per-token state
//! here is per row.

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
    /// Sequences one decode step may carry (the server's batch size).
    pub max_seqs: usize,
    /// Visits (layers the engine sees), one state table each.
    pub visits: usize,
    /// Short-conv width (a visit's conv state keeps `conv_k - 1` columns).
    pub conv_k: usize,
}

impl Dims {
    /// Elements between two blocks of the stack (block-major `[block, row, H]`).
    pub(crate) fn block_stride(&self) -> usize {
        self.max_rows * self.hidden
    }

    /// One visit's conv state: k, v, post-attention and post-MLP convs,
    /// `[channels, conv_k - 1]` fp32 each.
    pub(crate) fn conv_state_bytes(&self) -> usize {
        (2 * self.kv_dim + 2 * self.hidden) * (self.conv_k - 1) * 4
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
    /// `[3, rows, 64]` message-gate projections (z)
    pub gate_z: DevicePtr,
    /// block-residual chunk sums, `ops::block_res_sums_per_row` per row
    pub res_sums: DevicePtr,
    /// attention split partials: decode rows x splits, or prefill rows x 1
    pub attn_partials: DevicePtr,
    /// prefill relative-bias table `[rows, heads, rel_extent]` BF16
    pub rel_table: DevicePtr,
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
    /// prefill MoE: interleaved gate/up `[rows * top_k, 2 * inter]` and sort scratch
    pub gu: DevicePtr,
    pub moe_sort: DevicePtr,
    pub y: DevicePtr,
    pub moe_out: DevicePtr,
    pub mlp_conv: DevicePtr,
    pub ids: DevicePtr,
    // row metadata (visit 0 uploads)
    pub positions: DevicePtr,
    pub slots: DevicePtr,
    /// `[seqs, stride]` block tables (one shared table for a prefill chunk)
    pub block_table: DevicePtr,
    /// `[visits, max_seqs]` conv-state base pointers, a visit's own row each
    pub state_tables: DevicePtr,
    /// Each visit's fused-projection tables, one a row count up to
    /// `BATCHM_MAX_ROWS` (`proj.rs`)
    pub proj_tables: DevicePtr,
    /// The conv state of every graph-padding row (written, never read)
    pub pad_conv: DevicePtr,
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
            gate_z: a.take(3 * r * super::ops::MAX_GATE_RANK * f),
            res_sums: a.take(r * super::ops::block_res_sums_per_row(h) * f),
            rel_table: a.take(r * d.heads * super::ops::MAX_REL_EXTENT * b),
            attn_partials: a.take(
                super::ops::attn_partials_len(
                    r.max(d.max_seqs * super::ops::DECODE_ATTN_SPLITS),
                    d.heads,
                    d.q_dim / d.heads,
                ) * f,
            ),
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
            gu: a.take(r * d.top_k * 2 * d.inter * b),
            moe_sort: a.take(super::ops::moe_sort_bytes(r * d.top_k, d.experts)),
            y: a.take(r * d.top_k * h * b),
            moe_out: a.take(r * h * b),
            mlp_conv: a.take(r * h * b),
            ids: a.take(r * d.top_k * 4),
            positions: a.take(r * 4),
            slots: a.take(r * 8),
            // one table a row: up to MAX_VERIFY_ROWS verify rows of each sequence
            block_table: a
                .take(d.max_seqs * super::layer::MAX_VERIFY_ROWS * d.max_blocks_per_seq * 4),
            state_tables: a.take(d.visits * d.max_seqs * 8),
            proj_tables: a.take(d.visits * super::glue::BATCHM_MAX_ROWS * super::proj::TABLE_BYTES),
            pad_conv: a.take(d.conv_state_bytes()),
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

    /// Zeroed buffers.
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
            "Inkling2 shared buffers: {:.1} MB for {} rows, {} sequences",
            probe.used as f64 / 1e6,
            d.max_rows,
            d.max_seqs
        );
        Ok(Self::carve(&mut Arena { base, used: 0 }, d))
    }
}
