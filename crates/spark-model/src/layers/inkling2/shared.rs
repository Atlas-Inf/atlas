// SPDX-License-Identifier: AGPL-3.0-only

//! Buffers every Inkling 2 visit shares within one forward: the block stack,
//! the prelude's saved output, the previous pass's message states, scratch
//! for one row, and the token metadata visit 0 uploads (position, slot,
//! length, block table) for the other visits. Decode only, one row (the
//! prefill fallback runs decode per token); multi-row comes with the batched
//! kernels.

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
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Shared {
    // forward state
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
    // token metadata (visit 0 uploads)
    pub position: DevicePtr,
    pub slot: DevicePtr,
    pub seq_len: DevicePtr,
    pub block_table: DevicePtr,
    /// constant one-row conv bookkeeping: row_seq [0], seq_start [0, 1], slots [0]
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
        let (h, f, b) = (d.hidden, 4usize, 2usize);
        Self {
            blocks: a.take(d.stack_blocks * h * b),
            prelude_h: a.take(h * b),
            msg_blocks: a.take(d.stack_blocks * h * b),
            msg_h: a.take(h * b),
            res3: a.take(3 * h * f),
            msg4: a.take(4 * h * f),
            resid: a.take(h * f),
            mlp_in: a.take(h * f),
            logits: a.take(d.experts * f),
            weights: a.take(d.top_k * f),
            final_out: a.take(h * f),
            normed3: a.take(3 * h * b),
            q: a.take(d.q_dim * b),
            k_lin: a.take(d.kv_dim * b),
            v_lin: a.take(d.kv_dim * b),
            r: a.take(d.rel_dim * b),
            gate: a.take(d.q_dim * b),
            sink: a.take(d.heads * b),
            dyn_k: a.take(d.dyn_dim * b),
            dyn_v: a.take(d.dyn_dim * b),
            kc: a.take(d.kv_dim * b),
            vc: a.take(d.kv_dim * b),
            qn: a.take(d.q_dim * b),
            kn: a.take(d.kv_dim * b),
            vn: a.take(d.kv_dim * b),
            rn: a.take(d.rel_dim * b),
            attn: a.take(d.q_dim * b),
            gated: a.take(d.q_dim * b),
            proj: a.take(h * b),
            conv_out: a.take(h * b),
            mlp_normed: a.take(h * b),
            act: a.take(d.top_k * d.inter * b),
            y: a.take(d.top_k * h * b),
            moe_out: a.take(h * b),
            mlp_conv: a.take(h * b),
            ids: a.take(d.top_k * 4),
            position: a.take(4),
            slot: a.take(8),
            seq_len: a.take(4),
            block_table: a.take(d.max_blocks_per_seq * 4),
            row_seq: a.take(4),
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
        let s = Self::carve(&mut Arena { base, used: 0 }, d);
        // one-row conv bookkeeping
        gpu.copy_h2d(&0i32.to_le_bytes(), s.row_seq)?;
        gpu.copy_h2d(
            &[0i32.to_le_bytes(), 1i32.to_le_bytes()].concat(),
            s.seq_start,
        )?;
        gpu.copy_h2d(&0i32.to_le_bytes(), s.conv_slot)?;
        Ok(s)
    }
}
