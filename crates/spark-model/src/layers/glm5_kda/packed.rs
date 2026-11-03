// SPDX-License-Identifier: AGPL-3.0-only

//! Packed multi-prompt prefill (`layer::glm_packed`) for a KDA layer: every
//! row-local stage once over all prompts' rows, then each prompt's
//! convolution and recurrence over its own rows from its own (fresh) state.

use super::*;
use crate::layer::glm_packed::{GlmPackedOwner, tiled_rows};

impl Glm5KdaLayer {
    pub(super) fn prefill_packed(
        &self,
        hidden: DevicePtr,
        total: usize,
        owners: &mut [GlmPackedOwner<'_>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            tiled_rows(owners) == Some(total),
            "GLM packed prefill: owners do not tile {total} rows"
        );
        ensure!(
            ctx.midchunk_capture.is_none() && crate::layers::glm_sp::current().is_none(),
            "GLM packed prefill runs without a mid-chunk capture or sequence parallelism"
        );
        // The chunked FlashKDA recurrence borrows the packed buffer once an
        // owner's convolution has read it, which would clobber the next
        // owner's rows; the packing caps prompts below its threshold.
        ensure!(
            self.flash_prefill.is_none()
                || owners.iter().all(|o| !flash_prefill::eligible(
                    o.rows,
                    false,
                    ctx.graph_capture
                )),
            "GLM packed prefill: a prompt would take the FlashKDA recurrence"
        );
        let phase = self.forward_attention_rows(
            hidden,
            total,
            false,
            false,
            ctx,
            stream,
            &mut |projected, g1, beta| {
                self.packed_recurrent(projected, g1, beta, total, owners, ctx, stream)
            },
        )?;
        self.forward_ffn(phase, ctx, stream)
    }

    /// Pack every row once, then each owner's convolution and recurrence over
    /// its rows, as a single-sequence prefill of exactly those rows would run
    /// them (`sequence_recurrent_rows` on row-offset views).
    #[allow(clippy::too_many_arguments)]
    fn packed_recurrent(
        &self,
        projected: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        total: usize,
        owners: &mut [GlmPackedOwner<'_>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let p = self.heads * self.dim;
        let bf16 = 2usize;
        let packed = ctx.buffers.ssm_qkvz();
        ops::kda_pack_qkv(
            ctx.gpu,
            self.pack_k,
            projected,
            packed,
            total as u32,
            p as u32,
            stream,
        )?;
        let convolved = ctx.buffers.ssm_conv_out_f32();
        let core_out = ctx.buffers.attn_output();
        let row = PackedRowBytes::new(p, self.heads, bf16);
        for owner in owners.iter_mut() {
            let state = owner
                .state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("GLM-5 KDA expected SsmLayerState"))?;
            ensure!(!state.h_is_f16, "GLM-5 KDA requires FP32 recurrent state");
            let at = row.offsets(owner.row0);
            self.sequence_recurrent_rows(
                (packed.offset(at.packed), false),
                convolved.offset(at.packed),
                g1.offset(at.gate),
                beta.offset(at.beta),
                core_out.offset(at.gate),
                state,
                owner.rows,
                false,
                &mut None,
                ctx,
                stream,
            )?;
        }
        Ok(core_out)
    }
}

/// Byte strides of one row in the KDA recurrence buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedRowBytes {
    /// q|k|v packed and convolved rows (BF16).
    pub packed: usize,
    /// Gate and core-output rows (BF16, `heads * dim`).
    pub gate: usize,
    /// Beta rows (BF16, one per head).
    pub beta: usize,
}

impl PackedRowBytes {
    pub(super) fn new(p: usize, heads: usize, elem: usize) -> Self {
        Self {
            packed: 3 * p * elem,
            gate: p * elem,
            beta: heads * elem,
        }
    }

    /// Byte offsets of row `row0` in each buffer.
    pub(super) fn offsets(&self, row0: usize) -> Self {
        Self {
            packed: row0 * self.packed,
            gate: row0 * self.gate,
            beta: row0 * self.beta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PackedRowBytes;

    /// The same strides `verify_recurrent_rows` uses for a row offset.
    #[test]
    fn row_offsets_match_the_verify_strides() {
        let (heads, dim) = (32, 128);
        let p = heads * dim;
        let row = PackedRowBytes::new(p, heads, 2);
        assert_eq!(row.packed, 3 * p * 2);
        assert_eq!(row.gate, p * 2);
        assert_eq!(row.beta, heads * 2);
        let at = row.offsets(57);
        assert_eq!(at.packed, 57 * 3 * p * 2);
        assert_eq!(at.gate, 57 * p * 2);
        assert_eq!(at.beta, 57 * heads * 2);
        assert_eq!(row.offsets(0).packed, 0);
    }
}
