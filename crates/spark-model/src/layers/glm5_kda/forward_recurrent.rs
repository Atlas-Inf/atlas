// SPDX-License-Identifier: AGPL-3.0-only

//! Existing interleaved/fused snapshot paths, kept in their original stream order.

use super::*;

impl Glm5KdaLayer {
    pub(super) fn forward_recurrent(
        &self,
        projected: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        state: &mut SsmLayerState,
        tokens: usize,
        decode: bool,
        capture_verify_intermediates: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let m = tokens as u32;
        let p = self.heads * self.dim;
        let packed = ctx.buffers.ssm_qkvz();
        ops::kda_pack_qkv(ctx.gpu, self.pack_k, projected, packed, m, p as u32, stream)?;
        let convolved = ctx.buffers.ssm_conv_out_f32();
        let core_out = ctx.buffers.attn_output();
        if capture_verify_intermediates {
            self.verify_recurrent_rows(
                packed, convolved, g1, beta, core_out, state, 0, tokens, ctx, stream,
            )?;
        } else {
            self.sequence_recurrent_rows(
                packed, convolved, g1, beta, core_out, state, tokens, decode, ctx, stream,
            )?;
        }
        Ok(core_out)
    }

    /// Convolution + recurrence of one sequence over rows `[0, tokens)` of
    /// already-packed `packed`/`g1`/`beta` (prefill or single-row decode).
    #[allow(clippy::too_many_arguments)]
    fn sequence_recurrent_rows(
        &self,
        packed: DevicePtr,
        convolved: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        core_out: DevicePtr,
        state: &mut SsmLayerState,
        tokens: usize,
        decode: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let m = tokens as u32;
        let p = self.heads * self.dim;
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv_prefill_k,
            self.conv_prefill_tp_k,
            state.conv_state,
            packed,
            &self.weights.conv,
            DevicePtr::NULL,
            convolved,
            (3 * p) as u32,
            self.conv_width as u32,
            m,
            (3 * p) as u32,
            (3 * p) as u32,
            stream,
        )?;
        self.run_recurrent(
            convolved,
            g1,
            beta,
            state.h_state,
            core_out,
            m,
            decode,
            ctx,
            stream,
        )
    }

    /// Fused prefill chunk + verify owners: pack every row once, advance
    /// each owner over its own `rows` rows at `chunk + owner * rows`, then
    /// the chunk's own state over rows `[0, chunk)`. The owners go first
    /// because the chunk's flash recurrence borrows the packed buffer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_recurrent_passengers(
        &self,
        projected: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        state: &mut SsmLayerState,
        chunk: usize,
        owners: &mut [crate::layer::glm_long_owner::GlmLongOwner<'_>],
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let p = self.heads * self.dim;
        let packed = ctx.buffers.ssm_qkvz();
        ops::kda_pack_qkv(
            ctx.gpu,
            self.pack_k,
            projected,
            packed,
            (chunk + owners.len() * rows) as u32,
            p as u32,
            stream,
        )?;
        let convolved = ctx.buffers.ssm_conv_out_f32();
        let core_out = ctx.buffers.attn_output();
        self.owner_recurrent_rows(
            packed, convolved, g1, beta, core_out, chunk, owners, rows, ctx, stream,
        )?;
        self.sequence_recurrent_rows(
            packed, convolved, g1, beta, core_out, state, chunk, false, ctx, stream,
        )?;
        Ok(core_out)
    }

    /// Verify recurrence for rows `[row0, row0 + tokens)` of already-packed
    /// `packed`/`g1`/`beta`, advancing one owner's state and snapshotting it
    /// after every row but the last. Uses the single-launch snapshot kernels
    /// (unchanged recurrence and FP32 FMA order) when the owner's snapshot
    /// slabs are contiguous, else the original per-row launches. Convolution
    /// and recurrence are independent per row, so running all convolution
    /// rows before the recurrence rows is the same arithmetic.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_recurrent_rows(
        &self,
        packed: DevicePtr,
        convolved: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        core_out: DevicePtr,
        state: &mut SsmLayerState,
        row0: usize,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let records = !state.kda_records.is_null();
        ensure!(
            tokens >= 1
                && (records || state.h_state_intermediates.len() + 1 >= tokens)
                && state.conv_state_intermediates.len() >= tokens,
            "GLM-5 KDA verify needs K-1 h and K conv intermediates (h={}, conv={}, K={tokens})",
            state.h_state_intermediates.len(),
            state.conv_state_intermediates.len(),
        );
        let p = self.heads * self.dim;
        let bf16 = 2usize;
        let packed_row_bytes = 3 * p * bf16;
        let gate_row_bytes = p * bf16;
        let beta_row_bytes = self.heads * bf16;
        let (packed, convolved, g1, beta, core_out) = (
            packed.offset(row0 * packed_row_bytes),
            convolved.offset(row0 * packed_row_bytes),
            g1.offset(row0 * gate_row_bytes),
            beta.offset(row0 * beta_row_bytes),
            core_out.offset(row0 * gate_row_bytes),
        );
        let contiguous = |ptrs: &[DevicePtr], stride: usize| {
            ptrs.first().is_some_and(|base| {
                !base.is_null()
                    && ptrs[..tokens - 1]
                        .iter()
                        .enumerate()
                        .all(|(t, ptr)| ptr.0 == base.0 + (t * stride) as u64)
            })
        };
        self.verify_conv_rows(packed, convolved, state, 0, tokens, ctx, stream)?;
        if records {
            return self.records_recurrence(
                convolved,
                g1,
                beta,
                core_out,
                &[state.h_state],
                &[state.kda_records],
                tokens,
                ctx,
                stream,
            );
        }
        let fused_recurrent = self.recurrent_verify_snap_k.0 != 0
            && verify_batched_recurrent_snapshot_enabled()
            && contiguous(&state.h_state_intermediates, self.h_state_bytes);
        let m = tokens as u32;
        if fused_recurrent && self.recurrent_verify_owners_k.0 != 0 && self.dim == 128 {
            // The register-resident kernel with one owner: same states and
            // snapshots, output BF16 rounding may differ in rare elements.
            ops::kda_recurrent_verify_snap_owners(
                ctx.gpu,
                self.recurrent_verify_owners_k,
                convolved,
                g1,
                beta,
                self.weights.a_log.weight,
                self.weights.dt_bias.weight,
                core_out,
                &[state.h_state],
                &[state.h_state_intermediates[0]],
                self.h_state_bytes / size_of::<f32>(),
                m,
                self.heads as u32,
                self.dim as u32,
                self.lower_bound,
                stream,
            )?;
        } else if fused_recurrent {
            ops::kda_recurrent_verify_snap(
                ctx.gpu,
                self.recurrent_verify_snap_k,
                convolved,
                g1,
                beta,
                self.weights.a_log.weight,
                self.weights.dt_bias.weight,
                state.h_state,
                core_out,
                state.h_state_intermediates[0],
                self.h_state_bytes / size_of::<f32>(),
                m,
                self.heads as u32,
                self.dim as u32,
                self.lower_bound,
                stream,
            )?;
        } else {
            for t in 0..tokens {
                self.run_recurrent(
                    convolved.offset(t * packed_row_bytes),
                    g1.offset(t * gate_row_bytes),
                    beta.offset(t * beta_row_bytes),
                    state.h_state,
                    core_out.offset(t * gate_row_bytes),
                    1,
                    true,
                    ctx,
                    stream,
                )?;
                // A partial accept can select states after rows 0..K-2;
                // the post-row K-1 state is already canonical on full accept.
                if t + 1 < tokens {
                    ctx.gpu.copy_d2d_async(
                        state.h_state,
                        state.h_state_intermediates[t],
                        self.h_state_bytes,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// The snapshot verify convolution of `tokens` rows at `row0`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_conv_rows(
        &self,
        packed: DevicePtr,
        convolved: DevicePtr,
        state: &mut SsmLayerState,
        row0: usize,
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let p = self.heads * self.dim;
        let packed_row_bytes = 3 * p * 2;
        let (packed, convolved) = (
            packed.offset(row0 * packed_row_bytes),
            convolved.offset(row0 * packed_row_bytes),
        );
        let contiguous = |ptrs: &[DevicePtr], stride: usize| {
            ptrs.first().is_some_and(|base| {
                !base.is_null()
                    && ptrs[..tokens - 1]
                        .iter()
                        .enumerate()
                        .all(|(t, ptr)| ptr.0 == base.0 + (t * stride) as u64)
            })
        };
        let fused_conv = self.conv_prefill_tp_snap_k.0 != 0
            && verify_batched_conv_snapshot_enabled()
            && contiguous(&state.conv_state_intermediates, self.conv_state_bytes);
        let m = tokens as u32;
        if fused_conv {
            ops::conv1d_update_prefill_tp_snap(
                ctx.gpu,
                self.conv_prefill_tp_snap_k,
                state.conv_state,
                packed,
                &self.weights.conv,
                DevicePtr::NULL,
                convolved,
                state.conv_state_intermediates[0],
                self.conv_state_bytes / size_of::<f32>(),
                (3 * p) as u32,
                self.conv_width as u32,
                m,
                (3 * p) as u32,
                (3 * p) as u32,
                stream,
            )?;
        } else {
            for t in 0..tokens {
                ops::conv1d_update_prefill(
                    ctx.gpu,
                    self.conv_prefill_k,
                    self.conv_prefill_tp_k,
                    state.conv_state,
                    packed.offset(t * packed_row_bytes),
                    &self.weights.conv,
                    DevicePtr::NULL,
                    convolved.offset(t * packed_row_bytes),
                    (3 * p) as u32,
                    self.conv_width as u32,
                    1,
                    (3 * p) as u32,
                    (3 * p) as u32,
                    stream,
                )?;
                if t + 1 < tokens {
                    ctx.gpu.copy_d2d_async(
                        state.conv_state,
                        state.conv_state_intermediates[t],
                        self.conv_state_bytes,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Owner-batched verify recurrence: pack every owner's rows once, then
    /// advance each owner's own state over its own `rows` rows, in order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_recurrent_owners(
        &self,
        projected: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        owners: &mut [crate::layer::glm_long_owner::GlmLongOwner<'_>],
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let p = self.heads * self.dim;
        let packed = ctx.buffers.ssm_qkvz();
        ops::kda_pack_qkv(
            ctx.gpu,
            self.pack_k,
            projected,
            packed,
            (owners.len() * rows) as u32,
            p as u32,
            stream,
        )?;
        let convolved = ctx.buffers.ssm_conv_out_f32();
        let core_out = ctx.buffers.attn_output();
        self.owner_recurrent_rows(
            packed, convolved, g1, beta, core_out, 0, owners, rows, ctx, stream,
        )?;
        Ok(core_out)
    }

    /// Each owner's snapshot verify recurrence over its rows at
    /// `row0 + owner * rows`, in owner order.
    #[allow(clippy::too_many_arguments)]
    fn owner_recurrent_rows(
        &self,
        packed: DevicePtr,
        convolved: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        core_out: DevicePtr,
        row0: usize,
        owners: &mut [crate::layer::glm_long_owner::GlmLongOwner<'_>],
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.owner_batched_recurrence(
            packed, convolved, g1, beta, core_out, row0, owners, rows, ctx, stream,
        )? {
            return Ok(());
        }
        for (owner, input) in owners.iter_mut().enumerate() {
            let state = input
                .state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("GLM-5 KDA owner expected SsmLayerState"))?;
            ensure!(!state.h_is_f16, "GLM-5 KDA requires FP32 recurrent state");
            self.verify_recurrent_rows(
                packed,
                convolved,
                g1,
                beta,
                core_out,
                state,
                row0 + owner * rows,
                rows,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }
}
