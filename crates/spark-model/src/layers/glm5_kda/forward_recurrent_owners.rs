// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-owner verify recurrence launches: records-mode folds and the
//! owner-batched snapshot recurrence.

use super::*;

impl Glm5KdaLayer {
    /// Records-mode verify of up to four owners over `tokens` rows each
    /// (`--ssm-rollback-mode records`): the states are only read and each
    /// owner's fold records are written for `commit_kda_records`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn records_recurrence(
        &self,
        convolved: DevicePtr,
        g1: DevicePtr,
        beta: DevicePtr,
        core_out: DevicePtr,
        states: &[DevicePtr],
        records: &[DevicePtr],
        tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.recurrent_verify_rec_k.0 != 0 && self.dim == 128,
            "KDA records verify needs kda_recurrent_bf16_verify_rec_owners at dim 128"
        );
        ops::kda_recurrent_verify_snap_owners(
            ctx.gpu,
            self.recurrent_verify_rec_k,
            convolved,
            g1,
            beta,
            self.weights.a_log.weight,
            self.weights.dt_bias.weight,
            core_out,
            states,
            records,
            self.heads * ops::KDA_RECORD_FLOATS,
            tokens as u32,
            self.heads as u32,
            self.dim as u32,
            self.lower_bound,
            stream,
        )
    }
}

/// Owners one `kda_recurrent_bf16_verify_snap_owners` launch takes.
const OWNERS_PER_LAUNCH: usize = 4;

impl Glm5KdaLayer {
    /// Every owner's conv (per owner), then one owner-batched recurrence
    /// launch (`kda_recurrent_bf16_verify_snap_owners`, bit-identical to the
    /// per-owner snapshot kernel) when each owner's rollback slabs are
    /// contiguous. Returns false, having done nothing, otherwise.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn owner_batched_recurrence(
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
    ) -> Result<bool> {
        let records = owners.iter_mut().all(|o| {
            o.state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .is_some_and(|s| !s.kda_records.is_null())
        });
        if !(1..=crate::layer::glm_long_owner::MAX_OWNERS).contains(&owners.len())
            || rows < 2
            || self.dim != 128
            || (!records
                && (self.recurrent_verify_owners_k.0 == 0
                    || !verify_batched_recurrent_snapshot_enabled()))
        {
            return Ok(false);
        }
        let mut states = Vec::with_capacity(owners.len());
        for input in owners.iter_mut() {
            let state = input
                .state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("GLM-5 KDA owner expected SsmLayerState"))?;
            if records {
                states.push((state.h_state, state.kda_records));
                continue;
            }
            let contiguous = !state.h_is_f16
                && state.h_state_intermediates.len() + 1 >= rows
                && state.h_state_intermediates[..rows - 1]
                    .iter()
                    .enumerate()
                    .all(|(t, p)| {
                        p.0 == state.h_state_intermediates[0].0 + (t * self.h_state_bytes) as u64
                    });
            if !contiguous {
                return Ok(false);
            }
            states.push((state.h_state, state.h_state_intermediates[0]));
        }
        for (owner, input) in owners.iter_mut().enumerate() {
            let state = input
                .state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("GLM-5 KDA owner expected SsmLayerState"))?;
            self.verify_conv_rows(
                packed,
                convolved,
                state,
                row0 + owner * rows,
                rows,
                ctx,
                stream,
            )?;
        }
        let p = self.heads * self.dim;
        let (h, i): (Vec<_>, Vec<_>) = states.into_iter().unzip();
        if records {
            for (c, (h, i)) in h
                .chunks(OWNERS_PER_LAUNCH)
                .zip(i.chunks(OWNERS_PER_LAUNCH))
                .enumerate()
            {
                let r = row0 + c * OWNERS_PER_LAUNCH * rows;
                self.records_recurrence(
                    convolved.offset(r * 3 * p * 2),
                    g1.offset(r * p * 2),
                    beta.offset(r * self.heads * 2),
                    core_out.offset(r * p * 2),
                    h,
                    i,
                    rows,
                    ctx,
                    stream,
                )?;
            }
            return Ok(true);
        }
        // The kernel takes up to four owners; larger batches launch per four.
        for (c, (h, i)) in h
            .chunks(OWNERS_PER_LAUNCH)
            .zip(i.chunks(OWNERS_PER_LAUNCH))
            .enumerate()
        {
            let r = row0 + c * OWNERS_PER_LAUNCH * rows;
            ops::kda_recurrent_verify_snap_owners(
                ctx.gpu,
                self.recurrent_verify_owners_k,
                convolved.offset(r * 3 * p * 2),
                g1.offset(r * p * 2),
                beta.offset(r * self.heads * 2),
                self.weights.a_log.weight,
                self.weights.dt_bias.weight,
                core_out.offset(r * p * 2),
                h,
                i,
                self.h_state_bytes / size_of::<f32>(),
                rows as u32,
                self.heads as u32,
                self.dim as u32,
                self.lower_bound,
                stream,
            )?;
        }
        Ok(true)
    }
}
