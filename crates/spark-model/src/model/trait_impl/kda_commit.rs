// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::ops;
use crate::model::ssm_batched_copy::{
    StateCopy, batched_ssm_copy_enabled, copy_plan_as_strided_run, run_ssm_state_copies,
};
use atlas_core::config::LayerType;

impl TransformerModel {
    /// KDA records commit (`--ssm-rollback-mode records`): fold fold-records
    /// `rows` of every SSM layer into its h_state (which must already hold
    /// rows `0..rows.start`) and, when `rewind_conv`, restore conv_state from
    /// its snapshot after row `rows.end - 1`. Enqueued on `stream`: one
    /// `kda_commit_records` launch for all layers when the h_state and record
    /// pools are uniformly layer-strided (`copy_plan_as_strided_run` on the
    /// pointer pairs), else one per layer.
    pub(super) fn commit_kda_records(
        &self,
        seq: &mut SequenceState,
        rows: std::ops::Range<usize>,
        rewind_conv: bool,
        stream: u64,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;
        let ctas = ops::kda_commit_ctas()?;
        let kernel = ops::kda_commit_kernel(self.gpu.as_ref(), ctas)?;
        let heads = self.ssm_pool.h_bytes / (128 * 128 * 4);
        let conv_bytes = self.config.ssm_conv_state_bytes();
        let mut conv_plan = Vec::new();
        // Pointer pairs only (src = records, dst = h_state): `bytes: 1`
        // lets the strided-run check see addresses, not blob widths.
        let mut commits = Vec::new();
        let mut ssm_layer_idx = 0usize;
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            anyhow::ensure!(
                !ssm.kda_records.is_null() && rows.end <= self.ssm_pool.num_intermediates,
                "KDA records commit: layer {i} has no records for {} rows",
                rows.end
            );
            commits.push(StateCopy {
                src: ssm
                    .kda_records
                    .offset(rows.start * self.ssm_pool.kda_record_row_bytes),
                dst: ssm.h_state,
                bytes: 1,
            });
            if rewind_conv {
                conv_plan.push(StateCopy {
                    src: self
                        .ssm_pool
                        .conv_intermediate(ssm_layer_idx, seq.slot_idx, rows.end - 1),
                    dst: ssm.conv_state,
                    bytes: conv_bytes,
                });
            }
            ssm_layer_idx += 1;
        }
        let launch = |c: &StateCopy, state_pitch: usize, records_pitch: usize, layers: usize| {
            anyhow::ensure!(
                state_pitch.is_multiple_of(4) && records_pitch.is_multiple_of(4),
                "KDA records commit: layer pitches {state_pitch}/{records_pitch} not in floats"
            );
            ops::kda_commit_records(
                self.gpu.as_ref(),
                kernel,
                c.dst,
                state_pitch / 4,
                c.src,
                records_pitch / 4,
                heads * ops::KDA_RECORD_FLOATS,
                rows.len() as u32,
                heads as u32,
                layers as u32,
                ctas,
                stream,
            )
        };
        match copy_plan_as_strided_run(&commits).filter(|_| batched_ssm_copy_enabled()) {
            Some(run) => launch(&commits[0], run.dst_pitch, run.src_pitch, run.height)?,
            None => {
                for c in &commits {
                    launch(c, 0, 0, 1)?;
                }
            }
        }
        run_ssm_state_copies(self.gpu.as_ref(), &[], &conv_plan, stream)
    }
}
