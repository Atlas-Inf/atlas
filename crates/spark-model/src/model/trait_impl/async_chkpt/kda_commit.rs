// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl TransformerModel {
    /// KDA records commit (`--ssm-rollback-mode records`): fold fold-records
    /// `rows` of every SSM layer into its h_state (which must already hold
    /// rows `0..rows.start`) and, when `rewind_conv`, restore conv_state from
    /// its snapshot after row `rows.end - 1`. Enqueued on `stream`.
    pub(in crate::model::trait_impl) fn commit_kda_records(
        &self,
        seq: &mut SequenceState,
        rows: std::ops::Range<usize>,
        rewind_conv: bool,
        stream: u64,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;
        let kernel = self.gpu.kernel("kda", "kda_commit_records")?;
        let heads = self.ssm_pool.h_bytes / (128 * 128 * 4);
        let conv_bytes = self.config.ssm_conv_state_bytes();
        let mut conv_plan = Vec::new();
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
            ops::kda_commit_records(
                self.gpu.as_ref(),
                kernel,
                ssm.h_state,
                ssm.kda_records
                    .offset(rows.start * self.ssm_pool.kda_record_row_bytes),
                heads * ops::KDA_RECORD_FLOATS,
                rows.len() as u32,
                heads as u32,
                stream,
            )?;
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
        run_ssm_state_copies(self.gpu.as_ref(), &[], &conv_plan, stream)
    }
}
