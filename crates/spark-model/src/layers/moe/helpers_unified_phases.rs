// SPDX-License-Identifier: AGPL-3.0-only
//! Private construction phases; no partial-layout serving entry point.

use super::*;
#[path = "shared_gate_up_receipt.rs"]
mod receipt;
pub(in crate::layers::moe) use receipt::SharedGateUpReceipt;

#[cfg(test)]
pub(in crate::layers::moe) fn btile_shared_fixture(
    layer: &mut MoeLayer,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> Result<()> {
    layer.transpose_unified_shared_gate_up(gpu, config)
}

impl MoeLayer {
    pub(in crate::layers::moe) fn transpose_unified_shared_gate_up(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        // Invalidate before any fallible repeat; stamp only after both actual
        // transforms succeed. No new GPU operations or legacy failure paths.
        self.shared_gate_up_receipt = None;
        let h = config.hidden_size;
        let shared_inter = config.shared_expert_intermediate_size;
        // Shared expert (tiny, do unconditionally — fits regardless).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_gate_up_receipt = receipt::SharedGateUpReceipt::completed(
                gpu,
                shared_inter,
                h,
                self.shared_experts_scale_kind,
                self.shared_gate_t.expect("successful transform"),
                self.shared_up_t.expect("successful transform"),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn transpose_unified_down_owned(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        routed_group: usize,
        keep_originals: bool,
        keep_shared_originals: bool,
        mut release: Option<&mut dyn FnMut(usize, bool, DevicePtr) -> Result<()>>,
    ) -> Result<Vec<QuantizedWeight>> {
        let h = config.hidden_size;
        let inter = config.routed_inter_local();
        let shared_inter = config.shared_expert_intermediate_size;
        // ── Phase C: transpose down routed experts ──
        let down_src: Vec<QuantizedWeight> = self
            .weights
            .experts
            .iter()
            .map(|e| {
                if e.down_proj.is_null() {
                    QuantizedWeight::null()
                } else {
                    e.down_proj
                }
            })
            .collect();
        let down_t = self.transpose_experts_gpu(gpu, &down_src, h, inter, routed_group)?;
        self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        if !self.weights.shared_expert.down_proj.is_null() && shared_inter > 0 {
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                h,
                shared_inter,
            )?);
        }

        if !keep_originals {
            // ── Phase D: free down untransposed ──
            for (index, expert) in self.weights.experts.iter_mut().enumerate() {
                if !expert.down_proj.weight.is_null() {
                    if let Some(release) = release.as_mut() {
                        release(index, false, expert.down_proj.weight)?;
                        release(index, true, expert.down_proj.weight_scale)?;
                    } else {
                        gpu.free(expert.down_proj.weight)?;
                        gpu.free(expert.down_proj.weight_scale)?;
                    }
                    expert.down_proj.weight = DevicePtr::NULL;
                    expert.down_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !keep_shared_originals
                && !self.weights.shared_expert.down_proj.weight.is_null()
                && shared_inter > 0
            {
                gpu.free(self.weights.shared_expert.down_proj.weight)?;
                gpu.free(self.weights.shared_expert.down_proj.weight_scale)?;
                self.weights.shared_expert.down_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.down_proj.weight_scale = DevicePtr::NULL;
            }
        }
        Ok(down_t)
    }
}
