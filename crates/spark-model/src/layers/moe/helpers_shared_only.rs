// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl MoeLayer {
    /// Build per-expert swizzled SFB weight-scale tables for the CUTLASS grouped
    /// NVFP4 path (`ATLAS_HOLO_MOE_GROUPED_CUTLASS`). For each expert, swizzle the
    /// `[K/16,N]` `gate_ptrs_t`/`up_ptrs_t` scale into the CUTLASS SFB atom via
    /// `pack_weight_sfb`, then upload the per-expert pointer arrays. The grouped
    /// kernel pairs these with `gate_ptrs.packed` (`[N,K/2]`) + the real per-expert
    /// `scale2`. Requires FAST_MOE=full (gate_ptrs_t/up_ptrs_t present); no-op else.
    /// Shared-expert transposed twins only. CUTLASS grouped MoE keeps routed
    /// experts checkpoint-native, but the shared FP8 cache and prefill shared
    /// GEMMs still consume the transposed shared projections.
    pub fn transpose_shared_only(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_unified_shared_gate_up(gpu, config)?;
        let shared_inter = config.shared_expert_intermediate_size;
        if !self.weights.shared_expert.down_proj.is_null() && shared_inter > 0 {
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                config.hidden_size,
                shared_inter,
            )?);
        }
        Ok(())
    }

    /// Free the checkpoint routed-expert scales once CUTLASS owns swizzled
    /// copies (vLLM keeps only the swizzled scales too). Scale pointer tables
    /// are zeroed so any non-CUTLASS routed kernel faults instead of reading
    /// freed memory.
    pub(super) fn release_routed_scales(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut freed = 0usize;
        for expert in self.weights.experts.iter_mut() {
            for proj in [
                &mut expert.gate_proj,
                &mut expert.up_proj,
                &mut expert.down_proj,
            ] {
                if !proj.weight_scale.is_null() {
                    gpu.free(proj.weight_scale)?;
                    proj.weight_scale = DevicePtr::NULL;
                    freed += 1;
                }
            }
        }
        let zeros = vec![0u8; self.weights.experts.len() * 8];
        for table in [&self.gate_ptrs, &self.up_ptrs, &self.down_ptrs] {
            if !table.scale_ptrs.is_null() {
                gpu.copy_h2d(&zeros, table.scale_ptrs)?;
            }
        }
        self.routed_scales_released = true;
        tracing::info!(freed, "CUTLASS grouped: released checkpoint routed scales");
        Ok(())
    }
}
