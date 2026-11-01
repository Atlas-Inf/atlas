// SPDX-License-Identifier: AGPL-3.0-only
//! Construction-token access to existing shared/down phases, no new math.
//! Construction-token access to actual shared/down phases under helpers_a.
use super::*;

impl MoeLayer {
    pub(in crate::layers::moe) fn validate_btile_down(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        ordinal: usize,
        retirement: &crate::weight_loader::glm5::retirement::RetirementLog<'_>,
    ) -> Result<()> {
        self.validate_checkpoint_down(gpu, config, ordinal, retirement)?;
        Ok(())
    }
    pub(in crate::layers::moe) fn transpose_btile_shared(
        &mut self,
        token: &super::super::gate_up_repack::ConstructionToken,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        token.validate(self, gpu)?;
        self.transpose_unified_shared_gate_up(gpu, config)
    }

    pub(in crate::layers::moe) fn transpose_btile_down(
        &mut self,
        token: &super::super::gate_up_repack::ConstructionToken,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        ordinal: usize,
        retirement: &crate::weight_loader::glm5::retirement::RetirementLog<'_>,
    ) -> Result<Vec<QuantizedWeight>> {
        token.validate(self, gpu)?;
        // Actual tracked GS16 down phase; native GU and shared originals stay owned.
        self.transpose_checkpoint_down(gpu, config, ordinal, retirement)
    }
}
