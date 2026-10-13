// SPDX-License-Identifier: AGPL-3.0-only
//! Construction-only routed-down ownership handoff, not a serving converter.
use super::*;
use crate::weight_loader::glm5::retirement::RetirementLog;

impl MoeLayer {
    // Only the forthcoming private resident load session may call this in serving builds.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn transpose_checkpoint_down(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        ordinal: usize,
        log: &RetirementLog<'_>,
    ) -> Result<Vec<QuantizedWeight>> {
        let mut origins = self.validate_checkpoint_down(gpu, config, ordinal, log)?;
        let mut release = |expert, scale, ptr| {
            let (expected, origin) = origins
                .remove(&(expert, scale))
                .ok_or_else(|| anyhow::anyhow!("missing/consumed down retirement receipt"))?;
            anyhow::ensure!(ptr == expected, "changed down allocation before release");
            log.release_origin(log.store(), origin, gpu)
        };
        self.transpose_unified_down_owned(gpu, config, 16, false, true, Some(&mut release))
    }

    pub(super) fn validate_checkpoint_down(
        &self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        ordinal: usize,
        log: &RetirementLog<'_>,
    ) -> Result<
        std::collections::HashMap<
            (usize, bool),
            (DevicePtr, crate::weight_loader::glm5::retirement::Origin),
        >,
    > {
        use spark_runtime::weights::WeightDtype;
        anyhow::ensure!(
            self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4,
            "checkpoint retirement requires native NVFP4 down"
        );
        anyhow::ensure!(
            ordinal < config.num_hidden_layers && self.weights.experts.len() == config.num_experts,
            "checkpoint layer/expert ownership mismatch"
        );
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        anyhow::ensure!(
            h > 0 && inter > 0 && inter.is_multiple_of(16),
            "invalid checkpoint down geometry"
        );
        let mut origins = std::collections::HashMap::new();
        for (expert, weights) in self.weights.experts.iter().enumerate() {
            let q = weights.down_proj;
            anyhow::ensure!(
                q.is_null() != config.is_local_expert(expert),
                "checkpoint down locality mismatch"
            );
            if q.is_null() {
                continue;
            }
            let prefix = format!(
                "{}.mlp.experts.{expert}.down_proj",
                config.layer_prefix(ordinal)
            );
            for marker in [
                "weight_packed",
                "weight_global_scale",
                "scale",
                "weight_scale_inv",
            ] {
                anyhow::ensure!(
                    !log.store().contains(&format!("{prefix}.{marker}")),
                    "nonstandard down checkpoint marker {marker}"
                );
            }
            origins.insert(
                (expert, false),
                (
                    q.weight,
                    log.capture_exact(
                        &format!("{prefix}.weight"),
                        q.weight,
                        WeightDtype::UInt8,
                        &[h, inter / 2],
                        gpu,
                    )?,
                ),
            );
            origins.insert(
                (expert, true),
                (
                    q.weight_scale,
                    log.capture_exact(
                        &format!("{prefix}.weight_scale"),
                        q.weight_scale,
                        WeightDtype::FP8E4M3,
                        &[h, inter / 16],
                        gpu,
                    )?,
                ),
            );
        }
        Ok(origins)
    }
}
