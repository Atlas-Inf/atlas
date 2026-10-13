// SPDX-License-Identifier: AGPL-3.0-only
//! Opaque evidence minted only by the actual shared-only transpose phase.
use crate::weight_map::{QuantizedWeight, WeightQuantFormat};
use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
pub(in crate::layers::moe) struct SharedGateUpReceipt {
    backend: usize,
    weights: [QuantizedWeight; 2],
    regions: [(DevicePtr, usize); 4],
}
impl SharedGateUpReceipt {
    pub(super) fn completed(
        gpu: &dyn GpuBackend,
        n: usize,
        k: usize,
        format: WeightQuantFormat,
        gate: QuantizedWeight,
        up: QuantizedWeight,
    ) -> Option<Self> {
        // Unsupported legacy transforms retain their exact success behavior,
        // but never gain a B-tile shared-reader capability.
        if (n, k) != (2048, 4096)
            || format != WeightQuantFormat::Nvfp4
            || [gate, up]
                .iter()
                .any(|q| !q.weight_scale_2.is_finite() || q.has_per_row_scale2())
        {
            return None;
        }
        Some(Self {
            backend: gpu as *const dyn GpuBackend as *const () as usize,
            weights: [gate, up],
            regions: [
                (gate.weight, n * k / 2),
                (gate.weight_scale, n * k / 16),
                (up.weight, n * k / 2),
                (up.weight_scale, n * k / 16),
            ],
        })
    }
    pub(in crate::layers::moe) fn validate(
        &self,
        gpu: &dyn GpuBackend,
        gate: QuantizedWeight,
        up: QuantizedWeight,
    ) -> Result<[(DevicePtr, usize); 4]> {
        ensure!(
            self.backend == gpu as *const dyn GpuBackend as *const () as usize,
            "shared transform backend mismatch"
        );
        for (actual, expected) in [gate, up].iter().zip(self.weights.iter()) {
            ensure!(
                actual.weight == expected.weight
                    && actual.weight_scale == expected.weight_scale
                    && actual.weight_scale_2.to_bits() == expected.weight_scale_2.to_bits()
                    && actual.input_scale == expected.input_scale
                    && actual.weight_scale_2_vec == expected.weight_scale_2_vec,
                "stale shared transform receipt"
            );
        }
        Ok(self.regions)
    }
}
