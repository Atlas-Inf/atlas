// SPDX-License-Identifier: AGPL-3.0-only

//! Diagnostic bitwise comparison of staged K3 query rows against the scalar
//! reference (`ATLAS_GLM_K3_MLA_QUERY_COMPARE`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{COMPARE_FLAG, QueryPlan};

impl QueryPlan {
    pub(super) fn compare_rows(
        self,
        gpu: &dyn GpuBackend,
        stream: u64,
        stage: &str,
        candidate: DevicePtr,
        reference: DevicePtr,
        row_bytes: usize,
        attention_layer: usize,
        rank: usize,
    ) -> Result<()> {
        let candidate_bytes = row_bytes
            .checked_mul(self.rows)
            .ok_or_else(|| anyhow::anyhow!("{COMPARE_FLAG}: {stage} byte count overflow"))?;
        ensure!(
            candidate_bytes % 2 == 0,
            "{COMPARE_FLAG}: odd {stage} bytes"
        );
        gpu.synchronize(stream)?;
        let mut candidate_host = vec![0u8; candidate_bytes];
        let mut reference_host = vec![0u8; candidate_bytes];
        gpu.copy_d2h(candidate, &mut candidate_host)?;
        gpu.copy_d2h(reference, &mut reference_host)?;
        for (index, (actual, expected)) in candidate_host
            .chunks_exact(2)
            .zip(reference_host.chunks_exact(2))
            .enumerate()
        {
            let actual = u16::from_le_bytes([actual[0], actual[1]]);
            let expected = u16::from_le_bytes([expected[0], expected[1]]);
            let actual_f = f32::from_bits(u32::from(actual) << 16);
            let expected_f = f32::from_bits(u32::from(expected) << 16);
            ensure!(
                actual_f.is_finite() && expected_f.is_finite(),
                "{COMPARE_FLAG}: {stage} nonfinite row={} element={} candidate_bits={actual:04x} scalar_bits={expected:04x}",
                index / (row_bytes / 2),
                index % (row_bytes / 2),
            );
            ensure!(
                actual == expected,
                "{COMPARE_FLAG}: {stage} mismatch row={} element={} candidate_bits={actual:04x} scalar_bits={expected:04x}",
                index / (row_bytes / 2),
                index % (row_bytes / 2),
            );
        }
        tracing::info!(
            stage,
            rows = self.rows,
            bytes = candidate_bytes,
            attention_layer,
            rank,
            "GLM K3 query compare passed"
        );
        Ok(())
    }
}
