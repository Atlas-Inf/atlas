// SPDX-License-Identifier: AGPL-3.0-only

//! Selected-only construction check: every future drain width must exist.

use super::Qwen3AttentionLayer;
use anyhow::{Result, ensure};
use spark_runtime::gpu::KernelHandle;

pub(in crate::layers::qwen3_attention) fn validate_mla_handles(
    handles: [KernelHandle; 7],
) -> Result<()> {
    for (index, kernel) in handles.into_iter().enumerate() {
        ensure!(
            kernel.0 != 0,
            "independent MLA row{} kernel is unavailable",
            index + 2
        );
    }
    Ok(())
}

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention) fn independent_mla_handles(&self) -> [KernelHandle; 7] {
        [
            self.mla_batched_gemv_batch2_k,
            self.mla_batched_gemv_batch3_k,
            self.mla_batched_gemv_batch4_k,
            self.mla_batched_gemv_batch5_k,
            self.mla_batched_gemv_batch6_k,
            self.mla_batched_gemv_batch7_k,
            self.mla_batched_gemv_batch8_k,
        ]
    }

    pub(in crate::layers::qwen3_attention) fn validate_independent_kernels(&self) -> Result<()> {
        validate_mla_handles(self.independent_mla_handles())?;
        crate::model::glm_independent::validate_projection_handles(
            std::array::from_fn(|i| match i + 2 {
                2 => self.w4a16_gemv_batch2_k.0,
                3 => self.w4a16_gemv_batch3_k.0,
                rows => self.w4a16_batchm.kernel(rows as u32).0,
            }),
            self.dense_gemv_batchm_k.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_requirement_includes_every_later_drain_export() {
        let handles = std::array::from_fn(|i| KernelHandle((i + 2) as u64));
        validate_mla_handles(handles).unwrap();
        for missing in 0..7 {
            let mut broken = handles;
            broken[missing] = KernelHandle(0);
            let error = validate_mla_handles(broken).unwrap_err().to_string();
            assert!(error.contains(&format!("row{}", missing + 2)));
        }
    }
}
