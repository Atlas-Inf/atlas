// SPDX-License-Identifier: AGPL-3.0-only
//! Target-only cache ownership transaction; original and transposed weights survive.
use super::shared_fp8_cache::{SharedFp8Reserve, WEIGHT_BYTES, flags, target_plan, transaction};
use super::*;
use crate::weight_map::{Nvfp4Variant, WeightQuantFormat};
use spark_runtime::weights::{WeightDtype, WeightStore};

fn checkpoint_weight(
    store: &WeightStore,
    name: &str,
    ptr: DevicePtr,
    rows: usize,
    bytes: usize,
    scale: bool,
) -> Result<()> {
    let tensor = store.get(name)?;
    let elements = tensor
        .shape
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| anyhow::anyhow!("shared FP8 checkpoint shape overflow: {name}"))?;
    anyhow::ensure!(
        tensor.ptr == ptr
            && elements == bytes
            && tensor.shape.len() == 2
            && tensor.shape[0] == rows
            && if scale {
                matches!(tensor.dtype, WeightDtype::FP8E4M3 | WeightDtype::UInt8)
            } else {
                tensor.dtype == WeightDtype::UInt8
            },
        "shared FP8 checkpoint layout/extent mismatch: {name}"
    );
    Ok(())
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn maybe_cache_glm_target_shared_fp8(
        &mut self,
        store: &WeightStore,
        prefix: &str,
        config: &atlas_core::config::ModelConfig,
        layer: usize,
        target: bool,
        variant: Nvfp4Variant,
        gpu: &dyn GpuBackend,
        stream: u64,
        reserve: SharedFp8Reserve,
    ) -> Result<()> {
        // MTP deliberately exits before interpreting flags or target geometry.
        if !target {
            return Ok(());
        }
        self.install_glm_target_shared_fp8(
            store,
            prefix,
            config,
            layer,
            variant,
            gpu,
            stream,
            flags()?,
            reserve,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn install_glm_target_shared_fp8(
        &mut self,
        store: &WeightStore,
        prefix: &str,
        config: &atlas_core::config::ModelConfig,
        layer: usize,
        variant: Nvfp4Variant,
        gpu: &dyn GpuBackend,
        stream: u64,
        (enabled, verify): (bool, bool),
        reserve: SharedFp8Reserve,
    ) -> Result<()> {
        anyhow::ensure!(
            !verify || enabled,
            "shared FP8 VERIFY requires enabled cache"
        );
        let target = true;
        let Some(plan) = target_plan(config, layer, target, enabled)? else {
            return Ok(());
        };
        self.btile_storage.require_legacy()?;
        anyhow::ensure!(
            !gpu.stream_is_capturing(stream),
            "shared FP8 cache load cannot capture"
        );
        anyhow::ensure!(
            matches!(
                variant,
                Nvfp4Variant::Standard | Nvfp4Variant::CompressedTensors
            ),
            "shared FP8 cache requires checkpoint-native NVFP4"
        );
        anyhow::ensure!(
            self.shared_experts_scale_kind == WeightQuantFormat::Nvfp4
                && self.gate_nvfp4.is_none()
                && self.bf16_shared_expert.is_none()
                && self.fp8_shared_expert.is_none()
                && self.lora.is_none(),
            "shared FP8 cache requires native GLM shared weights and BF16 router"
        );
        anyhow::ensure!(
            self.shared_fp8_cache.layer.is_none(),
            "shared FP8 cache already initialized; refusing duplicate ownership"
        );
        anyhow::ensure!(
            self.shared_gate_fp8.is_none()
                && self.shared_up_fp8.is_none()
                && self.shared_down_fp8.is_none(),
            "shared FP8 cache refuses existing/partial cache"
        );
        let originals = [
            self.weights.shared_expert.gate_proj,
            self.weights.shared_expert.up_proj,
            self.weights.shared_expert.down_proj,
        ];
        let transposed = [self.shared_gate_t, self.shared_up_t, self.shared_down_t];
        let origins = self
            .shared_fp8_origins
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("shared FP8 cache requires captured load provenance"))?;
        let suffix = if variant == Nvfp4Variant::Standard {
            "weight"
        } else {
            "weight_packed"
        };
        let mut ranges = Vec::new();
        let mut derived_bf16_weights = 0;
        for (i, projection) in ["gate_proj", "up_proj", "down_proj"]
            .into_iter()
            .enumerate()
        {
            let rows = if i == 2 { 4096 } else { 2048 };
            let w = originals[i];
            let t = transposed[i]
                .ok_or_else(|| anyhow::anyhow!("shared FP8 requires retained T weights"))?;
            anyhow::ensure!(
                w.weight_scale_2.is_finite()
                    && !w.has_per_row_scale2()
                    && !t.has_per_row_scale2()
                    && w.weight_scale_2.to_bits() == t.weight_scale_2.to_bits(),
                "shared FP8 scalar scale2 contract"
            );
            let derived = origins[i].validate(&w, rows, WEIGHT_BYTES / rows)?;
            derived_bf16_weights += usize::from(derived);
            if !derived {
                checkpoint_weight(
                    store,
                    &format!("{prefix}.{projection}.{suffix}"),
                    w.weight,
                    rows,
                    WEIGHT_BYTES / 2,
                    false,
                )?;
                checkpoint_weight(
                    store,
                    &format!("{prefix}.{projection}.weight_scale"),
                    w.weight_scale,
                    rows,
                    WEIGHT_BYTES / 16,
                    true,
                )?;
            }
            for weight in [w, t] {
                for (ptr, bytes) in [
                    (weight.weight, WEIGHT_BYTES / 2),
                    (weight.weight_scale, WEIGHT_BYTES / 16),
                ] {
                    let range = super::m5_projections::span(ptr, bytes, 16)?;
                    for prior in &ranges {
                        super::m5_projections::disjoint(prior, &range)?;
                    }
                    ranges.push(range);
                }
            }
        }
        let converter = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        anyhow::ensure!(
            converter.0 != 0 && self.fp8_gemm_k.0 != 0 && self.w4a16_gemm_t.0 != 0,
            "shared FP8 cache requires conversion and both GEMM handles"
        );
        reserve.check(gpu.free_memory()?, plan.remaining_bytes)?;
        let outputs = transaction(
            || gpu.alloc(WEIGHT_BYTES),
            |p| gpu.free(p),
            |i, output| {
                let range = super::m5_projections::span(output, WEIGHT_BYTES, 16)?;
                for prior in &ranges {
                    super::m5_projections::disjoint(prior, &range)?;
                }
                ranges.push(range);
                let w = originals[i];
                let (n, k) = if i == 2 { (4096, 2048) } else { (2048, 4096) };
                let launched = ops::predequant_nvfp4_to_fp8(
                    gpu,
                    converter,
                    w.weight,
                    w.weight_scale,
                    w.weight_scale_2,
                    output,
                    n,
                    k,
                    stream,
                );
                let completed = gpu.synchronize(stream);
                launched?;
                completed?;
                if verify {
                    super::shared_fp8_cache_bytes::verify_predecoded(
                        gpu, &w, output, n as usize, k as usize, stream,
                    )?;
                    tracing::info!(
                        layer,
                        projection = i,
                        bytes = WEIGHT_BYTES,
                        "GLM shared FP8 resident byte oracle passed"
                    );
                }
                Ok(())
            },
        )?;
        // Successful allocations belong to this model's backend ledger, whose
        // sweep handles derived weights at teardown. No second allocation pass.
        self.shared_gate_fp8 = Some(outputs[0]);
        self.shared_up_fp8 = Some(outputs[1]);
        self.shared_down_fp8 = Some(outputs[2]);
        self.shared_fp8_cache.layer = Some(layer);
        self.shared_fp8_cache.verify = verify;
        tracing::info!(
            layer,
            bytes = 3 * WEIGHT_BYTES,
            total_bytes = plan.total_bytes,
            derived_bf16_weights,
            verify,
            "GLM target shared FP8 cache installed"
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "shared_fp8_cache_load_tests.rs"]
mod tests;
