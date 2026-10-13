// SPDX-License-Identifier: AGPL-3.0-only
//! Construction admission from actual layer/checkpoint owners, before writes.
use super::{kernels::validate_profile, native_source::NativeGateUpLayer};
use crate::{
    layers::moe::MoeLayer,
    weight_loader::glm5::retirement::RetirementLog,
    weight_map::{QuantizedWeight, WeightQuantFormat},
};
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::{gpu::GpuBackend, weights::WeightDtype};

pub(super) fn layer(layer: &MoeLayer, config: &ModelConfig) -> Result<()> {
    validate_profile(config)?;
    ensure!(
        config.scoring_func == "sigmoid"
            && layer.lora.is_none()
            && layer.experts_scale_kind == WeightQuantFormat::Nvfp4
            && layer.shared_experts_scale_kind == WeightQuantFormat::Nvfp4
            && layer.nvfp4_prequant_moe
            && !layer.gelu_activation
            && !layer.nvfp4_mmq_layout
            && !layer.hybrid_layout
            && !layer.unified_layout
            && layer.gate_ptrs_t.is_none()
            && layer.up_ptrs_t.is_none()
            && layer.down_ptrs_t.is_none()
            && layer.shared_gate_t.is_none()
            && layer.shared_up_t.is_none()
            && layer.shared_down_t.is_none()
            && layer.shared_gate_up_receipt.is_none()
            && layer.cutlass_grouped_host.is_none()
            && layer.bf16_shared_expert.is_none()
            && layer.fp8_shared_expert.is_none()
            && layer.bf16_gate_weight_ptrs.is_none()
            && layer.fp8_gate_weight_ptrs.is_none()
            && layer.gate_fp8.is_none()
            && layer.gate_nvfp4.is_none()
            && layer.shared_gate_fp8.is_none()
            && layer.shared_up_fp8.is_none()
            && layer.shared_down_fp8.is_none()
            && layer.pre_expert_norm.is_none()
            && layer.tid2eid_dev.is_none()
            && layer.correction_bias_dev.is_some()
            && layer.weights.shared_expert_gate.weight.is_null()
            && layer.down_t_scratch_packed.is_none()
            && layer.down_t_scratch_scale.is_none(),
        "incompatible resident construction state"
    );
    // Preserve the existing scalar/batched down and prequant producer math.
    // A partial secondary family cannot be repaired after in-place conversion.
    for handle in [
        layer.moe_expert_silu_down_shared_t_k,
        layer.moe_expert_silu_down_shared_batch2_t_k,
        layer.moe_expert_silu_down_shared_batch3_t_k,
        layer.moe_grouped_gemm_t_k64,
        layer.moe_w4a4_prequant_t_k64,
        layer.moe_w4a4_prequant_t_k64_compact,
        layer.moe_build_tile_worklist_k,
        layer.quantize_nvfp4_k,
        layer.silu_mul_quant_nvfp4_k,
        layer.moe_transpose_u8_batched_k,
    ] {
        ensure!(handle.0 != 0, "missing resident down/producer handle");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn checkpoint(
    q: QuantizedWeight,
    log: &RetirementLog<'_>,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    ensure!(
        q.weight_scale_2_vec.is_null(),
        "per-row native scalar unsupported"
    );
    for marker in [
        "weight_packed",
        "weight_global_scale",
        "scale",
        "weight_scale_inv",
    ] {
        ensure!(
            !log.store().contains(&format!("{prefix}.{marker}")),
            "nonstandard native projection"
        );
    }
    log.capture_exact(
        &format!("{prefix}.weight"),
        q.weight,
        WeightDtype::UInt8,
        &[n, k / 2],
        gpu,
    )?;
    log.capture_exact(
        &format!("{prefix}.weight_scale"),
        q.weight_scale,
        WeightDtype::FP8E4M3,
        &[n, k / 16],
        gpu,
    )?;
    let scalar_name = format!("{prefix}.weight_scale_2");
    let scalar = log.store().get(&scalar_name)?;
    log.capture_exact(&scalar_name, scalar.ptr, WeightDtype::FP32, &[1], gpu)?;
    let mut bytes = [0; 4];
    gpu.copy_d2h_on_stream(scalar.ptr, &mut bytes, stream)?;
    let value = f32::from_le_bytes(bytes);
    ensure!(
        value.is_finite() && value > 0.0 && value.to_bits() == q.weight_scale_2.to_bits(),
        "native scalar provenance mismatch"
    );
    Ok(())
}

pub(super) fn sources(
    layer: &MoeLayer,
    source: &NativeGateUpLayer<'_, '_>,
    log: &RetirementLog<'_>,
    config: &ModelConfig,
    ordinal: usize,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    for p in source.projections() {
        let e = &layer.weights.experts[p.expert];
        let q = if p.is_up { e.up_proj } else { e.gate_proj };
        ensure!(
            q.weight == p.packed.ptr
                && q.weight_scale == p.scales.ptr
                && q.weight_scale_2.to_bits() == p.scalar_bits
                && q.weight_scale_2_vec.is_null(),
            "native GU view mismatch"
        );
    }
    let prefix = config.layer_prefix(ordinal);
    for (e, expert) in layer.weights.experts.iter().enumerate() {
        if config.is_local_expert(e) {
            checkpoint(
                expert.down_proj,
                log,
                &format!("{prefix}.mlp.experts.{e}.down_proj"),
                4096,
                2048,
                gpu,
                stream,
            )?;
        }
    }
    for (name, q, n, k) in [
        (
            "gate_proj",
            layer.weights.shared_expert.gate_proj,
            2048,
            4096,
        ),
        ("up_proj", layer.weights.shared_expert.up_proj, 2048, 4096),
        (
            "down_proj",
            layer.weights.shared_expert.down_proj,
            4096,
            2048,
        ),
    ] {
        checkpoint(
            q,
            log,
            &format!("{prefix}.mlp.shared_experts.{name}"),
            n,
            k,
            gpu,
            stream,
        )?;
    }
    Ok(())
}
