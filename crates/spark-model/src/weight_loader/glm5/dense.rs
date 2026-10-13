// SPDX-License-Identifier: AGPL-3.0-only
//! Preserve checkpoint-native ModelOpt dense FFNs; retain the BF16 bring-up path.
use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_map::{
    Nvfp4Variant, QuantizeCtx, QuantizedWeight, dense_auto, quantize_to_nvfp4, quantized,
    scalar_f32,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn load_projection(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<QuantizedWeight> {
    let name = format!("{prefix}.weight");
    let weight = store.get(&name)?;
    if weight.dtype == WeightDtype::UInt8 {
        // NVIDIA 423acf37583782c51c142d145aef733d72943d93: row-major
        // [N,K/2] E2M1 bytes, [N,K/16] E4M3 scales, scalar F32 multiplier.
        // These are the same buffers used by the existing Standard expert
        // loader. Dequantizing then requantizing would change the checkpoint.
        ensure!(
            matches!(variant, Nvfp4Variant::Standard),
            "{prefix}: packed dense FFN requires ModelOpt NVFP4"
        );
        ensure!(
            n > 0 && k > 0 && k.is_multiple_of(16) && weight.shape == [n, k / 2],
            "{prefix}: invalid packed dense weight shape {:?}, expected [{n}, {}]",
            weight.shape,
            k / 2
        );
        let scales = store.get(&format!("{prefix}.weight_scale"))?;
        ensure!(
            scales.dtype == WeightDtype::FP8E4M3 && scales.shape == [n, k / 16],
            "{prefix}: invalid dense NVFP4 block scales"
        );
        // scalar_f32 checks dtype and element count, including scalar [] shape.
        let result = quantized(store, prefix, gpu)?;
        ensure!(
            result.weight_scale_2.is_finite() && result.weight_scale_2 > 0.0,
            "{prefix}: invalid dense NVFP4 global scale"
        );
        let input_name = format!("{prefix}.input_scale");
        if store.contains(&input_name) {
            let input_scale = scalar_f32(store, &input_name, gpu)?;
            ensure!(
                input_scale.is_finite() && input_scale > 0.0,
                "{prefix}: invalid dense NVFP4 input scale"
            );
        }
        return Ok(result);
    }

    let dense = dense_auto(store, &name, gpu)?;
    quantize_to_nvfp4(
        &dense,
        n,
        k,
        gpu,
        qctx.absmax_k,
        qctx.quantize_k,
        qctx.stream,
    )
}
