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

    // BF16 (bring-up) is the store's own tensor; FP4-Local's MXFP8 expands
    // exactly to a fresh BF16 copy, freed once quantized (quantization
    // synchronizes). Its MXFP8 source goes with the layer (`fp4l`).
    let dense = dense_auto(store, &name, gpu)?;
    let q = quantize_to_nvfp4(
        &dense,
        n,
        k,
        gpu,
        qctx.absmax_k,
        qctx.quantize_k,
        qctx.stream,
    )?;
    if dense.weight != weight.ptr {
        gpu.free(dense.weight)?;
    }
    Ok(q)
}

/// FP4-Local's MXFP8 dense MLP at `prefix` (`...mlp`) as native twins
/// (`ATLAS_GLM_FP4L_DENSE_MXFP8`, `layers::dense_ffn_native`): the whole
/// (replicated) gate / up `[inter, h]` and down `[h, inter]` copied from the
/// checkpoint's bytes, with W8A8 operands over them when prefill runs W8A8
/// (on the pinned kernel: chunk- and history-invariant rows). `None` when the switch is off or the
/// checkpoint ships them in another format (logged once). Frees what it built
/// on error. Synchronizes the default stream.
pub(super) fn native_dense(
    policy: crate::layers::glm_native::NativePolicy,
    store: &WeightStore,
    prefix: &str,
    [h, inter]: [usize; 2],
    gpu: &dyn GpuBackend,
) -> Result<Option<crate::layers::dense_ffn::DenseNative>> {
    use crate::layers::ops::{Fp8bPrefill, MxW8a8};
    use crate::weight_map::{mxfp8_rows_twin, raw_mxfp8};
    if !policy.dense_mxfp8 {
        return Ok(None);
    }
    if raw_mxfp8(store, &format!("{prefix}.gate_proj"))?.is_none() {
        super::mode::require_format(false, &format!("{prefix} MXFP8 weights"))?;
        static IGNORED: std::sync::Once = std::sync::Once::new();
        IGNORED.call_once(|| {
            tracing::warn!(
                "{} set, but {prefix}.gate_proj is not MXFP8 in this checkpoint: \
                 the dense MLP keeps its NVFP4 copies",
                crate::layers::glm_native::NativePolicy::DENSE
            )
        });
        return Ok(None);
    }
    let w8a8 = Fp8bPrefill::from_env()? == Fp8bPrefill::W8a8;
    let stream = gpu.default_stream();
    let mut twins = Vec::with_capacity(3);
    let mut operands = Vec::with_capacity(3);
    let built = (|| -> Result<()> {
        for (name, shape) in [
            ("gate_proj", [inter, h]),
            ("up_proj", [inter, h]),
            ("down_proj", [h, inter]),
        ] {
            let stem = format!("{prefix}.{name}");
            let raw = raw_mxfp8(store, &stem)?
                .ok_or_else(|| anyhow::anyhow!("{stem}: not MXFP8 beside an MXFP8 gate"))?;
            ensure!(
                [raw.n, raw.k] == shape,
                "{stem}: checkpoint MXFP8 [{}, {}] is not the config's {shape:?}",
                raw.n,
                raw.k
            );
            let (twin, _, _) = mxfp8_rows_twin(gpu, &[(raw, &[raw.n][..])], 0, 1, stream)?;
            twins.push(twin);
            if w8a8 {
                // The pinned kernel (as KDA's): no row bucket tuned at load.
                operands.push(
                    MxW8a8::from_twin(gpu, &twin, shape, 0, stream)?
                        .pinned_timed(super::kda_mx::PIN_REF_ROWS, stream),
                );
            }
        }
        Ok(())
    })();
    if let Err(e) = built {
        for w in operands {
            w.free(gpu)?;
        }
        for t in twins {
            gpu.free(t.data)?;
            gpu.free(t.scales)?;
        }
        return Err(e);
    }
    Ok(Some(crate::layers::dense_ffn::DenseNative {
        gate: twins[0],
        up: twins[1],
        down: twins[2],
        w8a8: w8a8.then(|| [operands[0], operands[1], operands[2]]),
    }))
}

#[cfg(test)]
#[path = "dense_native_tests.rs"]
mod native_tests;
