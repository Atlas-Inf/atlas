// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 → NVFP4 runtime requantization, for the experts the qwen4_exp loader
//! asks for through `quantized_any` (routed + shared MoE experts).
//!
//! Interim until native EXL3 MoE (M5) replaces it: dequantize the packed
//! linear to a transient BF16 buffer, run the SAME runtime NVFP4 quantization
//! the `Nvfp4Variant::Bf16Raw` arm of `quantized_any` runs, free the transient
//! and `reclaim` the four packed tensors.

use anyhow::{Result, bail};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::ops::{Exl3Kernels, exl3_dense_bf16_nk};
use crate::weight_map::exl3::exl3_from_store;
use crate::weight_map::loaders_fp8::quantize_to_nvfp4;
use crate::weight_map::{DenseWeight, QuantizeCtx, QuantizedWeight};

/// Load linear `prefix` as NVFP4 from its EXL3 packing (see the module doc).
pub(crate) fn quantized_from_exl3(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    qctx: QuantizeCtx,
) -> Result<QuantizedWeight> {
    let w = exl3_from_store(store, prefix, gpu)?;
    if w.shape.out_features != n || w.shape.in_features != k {
        bail!(
            "EXL3: {prefix} is {}x{} (out x in), the loader asked for {n}x{k}",
            w.shape.out_features,
            w.shape.in_features
        );
    }
    let kernels = Exl3Kernels::resolve(gpu)?;
    let bf16 = gpu.alloc(n * k * 2)?;
    let q = exl3_dense_bf16_nk(gpu, &kernels, &w, bf16, qctx.stream).and_then(|()| {
        quantize_to_nvfp4(
            &DenseWeight { weight: bf16 },
            n,
            k,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        )
    });
    // The transient must go whether the quantize worked or not.
    gpu.free(bf16)?;
    let q = q?;
    for suffix in ["trellis", "suh", "svh", "mul1"] {
        store.reclaim(gpu, &format!("{prefix}.{suffix}"))?;
    }
    Ok(q)
}
