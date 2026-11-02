// SPDX-License-Identifier: AGPL-3.0-only
//! Provenance captured at the existing quantization seam, before BF16 is freed.
use super::*;
use crate::weight_map::{Nvfp4Variant, QuantizeCtx, quantized_any};
use spark_runtime::weights::{WeightDtype, WeightStore};

#[derive(Clone, Copy)]
pub(crate) struct SharedFp8Origin {
    weight: QuantizedWeight,
    n: usize,
    k: usize,
    derived_bf16: bool,
}

impl SharedFp8Origin {
    pub(super) fn validate(&self, weight: &QuantizedWeight, n: usize, k: usize) -> Result<bool> {
        anyhow::ensure!(
            self.n == n
                && self.k == k
                && self.weight.weight == weight.weight
                && self.weight.weight_scale == weight.weight_scale
                && self.weight.weight_scale_2.to_bits() == weight.weight_scale_2.to_bits()
                && self.weight.weight_scale_2_vec == weight.weight_scale_2_vec
                && self.weight.input_scale == weight.input_scale,
            "shared FP8 generated-weight provenance mismatch"
        );
        Ok(self.derived_bf16)
    }

    #[cfg(test)]
    pub(super) fn native_fixture(weight: QuantizedWeight, n: usize, k: usize) -> Self {
        Self {
            weight,
            n,
            k,
            derived_bf16: false,
        }
    }
}

/// Whether `quantized_any` will DERIVE this shared projection's NVFP4 at load
/// (`true`: a BF16 source, or FP4-Local's FP8 one, requantized) rather than
/// hand back the checkpoint's own packed NVFP4 (`false`). Anything else, or
/// any mismatch of geometry, dtype or markers, is refused.
fn source_is_runtime_quantized(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
) -> Result<bool> {
    anyhow::ensure!(
        matches!((n, k), (2048, 4096) | (4096, 2048)),
        "shared FP8 origin requires exact GU/down geometry"
    );
    let has = |suffix| store.contains(&format!("{prefix}.{suffix}"));
    let has_packed = has("weight_packed");
    let has_scale = has("weight_scale");
    let has_inv = has("weight_scale_inv");
    if !has_packed && !has_scale && !has_inv {
        for marker in [
            "weight_scale_2",
            "weight_global_scale",
            "input_scale",
            "input_global_scale",
        ] {
            anyhow::ensure!(
                !has(marker),
                "shared FP8 BF16 source has unexpected {marker}"
            );
        }
        let tensor = store.get(&format!("{prefix}.weight"))?;
        anyhow::ensure!(
            tensor.dtype == WeightDtype::BF16 && tensor.shape == [n, k],
            "shared FP8 derived source must be BF16 [{n},{k}], got {:?} {:?}",
            tensor.dtype,
            tensor.shape
        );
        let bytes = tensor
            .shape
            .iter()
            .try_fold(2usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| anyhow::anyhow!("shared FP8 BF16 source extent overflow"))?;
        super::m5_projections::span(tensor.ptr, bytes, 16)?;
        return Ok(true);
    }
    // nvidia GLM-5.3-Flash-FP4-Local: E4M3 `[n, k]` with FP8_PB_WO's F32
    // 128x128 grid (or MXFP8's UE8M0 per 32). `quantized_any` routes it
    // through `quantized_from_fp8` (FP8 -> BF16 -> NVFP4), the same runtime
    // requantization a BF16 source gets, so it is derived like one.
    if !has_packed
        && let Ok(weight) = store.get(&format!("{prefix}.weight"))
        && weight.dtype == WeightDtype::FP8E4M3
    {
        for marker in ["weight_scale_inv", "weight_scale_2", "weight_global_scale"] {
            anyhow::ensure!(
                !has(marker),
                "shared FP8 E4M3 source has unexpected {marker}"
            );
        }
        let scale = store.get(&format!("{prefix}.weight_scale"))?;
        let block = scale.dtype == WeightDtype::FP32
            && crate::weight_map::block_scale_grid(&scale.shape)
                .is_some_and(|(sn, sk)| n.is_multiple_of(sn) && k.is_multiple_of(sk));
        anyhow::ensure!(
            weight.shape == [n, k]
                && (block || crate::weight_map::mxfp8_geometry(store, prefix)?.is_some()),
            "shared FP8 E4M3 source must be [{n},{k}] with an FP8 block or MXFP8 scale, \
             got {:?} + {:?} {:?}",
            weight.shape,
            scale.dtype,
            scale.shape
        );
        return Ok(true);
    }
    // Native NVFP4 remains allowed.
    anyhow::ensure!(!has_inv && has_scale, "shared FP8 native source markers");
    let weight = store.get(&format!(
        "{prefix}.{}",
        if has_packed {
            "weight_packed"
        } else {
            "weight"
        }
    ))?;
    let scale = store.get(&format!("{prefix}.weight_scale"))?;
    anyhow::ensure!(
        weight.dtype == WeightDtype::UInt8
            && weight.shape == [n, k / 2]
            && matches!(scale.dtype, WeightDtype::UInt8 | WeightDtype::FP8E4M3)
            && scale.shape == [n, k / 16],
        "shared FP8 native source layout/dtype"
    );
    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_glm_shared_fp8_weight(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    target: bool,
) -> Result<(QuantizedWeight, Option<SharedFp8Origin>)> {
    let track = target && super::shared_fp8_cache::flags()?.0;
    load_with_origin(store, prefix, n, k, gpu, variant, qctx, track)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_with_origin(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    track: bool,
) -> Result<(QuantizedWeight, Option<SharedFp8Origin>)> {
    if track {
        anyhow::ensure!(
            matches!(
                variant,
                Nvfp4Variant::Standard | Nvfp4Variant::CompressedTensors
            ),
            "shared FP8 origin requires native NVFP4 model variant"
        );
    }
    let derived_bf16 = if track {
        source_is_runtime_quantized(store, prefix, n, k)?
    } else {
        false
    };
    let weight = quantized_any(store, prefix, n, k, gpu, variant, qctx)?;
    Ok((
        weight,
        track.then_some(SharedFp8Origin {
            weight,
            n,
            k,
            derived_bf16,
        }),
    ))
}

impl MoeLayer {
    pub(crate) fn set_shared_fp8_origins(
        &mut self,
        origins: [Option<SharedFp8Origin>; 3],
    ) -> Result<()> {
        anyhow::ensure!(
            self.shared_fp8_origins.is_none() && self.shared_fp8_cache.layer.is_none(),
            "shared FP8 provenance already installed"
        );
        self.shared_fp8_origins = match origins {
            [None, None, None] => None,
            [Some(a), Some(b), Some(c)] => Some([a, b, c]),
            _ => anyhow::bail!("partial shared FP8 load provenance"),
        };
        Ok(())
    }
}

#[cfg(test)]
#[path = "shared_fp8_origin_tests.rs"]
mod tests;
