// SPDX-License-Identifier: AGPL-3.0-only

//! FP4-Local's FP8 block shared experts as native twins
//! (`ATLAS_GLM_FP4L_SHARED_FP8`, served by `layers::moe::shared_native`).

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::glm_native::NativePolicy;
use crate::layers::moe::shared_native::SharedNative;
use crate::layers::ops::Fp8bPrefill;
use crate::tp_shard::TpShardKind;
use crate::weight_map::{fp8_block_twin, raw_fp8_block};

/// The whole (replicated) gate / up / down of the shared expert at `prefix`
/// (`...mlp.shared_experts`) from the checkpoint's FP8 block bytes, or
/// `None` when the switch is off or the checkpoint ships it in another format
/// (the MTP layer's MXFP8, the NVFP4 release's BF16; logged once). Frees what
/// it built on error.
pub(super) fn shared_native_twins(
    policy: NativePolicy,
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<SharedNative>> {
    if !policy.shared_fp8 {
        return Ok(None);
    }
    if raw_fp8_block(store, &format!("{prefix}.gate_proj"))?.is_none() {
        static IGNORED: std::sync::Once = std::sync::Once::new();
        IGNORED.call_once(|| {
            tracing::warn!(
                "{} set, but {prefix}.gate_proj is not FP8 block in this checkpoint: \
                 its shared expert keeps the NVFP4 copies",
                NativePolicy::SHARED
            )
        });
        return Ok(None);
    }
    let (h, inter) = (config.hidden_size, config.shared_expert_intermediate_size);
    let stream = gpu.default_stream();
    let mut built = Vec::with_capacity(3);
    let load = (|| -> Result<()> {
        for (name, shape) in [
            ("gate_proj", [inter, h]),
            ("up_proj", [inter, h]),
            ("down_proj", [h, inter]),
        ] {
            let stem = format!("{prefix}.{name}");
            let raw = raw_fp8_block(store, &stem)?
                .ok_or_else(|| anyhow::anyhow!("{stem}: not FP8 block beside an FP8 block gate"))?;
            ensure!(
                [raw.n, raw.k] == shape,
                "{stem}: checkpoint FP8 block [{}, {}] is not the config's {shape:?}",
                raw.n,
                raw.k
            );
            built.push(fp8_block_twin(gpu, raw, TpShardKind::Replicated, 0, 1, stream)?.0);
        }
        Ok(())
    })();
    if let Err(e) = load {
        for t in built {
            t.free(gpu)?;
        }
        return Err(e);
    }
    Ok(Some(SharedNative {
        gate: built[0],
        up: built[1],
        down: built[2],
        prefill: Fp8bPrefill::from_env()?,
    }))
}

#[cfg(test)]
#[path = "shared_native_tests.rs"]
mod tests;
