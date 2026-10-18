// SPDX-License-Identifier: AGPL-3.0-only

//! MTP routed experts of nvidia/Qwen3.8-Flash-Next-FP4-Local.
//!
//! That pack quantizes `mtp.layers.0.mlp.experts` as W4A16 NVFP4 with plain
//! E4M3 block scales — the exact per-expert format of the trunk experts — so
//! they load through the same `quantized_any` path, uploaded as shipped with
//! no requantization (the NVFP4 pack's FP8 MTP experts need a BF16 round
//! trip; these do not).

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use crate::weight_map::{ExpertWeight, Nvfp4Variant, QuantizeCtx, quantized_any};

pub(super) fn experts(
    store: &WeightStore,
    mlp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
) -> Result<Vec<ExpertWeight>> {
    let (h, inter) = (config.hidden_size, config.moe_intermediate_size);
    let qctx = QuantizeCtx {
        absmax_k,
        quantize_k,
        stream,
    };
    let load = |prefix: String, n: usize, k: usize| {
        quantized_any(store, &prefix, n, k, gpu, Nvfp4Variant::Standard, qctx)
            .with_context(|| format!("qwen4_exp MTP: NVFP4 {prefix}"))
    };
    (0..config.num_experts)
        .map(|e| {
            let p = |proj: &str| format!("{mlp}.experts.{e}.{proj}_proj");
            Ok(ExpertWeight {
                gate_proj: load(p("gate"), inter, h)?,
                up_proj: load(p("up"), inter, h)?,
                down_proj: load(p("down"), h, inter)?,
            })
        })
        .collect()
}
