// SPDX-License-Identifier: AGPL-3.0-only

//! The per-layer MoE block: 512 routed experts, top-10, plus a shared expert
//! and its sigmoid gate.
//!
//! This is `load_moe_qwen35` verbatim. Both the naming
//! (`mlp.gate`, `mlp.shared_expert.{gate,up,down}_proj`,
//! `mlp.shared_expert_gate`, `mlp.experts.{e}.{gate,up,down}_proj`) and the
//! on-disk quantization (standard ModelOpt NVFP4: packed E2M1 `weight`,
//! per-16 E4M3 `weight_scale`, per-tensor F32 `weight_scale_2`) are identical
//! to Qwen3.5/3.6 MoE. The expert COUNT and widths differ — 512 x 640 here
//! against 256 x 512 there — but those come from `config`, not from the
//! loader, so nothing needs re-deriving.
//!
//! The router (`mlp.gate`) is left BF16 and runtime-quantized to NVFP4 only
//! for the non-native-FP8 path, matching qwen35. Its precision matters more
//! than its size: at 512 experts the top-10 weights cluster tightly, and a
//! 4-bit ULP wider than that spread cannot tell them apart.
//!
//! FP4-Local keeps the shared expert MXFP8: it is the one expert every token
//! goes through, so the BF16-expand + NVFP4 requant the fused path would
//! impose is not paid here. The routed experts stay on `quantized_any`
//! exactly as every other pack.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::ops::{Mxfp8Proj, Mxfp8ProjKernels, Mxfp8Scratch};
use crate::layers::{FfnComponent, MoeLayer};
use crate::weight_map::{Mxfp8Weight, Nvfp4Variant, load_moe_qwen35, quantize_to_nvfp4};

use super::mx8_common;

/// `Some` when the shared expert sits in the store as checkpoint-native
/// MXFP8 (FP4-Local): E4M3 `weight` + U8 E8M0 `weight_scale` per 32 inputs.
fn mxfp8_shared(store: &WeightStore, lp: &str, inter: usize, h: usize) -> Result<bool> {
    let key = format!("{lp}.mlp.shared_expert.gate_proj.weight");
    let Ok(w) = store.get(&key) else {
        return Ok(false);
    };
    if w.dtype != WeightDtype::FP8E4M3 {
        return Ok(false);
    }
    for (proj, n, k) in [
        ("gate_proj", inter, h),
        ("up_proj", inter, h),
        ("down_proj", h, inter),
    ] {
        let [_, _] = mx8_common::module(store, &format!("{lp}.mlp.shared_expert.{proj}"), n, k)?;
    }
    Ok(true)
}

pub(super) fn build_moe(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    mx8_scratch: &mut Option<Mxfp8Scratch>,
) -> Result<FfnComponent> {
    let h = config.hidden_size;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let inter = config.shared_expert_intermediate_size;
    let native_shared = mxfp8_shared(store, lp, inter, h)?;

    let weights = load_moe_qwen35(
        store,
        lp,
        config.num_experts,
        gpu,
        config,
        variant,
        absmax_k,
        quantize_k,
        stream,
        false,
        native_shared,
    )
    .with_context(|| format!("qwen4_exp: MoE block at {lp}"))?;

    let gate_nvfp4 = Some(quantize_to_nvfp4(
        &weights.gate,
        config.num_experts,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?);

    let mut moe = MoeLayer::new(weights, config.num_experts, gate_nvfp4, gpu, config)?;
    if native_shared {
        let m = |proj: &str, n: usize, k: usize| -> Result<Mxfp8Proj> {
            let [w, s] =
                mx8_common::module(store, &format!("{lp}.mlp.shared_expert.{proj}"), n, k)?;
            Ok(Mxfp8Proj {
                w: Mxfp8Weight { data: w, scales: s },
                n: n as u32,
                k: k as u32,
            })
        };
        // Gate‖up row-concatenated `[2*inter, h]` twin: one `mxfp8_proj` pass
        // then `fused_silu_mul` on the `[m, 2*inter]` layout for the ≤32-row
        // bands; the unfused trio stays for the prefill arm.
        let [gw, gs] = mx8_common::module(
            store,
            &format!("{lp}.mlp.shared_expert.gate_proj"),
            inter,
            h,
        )?;
        let [uw, us] =
            mx8_common::module(store, &format!("{lp}.mlp.shared_expert.up_proj"), inter, h)?;
        let gate_up = {
            let w = gpu.alloc(2 * inter * h)?;
            gpu.copy_d2d(gw, w, inter * h)?;
            gpu.copy_d2d(uw, w.offset(inter * h), inter * h)?;
            let s = gpu.alloc(2 * inter * (h / 32))?;
            gpu.copy_d2d(gs, s, inter * (h / 32))?;
            gpu.copy_d2d(us, s.offset(inter * (h / 32)), inter * (h / 32))?;
            let out = gpu.alloc(32 * 2 * inter * 2)?;
            Some(crate::layers::moe::Mxfp8GateUp {
                proj: Mxfp8Proj {
                    w: Mxfp8Weight { data: w, scales: s },
                    n: (2 * inter) as u32,
                    k: h as u32,
                },
                gate_up_out: out,
                silu_mul: crate::layers::try_kernel(gpu, "gemm", "fused_silu_mul"),
            })
        };
        // The shared MXFP8 BF16 prefill scratch: the MoE runs after the
        // layer's attention/GDN on the same stream, so the transient buffer
        // is never live twice.
        let scratch = mx8_common::scratch_for(mx8_scratch, config, gpu)?;
        moe.set_mxfp8_shared_expert(
            m("gate_proj", inter, h)?,
            m("up_proj", inter, h)?,
            m("down_proj", h, inter)?,
            gate_up,
            Mxfp8ProjKernels::resolve(gpu)?,
            scratch,
        )?;
    }
    Ok(FfnComponent::Moe(moe))
}
