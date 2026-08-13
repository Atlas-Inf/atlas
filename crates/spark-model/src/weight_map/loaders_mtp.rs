// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Slice a stacked + fused MTP MoE expert layout into per-expert
/// `DenseExpertWeight`s via DevicePtr offsets (zero-copy).
///
/// Expects two BF16 tensors in `store`:
///   `{mlp}.experts.gate_up_proj` shape `[E, 2*I, H]`
///       — first `I` rows of axis 1 are gate, next `I` rows are up
///   `{mlp}.experts.down_proj`    shape `[E, H, I]`
///
/// Each expert's `gate`, `up`, `down` are contiguous sub-tensors of the
/// stacked allocations, so we hand back DenseWeights pointing into the
/// same underlying GPU memory. The WeightStore retains ownership of the
/// stacked allocations; the offset pointers are aliases and must NEVER
/// be passed to `gpu.free()` (the loader doesn't, and ModelWeights drops
/// the WeightStore last in any case).
pub(super) fn load_mtp_experts_stacked(
    store: &WeightStore,
    mlp: &str,
    num_experts: usize,
) -> Result<Vec<DenseExpertWeight>> {
    let gate_up = store.get(&format!("{mlp}.experts.gate_up_proj"))?;
    let down = store.get(&format!("{mlp}.experts.down_proj"))?;

    ensure!(
        gate_up.shape.len() == 3,
        "MTP stacked experts.gate_up_proj: expected 3D [E,2I,H], got {:?}",
        gate_up.shape
    );
    ensure!(
        down.shape.len() == 3,
        "MTP stacked experts.down_proj: expected 3D [E,H,I], got {:?}",
        down.shape
    );
    ensure!(
        gate_up.shape[0] == num_experts,
        "MTP stacked experts.gate_up_proj: expert dim {} != num_experts {num_experts}",
        gate_up.shape[0]
    );
    ensure!(
        down.shape[0] == num_experts,
        "MTP stacked experts.down_proj: expert dim {} != num_experts {num_experts}",
        down.shape[0]
    );

    let two_inter = gate_up.shape[1];
    let hidden = gate_up.shape[2];
    ensure!(
        two_inter % 2 == 0,
        "MTP stacked experts.gate_up_proj: 2nd dim must be even (gate+up fused), got {two_inter}"
    );
    let intermediate = two_inter / 2;

    ensure!(
        down.shape[1] == hidden,
        "MTP stacked: gate_up_proj.hidden ({hidden}) != down_proj.hidden ({})",
        down.shape[1]
    );
    ensure!(
        down.shape[2] == intermediate,
        "MTP stacked: down_proj.intermediate ({}) != gate_up_proj/2 ({intermediate})",
        down.shape[2]
    );

    // Stacked tensors must be BF16 — the per-expert split path also returns
    // BF16 (norm/gate dense or dequanted projections), so we keep the
    // contract uniform downstream.
    ensure!(
        matches!(gate_up.dtype, WeightDtype::BF16),
        "MTP stacked experts.gate_up_proj: expected BF16, got {:?}",
        gate_up.dtype
    );
    ensure!(
        matches!(down.dtype, WeightDtype::BF16),
        "MTP stacked experts.down_proj: expected BF16, got {:?}",
        down.dtype
    );

    let elt = WeightDtype::BF16.byte_size();
    let half_bytes = intermediate * hidden * elt;
    let gate_up_stride = two_inter * hidden * elt;
    let down_stride = hidden * intermediate * elt;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let base_gu = gate_up.ptr.offset(e * gate_up_stride);
        experts.push(DenseExpertWeight {
            gate_proj: DenseWeight { weight: base_gu },
            up_proj: DenseWeight {
                weight: base_gu.offset(half_bytes),
            },
            down_proj: DenseWeight {
                weight: down.ptr.offset(e * down_stride),
            },
        });
    }
    Ok(experts)
}

// ── Qwen3.5-MoE weight loaders ──
// Two NVFP4 naming conventions exist:
//   Standard (nvidia/txn545):  weight, weight_scale, weight_scale_2, input_scale
//   Sehyo (compressed-tensors): weight_packed, weight_scale, weight_global_scale, input_global_scale
// Additionally, Sehyo quantizes attention/SSM projections; standard keeps them BF16.

/// Weight quantization variant (on-disk format).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nvfp4Variant {
    /// Standard ModelOpt: weight, weight_scale, weight_scale_2, input_scale.
    /// Attention/SSM projections are BF16 dense.
    Standard,
    /// Sehyo/compressed-tensors: weight_packed, weight_global_scale, input_global_scale.
    /// Attention/SSM projections are NVFP4 quantized.
    CompressedTensors,
    /// FP8 block-scaled (e.g. Qwen/Qwen3.5-35B-A3B-FP8, Qwen/Qwen3.6-35B-A3B-FP8):
    /// weight (float8_e4m3fn) + weight_scale_inv (BF16 per-`[128,128]`-block).
    ///
    /// Loaded **NATIVELY as FP8** in Qwen3 and Qwen3.5/3.6 model families.
    /// Attention uses `w8a16_gemv` (decode) + `w8a16_gemm` (prefill).
    /// MoE uses the FP8 fused grouped-GEMM batch1/2/3 path.
    /// SSM uses `w8a16_gemv` decode + `fp8_gemm_n128` prefill (single-scale).
    /// No silent FP8→BF16→NVFP4 triple-conversion.
    ///
    /// Historical note: the variant name retains "Dequanted" because the
    /// `Bf16Raw` cousin and the pre-2026-05-24 NVFP4 detour did dequant on
    /// load. The dispatch tables in `qwen35/load_layers.rs` (`LayerType::
    /// FullAttention if native_fp8`) and `qwen3.rs` (line 176) now branch
    /// to native FP8 paths when `quant_format == QuantFormat::Fp8`.
    Fp8Dequanted,
    /// Raw BF16/FP16 fine-tunes (e.g. samuelcardillo/Carnice-MoE-35B-A3B):
    /// only `.weight` tensors exist (no quantization metadata). Runtime-quantize
    /// from BF16 to NVFP4 at load time. Quality is suboptimal vs. a
    /// pre-calibrated NVFP4 release — the user gets a warning at startup.
    Bf16Raw,
}

/// Lightning MTP: `mtp.layers.0.{enorm,hnorm,eh_proj,mixer.{q,k,v,o}_proj}`
/// plus `mtp.layers.1` ReLU² MoE (up/down only, 128 experts). No q/k RMSNorm.
/// Missing norms are filled with BF16 ones so the Qwen MTP head can load.
/// ReLU² experts reuse `up_proj` as `gate_proj` — SwiGLU is not ReLU²; accept
/// rate must be measured and this mapping replaced if it stays near zero.
pub fn load_mtp_lightning(
    store: &WeightStore,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    hidden: usize,
) -> Result<MtpWeights> {
    if !store.contains("mtp.layers.0.eh_proj.weight") {
        anyhow::bail!("Lightning MTP eh_proj missing");
    }
    let ones = |n: usize| -> Result<DenseWeight> {
        let mut host = vec![0u8; n * 2];
        for i in 0..n {
            host[i * 2] = 0x00;
            host[i * 2 + 1] = 0x3c; // BF16 1.0
        }
        let ptr = gpu.alloc(host.len())?;
        gpu.copy_h2d(&host, ptr)?;
        Ok(DenseWeight { weight: ptr })
    };
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let up = dense(store, &format!("mtp.layers.1.mixer.experts.{e}.up_proj.weight"))?;
        let down = dense(store, &format!("mtp.layers.1.mixer.experts.{e}.down_proj.weight"))?;
        experts.push(DenseExpertWeight {
            gate_proj: up,
            up_proj: up,
            down_proj: down,
        });
    }
    let shared_up = dense(store, "mtp.layers.1.mixer.shared_experts.up_proj.weight")?;
    Ok(MtpWeights {
        pre_fc_norm_embedding: dense(store, "mtp.layers.0.enorm.weight")?,
        pre_fc_norm_hidden: dense(store, "mtp.layers.0.hnorm.weight")?,
        fc: dense(store, "mtp.layers.0.eh_proj.weight")?,
        input_layernorm: dense(store, "mtp.layers.0.norm.weight")?,
        q_proj: dense(store, "mtp.layers.0.mixer.q_proj.weight")?,
        k_proj: dense(store, "mtp.layers.0.mixer.k_proj.weight")?,
        v_proj: dense(store, "mtp.layers.0.mixer.v_proj.weight")?,
        o_proj: dense(store, "mtp.layers.0.mixer.o_proj.weight")?,
        q_norm: ones(hidden)?,
        k_norm: ones(hidden)?,
        post_attn_layernorm: dense(store, "mtp.layers.1.norm.weight")?,
        moe_gate: dense(store, "mtp.layers.1.mixer.gate.weight")?,
        shared_expert: DenseExpertWeight {
            gate_proj: shared_up,
            up_proj: shared_up,
            down_proj: dense(store, "mtp.layers.1.mixer.shared_experts.down_proj.weight")?,
        },
        shared_expert_gate: ones(hidden)?,
        experts,
        dense_ffn: None,
        norm: dense(store, "mtp.layers.1.final_layernorm.weight")?,
    })
}

