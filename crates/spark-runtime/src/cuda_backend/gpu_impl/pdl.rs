// SPDX-License-Identifier: AGPL-3.0-only

//! Programmatic dependent launch (PDL) gating: which kernel targets and
//! kernels are launched with PDL under `ATLAS_PDL=1`.

use std::sync::OnceLock;

/// Kernel targets whose every copy of a [`PDL_KERNELS`] entry starts with
/// `atlas_pdl_enter()`. Launching any other copy with PDL would let it read its
/// predecessor's output early, so `ATLAS_PDL=1` is honoured only for these.
const PDL_TARGETS: &[&str] = &["glm-5.3-flash"];

static PDL_TARGET: OnceLock<bool> = OnceLock::new();

/// Record the served kernel target before any kernel handle is resolved.
pub fn configure_pdl(target_model: &str) {
    let allowed = PDL_TARGETS.contains(&target_model);
    if !allowed && std::env::var("ATLAS_PDL").as_deref() == Ok("1") {
        tracing::warn!(
            "ATLAS_PDL=1 ignored: kernel target {target_model} has no PDL-entered kernels"
        );
    }
    let _ = PDL_TARGET.set(allowed);
}

/// `ATLAS_PDL=1` on a PDL-ready target: launch the kernels below with
/// programmatic dependent launch. Read once, after [`configure_pdl`].
pub fn pdl_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_PDL").as_deref() == Ok("1")
            && PDL_TARGET.get().copied().unwrap_or(false)
    })
}

/// Kernels whose every copy starts with `atlas_pdl_enter()`
/// (kernels/gb10/common/atlas_pdl.cuh), or with `atlas_pdl_enter_touch(..)`,
/// which before its wait reads only two of the kernel's immutable weight
/// parameters (kernels/gb10/glm-5.3-flash/nvfp4/atlas_pdl_touch.cuh). A kernel
/// launched with PDL must not read its predecessor's output before that wait,
/// so only these qualify.
pub(super) const PDL_KERNELS: &[&str] = &[
    "w4a16_gemv_tc8",
    "w4a16_gemv_tc8_ld",
    "w4a16_gemv_tc8_touch",
    "w4a16_gemv_tc16_touch",
    "w4a16_gemv_tc32_touch",
    "w4a16_gemv_batch2_touch",
    "w4a16_gemv_batch3_touch",
    "w4a16_gemv_batch5_qkv_touch",
    "w4a16_gemv_tc8_pair_touch",
    "w4a16_gemv_tc16_pair_touch",
    "w4a16_gemv_tc32_pair_touch",
    "mxfp8_gemv_tc8",
    "mxfp8_gemv_tc8_touch",
    "mxfp8_gemv_tc16_touch",
    "mxfp8_gemv_tc32_touch",
    "mxfp8_gemv_tc8_grouped",
    "mxfp8_gemv_tc16_grouped",
    "dense_gemv_bf16_batchm",
    "dense_gemv_bf16_batchm_ahead",
    "dense_gemv_bf16_batch5",
    "dense_gemv_bf16_batch5_dual",
    "dense_gemv_bf16_batch5_triple_n",
    "dense_gemv_bf16_batchm_dual",
    "dense_gemv_bf16_batchm_dual_k128",
    "dense_gemv_bf16_batchm_triple_n",
    "rms_norm_vanilla",
    "rms_norm_vanilla_regs",
    "bf16_add_inplace",
    "glm_hc_decode_partial_bf16",
    "glm_hc_decode_post_partial_bf16",
    "glm_hc_decode_finalize_bf16",
    "glm_hc_decode_post_bf16",
    "glm_hc_decode_partial_rows_bf16",
    "glm_hc_decode_post_partial_rows_bf16",
    "glm_hc_decode_partial_ilp_bf16",
    "glm_hc_decode_post_partial_ilp_bf16",
    "glm_hc_decode_finalize_ilp_bf16",
    "hc_post_bf16",
    "moe_topk_sigmoid_batched",
    "moe_sort_by_expert",
    "moe_sort_by_expert_scan",
    "moe_build_tile_worklist",
    "moe_build_tile_worklist_scan",
    "quantize_bf16_to_nvfp4",
    "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up",
    "silu_mul_quant_nvfp4",
    "moe_w4a4_grouped_gemm_prequant_t_k128",
    "glm_moe_decode_m16_gate_up_silu_k128w",
    "glm_moe_decode_m16_k128w",
    "glm_moe_decode_m16_k128w_zskip",
    "glm_moe_decode_m16s_gate_up_silu_k128w",
    "glm_moe_decode_m16s_k128w",
    "glm_moe_decode_m16s_k128w_zskip",
    "glm_moe_decode_m32s_gate_up_silu_k128w",
    "glm_moe_decode_m32s_k128w",
    "glm_moe_decode_m32s_k128w_zskip",
    "glm_moe_decode_m16s_gate_up_silu_k128w_l2pf",
    "glm_moe_decode_m16s_k128w_l2pf",
    "glm_moe_decode_m16s_k128w_zskip_l2pf",
    "glm_moe_decode_m32s_gate_up_silu_k128w_l2pf",
    "glm_moe_decode_m32s_k128w_l2pf",
    "glm_moe_decode_m32s_k128w_zskip_l2pf",
    "moe_unpermute_reduce_indexed_ep",
    "moe_unpermute_reduce_indexed_ep_vec8",
    "moe_batched_blend",
    "moe_unpermute_blend_ep_vec8",
    "moe_silu_mul",
    "kda_pack_qkv",
    "causal_conv1d_update_prefill_tp_snap",
    "kda_recurrent_bf16_verify_rec_owners",
    "kda_commit_records",
    "kda_sigmoid_gated_rms_norm",
];

#[cfg(test)]
#[path = "pdl_tests.rs"]
mod tests;
