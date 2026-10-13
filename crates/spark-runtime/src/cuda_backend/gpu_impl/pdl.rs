// SPDX-License-Identifier: AGPL-3.0-only

//! Programmatic dependent launch (PDL) gating: which kernel targets and
//! kernels are launched with PDL, and under which switch.
//!
//! Each PDL target names its switch and its own kernel list: GLM-5.3-Flash
//! under `ATLAS_PDL=1` ([`PDL_KERNELS`]), Qwen3.8-Flash-Next under
//! `ATLAS_QWEN4EXP_PDL=1` ([`QWEN4EXP_PDL_KERNELS`]). A list is its target's
//! own: a `common/` kernel that waits for one target's sake is launched with
//! PDL only where a target lists it (without the launch attribute its
//! `griddepcontrol` instructions are no-ops).

use std::sync::OnceLock;

/// A kernel target with PDL-entered kernels: its name, the switch that turns
/// PDL on for it, and the kernels it then launches with PDL. Every copy of a
/// listed kernel the target serves starts with `atlas_pdl_enter()`; launching
/// any other kernel with PDL would let it read its predecessor's output early.
pub(super) struct PdlTarget {
    pub(super) model: &'static str,
    pub(super) switch: &'static str,
    pub(super) kernels: &'static [&'static str],
}

pub(super) const PDL_TARGETS: &[PdlTarget] = &[
    PdlTarget {
        model: "glm-5.3-flash",
        switch: "ATLAS_PDL",
        kernels: PDL_KERNELS,
    },
    PdlTarget {
        model: "qwen3.8-flash-next",
        switch: "ATLAS_QWEN4EXP_PDL",
        kernels: QWEN4EXP_PDL_KERNELS,
    },
];

static PDL_TARGET: OnceLock<Option<&'static PdlTarget>> = OnceLock::new();

/// Record the served kernel target before any kernel handle is resolved.
pub fn configure_pdl(target_model: &str) {
    for other in PDL_TARGETS.iter().filter(|t| t.model != target_model) {
        if std::env::var(other.switch).as_deref() == Ok("1") {
            tracing::warn!(
                "{}=1 ignored: it enables PDL for kernel target {}, not {target_model}",
                other.switch,
                other.model
            );
        }
    }
    let _ = PDL_TARGET.set(PDL_TARGETS.iter().find(|t| t.model == target_model));
}

fn target() -> Option<&'static PdlTarget> {
    PDL_TARGET.get().copied().flatten()
}

/// The served target's PDL switch is `1`: launch its listed kernels with
/// programmatic dependent launch. Read once, after [`configure_pdl`].
pub fn pdl_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| target().is_some_and(|t| std::env::var(t.switch).as_deref() == Ok("1")))
}

/// Whether `func_name` launches with PDL: PDL is on and the served target
/// lists it.
pub(super) fn pdl_kernel(func_name: &str) -> bool {
    pdl_enabled() && target().is_some_and(|t| t.kernels.contains(&func_name))
}

/// GLM-5.3-Flash: kernels whose every copy starts with `atlas_pdl_enter()`
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

/// Qwen3.8-Flash-Next (`ATLAS_QWEN4EXP_PDL=1`): the single-token decode chain
/// of a GDN layer, an attention layer and the MoE, as `atlas_pdl_enter()`
/// kernels. Not listed, on purpose: the RDMA one-shot all-reduce
/// (`rdma_oneshot_bf16`, see its kernel note), the routed-expert GEMVs, the
/// QSA indexer and cuBLASLt. A kernel after an unlisted one (or after a copy
/// or memset) simply launches when its predecessor completes.
pub(super) const QWEN4EXP_PDL_KERNELS: &[&str] = &[
    // mHC (ATLAS_QWEN4EXP_HC_FAST) and the fused seam.
    "hc_pre_stage_vec",
    "hc_pre_down_vec",
    "hc_pre_down_vec8",
    "hc_pre_finish_vec",
    "hc_pre_finish_vec8",
    "hc_post_vec",
    "hc_post_stage_vec",
    // GDN mixer: projections, the four small kernels and their fused twin.
    "dense_gemv_bf16",
    "dense_gemv_fp8w",
    "w4a16_gemv",
    "w4a16_gemv_sw",
    "dense_gemv_ba_gates",
    "causal_conv1d_update_l2norm_f32",
    "gated_delta_rule_decode_f32",
    "gated_rms_norm_f32_input_sigmoid",
    "qwen4exp_gdn_decode_fused",
    // Attention mixer.
    "w4a16_gemv_qg",
    "w4a16_gemv_dual",
    "rms_norm",
    "rope_forward_mrope_interleaved",
    "reshape_and_cache_flash",
    "paged_decode_attn",
    "sigmoid_gate_mul",
    // MoE glue around the expert GEMVs, and the fused EP blend + post.
    "moe_topk_softmax",
    "moe_weighted_sum_blend",
    "moe_batched_blend",
    "moe_blend_hc_post",
    "bf16_add_inplace",
];

#[cfg(test)]
#[path = "pdl_tests.rs"]
mod tests;
