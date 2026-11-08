// SPDX-License-Identifier: AGPL-3.0-only

//! Programmatic dependent launch (PDL) gating: which kernel targets and
//! kernels are launched with PDL, and under which switch.
//!
//! Each PDL target names its switches, each with its own kernel list:
//! GLM-5.3-Flash under `ATLAS_PDL=1` ([`PDL_KERNELS`]), Qwen3.8-Flash-Next
//! under `ATLAS_QWEN4EXP_PDL=1` ([`QWEN4EXP_PDL_KERNELS`]), Qwen3.8-27B under
//! `ATLAS_QWEN38_PDL=1` (prefill, [`QWEN38_PDL_KERNELS`]) and, separately,
//! `ATLAS_QWEN38_PDL_DECODE=1` (the decode chain,
//! [`QWEN38_PDL_DECODE_KERNELS`]). A kernel launches with PDL when a switch
//! that is on lists it. A list is its target's own: a `common/` kernel that
//! waits for one target's sake is launched with PDL only where a target lists
//! it (without the launch attribute its `griddepcontrol` instructions are
//! no-ops).

use std::sync::OnceLock;

/// A kernel target with PDL-entered kernels: its name and its switches. Every
/// copy of a kernel any of its lists names, in the kernels the target serves,
/// starts with `atlas_pdl_enter()`; launching any other kernel with PDL would
/// let it read its predecessor's output early.
pub(super) struct PdlTarget {
    pub(super) model: &'static str,
    pub(super) switches: &'static [PdlSwitch],
}

/// One PDL switch of a target: set to `1`, the kernels it lists launch with
/// PDL. A target's switches are independent; a kernel on two lists launches
/// with PDL when either is on.
pub(super) struct PdlSwitch {
    pub(super) name: &'static str,
    pub(super) kernels: &'static [&'static str],
}

impl PdlTarget {
    /// Whether any of this target's lists names `func_name`.
    #[cfg(test)]
    pub(super) fn lists(&self, func_name: &str) -> bool {
        self.switches.iter().any(|s| s.kernels.contains(&func_name))
    }

    /// Whether `func_name` launches with PDL when the switches `on` reports
    /// as set are on.
    fn pdl_kernel_with(&self, func_name: &str, on: impl Fn(&str) -> bool) -> bool {
        self.switches
            .iter()
            .any(|s| on(s.name) && s.kernels.contains(&func_name))
    }
}

pub(super) const PDL_TARGETS: &[PdlTarget] = &[
    PdlTarget {
        model: "glm-5.3-flash",
        switches: &[PdlSwitch {
            name: "ATLAS_PDL",
            kernels: PDL_KERNELS,
        }],
    },
    PdlTarget {
        model: "qwen3.8-flash-next",
        switches: &[PdlSwitch {
            name: "ATLAS_QWEN4EXP_PDL",
            kernels: QWEN4EXP_PDL_KERNELS,
        }],
    },
    PdlTarget {
        model: "qwen3.8-27b",
        switches: &[
            PdlSwitch {
                name: "ATLAS_QWEN38_PDL",
                kernels: QWEN38_PDL_KERNELS,
            },
            PdlSwitch {
                name: "ATLAS_QWEN38_PDL_DECODE",
                kernels: QWEN38_PDL_DECODE_KERNELS,
            },
        ],
    },
];

fn switch_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

static PDL_TARGET: OnceLock<Option<&'static PdlTarget>> = OnceLock::new();

/// Record the served kernel target before any kernel handle is resolved.
pub fn configure_pdl(target_model: &str) {
    for other in PDL_TARGETS.iter().filter(|t| t.model != target_model) {
        for switch in other.switches.iter().filter(|s| switch_on(s.name)) {
            tracing::warn!(
                "{}=1 ignored: it enables PDL for kernel target {}, not {target_model}",
                switch.name,
                other.model
            );
        }
    }
    let _ = PDL_TARGET.set(PDL_TARGETS.iter().find(|t| t.model == target_model));
}

fn target() -> Option<&'static PdlTarget> {
    PDL_TARGET.get().copied().flatten()
}

/// One of the served target's PDL switches is `1`: launch the kernels it
/// lists with programmatic dependent launch. Read once, after
/// [`configure_pdl`].
pub fn pdl_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| target().is_some_and(|t| t.switches.iter().any(|s| switch_on(s.name))))
}

/// Whether `func_name` launches with PDL: a switch of the served target that
/// is on lists it.
pub(super) fn pdl_kernel(func_name: &str) -> bool {
    pdl_enabled() && target().is_some_and(|t| t.pdl_kernel_with(func_name, switch_on))
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
    "w4a16_gemv_tc8_parts",
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
    "mxfp8_gemv_tc8_tiled",
    "mxfp8_gemv_tc8_touch",
    "mxfp8_gemv_tc16_touch",
    "mxfp8_gemv_tc32_touch",
    "mxfp8_gemv_tc8_grouped",
    "mxfp8_gemv_tc16_grouped",
    // The tiled, block-FP8 and multi-projection MXFP8 tiers (PDL entry +
    // touch, as SparkGLM's FP4-Local native paths run them).
    "mxfp8_gemv_tc8_tiled",
    "fp8b_gemv_tc8",
    "fp8b_gemv_tc16",
    "fp8b_gemv_tc32",
    "fp8b_gemv_tc8_tiled",
    "fp8b_gemv_tc16_tiled",
    "fp8b_gemv_tc32_tiled",
    "fp8b_gemv_tc8_ld",
    "fp8b_gemv_tc16_ld",
    "fp8b_gemv_tc32_ld",
    "mxfp8_gemv_tc8_multi",
    "mxfp8_gemv_tc16_multi",
    "mxfp8_gemv_tc32_multi",
    "mxfp8_gemv_tc8_multi_tiled",
    "mxfp8_gemv_tc16_multi_tiled",
    "mxfp8_gemv_tc32_multi_tiled",
    "fp8b_gemv_tc8_multi",
    "fp8b_gemv_tc16_multi",
    "fp8b_gemv_tc32_multi",
    "fp8b_gemv_tc8_multi_tiled",
    "fp8b_gemv_tc16_multi_tiled",
    "fp8b_gemv_tc32_multi_tiled",
    "dense_gemv_bf16_batchm",
    "dense_gemv_bf16_batchm_ahead",
    "dense_gemv_bf16_batchm_f32",
    "dense_gemv_bf16_batchm_ahead_f32",
    "dense_gemv_bf16_batch5",
    "dense_gemv_bf16_batch5_dual",
    "dense_gemv_bf16_batch5_triple_n",
    "dense_gemv_bf16_batchm_dual",
    "dense_gemv_bf16_batchm_dual_k128",
    "dense_gemv_bf16_batchm_triple_n",
    "dense_gemv_bf16_tc8",
    "dense_gemv_bf16_tc16",
    "dense_gemv_bf16_tc32",
    "dense_gemv_bf16_tc8_f32",
    "dense_gemv_bf16_tc16_f32",
    "dense_gemv_bf16_tc32_f32",
    "dense_gemv_bf16_tc8_grouped",
    "dense_gemv_bf16_tc16_grouped",
    "dense_gemv_bf16_tc32_grouped",
    "dense_gemv_bf16_tc8_planes",
    "dense_gemv_bf16_tc16_planes",
    "dense_gemv_bf16_tc32_planes",
    "glm_mla_down_fuse_tc8",
    "glm_mla_down_fuse_tc16",
    "glm_mla_down_fuse_tc32",
    "glm_mla_down_fuse_fp8b_tc8",
    "glm_mla_down_fuse_fp8b_tc16",
    "glm_mla_down_fuse_fp8b_tc32",
    "glm_mla_mid_fuse",
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
    "moe_topk_sigmoid_batched_f32",
    "moe_sort_by_expert",
    "moe_sort_by_expert_scan",
    "moe_route_mask_rows",
    "mla_cache_assemble_batched",
    "reshape_and_cache_flash",
    "glm_latent_cache_write_fp8g128",
    "glm_latent_dequant_fp8g128",
    "glm_index_layernorm_bf16",
    "glm_index_tail_write_bf16",
    "glm_index_kpool_finalize_bf16",
    // The DFlash drafter's conv and residual add (each enters first).
    "dflash2_grouped_dynamic_causal_conv",
    "bf16_residual_add",
    "moe_build_tile_worklist",
    "moe_build_tile_worklist_scan",
    "quantize_bf16_to_nvfp4",
    "quantize_bf16_to_nvfp4_weight_rne",
    "glm_moe_router_quant",
    "glm_moe_route_select",
    "glm_moe_route_one",
    "moe_w4a4_grouped_gemm_prequant_t_k64_vecscale_compact_gate_up",
    "silu_mul_quant_nvfp4",
    "silu_mul_quant_nvfp4_by_expert",
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
    "glm_moe_decode_m16s_gate_up_silu_k128w_l2pf_p",
    "glm_moe_decode_m16s_k128w_zskip_l2pf_p",
    "glm_moe_decode_m32s_gate_up_silu_k128w_l2pf_p",
    "glm_moe_decode_m32s_k128w_zskip_l2pf_p",
    "glm_moe_decode_m64s_gate_up_silu_k128w_l2pf_p",
    "glm_moe_decode_m16s_a16_k128w_l2pf_p",
    "glm_moe_decode_m16s_a16_gate_up_silu_k128w_l2pf_p",
    "glm_moe_decode_m32s_a16_k128w_l2pf_p",
    "glm_moe_decode_m32s_a16_gate_up_silu_k128w_l2pf_p",
    "glm_moe_decode_m64s_a16_k128w_l2pf_p",
    "glm_moe_decode_m64s_a16_gate_up_silu_k128w_l2pf_p",
    "glm_moe_decode_m64s_k128w_zskip_l2pf_p",
    "moe_unpermute_reduce_indexed_ep",
    "moe_unpermute_reduce_indexed_ep_vec8",
    "moe_unpermute_reduce_indexed_ep_vec8_e4m3",
    "moe_unpermute_rs_fused_vec8",
    "moe_unpermute_rs_fused_vec8_e4m3",
    "moe_batched_blend",
    "moe_unpermute_blend_ep_vec8",
    "glm_hc_moe_blend_post_bf16",
    "moe_silu_mul",
    "moe_silu_mul_touch",
    "kda_pack_qkv",
    "causal_conv1d_update_prefill_tp_snap",
    "kda_recurrent_bf16_verify_rec_owners",
    "kda_commit_records",
    "kda_commit_records_stride",
    "kda_sigmoid_gated_rms_norm",
    // The rank-split drafter's and dense TP split's joins after a chain
    // gather (18 a prose step).
    "rank_split_join_bf16",
];

/// Qwen3.8-Flash-Next (`ATLAS_QWEN4EXP_PDL=1`): the single-token decode chain
/// of a GDN layer, an attention layer and the MoE, as `atlas_pdl_enter()`
/// kernels. Not listed, on purpose: the RDMA one-shot all-reduce
/// (`rdma_oneshot_bf16` and `rdma_chain_bf16`, see their kernel note), the
/// routed-expert GEMVs, the QSA indexer and cuBLASLt. A kernel after an
/// unlisted one (or after a copy or memset) simply launches when its
/// predecessor completes.
pub(super) const QWEN4EXP_PDL_KERNELS: &[&str] = &[
    // The corrected-RNE weight converter of the draft copies enters first.
    "quantize_bf16_to_nvfp4_weight_rne",
    // mHC (ATLAS_QWEN4EXP_HC_FAST) and the fused seam.
    "hc_pre_stage_vec",
    "hc_pre_down_vec",
    "hc_pre_down_vec8",
    "hc_pre_finish_vec",
    "hc_pre_finish_vec8",
    "hc_pre_down_vec_rows",
    "hc_pre_finish_vec_rows",
    "hc_pre_down_vec_wide",
    "hc_post_vec",
    "hc_post_stage_vec",
    "hc_mma_down",
    "hc_mma_down_rows",
    "hc_mma_finish",
    "hc_mma_finish_rows",
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
    "rope_forward_mrope_interleaved_yarn",
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

/// Qwen3.8-27B (`ATLAS_QWEN38_PDL=1`, kernels of qwen3.6-27b's tree): the
/// dense prefill GEMMs with their activation quantizers and the GDN BA gates
/// (`dense_bf16mma_prefill.cu`, `gdn_ba_gates_prefill.cu`), and the prefill
/// glue that already enters first. The prefill norms, the GDN scan and conv,
/// and prefill attention are not PDL-entered: a listed kernel after one of
/// them launches when it completes. The decode chain has its own switch
/// ([`QWEN38_PDL_DECODE_KERNELS`]).
pub(super) const QWEN38_PDL_KERNELS: &[&str] = &[
    "dense_bf16mma_gate_up_silu",
    "dense_bf16mma_nvfp4",
    "dense_bf16mma_fp8",
    "dense_w4a4_gate_up_silu",
    "dense_w4a4_nvfp4",
    "dense_w4a8_gate_up_silu",
    "dense_w4a8_nvfp4",
    "dense_w4a8_gate_up_silu_mx",
    "dense_w4a8mx_nvfp4",
    "dense_w4a8ws_gate_up_silu",
    "dense_w4a8ws_nvfp4",
    "dense_w4a4ws_gate_up_silu",
    "dense_w4a4ws_nvfp4",
    "dense_w4a4wsl_gate_up_silu",
    "dense_w4a4wsl_nvfp4",
    "dense_w8a8_fp8",
    "dense_w8a8_fp8t",
    "dl_quant_nvfp4",
    "dl_quant_fp8",
    "gdn_ba_gates_prefill",
    "gdn_ba_gates_prefill_small",
    "bf16_residual_add",
    "rope_forward_mrope_interleaved",
    "reshape_and_cache_flash",
];

/// Qwen3.8-27B decode (`ATLAS_QWEN38_PDL_DECODE=1`, independent of
/// `ATLAS_QWEN38_PDL`): the single-token chain of a GDN layer, an attention
/// layer (FP8 and BF16 KV) and the dense FFN, the final norm and the LM head,
/// each `atlas_pdl_enter()` first. PDL is a property of the kernel, so a
/// listed kernel also launches with PDL where prefill or verify calls it.
/// Not listed: argmax and sampling (after the graph, behind a copy) and the
/// arms a default decode does not take (`--gdn-fused-norm`,
/// `ATLAS_NO_DECODE_SPLIT_SILU`, the TC/batch GEMV tiers).
pub(super) const QWEN38_PDL_DECODE_KERNELS: &[&str] = &[
    // Layer input norm and the residual + post-mixer norm.
    "rms_norm_residual",
    "residual_add_rms_norm",
    // GDN mixer: projections (FP8 block / NVFP4 / per-row FP8 / BF16), the
    // four small kernels.
    "w8a16_gemv",
    "w4a16_gemv",
    "w4a16_gemv_sw",
    "dense_gemv_fp8w",
    "dense_gemv_bf16",
    "dense_gemv_ba_gates",
    "causal_conv1d_update_l2norm_f32",
    "gated_delta_rule_decode_f32",
    "gated_rms_norm_f32_input",
    // Their fused twin (ATLAS_QWEN38_DECODE_FUSE).
    "qwen35_gdn_decode_fused",
    // Attention mixer: Q+gate and K/V GEMVs, q/k norm, MRoPE, the KV write
    // and paged decode attention (split-K and its reduce), the output gate.
    "w4a16_gemv_qg",
    "w4a16_gemv_dual",
    "rms_norm",
    "rope_forward_mrope_interleaved",
    "reshape_and_cache_flash",
    "reshape_and_cache_flash_fp8",
    "paged_decode_attn",
    "paged_decode_attn_fp8",
    "paged_decode_attn_splitk_fp8",
    "paged_decode_attn_reduce_fp8",
    "sigmoid_gate_mul",
    // Dense FFN: gate/up, SiLU-mul, down (the plain and `_sw` arms), and the
    // residual add.
    "w4a16_gemv_dual_sw",
    "moe_silu_mul",
    "bf16_residual_add",
    // The row-invariant tensor-core twins of the GEMVs above
    // (ATLAS_SKINNY_MMA), which take their place in decode and verify.
    "skinny_w4a16_t1",
    "skinny_w4a16_t2",
    "skinny_qg_t1",
    "skinny_qg_t2",
    "skinny_w8a16_t1",
    "skinny_w8a16_t2",
    // Their wide arms past 16 rows (batched decode and verify).
    "skinny_w4a16_w4",
    "skinny_w4a16_w8",
    "skinny_w4a16_w16",
    "skinny_w4a16_x8",
    "skinny_qg_w4",
    "skinny_qg_w8",
    "skinny_qg_w16",
    "skinny_qg_x8",
    "skinny_w8a16_w4",
    "skinny_w8a16_w8",
    "skinny_w8a16_w16",
];

#[cfg(test)]
#[path = "pdl_tests.rs"]
mod tests;
