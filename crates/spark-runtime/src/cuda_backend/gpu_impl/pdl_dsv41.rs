// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1-Flash's PDL list (`ATLAS_DSV41_PDL=1`).

/// The kernels of a decode step and of a verify's few rows: each starts
/// with `atlas_pdl_enter()` (`dsv41_common.cuh` includes it). A decode step
/// is about 1,100 launches and a verify 1,500, most of them a few
/// microseconds long, and without PDL each waits a couple of microseconds
/// for the one before to leave the GPU: 2 ms of a 29 ms step. The kernels
/// only a prefill's batch runs are left out, and do not wait: with them in,
/// a 32K prefill was 2.5% slower. The last is the one kernel of the
/// DeepSeek-V4 tree this target compiles that waits; V4.1 does not launch
/// it.
pub(super) const DSV41_PDL_KERNELS: &[&str] = &[
    "dsv41_accumulate",
    "dsv41_attn_front",
    "dsv41_attn_index",
    "dsv41_attn_logits",
    "dsv41_attn_logits_tc",
    "dsv41_attn_output",
    "dsv41_attn_output_tc_few",
    "dsv41_attn_schedule",
    "dsv41_compress_pool",
    "dsv41_engram_gate",
    "dsv41_engram_rows",
    "dsv41_expert_act",
    "dsv41_expert_in",
    "dsv41_expert_out",
    "dsv41_experts_act",
    "dsv41_experts_in",
    "dsv41_experts_out",
    "dsv41_experts_plan",
    "dsv41_fp4_gemv",
    "dsv41_fp4_gemv_tc8",
    "dsv41_fp4_gemv_tc8_f32",
    "dsv41_fp4_quant_dequant",
    "dsv41_fp8_gemv_groups",
    "dsv41_fp8_gemv_groups_q",
    "dsv41_fp8_gemv_pieces",
    "dsv41_fp8_gemv_pieces_q",
    "dsv41_fp8_gemv_tc8",
    "dsv41_fp8_gemv_tc8_f32",
    "dsv41_fp8_gemv_tc8_f32_q",
    "dsv41_fp8_gemv_tc8_q",
    "dsv41_fp8_gemv_two",
    "dsv41_fp8_quant_dequant",
    "dsv41_gemv_f32",
    "dsv41_hc_collapse",
    "dsv41_hc_expand",
    "dsv41_hc_expand_parts",
    "dsv41_hc_mix_finish",
    "dsv41_hc_mix_sums",
    "dsv41_hc_site_finish",
    "dsv41_hc_site_sums",
    "dsv41_hc_start",
    "dsv41_index_candidates",
    "dsv41_index_choose",
    "dsv41_index_score",
    "dsv41_markov_round",
    "dsv41_moe_pick",
    "dsv41_moe_route",
    "dsv41_moe_scores",
    "dsv41_moe_soften",
    "dsv41_parts_round",
    "dsv41_parts_rows",
    "dsv41_rotate",
    "dsv41_round_bf16",
    "dsv41_slice_columns",
    "dsv41_slots_sum",
    "dsv41_sparse_attn",
    "dsv41_stream_mean",
    "dsv41_swiglu",
    "mla_cache_assemble_batched",
];
