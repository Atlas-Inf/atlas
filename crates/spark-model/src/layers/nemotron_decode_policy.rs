// SPDX-License-Identifier: AGPL-3.0-only

/// Return whether one Lightning AR component should use its batched path.
///
/// The component override is diagnostic and exact-valued: `"1"` enables,
/// any other supplied value disables. When absent, the shared policy applies.
pub(crate) fn decode_multi_seq_batched(
    shared_value: Option<&str>,
    component_value: Option<&str>,
) -> bool {
    match component_value {
        Some(value) => value == "1",
        None => matches!(shared_value, Some("1")),
    }
}

#[cfg(test)]
mod tests {
    use super::decode_multi_seq_batched;

    #[test]
    fn shared_policy_requires_exact_one() {
        assert!(!decode_multi_seq_batched(None, None));
        assert!(!decode_multi_seq_batched(Some("0"), None));
        assert!(decode_multi_seq_batched(Some("1"), None));
        assert!(!decode_multi_seq_batched(Some("true"), None));
    }

    #[test]
    fn component_override_is_explicit_and_wins() {
        assert!(decode_multi_seq_batched(None, Some("1")));
        assert!(decode_multi_seq_batched(Some("0"), Some("1")));
        assert!(!decode_multi_seq_batched(Some("1"), Some("0")));
        assert!(!decode_multi_seq_batched(Some("1"), Some("true")));
    }

    #[test]
    fn verify_moe_preserves_k_width_gate_and_shares_row_independent_weights() {
        let layer = include_str!("nemotron_moe/transformer_layer.rs");
        let body = include_str!("nemotron_moe/decode_batched.rs");
        assert!(layer.contains("decode_verify_multi_shared(hidden, residual, ks"));
        assert!(body.contains("Some(ks)"));
        assert!(body.contains("normed.offset(row * h * bf16)"));
        assert!(body.contains("gate_logits.offset(row * num_experts as usize * bf16)"));
        assert!(body.contains("w4a16_gemv_batch32_k"));
        assert!(body.contains("moe_expert_gemv_wide("));
    }

    #[test]
    fn marlin_all_widths_can_replace_single_token_and_packed_verify() {
        let source = include_str!("nemotron_moe/decode_batched.rs");
        assert!(source.contains("ATLAS_LIGHTNING_MOE_MARLIN_ALL_WIDTHS"));
        assert!(source.contains("num_tokens <= 4 || marlin_all_widths"));
        assert!(source.contains("native_fp8 && !marlin_all_widths"));
    }

    #[test]
    fn lightning_multi_verify_keeps_full_mamba_stages_on_literal_m1() {
        let source = include_str!("nemotron_mamba2/trait_decode_verify_multi.rs");
        assert!(source.contains("if ctx.levers.lightning_mamba_exact_recurrence"));
        assert!(source.contains("ATLAS_LIGHTNING_MAMBA_EXACT_BATCHED"));
        assert!(source.contains("ATLAS_LIGHTNING_MAMBA_BATCH_OUT"));
        assert!(source.contains("ATLAS_LIGHTNING_MAMBA_EXACT_PERSISTENT"));
        assert!(source.contains("(r_total - row).min(16)"));
        assert!(source.contains("decode_out_proj_exact"));
        assert!(source.contains("self.decode("));
        assert!(source.contains("h_state_intermediates[t]"));
        assert!(source.contains("conv_state_intermediates[t]"));
    }

    #[test]
    fn layer_trace_covers_literal_m1_and_k4_without_graph_capture() {
        let trace = include_str!("../model/trait_impl/verify_layer_trace.rs");
        let m1 = include_str!("../model/trait_impl/decode_a3.rs");
        let k4 = include_str!("../model/trait_impl/verify_d.rs");
        assert!(trace.contains("ATLAS_LIGHTNING_VERIFY_LAYER_TRACE"));
        assert!(m1.contains("trace_lightning_hidden_rows(\"m1\""));
        assert!(k4.contains("\"k4\""));
        assert!(k4.contains("!super::verify_layer_trace::enabled()"));
    }

    #[test]
    fn literal_m1_oracle_preserves_every_logits_row_for_scheduler_processing() {
        let source = include_str!("../model/trait_impl/verify_d_serial.rs");
        assert!(source.contains("logits_base.offset((t + 1) * logits_row_bytes)"));
        assert!(source.contains("logits_base.offset(t * logits_row_bytes)"));
        assert!(source.contains("for t in 0..tokens.len()"));
    }

    #[test]
    fn marlin_one_wave_uses_fixed_m8_with_full_product_slot_capacity() {
        let source = include_str!("nemotron_moe/marlin_slots.rs");
        let ops = include_str!("ops/marlin_nvfp4.rs");
        assert!(source.contains("ATLAS_LIGHTNING_MOE_MARLIN_ONE_WAVE"));
        assert!(source.contains("let wave = if one_wave { num_tokens } else { 4 }"));
        assert!(source.contains("w4a16_gemv_batch32_k"));
        assert!(ops.contains("pub const MARLIN_SLOTS: i32 = 128"));
    }
}
