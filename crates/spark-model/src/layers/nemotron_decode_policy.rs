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
}
