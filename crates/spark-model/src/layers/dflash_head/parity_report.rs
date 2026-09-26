// SPDX-License-Identifier: AGPL-3.0-only

//! Pure comparisons for the generic-DFlash B×gamma parity diagnostic.
//!
//! Exact byte parity is not expected once M goes from γ to B·γ — GEMM tiling
//! and reduction order change — so generic DFlash reports instead of bailing
//! (the Lightning product keeps the fail-closed checks).

use half::bf16;

/// Count bitwise-mismatched BF16 elements and return the max |a − b| over
/// them. Length mismatches count the unpaired elements as mismatched.
pub(super) fn bf16_diff(actual: &[u8], expected: &[u8]) -> (usize, f32) {
    let common = actual.len().min(expected.len()) & !1;
    let mut mismatched = (actual.len().max(expected.len()) - common) / 2;
    let mut max_abs = 0.0f32;
    for i in (0..common).step_by(2) {
        let lhs = &actual[i..i + 2];
        let rhs = &expected[i..i + 2];
        if lhs == rhs {
            continue;
        }
        mismatched += 1;
        let a = bf16::from_bits(u16::from_le_bytes([lhs[0], lhs[1]])).to_f32();
        let b = bf16::from_bits(u16::from_le_bytes([rhs[0], rhs[1]])).to_f32();
        max_abs = max_abs.max((a - b).abs());
    }
    (mismatched, max_abs)
}

/// Per-sequence draft agreement: how many sequences match the oracle prefix
/// exactly, and how many oracle tokens the native drafts reproduce.
/// Returns `(sequences_exact, tokens_equal, tokens_total)`.
pub(super) fn draft_agreement(native: &[Vec<u32>], oracle: &[Vec<u32>]) -> (usize, usize, usize) {
    let mut sequences_exact = 0usize;
    let mut tokens_equal = 0usize;
    let mut tokens_total = 0usize;
    for (native_tokens, oracle_tokens) in native.iter().zip(oracle.iter()) {
        tokens_total += oracle_tokens.len();
        if native_tokens.get(..oracle_tokens.len()) == Some(oracle_tokens.as_slice()) {
            sequences_exact += 1;
            tokens_equal += oracle_tokens.len();
            continue;
        }
        tokens_equal += oracle_tokens
            .iter()
            .zip(native_tokens.iter())
            .take_while(|(lhs, rhs)| lhs == rhs)
            .count();
    }
    (sequences_exact, tokens_equal, tokens_total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn bf16_diff_identical_is_clean() {
        let bytes = bf16_bytes(&[1.0, -2.5, 0.0]);
        assert_eq!(bf16_diff(&bytes, &bytes), (0, 0.0));
    }

    #[test]
    fn bf16_diff_counts_and_measures() {
        let a = bf16_bytes(&[1.0, 2.0, 3.0]);
        let b = bf16_bytes(&[1.0, 2.5, 4.0]);
        let (mismatched, max_abs) = bf16_diff(&a, &b);
        assert_eq!(mismatched, 2);
        assert!((max_abs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bf16_diff_counts_length_mismatch() {
        let a = bf16_bytes(&[1.0, 2.0]);
        let b = bf16_bytes(&[1.0]);
        assert_eq!(bf16_diff(&a, &b), (1, 0.0));
    }

    #[test]
    fn draft_agreement_exact_and_prefix() {
        let native = vec![vec![10, 20, 30], vec![7, 9, 1]];
        let oracle = vec![vec![10, 20, 30], vec![7, 8]];
        assert_eq!(draft_agreement(&native, &oracle), (1, 4, 5));
    }

    #[test]
    fn draft_agreement_empty_is_zero() {
        assert_eq!(draft_agreement(&[], &[]), (0, 0, 0));
    }

    #[test]
    fn draft_agreement_shorter_native_is_not_exact() {
        let native = vec![vec![5]];
        let oracle = vec![vec![5, 6]];
        assert_eq!(draft_agreement(&native, &oracle), (0, 1, 2));
    }
}
