// SPDX-License-Identifier: AGPL-3.0-only

/// Return whether the Lightning AR multi-sequence path should batch.
///
/// Only the exact value `"1"` opts into batching. Missing, `"0"`, and every
/// other value keep the serial diagnostic path selected.
pub(crate) fn decode_multi_seq_batched(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

#[cfg(test)]
mod tests {
    use super::decode_multi_seq_batched;

    #[test]
    fn only_exact_one_enables_batched_decode() {
        assert!(!decode_multi_seq_batched(None));
        assert!(!decode_multi_seq_batched(Some("0")));
        assert!(decode_multi_seq_batched(Some("1")));
        assert!(!decode_multi_seq_batched(Some("true")));
    }
}
