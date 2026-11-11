// SPDX-License-Identifier: AGPL-3.0-only
//! Both model families must retain their prefill row boundaries after integration.
use super::chunk_align_for;

#[test]
fn qwen_granules_and_glm_sequence_parallel_rows_both_align() {
    assert_eq!(chunk_align_for(8193, 1, 64), 8192);
    assert_eq!(chunk_align_for(16399, 1, 64), 16384);
    assert_eq!(chunk_align_for(8193, 512, 4), 8192);
    assert_eq!(chunk_align_for(8703, 1024, 64), 8192);
    assert_eq!(chunk_align_for(63, 1, 64), 63);
    assert_eq!(chunk_align_for(7, 1, 4), 4);
}
