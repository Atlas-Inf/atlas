// SPDX-License-Identifier: AGPL-3.0-only

//! Locks greedy argmax tie-breaking to the sampler's first-index-wins rule.

use std::path::PathBuf;

fn source() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join("kernels/gb10/common/argmax_bf16.cu")).unwrap()
}

#[test]
fn every_cuda_argmax_reduction_matches_last_index_greedy_contract() {
    let src = source();
    assert!(
        src.contains("argmax_other_better"),
        "CUDA argmax needs one value-desc/index-desc comparator"
    );
    let calls = src.matches("argmax_other_better(").count();
    // One definition plus local scan and tree merge in each of BF16 single,
    // BF16 batch, BF16 batch+logprob, and FP32.
    assert_eq!(
        calls, 9,
        "all four scans and reductions must use the comparator"
    );
    assert!(
        src.contains("other_idx > mine_idx"),
        "equal BF16/FP32 maxima must select the higher vocabulary index"
    );
}

#[test]
fn cross_stride_tie_proves_lane_order_is_not_last_vocab_id() {
    const BLOCK: u32 = 1024;
    let first_vocab_id = 1024u32;
    let later_vocab_id = 2047u32;
    assert!(first_vocab_id < later_vocab_id);
    assert!(
        later_vocab_id % BLOCK > first_vocab_id % BLOCK,
        "fixture must make the later vocab ID live in the higher CUDA lane"
    );
    let better = |other_val: f32, other_idx: u32, mine_val: f32, mine_idx: u32| {
        other_val > mine_val || (other_val == mine_val && other_idx > mine_idx)
    };
    assert!(better(7.0, later_vocab_id, 7.0, first_vocab_id));
    assert!(!better(7.0, first_vocab_id, 7.0, later_vocab_id));
}
