// SPDX-License-Identifier: AGPL-3.0-only

//! Locks greedy argmax tie-breaking to the sampler's first-index-wins rule.

use std::path::PathBuf;

fn source() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join("kernels/gb10/common/argmax_bf16.cu")).unwrap()
}

#[test]
fn every_cuda_argmax_reduction_compares_value_then_vocab_index() {
    let src = source();
    assert!(
        src.contains("argmax_other_better"),
        "CUDA argmax needs one value-desc/index-asc comparator"
    );
    let calls = src.matches("argmax_other_better(").count();
    // One definition plus BF16 single, BF16 batch, BF16 batch+logprob, FP32.
    assert_eq!(calls, 5, "all four reductions must use the comparator");
    assert!(
        src.contains("other_idx < mine_idx"),
        "equal BF16/FP32 maxima must select the lower vocabulary index"
    );
}

#[test]
fn cross_stride_tie_proves_lower_thread_id_is_not_first_vocab_id() {
    const BLOCK: u32 = 1024;
    let first_vocab_id = 1023u32;
    let later_vocab_id = 1024u32;
    assert!(first_vocab_id < later_vocab_id);
    assert!(
        later_vocab_id % BLOCK < first_vocab_id % BLOCK,
        "fixture must make the later vocab ID live in the lower CUDA lane"
    );
    let better = |other_val: f32, other_idx: u32, mine_val: f32, mine_idx: u32| {
        other_val > mine_val || (other_val == mine_val && other_idx < mine_idx)
    };
    assert!(better(7.0, first_vocab_id, 7.0, later_vocab_id));
    assert!(!better(7.0, later_vocab_id, 7.0, first_vocab_id));
}
