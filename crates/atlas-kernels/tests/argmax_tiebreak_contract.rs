// SPDX-License-Identifier: AGPL-3.0-only

//! Locks greedy argmax tie-breaking to FIRST-index-wins — the rule llama.cpp's
//! greedy sampler and vLLM's `torch.argmax` share, and the rule every
//! committed BFCL record for Qwen3.6/3.8-27B was measured under. The
//! 2026-09-03 switch to last-index-wins (`argmax_other_better`, "value
//! descending, then higher vocabulary index") moved Qwen3.8-27B's BFCL
//! normalized score from 83.38 to 83.10 on gb10: BF16 logits tie often at
//! temperature 0, and on the nine flipped samples of the gate draw the old
//! rule answered five correctly against two. Reverting the kernel alone
//! restored all nine. The host paths (`verify_pipeline_helper::argmax`,
//! the greedy logit processor) follow the same rule so the engine does not
//! disagree with itself.

use std::path::PathBuf;

fn source() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join("kernels/gb10/common/argmax_bf16.cu")).unwrap()
}

#[test]
fn cuda_argmax_keeps_the_first_strict_maximum() {
    let src = source();
    assert!(
        !src.contains("argmax_other_better"),
        "the last-index-wins comparator must not come back silently"
    );
    assert!(
        !src.contains("other_idx > mine_idx"),
        "equal maxima must never select the higher vocabulary index"
    );
    // Every scan advances only on a strictly greater value: first max wins.
    assert!(
        src.matches("if (v > local_max)").count() >= 3,
        "each strided scan must keep the first strict maximum"
    );
}

#[test]
fn first_index_wins_reference_rule() {
    // The rule the kernel's strided scan + lower-tid tree merge implements and
    // the host helpers (`spark_runtime::sampler::argmax_first_wins_f32`) share:
    // advance only on a strictly greater value.
    let v = [1.0f32, 7.0, 3.0, 7.0, 2.0];
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    assert_eq!(best, 1);
}

/// The scan above only guarantees first-wins WITHIN one lane's stride class.
///
/// `if (v > local_max)` keeps the first strict max of the indices
/// `tid, tid+stride, tid+2*stride, ...`. Two equal maxima in DIFFERENT lanes
/// are then resolved by the tree merge, and a plain `>` merge picks the lower
/// LANE — which is a higher index whenever the lower index sits in the higher
/// lane. With stride 1024 and the max at index 1029 (lane 5) and 2050 (lane 2),
/// `>` returns 2050 while `argmax_first_wins_f32` returns 1029.
///
/// So the kernel needs the tie clause in every merge, not just the strict
/// comparison in the scans. This test is the guard for that clause: it was
/// absent until 2026-09-16, and the file's "same index as n sequential calls"
/// claim was asserted rather than earned.
#[test]
fn cuda_argmax_merge_is_index_exact_on_equal_maxima() {
    let src = source();
    let merges = src.matches("s_idx[tid + s] < s_idx[tid]").count();
    assert!(
        merges >= 4,
        "every argmax merge must prefer the lower INDEX on equal maxima; found \
         {merges} tie clauses, expected one per merge site (argmax_bf16, \
         argmax_bf16_batch, argmax_bf16_batch_lp, argmax_fp32). A plain `>` \
         merge resolves ties to the lowest LANE, which is a HIGHER index when \
         the lower index is in the higher lane — and that divergence is \
         serial-vs-verify, not cosmetic."
    );
    // NEGATIVE half: the detector must fire on the pre-2026-09-16 shape, or
    // this test would pass on a file whose merge had been reverted.
    let legacy = "if (s_val[tid + s] > s_val[tid]) { s_val[tid] = s_val[tid + s]; }";
    assert_eq!(legacy.matches("s_idx[tid + s] < s_idx[tid]").count(), 0);
}
