// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 launch-limit predicate tests. Split from `preflight.rs` (500-LoC cap).

use super::{
    glm5_concurrency_supported, glm5_context_supported, glm5_dual_spark_parallelism,
    glm5_long_context_concurrency_supported,
};

#[test]
fn glm5_accepts_ep_fallback_and_overlapping_tp2_only() {
    assert!(glm5_dual_spark_parallelism(2, 1, 2));
    assert!(glm5_dual_spark_parallelism(2, 2, 2));
    assert!(!glm5_dual_spark_parallelism(2, 2, 1));
    assert!(!glm5_dual_spark_parallelism(4, 2, 2));
    assert!(!glm5_dual_spark_parallelism(2, 4, 2));
}

#[test]
fn glm5_concurrency_is_bounded_and_requires_ep_v2() {
    assert!(glm5_concurrency_supported(1, 1, false, false));
    assert!(glm5_concurrency_supported(3, 5, true, false));
    assert!(!glm5_concurrency_supported(2, 5, false, false));
    assert!(!glm5_concurrency_supported(4, 5, true, false));
    assert!(!glm5_concurrency_supported(3, 2, true, false));
    assert!(!glm5_concurrency_supported(3, 6, true, false));
    assert!(!glm5_concurrency_supported(4, 4, true, false));
    assert!(glm5_concurrency_supported(4, 4, true, true));
    assert!(!glm5_concurrency_supported(4, 5, true, true));
    assert!(!glm5_concurrency_supported(4, 4, false, true));
}

#[test]
fn glm5_long_context_stays_within_model_limit_and_uses_chunking() {
    assert!(glm5_context_supported(100_000, 1024, 1_048_576));
    assert!(!glm5_context_supported(1_048_577, 1024, 1_048_576));
    assert!(!glm5_context_supported(100_000, 0, 1_048_576));
}

#[test]
fn glm5_long_context_concurrency_requires_explicit_sparse_decode() {
    assert!(glm5_long_context_concurrency_supported(
        100_000, 2048, 1, 1, false, false
    ));
    assert!(!glm5_long_context_concurrency_supported(
        100_000, 2048, 2, 2, false, false
    ));
    assert!(glm5_long_context_concurrency_supported(
        2048, 2048, 3, 5, false, false
    ));
    assert!(glm5_long_context_concurrency_supported(
        16384, 2048, 3, 3, true, false
    ));
    assert!(glm5_long_context_concurrency_supported(
        16384, 2048, 2, 5, true, false
    ));
    assert!(!glm5_long_context_concurrency_supported(
        16384, 2048, 4, 4, true, false
    ));
    assert!(!glm5_long_context_concurrency_supported(
        16384, 2048, 1, 5, true, false
    ));
}

#[test]
fn glm5_long_c4_requires_its_own_bounded_opt_in() {
    assert!(glm5_long_context_concurrency_supported(
        16384, 2048, 4, 4, true, true
    ));
    for (context, active, admitted, sparse) in [
        (16385, 4, 4, true),
        (16384, 4, 5, true),
        (16384, 5, 5, true),
        (16384, 4, 4, false),
    ] {
        assert!(!glm5_long_context_concurrency_supported(
            context, 2048, active, admitted, sparse, true
        ));
    }
}
