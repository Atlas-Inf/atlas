// SPDX-License-Identifier: AGPL-3.0-only

//! The shapes `skinny_mma_microtest` runs (split from the example for the
//! 500-line cap; moved as is).

use spark_model::layers::ops::skinny_mma::Family;

/// `(label, family, N, K, heads, head_dim)`; heads/head_dim only for Q+gate.
pub(super) const SHAPES: &[(&str, Family, usize, usize, u32, u32)] = &[
    (
        "w4 gate/up  [17408 x 5120]",
        Family::W4a16,
        17408,
        5120,
        0,
        0,
    ),
    (
        "w4 down     [5120 x 17408]",
        Family::W4a16,
        5120,
        17408,
        0,
        0,
    ),
    (
        "w4 gdn qkvz [16384 x 5120]",
        Family::W4a16,
        16384,
        5120,
        0,
        0,
    ),
    (
        "w4 o_proj   [ 5120 x 6144]",
        Family::W4a16,
        5120,
        6144,
        0,
        0,
    ),
    (
        "w4 k/v      [ 1024 x 5120]",
        Family::W4a16,
        1024,
        5120,
        0,
        0,
    ),
    (
        "w4 lm_head [248077 x 5120]",
        Family::W4a16,
        248_077,
        5120,
        0,
        0,
    ),
    (
        "w4 tail     [10307 x 2560]",
        Family::W4a16,
        10307,
        2560,
        0,
        0,
    ),
    (
        "qg q+gate   [12288 x 5120]",
        Family::Qg,
        12288,
        5120,
        24,
        256,
    ),
    ("qg tail     [  500 x 1024]", Family::Qg, 500, 1024, 5, 50),
    // Inkling 2 Mini's Spark projections (hidden 2,304) and vocabulary head
    (
        "w4 ink q    [ 3072 x 2304]",
        Family::W4a16,
        3072,
        2304,
        0,
        0,
    ),
    (
        "w4 ink k/v  [ 1024 x 2304]",
        Family::W4a16,
        1024,
        2304,
        0,
        0,
    ),
    (
        "w4 ink o    [ 2304 x 3072]",
        Family::W4a16,
        2304,
        3072,
        0,
        0,
    ),
    ("w4 ink rel  [  384 x 2304]", Family::W4a16, 384, 2304, 0, 0),
    (
        "w4 ink head [200058 x 2304]",
        Family::W4a16,
        200_058,
        2304,
        0,
        0,
    ),
    (
        "w8 gdn qkvz [16384 x 5120]",
        Family::W8a16,
        16384,
        5120,
        0,
        0,
    ),
    (
        "w8 gdn out  [ 5120 x 6144]",
        Family::W8a16,
        5120,
        6144,
        0,
        0,
    ),
    (
        "w8 tail     [10308 x 2048]",
        Family::W8a16,
        10308,
        2048,
        0,
        0,
    ),
];
