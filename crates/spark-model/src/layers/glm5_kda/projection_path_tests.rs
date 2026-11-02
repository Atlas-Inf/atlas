// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn glm_kda_selects_nvfp4_gemv_for_decode_and_gemm_for_prefill() {
    assert_eq!(
        ProjectionPath::for_forward(true, true, 1000),
        ProjectionPath::DecodeGemv
    );
    assert_eq!(
        ProjectionPath::for_forward(false, true, 1000),
        ProjectionPath::PrefillTransposedM128
    );
    assert_eq!(
        ProjectionPath::for_forward(false, true, 128),
        ProjectionPath::PrefillBase
    );
    assert_eq!(
        ProjectionPath::for_forward(false, false, 1000),
        ProjectionPath::PrefillBase
    );
}

#[test]
fn glm_kda_dense_takes_cutlass_only_for_short_k_in_whole_k16_steps() {
    // f_b / g_b (K = dim = 128) on CUTLASS rows.
    assert_eq!(DensePath::for_forward(8196, 128, true), DensePath::Cutlass);
    assert_eq!(DensePath::for_forward(300, 64, true), DensePath::Cutlass);
    // CUTLASS off or short rows (`bf16_gemm_cutlass_rows` false).
    assert_eq!(
        DensePath::for_forward(8196, 128, false),
        DensePath::Pipelined
    );
    // beta / f_a / g_a (K = hidden) stay on the pipelined kernel.
    assert_eq!(
        DensePath::for_forward(8196, 4096, true),
        DensePath::Pipelined
    );
    // A K residue groups differently in CUTLASS: not bit-identical.
    for k in [40, 104, 120] {
        assert_eq!(DensePath::for_forward(8196, k, true), DensePath::Pipelined);
    }
    assert_eq!(DensePath::for_forward(1, 128, true), DensePath::Gemv);
}
