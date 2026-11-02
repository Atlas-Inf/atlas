// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_multi_seq — batched-projection SSM mixer.

/// Tensor-core mixer projections for wide decode batches. **ON by default at
/// n>=9**; `ATLAS_SSM_TC_PROJ=&lt;n&gt;` moves the threshold, `=0` disables.
///
/// WHY: the mixer's qkvz/out_proj run through `w4a16_gemv_batchm`, a SCALAR-FMA
/// kernel. It reads the weights once for all n rows, but its arithmetic scales
/// with n, so it stops being weight-bound at about M=3 and measures 2.3-3.0
/// TFLOP/s at M=16. `w4a16_gemm_t` is an M64/N128 FP8-MMA tile GEMM on the
/// SAME weights and is roughly flat in M. This is the same class of fix as the
/// wide dense-FFN arm (+30% at C=16) applied to the other half of the mixer.
///
/// Costs nothing to try: `qkvz_nvfp4_t` / `out_proj_nvfp4_t` are transposed
/// NVFP4 twins ALREADY built at load (init.rs) and already used by the SSM
/// PREFILL path, so there is no repack, no new buffer and no extra VRAM.
///
/// MEASURED 2 reps/cell, coherence identical:
///   GEMV (old):   C=8 57.8 | C=16 79.6
///   TC n>=9:      C=8 57.6 | C=16 **86.9  (+9.2%)**
///   TC n>=5:      C=8 54.9 (**regresses**) | C=16 86.4
/// So the mixer's crossover is 9 — NOT the FFN's 5. Different shapes, different
/// crossover; do not assume one transfers to the other.
///
/// ACCURACY DEBT: `w4a16_gemm_t` dequants the FP4 weight to E4M3 and converts
/// the BF16 activations to E4M3 (W4A8) where the GEMV path is W4A16, so this
/// CAN move a greedy token. It is the production SSM PREFILL path for these
/// exact two weights, and the coherence smoke is identical, but a BFCL gate is
/// owed before this merges. Read ONCE — this site runs under graph capture.
#[path = "ssm_batched/impl.rs"]
mod batched_impl;
