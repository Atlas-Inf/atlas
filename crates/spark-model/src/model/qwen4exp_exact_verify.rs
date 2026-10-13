// SPDX-License-Identifier: AGPL-3.0-only
//! Bit-exact K=2/3/4 MTP verify for qwen4_exp (Qwen3.8-Flash-Next)
//! (`ATLAS_QWEN4EXP_EXACT_VERIFY=1`, default off; both ranks must agree,
//! `startup_parity`).
//!
//! With the switch, row `i` of a single-sequence verify (`verify_b.rs` K=2,
//! `verify_c.rs` K=3, `verify_c2.rs` K=4) computes exactly the bits the serial
//! decode step at that position computes: same kernels, same accumulation
//! order, same inputs. A T=0 generation with `--speculative --num-drafts
//! 1|2|3` (K=3/4 need `ATLAS_QWEN4EXP_MTP_DEPTH`, `qwen4exp_mtp_depth.rs`) then
//! commits the tokens, and the logits bytes, a non-speculative run commits,
//! so speculation is lossless and the scheduler lets it run inside
//! `<think>` (`Model::verify_bit_exact`, `mtp_gate::spec_think_for`).
//!
//! What the verify forward runs, row by row against serial decode (default
//! profile: BF16 GDN projections, NVFP4 attention/experts, BF16 KV, TP1 or
//! TP=EP=2):
//!
//! | op | serial | verify | per row |
//! |---|---|---|---|
//! | embed | row copy | row copy | equal |
//! | mHC pre/post/head | `hc_pre_split` T=1 | T=K, per-token blocks | equal (also `_vec`) |
//! | PLE (layer 1) | projections `dense_gemm_bf16_pipelined` M=1 | M=K | equal |
//! | GDN qkvz, out_proj | `dense_gemv_bf16` | `_batch2` / `_batchm` | equal |
//! | GDN (FP8 opt-in) | `w8a16_gemv` | `w8a16_gemv_batch4` | equal |
//! | GDN ba + gates | `dense_gemv_ba_gates` | `dense_gemm_ba_gates_prefill` | equal |
//! | GDN conv, recurrence, gated norm | FP32 conv, `decode_f32`, `_f32_input` norm | BF16 conv, `wy2`, BF16 norm | **differ** -> the exact chain |
//! | attention Q+gate | `w4a16_gemv_qg` | `w4a16_gemv_qg_batch2/3`; 4 rows `w4a16_gemv_batch4_os` | 2/3 equal; 4 **differ** -> `w4a16_gemv_qg_batch4` |
//! | attention K/V | `w4a16_gemv_dual` | `w4a16_gemv_dual_batch2/3`; 4 rows `_batch4_os` | **differ** (2/3) -> `w4a16_gemv_batch2/3/4` per projection |
//! | q/k norm, rope, KV write, gate | per row | strided, per row | equal |
//! | QSA select / attention | bs=1 | per row (`qsa_rows.rs`), BF16 paged decode has no split-K | equal |
//! | attention o_proj | `w4a16_gemv_sw` | `w4a16_gemv_batch2/3`; 4 rows DP4A / TC tier when opted in | equal -> scalar `w4a16_gemv_batch4` |
//! | MoE, attention layers | `ffn.forward` | per row `ffn.forward` | equal |
//! | MoE, GDN layers | `ffn.forward` | `forward_k2/k3` | **differ** (top-k ties, originals-layout kernels, no +-10 clamp) -> per row `ffn.forward` |
//! | TP all-reduces | `[1,h]` | `[K,h]` | equal (one commutative BF16 add) |
//! | LM head (BF16) | `dense_gemv_bf16` | 2 rows per row or `_batchm`; 3/4 rows scalar `dense_gemm_bf16` | 3/4 **differ** (1 output in ~12k) -> `dense_gemv_bf16_batchm` rows |
//!
//! The three differing ops move to the serial arithmetic under the switch
//! ([`ModelLevers::qwen4exp_exact_verify`](crate::layers::ops::ModelLevers)):
//! the GDN chain takes the `--exact-verify` per-token arm
//! (`qwen3_ssm::verify_exact_for`), K/V take `w4a16_gemv_batchN` (byte-
//! identical rows, one weight pass each), and the GDN layers' MoE runs each
//! row through `ffn.forward` (`trait_decode_batched_hc.rs`). Per-kernel row
//! parity and cost: `scripts/dev/qwen4exp_exact_verify_bench.cu`.
//!
//! The 4-row arms: `multi_seq/qkv_exact4.rs` (Q/K/V), `attn/o_proj.rs`
//! (scalar batch4 o_proj), `HeadArith::batched` + `lm_head_batched` (head).
//!
//! `ATLAS_QWEN4EXP_EXACT_VERIFY_CHECK=1` (diagnostic, default off; both ranks)
//! proves it on a live model: before each K=2/3/4 verify, the K tokens run
//! through serial decode on the live state, their logits and final hidden
//! rows are kept, the state is put back (GDN h/conv copied back, PLE carry
//! restored from its aux blob, QSA ingest rewound as a reject does), and the
//! verify then runs as usual. Its rows are compared byte for byte and every
//! mismatch is logged (`EXACT_VERIFY_CHECK`), with a running summary. Row
//! `i` is checked whether or not the drafts were accepted: the serial run
//! decodes the drafts too. Works with the switch off as well, which measures
//! how far the default verify is from serial. Roughly triples a verify step.

use std::sync::OnceLock;

use anyhow::{Result, bail};

/// `ATLAS_QWEN4EXP_EXACT_VERIFY=1`, read once.
pub(crate) fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_EXACT_VERIFY").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_EXACT_VERIFY_CHECK=1`, read once.
pub(crate) fn check_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_EXACT_VERIFY_CHECK").as_deref() == Ok("1"))
}

/// The model's `qwen4exp_exact_verify` lever: the switch, on a qwen4_exp
/// model. Refused beside the FP16 h-state, whose pool the exact GDN chain's
/// FP32 kernels must not read.
pub(crate) fn lever(model_type: &str) -> Result<bool> {
    lever_from(
        requested(),
        model_type,
        crate::layers::qwen3_ssm::ssm_h_fp16_enabled(),
    )
}

fn lever_from(requested: bool, model_type: &str, h_f16: bool) -> Result<bool> {
    if !requested || model_type != "qwen4_exp" {
        return Ok(false);
    }
    if h_f16 {
        bail!(
            "ATLAS_QWEN4EXP_EXACT_VERIFY=1 needs the FP32 GDN h-state: the exact \
             verify chain is serial decode's FP32 kernels. Drop --ssm-h-dtype f16."
        );
    }
    tracing::info!(
        "qwen4_exp exact verify ON (ATLAS_QWEN4EXP_EXACT_VERIFY=1): K=2/3/4 verify rows \
         run serial decode's arithmetic; speculation may run inside <think>"
    );
    Ok(true)
}

/// Whether the check runs for this model (both ranks: `startup_parity`).
pub(crate) fn check_active(model_type: &str) -> bool {
    check_requested() && model_type == "qwen4_exp"
}

#[cfg(test)]
mod tests {
    use super::lever_from;

    #[test]
    fn lever_is_qwen4exp_only_and_refuses_the_f16_h_state() {
        assert!(!lever_from(false, "qwen4_exp", false).unwrap());
        assert!(!lever_from(true, "qwen3_next", false).unwrap());
        assert!(lever_from(true, "qwen4_exp", false).unwrap());
        assert!(lever_from(true, "qwen4_exp", true).is_err());
        // Off, the FP16 h-state is none of its business.
        assert!(!lever_from(false, "qwen4_exp", true).unwrap());
    }
}
