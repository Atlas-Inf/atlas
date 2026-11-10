// SPDX-License-Identifier: AGPL-3.0-only
//! Bit-exact single-sequence verify for the dense hybrid qwen3_5
//! (Qwen3.8-27B): MTP K=2/3/4 and the DFlash K=γ verify up to
//! [`DFLASH_MAX_ROWS`] rows (`ATLAS_EXACT_VERIFY=1`, default off; both ranks
//! must agree, `startup_parity`). The qwen4_exp switch
//! (`ATLAS_QWEN4EXP_EXACT_VERIFY`, `qwen4exp_exact_verify.rs`) is separate
//! and unchanged; both set the generic
//! [`ModelLevers::exact_verify`](crate::layers::ops::ModelLevers), which the
//! ops the two models share read.
//!
//! With the switch, row `i` of a single-sequence verify (`verify_b.rs` K=2,
//! `verify_c.rs` K=3, `verify_c2.rs` K=4, `verify_d.rs` K=γ for DFlash)
//! computes the bits the serial decode step at that position computes, so
//! `Model::verify_bit_exact` is true and the scheduler lets MTP or DFlash
//! speculate inside `<think>` (`mtp_gate::spec_think_for`, no
//! `ATLAS_DFLASH_SPEC_THINK` needed): lossless at T=0, and exact in
//! distribution at T>0 because the rows are accepted by sequential
//! sampling. With `ATLAS_SKINNY_MMA=1` as well the batched multi-sequence
//! decode and verify are exact too ([`batch_lever`], the section below), so
//! `batch_verify_bit_exact` holds and a concurrent batch speculates inside
//! `<think>`; without it only ONE active sequence does. The MTP depth is capped at
//! [`MAX_DRAFTS`]; beside a DFlash drafter γ is capped at
//! [`DFLASH_MAX_DRAFTS`] (`ModelLevers::exact_verify_rows` = 16, which also
//! widens the attention split pin and admits the 5..=16-row arms below).
//!
//! Row by row against serial decode (nvidia/Qwen3.8-27B-NVFP4: NVFP4 MLP and
//! attention, FP8 GDN projections (per-tensor scale broadcast to the block
//! grid), NVFP4 LM head, GB10). "K" is the verify width (rows):
//!
//! | op | serial | verify (default) | under the switch |
//! |---|---|---|---|
//! | GDN qkvz, out_proj (FP8) | `w8a16_gemv` | `w8a16_gemv_batch4`; 5..16 `batch16` | `exact_rows_proj`: `batch4` to 4 rows, `w8a16_gemv_batch16` 5..16 (rows are `w8a16_gemv`'s) |
//! | GDN ba + gates | `dense_gemv_ba_gates` | `dense_gemm_ba_gates_prefill` | equal (same reduction, token-parallel) |
//! | GDN conv, recurrence, norm | FP32 conv, `decode_f32[_norm]` (or the one-launch `ATLAS_QWEN38_DECODE_FUSE` twin) | BF16 conv, WY / wyN | the `--exact-verify` per-token chain (`_snap` twins), any K |
//! | attention Q+gate | `w4a16_gemv_qg` | `qg_batch2/3`; 4..8 `batch4/8_os`; 9+ tile GEMM | 2/3 equal; 4..16 `qg_batch4` chunks, the last 1..3 rows `qg[_batch2/3]` (`qkv_exact4.rs`) |
//! | attention K/V | `w4a16_gemv_dual` | `dual_batch2/3`; 4..8 `batch4/8_os`; 9+ tile GEMM | `w4a16_gemv_batch2/3/N` per projection, batch16 past 8 rows |
//! | q/k norm, MRoPE, KV write, gate | per row | strided, per row | equal (text positions: MRoPE == RoPE) |
//! | attention core | split-K from `max(max_batch, 1)` | from `max(max_batch, K)` | split count pinned to `max(max_batch, 4)` (16 beside DFlash) for both |
//! | o_proj | `w4a16_gemv[_sw]` | `batch2/3`; 4+ rows tile GEMM | scalar `w4a16_gemv_batchN` (batch16 past 8 rows) |
//! | attention q+gate, k, v, o (native FP8, `ATLAS_DENSE_NATIVE_FP8_ATTN`) | `w8a16_gemv` (+ `deinterleave_qg`) | one `w8a16_gemv_batch4`/`batch16` (`ops::fp8_rows_plan`) | equal on GB10 (rows are `w8a16_gemv`'s bytes) |
//! | dense FFN | `dual[_sw]` + SiLU + `gemv[_sw]` | `dual_batch2/3`, batchm 4..8; GDN layers 9+ `forward_prefill` | scalar `w4a16_gemv_batch2/3/N` per projection, batch16 past 8 rows (`dense_ffn_exact.rs`) |
//! | residual/RMS norms, final norm | 1 row | `[K, h]`, one block a row | equal |
//! | LM head (BF16) | `dense_gemv_bf16` | 2 rows `_batch2`; 3+ tile GEMM | `dense_gemv_bf16_batchm` |
//! | LM head (NVFP4) | `w4a16_gemv` | `batch2`; batchm tier / DP4A; 9+ wide or tile | scalar tier; 9..16 rows `w4a16_gemv_batch16` |
//!
//! Every verify arm above is a scalar template whose row `r` is the
//! single-row kernel's bytes; no tensor-core or DP4A tier can be picked
//! under the switch. `ATLAS_WIDE_GEMV=1` (`layers/ops/wide_gemv.rs`) serves
//! the 5..=16-row NVFP4, FP8 and Q+gate arms with one `wide_gemv.cu` launch
//! each instead: the same rows' bytes, fewer instructions a weight byte.
//! `ATLAS_SKINNY_MMA=1` (`layers/ops/skinny_mma.rs`) goes further: serial
//! decode itself and every one of these NVFP4, Q+gate and FP8 arms run one
//! row-invariant tensor-core kernel family, so the rows stay serial decode's
//! bytes while serial decode's own arithmetic changes. Refused beside it: the FP16 h-state (the exact GDN chain
//! reads FP32), `ATLAS_DECODE_FFN_VIA_GEMM` and `ATLAS_NO_DECODE_SPLIT_SILU`
//! (serial decode's FFN would then take another arithmetic).
//!
//! `ATLAS_ATTN_SHARED_SCAN=1` (`layers/ops/shared_scan.rs`) does the same for
//! the attention core over a BF16 KV cache: serial decode and every verify
//! width run `attn_shared_scan`, which reads a KV head's keys and values once
//! for the rows that share its blocks and scores them on tensor cores. The
//! positions are cut by the row's own length and every sum is per query, so a
//! row's bytes do not depend on the launch's rows
//! (`examples/attn_shared_scan_check.rs`); serial decode's own attention
//! arithmetic changes with the switch.
//!
//! The DFlash path's target-side extras do not touch the target's
//! arithmetic: the hidden-row capture for the drafter's context
//! (`kgamma_dflash_capture`, `try_dflash_capture` in serial decode) only
//! copies residual rows out, and the drafter's context commit, warm context
//! (`ATLAS_DFLASH_WARM_CTX`) and proposal run on drafter state. The proposal
//! itself is not exact and need not be: it only proposes.
//!
//! ## Several sequences (`ATLAS_EXACT_VERIFY=1` + `ATLAS_SKINNY_MMA=1`)
//!
//! A batched multi-sequence decode (`decode_a2`, one row a sequence) or
//! verify (`verify_e`, `Σ ks` rows, DFlash or MTP) runs the same layer bodies
//! as the single-sequence verify above at a wider row count, up to
//! [`BATCH_MAX_ROWS`]. Under [`batch_lever`] (`ModelLevers::dense_batch_exact`)
//! every op whose arithmetic was keyed on the row count takes, at every
//! width, the arm whose rows are serial decode's bytes:
//!
//! | op | batched (default) | under the batch lever |
//! |---|---|---|
//! | NVFP4, Q+gate and FP8 projections past 16 rows (attention q/k/v/o, GDN qkvz/out, dense FFN) | tile GEMMs (`forward_prefill`, `wide_verify_gemm`, `w8a16_gemm`) | the skinny kernels at any row count (`skinny_mma::MAX_ROWS`: one `x8` / `w4..w16` launch, the weight streamed once; a row's bits never depend on the launch's rows) |
//! | dense FFN of a 5+-row GDN decode step | `forward_prefill` tile GEMMs | `forward_km`'s row-exact arm (`dense_ffn_exact.rs`) |
//! | dense FFN past 16 rows (attention and GDN layers) | `forward_prefill` | the same row-exact arm |
//! | attention core | split count from `max(pin, rows)` | pin raised to [`BATCH_MAX_ROWS`] (on this model one split either way) |
//! | GDN conv, recurrence, norm | per-sequence exact chain / strided twin | unchanged (already per-sequence exact) |
//! | LM head (BF16), batched decode and mixed forward | `dense_gemm` tile GEMM | `dense_gemv_bf16` rows (`head_project`) |
//! | LM head (BF16), past 8 rows | `dense_gemv_bf16_batchm` 8-row passes | `qwen4exp_bf16_rows16/32` passes (rows `dense_gemv_bf16`'s bytes, `ops::Qwen4ExpWideRows`) |
//!
//! The scheduler half: a bit-exact batched DFlash verify picks each
//! sequence's rows through serial decode's logits pipeline
//! (`verify_dflash_batch_step`, as `verify_dflash_step`'s `exact_picks`
//! does for one sequence), so the thinking budget's `</think>`, the
//! watchdogs and the penalties land on the token serial decode puts them on.
//!
//! `ATLAS_BATCH_EXACT_CHECK=1` (diagnostic, default off) is the qwen4_exp
//! batching check (`trait_impl/batch_fast_check.rs`) on this model: before
//! every batched decode step and batched verify, each sequence's rows are
//! decoded serially on its live state, the state is put back, and the
//! batched step's logits and final hidden rows are compared byte for byte
//! (`BATCH_FAST_CHECK` lines).
//!
//! `ATLAS_EXACT_VERIFY_CHECK=1` (diagnostic, default off) is the qwen4_exp
//! check (`trait_impl/exact_verify_check.rs`) on this model: before each
//! K=2/3/4 verify, and each DFlash K=γ verify of up to [`DFLASH_MAX_ROWS`]
//! rows, the tokens are decoded serially on the live state, the state is put
//! back, and the verify's logits and final hidden rows are compared byte for
//! byte (`EXACT_VERIFY_CHECK` lines, summary every 64 steps). Works with the
//! switch off too, which measures the default gap.

use std::sync::OnceLock;

use anyhow::{Result, bail};

use crate::layers::ops::ModelLevers;

/// Widest single-sequence MTP verify the switch makes exact (`verify_c2.rs`).
pub const MAX_ROWS: usize = 4;
/// The MTP depth that verify serves.
pub const MAX_DRAFTS: usize = MAX_ROWS - 1;
/// Widest single-sequence DFlash (K=γ, `verify_d.rs`) verify the switch makes
/// exact: every row-exact twin below serves it in one launch (the scalar
/// `w4a16_gemv_batch16` / `w8a16_gemv_batch16` templates).
pub const DFLASH_MAX_ROWS: usize = 16;
/// The DFlash γ that verify serves.
pub const DFLASH_MAX_DRAFTS: usize = DFLASH_MAX_ROWS - 1;
/// Widest batched multi-sequence step the batch lever makes exact: the
/// batched verify's row cap (`trait_impl/verify_e2.rs`, `VERIFY_ROW_CAP`).
pub const BATCH_MAX_ROWS: usize = 128;
/// The dense hybrid the switch is for.
const MODEL_TYPE: &str = "qwen3_5";

/// Whether an `ATLAS_EXACT_VERIFY` value asks for the switch: exactly `1`.
pub fn requested_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// `ATLAS_EXACT_VERIFY=1`, read once.
pub fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| requested_from(std::env::var("ATLAS_EXACT_VERIFY").ok().as_deref()))
}

/// `ATLAS_EXACT_VERIFY_CHECK=1`, read once.
pub(crate) fn check_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| requested_from(std::env::var("ATLAS_EXACT_VERIFY_CHECK").ok().as_deref()))
}

/// Settings beside which serial decode would not run the arithmetic the
/// exact verify rows reproduce.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Conflicts {
    /// `--ssm-h-dtype f16`: the exact GDN chain's FP32 kernels must not read
    /// the FP16 pool.
    pub h_f16: bool,
    /// `ATLAS_DECODE_FFN_VIA_GEMM=1`: serial decode's FFN runs tile GEMMs.
    pub ffn_via_gemm: bool,
    /// `ATLAS_NO_DECODE_SPLIT_SILU`: serial decode fuses SiLU into down.
    pub no_split_silu: bool,
}

/// The model's dense exact-verify lever: the switch, on a qwen3_5 model.
pub(crate) fn lever(model_type: &str, levers: &ModelLevers) -> Result<bool> {
    lever_from(
        requested(),
        model_type,
        Conflicts {
            h_f16: crate::layers::qwen3_ssm::ssm_h_fp16_enabled(),
            ffn_via_gemm: levers.decode_ffn_via_gemm,
            no_split_silu: std::env::var_os("ATLAS_NO_DECODE_SPLIT_SILU").is_some(),
        },
    )
}

pub(crate) fn lever_from(requested: bool, model_type: &str, c: Conflicts) -> Result<bool> {
    if !requested {
        return Ok(false);
    }
    if model_type != MODEL_TYPE {
        tracing::warn!(
            "ATLAS_EXACT_VERIFY=1 ignored: model_type {model_type} is not {MODEL_TYPE} \
             (qwen4_exp has ATLAS_QWEN4EXP_EXACT_VERIFY)"
        );
        return Ok(false);
    }
    let refuse = [
        (
            c.h_f16,
            "--ssm-h-dtype f16 (the exact GDN chain reads the FP32 h-state)",
        ),
        (
            c.ffn_via_gemm,
            "ATLAS_DECODE_FFN_VIA_GEMM=1 (serial FFN runs tile GEMMs)",
        ),
        (
            c.no_split_silu,
            "ATLAS_NO_DECODE_SPLIT_SILU (serial FFN fuses SiLU into down)",
        ),
    ];
    if let Some((_, why)) = refuse.iter().find(|(on, _)| *on) {
        bail!("ATLAS_EXACT_VERIFY=1 cannot run beside {why}: drop one of the two.");
    }
    tracing::info!(
        "exact verify ON (ATLAS_EXACT_VERIFY=1): single-sequence verify rows (MTP K=2/3/4, \
         DFlash K=γ up to {DFLASH_MAX_ROWS}) run serial decode's arithmetic; speculation may \
         enter <think> (one sequence)"
    );
    Ok(true)
}

/// Both switches the batch lever needs are set (`ATLAS_EXACT_VERIFY=1`,
/// `ATLAS_SKINNY_MMA=1`): load-time sites that run before the model's levers
/// exist (the head's wide BF16 tiers) ask this.
pub(crate) fn batch_requested() -> bool {
    requested() && crate::layers::ops::skinny_mma::requested()
}

/// The model's batch lever (`ModelLevers::dense_batch_exact`): the dense
/// exact verify with the skinny kernels active, whose rows are invariant to
/// the launch's row count at any width the batch lever serves.
pub(crate) fn batch_lever(levers: &ModelLevers) -> bool {
    batch_lever_from(
        levers.dense_exact_verify(),
        crate::layers::ops::skinny_mma::requested(),
        crate::layers::ops::skinny_mma::active().is_some(),
    )
}

pub(crate) fn batch_lever_from(dense_exact: bool, skinny_requested: bool, skinny: bool) -> bool {
    if !(dense_exact && skinny_requested) {
        return false;
    }
    if !skinny {
        tracing::warn!(
            "ATLAS_EXACT_VERIFY=1 + ATLAS_SKINNY_MMA=1: the skinny kernels did not resolve, so \
             the batched multi-sequence steps stay inexact (one sequence speculates in <think>)"
        );
        return false;
    }
    tracing::info!(
        "exact batching ON (ATLAS_EXACT_VERIFY=1 + ATLAS_SKINNY_MMA=1): every row of a batched \
         multi-sequence decode or verify, up to {BATCH_MAX_ROWS} rows, runs serial decode's \
         arithmetic; a concurrent batch speculates inside <think>"
    );
    true
}

/// `ATLAS_BATCH_EXACT_CHECK=1`, read once.
pub(crate) fn batch_check_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| requested_from(std::env::var("ATLAS_BATCH_EXACT_CHECK").ok().as_deref()))
}

/// Whether the batched-step check (`trait_impl/batch_fast_check.rs`) runs
/// for this model under the batch lever.
pub(crate) fn batch_check_active(model_type: &str, lever: bool) -> bool {
    lever && check_from(batch_check_requested(), model_type)
}

/// Whether the serial check runs for this model: the qwen4_exp check, or
/// `ATLAS_EXACT_VERIFY_CHECK` on a qwen3_5 model.
pub(crate) fn check_active(model_type: &str) -> bool {
    super::qwen4exp_exact_verify::check_active(model_type)
        || check_from(check_requested(), model_type)
}

/// Whether the serial check runs before a DFlash K=γ verify
/// (`verify_d.rs`): `ATLAS_EXACT_VERIFY_CHECK` on a qwen3_5 model only, so
/// the qwen4_exp check keeps the K=2/3/4 verifies it was built for.
pub(crate) fn kgamma_check_active(model_type: &str) -> bool {
    check_from(check_requested(), model_type)
}

fn check_from(requested: bool, model_type: &str) -> bool {
    requested && model_type == MODEL_TYPE
}

/// The widest exact verify under the dense lever: the K=γ verify's beside a
/// DFlash drafter, the MTP K=4 verify's otherwise
/// (`ModelLevers::exact_verify_rows`).
pub(crate) fn max_rows(dflash: bool) -> usize {
    if dflash { DFLASH_MAX_ROWS } else { MAX_ROWS }
}

/// `Model::verify_bit_exact`: the qwen4_exp lever, or the dense one (MTP
/// K=2/3/4, or a DFlash K=γ verify of up to [`DFLASH_MAX_ROWS`] rows).
pub(crate) fn verify_bit_exact(levers: &ModelLevers) -> bool {
    levers.qwen4exp_exact_verify || levers.dense_exact_verify()
}

/// `Model::verify_max_drafts` from the layers' own ceiling: under the dense
/// lever at most the widest exact verify's drafts ([`MAX_DRAFTS`], or
/// [`DFLASH_MAX_DRAFTS`] beside a DFlash drafter).
pub(crate) fn max_drafts(
    levers: &ModelLevers,
    dflash: bool,
    layers: Option<usize>,
) -> Option<usize> {
    if levers.dense_exact_verify() {
        let cap = max_rows(dflash) - 1;
        Some(layers.map_or(cap, |d| d.min(cap)))
    } else {
        layers
    }
}

#[cfg(test)]
#[path = "exact_verify_tests.rs"]
mod tests;
