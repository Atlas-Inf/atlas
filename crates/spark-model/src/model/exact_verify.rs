// SPDX-License-Identifier: AGPL-3.0-only
//! Bit-exact K=2/3/4 MTP verify for the dense hybrid qwen3_5 (Qwen3.8-27B)
//! (`ATLAS_EXACT_VERIFY=1`, default off; both ranks must agree,
//! `startup_parity`). The qwen4_exp switch (`ATLAS_QWEN4EXP_EXACT_VERIFY`,
//! `qwen4exp_exact_verify.rs`) is separate and unchanged; both set the
//! generic [`ModelLevers::exact_verify`](crate::layers::ops::ModelLevers),
//! which the ops the two models share read.
//!
//! With the switch, row `i` of a single-sequence verify (`verify_b.rs` K=2,
//! `verify_c.rs` K=3, `verify_c2.rs` K=4) computes the bits the serial decode
//! step at that position computes, so `Model::verify_bit_exact` is true and
//! the scheduler lets MTP speculate inside `<think>` (`mtp_gate::
//! spec_think_for`): lossless at T=0, and exact in distribution at T>0
//! because the rows are accepted by sequential sampling. Only one active
//! sequence: the multi-sequence verify is NOT exact
//! (`batch_verify_bit_exact` stays false). Not with a DFlash drafter, whose
//! verify is wider than 4 rows; the MTP depth is capped at 3 drafts.
//!
//! Row by row against serial decode (nvidia/Qwen3.8-27B-NVFP4: NVFP4 MLP and
//! attention, FP8 block-scaled GDN projections, BF16 LM head, GB10):
//!
//! | op | serial | verify (default) | under the switch |
//! |---|---|---|---|
//! | GDN qkvz, out_proj (FP8) | `w8a16_gemv` | `w8a16_gemv_batch4` | `exact_rows_proj` (same kernel; also NVFP4/BF16 copies) |
//! | GDN ba + gates | `dense_gemv_ba_gates` | `dense_gemm_ba_gates_prefill` | equal (same reduction, token-parallel) |
//! | GDN conv, recurrence, norm | FP32 conv, `decode_f32[_norm]` | BF16 conv, WY | the `--exact-verify` per-token chain (`_snap` twins) |
//! | attention Q+gate | `w4a16_gemv_qg` | `qg_batch2/3`; 4 rows `batch4_os` | 2/3 equal; 4 rows `qg_batch4` (`qkv_exact4.rs`) |
//! | attention K/V | `w4a16_gemv_dual` | `dual_batch2/3`; 4 rows `batch4_os` | `w4a16_gemv_batch2/3/4` per projection |
//! | q/k norm, MRoPE, KV write, gate | per row | strided, per row | equal (text positions: MRoPE == RoPE) |
//! | attention core | split-K from `max(max_batch, 1)` | from `max(max_batch, K)` | split count pinned to `max(max_batch, 4)` for both |
//! | o_proj | `w4a16_gemv[_sw]` | `batch2/3`; 4 rows tile GEMM | 4 rows scalar `w4a16_gemv_batch4` |
//! | attention q+gate, k, v, o (native FP8, `ATLAS_DENSE_NATIVE_FP8_ATTN`) | `w8a16_gemv` (+ `deinterleave_qg`) | one `w8a16_gemv_batch4`, M = K (`ops::fp8_rows_plan`) | equal (rows are `w8a16_gemv`'s bytes) |
//! | dense FFN | `dual[_sw]` + SiLU + `gemv[_sw]` | `dual_batch2/3`, batchm 4 | scalar `w4a16_gemv_batch2/3/4` per projection |
//! | residual/RMS norms, final norm | 1 row | `[K, h]`, one block a row | equal |
//! | LM head (BF16) | `dense_gemv_bf16` | 2 rows `_batch2`; 3/4 tile GEMM | 3/4 rows `dense_gemv_bf16_batchm` |
//! | LM head (NVFP4) | `w4a16_gemv` | `batch2`; batchm tier | scalar tier (no DP4A / tensor core) |
//!
//! Every verify arm above is a scalar template whose row `r` is the
//! single-row kernel's bytes; no tensor-core or DP4A tier can be picked
//! under the switch. Refused beside it: the FP16 h-state (the exact GDN chain
//! reads FP32), `ATLAS_DECODE_FFN_VIA_GEMM` and `ATLAS_NO_DECODE_SPLIT_SILU`
//! (serial decode's FFN would then take another arithmetic).
//!
//! `ATLAS_EXACT_VERIFY_CHECK=1` (diagnostic, default off) is the qwen4_exp
//! check (`trait_impl/exact_verify_check.rs`) on this model: before each
//! K=2/3/4 verify the tokens are decoded serially on the live state, the
//! state is put back, and the verify's logits and final hidden rows are
//! compared byte for byte (`EXACT_VERIFY_CHECK` lines, summary every 64
//! steps). Works with the switch off too, which measures the default gap.

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
        "exact verify ON (ATLAS_EXACT_VERIFY=1): K=2/3/4 single-sequence verify rows run \
         serial decode's arithmetic; MTP may speculate inside <think> (one sequence)"
    );
    Ok(true)
}

/// Whether the serial check runs for this model: the qwen4_exp check, or
/// `ATLAS_EXACT_VERIFY_CHECK` on a qwen3_5 model.
pub(crate) fn check_active(model_type: &str) -> bool {
    super::qwen4exp_exact_verify::check_active(model_type)
        || check_from(check_requested(), model_type)
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

/// `Model::verify_bit_exact`: the qwen4_exp lever, or the dense one without
/// a DFlash drafter (its verify is wider than [`MAX_ROWS`]).
pub(crate) fn verify_bit_exact(levers: &ModelLevers, dflash: bool) -> bool {
    levers.qwen4exp_exact_verify || (levers.dense_exact_verify() && !dflash)
}

/// `Model::verify_max_drafts` from the layers' own ceiling: under the dense
/// lever (no DFlash) at most [`MAX_DRAFTS`], the widest exact verify.
pub(crate) fn max_drafts(
    levers: &ModelLevers,
    dflash: bool,
    layers: Option<usize>,
) -> Option<usize> {
    if levers.dense_exact_verify() && !dflash {
        Some(layers.map_or(MAX_DRAFTS, |d| d.min(MAX_DRAFTS)))
    } else {
        layers
    }
}

#[cfg(test)]
#[path = "exact_verify_tests.rs"]
mod tests;
