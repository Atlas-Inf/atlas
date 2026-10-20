// SPDX-License-Identifier: AGPL-3.0-only
//! Exact multi-sequence batching for qwen4_exp (Qwen3.8-Flash-Next)
//! (`ATLAS_QWEN4EXP_BATCH_FAST=1`, default off; both ranks must agree,
//! `startup_parity`).
//!
//! With the switch, row `i` of a batched multi-sequence decode step computes
//! exactly the bits a single-sequence decode of row `i`'s sequence computes:
//! the same kernels per row, or batched kernels whose rows are byte-identical
//! to them, and collectives whose per-element sums are the per-row sums. The
//! batched step therefore commits the tokens and logits that C1 commits, at
//! any concurrency, while reading the weights once per step where it can.
//!
//! What each op of the batched decode runs under the switch, against the
//! single-row decode (`decode()`, TP1 or TP=EP=2, BF16 GDN projections,
//! NVFP4 attention/experts, BF16 KV):
//!
//! | op | serial | batched (switch on) | per row |
//! |---|---|---|---|
//! | embed, PLE, mHC pre/post/head | per row / T=1 | per row / T=n (`hc_pre_split`, `_vec`; past 8 rows in 8-row split chunks, under `ATLAS_QWEN4EXP_HC_FAST` row-grouped launches of up to 32 (`hc_pre_*_vec_rows`), not the GEMM collapse) | equal |
//! | GDN qkvz, out_proj (BF16) | `dense_gemv_bf16` | `dense_gemv_bf16_batchm` (<= 8 rows a launch; 9..32 `qwen4exp_bf16_rows16/32`, `ops::Qwen4ExpWideRows`) instead of cuBLASLt | equal |
//! | GDN qkvz, out_proj (FP8 opt-in) | `w8a16_gemv` | `w8a16_gemv_batch4/16` | equal |
//! | GDN ba+gates, conv, recurrence, norm | per row | the per-sequence loop (the strided batched recurrence is skipped) | equal |
//! | attention Q+gate | `w4a16_gemv_qg` (one accumulator, scale folded in) | `qg_batch2/3/4` in <= 4-row chunks, past 8 rows `qwen4exp_qg_rows16/32` — never the scalar template the default 4+-row arm runs | equal |
//! | attention K/V | `w4a16_gemv_dual` | `w4a16_gemv_batch2..8` per projection (as exact verify), <= 8 rows a launch, past 8 rows `batch16` | equal |
//! | QSA select / attention | bs=1 | per row (`qsa_rows.rs`), also for rows of several sequences | equal |
//! | attention o_proj | `w4a16_gemv_sw` | `w4a16_gemv_batch2..8` | equal |
//! | MoE router + top-k | `w4a16_gemv_sw`, `moe_topk_softmax` | router `w4a16_gemv_batchN` (row pair; `batch16` past 8 rows), top-k per row | equal |
//! | MoE experts | `moe_expert_{gate_up,silu_down}_shared` | the same kernels per row, or their row-gridded twins (`qwen4exp_moe_rows.cu`) | equal |
//! | MoE EP all-reduce + shared blend | `[1,h]` then `moe_batched_blend` T=1 | ONE `[n,h]` all-reduce, `moe_batched_blend` T=n | equal (one commutative BF16 add; the blend is a block per row) |
//! | TP all-reduces | `[1,h]` | `[n,h]` | equal |
//! | LM head (BF16, optionally vocab-split) | `dense_gemv_bf16` | `dense_gemv_bf16_batchm` (<= 8 rows a launch; 9..32 `qwen4exp_bf16_rows16/32`) instead of `dense_gemm_bf16` | equal |
//!
//! The switch also keeps a batch with an ACTIVE QSA selection (a sequence
//! past the 2051-token inert bound) on the batched step, served row by row
//! inside it, where the default sends the whole batch to per-sequence
//! decode under a parallel communicator.
//!
//! With `ATLAS_QWEN4EXP_EXACT_VERIFY=1` as well, the batched multi-sequence
//! MTP verify (`verify_e`) runs under a parallel communicator (TP2), with
//! every row on the exact-verify arithmetic, so speculation stays on at
//! C2..C8 (`ATLAS_MTP_MAX_SEQS` > 1) without giving up exactness.
//!
//! `ATLAS_QWEN4EXP_BATCH_FAST_CHECK=1` (diagnostic, default off; both ranks)
//! proves it on a live model: before each batched decode step, every row's
//! token runs through single-sequence decode on its live state, the logits
//! and final hidden rows are kept, the state is put back, and the batched
//! step's rows are compared byte for byte (`BATCH_FAST_CHECK` lines).
//!
//! `ATLAS_QWEN4EXP_BATCH_FAST_BISECT=<mask>` (diagnostic, both ranks) forces
//! components of a batched step back to one single-row launch per row, as C1
//! runs them, to find the one a CHECK mismatch comes from: 1 MoE experts per
//! row, 2 MoE `forward` per row (per-row all-reduce too), 4 mHC collapse one
//! row a launch, 8 GDN mixer through `ssm_forward` per sequence, 16 attention
//! Q/K/V and o_proj per row, 32 attention core per row, 64 LM head per row
//! (`ops::BISECT_*`).
//!
//! `ATLAS_QWEN4EXP_BATCH_SMALL=1` (default off; needs the lane) takes the
//! small kernels the table above still runs once per row into one launch
//! each, every row the bytes of its single-row launch
//! (`scripts/dev/qwen4exp_batch_small_bench.cu`):
//!
//! | per row (lane) | one launch (BATCH_SMALL) |
//! |---|---|
//! | `moe_topk_softmax` | `moe_topk_softmax_rows`, a block per row, the single-row lower-index tie-break (not `moe_topk_softmax_batched`'s) |
//! | `moe_weighted_sum_blend` | `moe_weighted_sum_blend_rows`, blockIdx.y the row |
//! | GDN `dense_gemv_ba_gates`, `causal_conv1d_update_l2norm_f32`, `gated_delta_rule_decode_f32`, `gated_rms_norm_f32_input_sigmoid` per sequence | `qwen4exp_gdn_decode_fused_rows`, up to 8 sequences a launch, each on its own state: the exact fused step of `ATLAS_QWEN4EXP_DECODE_FUSE` |
//! | exact MTP verify (`ATLAS_QWEN4EXP_EXACT_VERIFY`), per sequence and token: conv, conv rollback copy, recurrence, gated norm, H rollback copy | `qwen4exp_gdn_verify_fused_rows`, up to 8 sequences of up to 8 tokens a launch, H and the conv windows in registers across the tokens (single-sequence verify too) |
//!
//! An 8-row decode step drops from 8 x (48 + 48 + 36 x 4) = 1,920 of these
//! launches to 48 + 48 + 36 = 132 (240 a row to 16.5); an 8-sequence K=2
//! verify's GDN chain from 36 x 8 x 8 = 2,304 to 36. It changes no
//! collective, so the ranks need not agree on it (no startup-parity entry).
//!
//! Refused beside `ATLAS_W4A16_TC=1`: the tensor-core GEMV tiers it selects
//! are not the scalar GEMV's arithmetic.

use std::sync::OnceLock;

use anyhow::{Result, bail};

/// `ATLAS_QWEN4EXP_BATCH_FAST=1`, read once.
pub(crate) fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_BATCH_FAST").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_BATCH_FAST_BISECT=<mask>` (diagnostic, default 0): the
/// `ops::BISECT_*` components of a batched step forced to one single-row
/// launch per row, to find which one parts from C1. Read once.
pub(crate) fn bisect_requested() -> u32 {
    static MASK: OnceLock<u32> = OnceLock::new();
    *MASK.get_or_init(|| {
        parse_mask(
            std::env::var("ATLAS_QWEN4EXP_BATCH_FAST_BISECT")
                .ok()
                .as_deref(),
        )
    })
}

/// A decimal or `0x` hex mask; anything else (and unset) is 0.
fn parse_mask(v: Option<&str>) -> u32 {
    v.map(str::trim)
        .and_then(|v| {
            v.strip_prefix("0x")
                .map_or_else(|| v.parse().ok(), |h| u32::from_str_radix(h, 16).ok())
        })
        .unwrap_or(0)
}

/// `ATLAS_QWEN4EXP_BATCH_SMALL=1`, read once.
pub(crate) fn small_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_BATCH_SMALL").as_deref() == Ok("1"))
}

/// The model's `qwen4exp_batch_small` lever, asked once the lane is on.
pub(crate) fn small_lever() -> bool {
    let on = small_requested();
    if on {
        tracing::info!(
            "qwen4_exp exact small-kernel batching ON (ATLAS_QWEN4EXP_BATCH_SMALL=1): MoE \
             top-k, MoE blend and the GDN step one launch over the rows"
        );
    }
    on
}

/// `ATLAS_QWEN4EXP_MOE_UNITS=1`, read once: under the lane, the rows pair's
/// routed + shared experts go through `qwen4exp_moe_c8.cu`'s expert units
/// (each weight decoded once for every row that picked it, SiLU in the
/// gate/up epilogue, down's lane chains rebalanced) -- the same bytes as the
/// rows pair. `startup_parity` carries it anyway, so an A/B never mixes arms.
pub(crate) fn units_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MOE_UNITS").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_MOE_TC=1`, read once: the expert units on TENSOR CORES
/// (`qwen4exp_moe_c8_tc.cu`) -- exactness contract (b), a new numerics
/// baseline: BF16 weights `lut * dec(scale)` (exact), BF16 activations,
/// FP32 MMA accumulation, scale2 per output. Row-invariant, and serial
/// decode's single-row MoE switches to the same kernels (`forward_row_local`),
/// so speculation stays exact against serial decode under the switch; NOT
/// today's bytes. Takes precedence over `ATLAS_QWEN4EXP_MOE_UNITS`. Both
/// ranks must agree on it (each rank's routed sums meet in the EP all-reduce).
pub(crate) fn tc_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MOE_TC").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_MOE_TC_V1=1`, read once: under `ATLAS_QWEN4EXP_MOE_TC`,
/// the tensor-core units as first shipped (for A/B against the current
/// ones, which write the same bytes).
pub(crate) fn tc_v1_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MOE_TC_V1").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_MOE_TC_V2=1`, read once: under `ATLAS_QWEN4EXP_MOE_TC`,
/// the v2 units (separate gate/up and down launches) instead of v3's one
/// persistent launch (`qwen4exp_moe_c8_tc3.cu`), for A/B: the same bytes.
pub(crate) fn tc_v2_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MOE_TC_V2").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_MOE_NO_CLAMP=1`, read once: the qwen4_exp decode / verify
/// MoE kernels drop the routed SwiGLU clamp (+-10) -- the serial-decode
/// single-row silu/down, the rows pair, the units and the TC units -- which
/// the checkpoint does not declare (`swiglu_limit` null; the vLLM reference
/// is plain SiLU) and the prefill, K=2/K=3 and `_t` arms never applied.
/// Not today's bytes where the clamp bit (it fired on 0.002% of layer 47's
/// gate/up values in a C8 prose capture, `scripts/dev/qwen4exp_moe_fidelity.cu`);
/// every decode path switches together, so speculation stays exact.
pub(crate) fn no_clamp_requested(model_type: &str) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    model_type == "qwen4_exp"
        && *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_MOE_NO_CLAMP").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_MOE_ROUTE_DUMP=<file>` (diagnostic, local only: its
/// `.bin` holds request activations; `layers/moe/route_dump.rs`): each
/// rows-pair MoE launch appends its `[rows, top_k]` expert ids to `<file>` as
/// one line (synchronizing the stream). `scripts/dev/qwen4exp_moe_c8_bench.cu`
/// replays the lines (`ROUTES=<file>`). Read once.
pub(crate) fn route_dump_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        let path = std::env::var("ATLAS_QWEN4EXP_MOE_ROUTE_DUMP")
            .ok()
            .filter(|p| !p.is_empty());
        if let Some(p) = &path {
            tracing::warn!(
                "ATLAS_QWEN4EXP_MOE_ROUTE_DUMP={p}: writing MoE routes and input \
                 activations of every request to disk; local diagnosis only"
            );
        }
        path
    })
    .as_deref()
}

/// `ATLAS_QWEN4EXP_BATCH_FAST_CHECK=1`, read once.
pub(crate) fn check_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_BATCH_FAST_CHECK").as_deref() == Ok("1"))
}

/// The model's `qwen4exp_batch_fast` lever: the switch, on a qwen4_exp model.
pub(crate) fn lever(model_type: &str) -> Result<bool> {
    if model_type == "qwen4_exp" {
        warn_inert_numerics(requested(), tc_requested(), units_requested());
    }
    lever_from(
        requested(),
        model_type,
        crate::layers::w4a16_gemv_tiers::tc_requested(),
    )
}

/// Log the MoE switches that do not do what they say on their own: the
/// units only exist under the lane, and prefill's BF16 routed experts are
/// MoE_TC decode's numerics, so without it prefill and decode differ.
fn warn_inert_numerics(lane: bool, tc: bool, units: bool) {
    if !lane && (tc || units) {
        tracing::warn!(
            "ATLAS_QWEN4EXP_MOE_TC / _MOE_UNITS ignored: they need ATLAS_QWEN4EXP_BATCH_FAST=1"
        );
    }
    if crate::layers::moe::tcp_requested() && !(lane && tc) {
        tracing::warn!(
            "ATLAS_QWEN4EXP_PREFILL_MOE_BF16 (or _PREFILL_BF16_PROJ) without \
             ATLAS_QWEN4EXP_BATCH_FAST=1 + ATLAS_QWEN4EXP_MOE_TC=1: prefill's routed experts \
             take the TC decode numerics, decode does not"
        );
    }
}

fn lever_from(requested: bool, model_type: &str, w4a16_tc: bool) -> Result<bool> {
    if !requested || model_type != "qwen4_exp" {
        return Ok(false);
    }
    if w4a16_tc {
        bail!(
            "ATLAS_QWEN4EXP_BATCH_FAST=1 needs the scalar W4A16 GEMV tiers, whose rows \
             are byte-identical to a GEMV a row; ATLAS_W4A16_TC=1 swaps in tensor-core \
             tiers that are not. Unset ATLAS_W4A16_TC."
        );
    }
    tracing::info!(
        "qwen4_exp exact batching ON (ATLAS_QWEN4EXP_BATCH_FAST=1): every row of a \
         batched multi-sequence step runs single-sequence decode's arithmetic"
    );
    Ok(true)
}

/// Whether the check runs for this model (both ranks: `startup_parity`).
pub(crate) fn check_active(model_type: &str, lever: bool) -> bool {
    lever && check_requested() && model_type == "qwen4_exp"
}

#[cfg(test)]
mod tests {
    use super::{lever_from, parse_mask};

    #[test]
    fn bisect_mask_parses_decimal_and_hex() {
        assert_eq!(parse_mask(None), 0);
        assert_eq!(parse_mask(Some("16")), 16);
        assert_eq!(parse_mask(Some(" 0x7f ")), 0x7f);
        assert_eq!(parse_mask(Some("moe")), 0);
    }

    #[test]
    fn lever_is_qwen4exp_only_and_refuses_tensor_core_gemv_tiers() {
        assert!(!lever_from(false, "qwen4_exp", false).unwrap());
        assert!(!lever_from(true, "qwen3_next", false).unwrap());
        assert!(lever_from(true, "qwen4_exp", false).unwrap());
        assert!(lever_from(true, "qwen4_exp", true).is_err());
        // Off, the tensor-core tiers are none of its business.
        assert!(!lever_from(false, "qwen4_exp", true).unwrap());
    }
}
