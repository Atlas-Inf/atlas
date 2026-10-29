// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use super::while_on;
use crate::model::{graph_flags, qwen4exp_batch_fast, qwen4exp_exact_verify, qwen4exp_mtp_depth};

/// The table's trailing entries, split for the 500-LoC cap. The caller
/// chains them after [`super::SETTINGS`], so the gathered order is
/// unchanged.
pub(crate) const SETTINGS_TAIL: &[(&str, fn() -> Result<u64>)] = &[
    ("ATLAS_QWEN4EXP_EXACT_VERIFY_CHECK", || {
        Ok(qwen4exp_exact_verify::check_requested() as u64)
    }),
    // The qwen4_exp verify width ceiling: the K the head dispatches and the
    // worker mirrors.
    ("ATLAS_QWEN4EXP_MTP_DEPTH", || {
        Ok(qwen4exp_mtp_depth::requested() as u64)
    }),
    // The dynamic MTP depth sizes every <= 8-wide slot's verify pools for
    // `--num-drafts` (`speculative::deep_depth`): a worker without it would
    // refuse the head's deep windows.
    ("ATLAS_MTP_DYNAMIC_DEPTH", || {
        Ok(crate::speculative::deep_depth::enabled() as u64)
    }),
    // The widest batch whose slots' verify pools are sized for deep drafts:
    // a worker sized narrower refuses the head's deep windows.
    ("ATLAS_MTP_DEEP_MAX_SEQS", || {
        while_on(
            crate::speculative::deep_depth::enabled(),
            crate::speculative::deep_depth::deep_max_seqs,
        )
    }),
    // The qwen4_exp exact batching lane: the batched MoE all-reduce shape,
    // the batched QSA-active step (instead of per-sequence decode), the
    // batched multi-sequence verify under TP, and the check's serial steps.
    ("ATLAS_QWEN4EXP_BATCH_FAST", || {
        Ok(qwen4exp_batch_fast::requested() as u64)
    }),
    ("ATLAS_QWEN4EXP_BATCH_FAST_CHECK", || {
        Ok(qwen4exp_batch_fast::check_requested() as u64)
    }),
    // Per-row MoE / GDN arms change the collective count.
    ("ATLAS_QWEN4EXP_BATCH_FAST_BISECT", || {
        Ok(qwen4exp_batch_fast::bisect_requested() as u64)
    }),
    // Which MXFP8 GEMV family sums the decode/verify projections: two
    // summation orders in one out_proj all-reduce.
    ("ATLAS_QWEN4EXP_MXFP8_DECODE", || {
        Ok(crate::layers::ops::mxfp8_decode_tc() as u64)
    }),
    // Whether prefill projections are W8A8 or BF16-expansion GEMMs: the
    // rows a rank writes into the prefix cache and the SP gathers.
    ("ATLAS_QWEN4EXP_PREFILL_W8A8", || {
        Ok(crate::layers::ops::qwen4exp_w8a8::enabled() as u64)
    }),
    ("ATLAS_SSM_SAVE_DUMP", || {
        Ok(graph_flags::ssm_save_dump() as u64)
    }),
    ("ATLAS_LIGHTNING_VERIFY_LAYER_TRACE", || {
        Ok(graph_flags::verify_layer_trace() as u64)
    }),
    ("ATLAS_MS_PROFILE", || Ok(graph_flags::ms_profile() as u64)),
    ("ATLAS_DFLASH_DEBUG_NO_GRAPH", || {
        Ok(graph_flags::dflash_debug_no_graph() as u64)
    }),
    ("ATLAS_K2_DIAG", || Ok(graph_flags::k2_diag() as u64)),
];
