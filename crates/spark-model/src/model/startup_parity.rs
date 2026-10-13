// SPDX-License-Identifier: AGPL-3.0-only

//! Startup agreement on the settings every rank must run with the same value.
//!
//! Each rank reads its switches from its own environment and command line,
//! and the worker mirrors the head command by command. A rank whose value of
//! one of these differs (a stale profile on one node) issues other
//! collectives or command words than its peer, takes another share of work
//! the pair splits, or fills a cache differently. The pair then deadlocks or
//! pairs unrelated collectives at some prompt-dependent step (one rank's
//! restore-depth reduction read as the other's KV-admission vote, which
//! refuses a cold prompt), or serves wrong numerics silently.
//!
//! So every rank gathers every rank's values once, right after the
//! communicator comes up, and fails on any difference, naming the setting and
//! both values. The gather is itself a collective: it runs whatever the
//! settings are, and its size is the length of `SETTINGS` plus the
//! caller's, which no setting changes. Only a build changes it, so a gather
//! of one word, the table's id, goes first and fails ranks on different
//! builds before their settings gathers could mispair. A rank whose parser
//! refuses a value has no settings to gather: it takes part in that first
//! gather with `REFUSED`, which ends the agreement there on every rank.
//!
//! `ATLAS_STARTUP_PARITY=warn` logs a disagreement or a refusal and boots
//! anyway; the gathers are the same, and ranks on different builds still
//! fail.
//!
//! # What belongs in the table
//!
//! A setting whose mismatch changes the command words or collectives of a
//! step (their count, order, size or transport), the share of a split
//! computation a rank takes, the lane both ranks must take through a step,
//! or a cache both ranks must fill alike. Its value comes from the parser the
//! feature itself reads, never from a second reading of the variable. A
//! setting that only matters under a switch reads 0 while that switch is off
//! (`while_on`), so stale leftovers do not fail a boot.
//!
//! Not here: what the ranks already reconcile (`ATLAS_KV_MAX_BLOCKS` takes
//! the pair's minimum; the RDMA pair compares its capacity and one-shot
//! settings at bootstrap), what is local to a rank (the drafter, the rails,
//! logging, and the scheduler, which only the head runs: the worker follows
//! its commands), and kernel choices that only reorder one rank's own
//! arithmetic. Not here either, and not reconciled: the switches that pick
//! the routed-MoE arm of a row count off the expert-TP lane
//! (`ATLAS_MOE_DECODE_ARM`, `ATLAS_GLM_C3_GROUPED_MOE`, ...). On that lane
//! every arm but the K=5 one is the grouped prefill MoE, so only the K=5
//! switches are carried.

use anyhow::{Result, anyhow, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::radix_tree::{glm_pc_evict_enabled, snap_evict_alpha, snap_evict_legacy};

use super::glm_long_verify::{oracle_enabled, serial_diagnostic};
use super::trait_impl::finish_leaf;
use super::trait_impl::prefill_b::pc_policy as pc;
use super::{
    decode_pieces, glm_c4, glm_independent, glm_vocab_split, graph_flags, mtp_carry,
    qwen4exp_batch_fast, qwen4exp_exact_verify, qwen4exp_lmhead_split, qwen4exp_mtp_depth,
    verify_pieces,
};
use crate::layer::glm_long_owner;
use crate::layers::dflash_head::rank_split;
use crate::layers::qwen3_attention::{
    glm_mla_multi_seq_enabled, glm_multi_seq_sparse_enabled, glm_multi_seq_sparse_graphs_enabled,
    grouped_routed_decode_enabled, grouped_routed_decode_min, index_split_words,
    pairwise_moe_decode_enabled, write_floor_legacy,
};
use crate::layers::{self, glm_kv_shard, glm_sp, moe, ops, qwen3_ssm, w4a16_gemv_tiers};
use crate::speculative::glm_repair_policy;
use crate::weight_loader::qwen4_exp::GdnProjections;

/// One agreed setting: its name and a rank's value.
pub type Setting = (&'static str, u64);

/// A setting that only matters while its switch is `on`.
fn while_on(on: bool, value: fn() -> usize) -> Result<u64> {
    Ok(if on { value() as u64 } else { 0 })
}

/// `ATLAS_STARTUP_PARITY=warn`: log what [`agree`] would fail on and boot.
fn warn_only() -> bool {
    std::env::var("ATLAS_STARTUP_PARITY").as_deref() == Ok("warn")
}

/// The model type the GLM parsers are asked about.
const GLM: &str = "glm5_next";

/// The settings read in this crate: name, and this process's value from the
/// feature's own parser. One entry per setting.
const SETTINGS: &[(&str, fn() -> Result<u64>)] = &[
    // A rank that would only warn beside one that fails.
    ("ATLAS_STARTUP_PARITY=warn", || Ok(warn_only() as u64)),
    // The head's command words.
    ("ATLAS_EP_PROTOCOL=v2", || {
        Ok(super::ep_protocol_v2_requested() as u64)
    }),
    // The passes of a prefill chunk and the rows each rank runs.
    ("ATLAS_NO_TAIL_SPLIT", || {
        Ok(pc::tail_split_disabled() as u64)
    }),
    ("ATLAS_GLM_PREFILL_SP", || Ok(glm_sp::requested() as u64)),
    // Sequence-parallel prefill needs the BF16 highway, which needs the
    // DFlash lane.
    ("ATLAS_GLM_HC_BF16", || Ok(ops::hc_bf16_for(GLM) as u64)),
    ("ATLAS_GLM_INDEX_SPLIT", || Ok(index_split_words()?[0])),
    ("ATLAS_GLM_INDEX_SPLIT_MIN_CTX", || {
        Ok(index_split_words()?[1])
    }),
    (
        "ATLAS_GLM_INDEX_SPLIT_CHECK",
        || Ok(index_split_words()?[2]),
    ),
    // Prefix-cache policy: the min-reductions of a prefill, its restore
    // depth (so its row range) and the head's cache-sequence command.
    ("ATLAS_GLM_PC_EVICT", || Ok(glm_pc_evict_enabled() as u64)),
    ("ATLAS_GLM_PC_BRANCH", || {
        Ok(pc::glm_pc_branch_enabled() as u64)
    }),
    ("ATLAS_GLM_PC_BRANCH_MIN", || {
        while_on(pc::glm_pc_branch_enabled(), pc::glm_pc_branch_min_tokens)
    }),
    // In effect, that is with the preconditions `finish_leaf::flag` checks.
    ("ATLAS_GLM_PC_FINISH_LEAF", || {
        Ok(finish_leaf::enabled() as u64)
    }),
    ("ATLAS_GLM_PC_FINISH_LEAF_BLOCKS", || {
        while_on(finish_leaf::enabled(), finish_leaf::span_blocks)
    }),
    ("ATLAS_MARCONI_EXACT", || {
        Ok(mtp_carry::marconi_exact() as u64)
    }),
    ("ATLAS_MARCONI_MIN_TOKENS", || {
        Ok(mtp_carry::marconi_min_tokens() as u64)
    }),
    ("ATLAS_MARCONI_PREFILL_ONLY", || {
        Ok(mtp_carry::marconi_prefill_only() as u64)
    }),
    // The snapshots a rank has to restore from: the decode checkpoints it
    // takes and the ones a full pool evicts.
    ("ATLAS_DECODE_CKPT_BLOCKS", || {
        while_on(
            !mtp_carry::marconi_prefill_only(),
            mtp_carry::decode_ckpt_blocks,
        )
    }),
    ("ATLAS_SNAP_EVICT_LEGACY", || {
        Ok((!glm_pc_evict_enabled() && snap_evict_legacy()) as u64)
    }),
    ("ATLAS_SNAP_EVICT_ALPHA", || {
        let unused = glm_pc_evict_enabled() || snap_evict_legacy();
        Ok(if unused {
            0
        } else {
            snap_evict_alpha().to_bits()
        })
    }),
    // Which cached rows a prefill rewrites, so what each rank attends.
    ("ATLAS_GLM_PC_WRITE_FLOOR", || {
        Ok(pc::glm_pc_write_floor_enabled() as u64)
    }),
    ("ATLAS_GLM_KV_WRITE_FLOOR_LEGACY", || {
        Ok(write_floor_legacy() as u64)
    }),
    // Which rank stores each block's latents, and the exchanges the shard's
    // attention runs (the pool-size gather checks them again:
    // `glm_kv_shard::agreed_blocks`).
    ("ATLAS_GLM_KV_SHARD", || {
        Ok(glm_kv_shard::requested()? as u64)
    }),
    ("ATLAS_GLM_KV_SHARD_COMPACT", || {
        Ok(glm_kv_shard::MergeTuning::get()?.compact as u64)
    }),
    ("ATLAS_GLM_KV_SHARD_OVERLAP", || {
        Ok(glm_kv_shard::MergeTuning::get()?.overlap as u64)
    }),
    ("ATLAS_GLM_KV_SHARD_CHECK", || {
        Ok(glm_kv_shard::MergeTuning::get()?.check as u64)
    }),
    // Work the pair splits, and the exchanges that stand in for a reduce.
    ("ATLAS_GLM_SHARED_TP_SPLIT", || {
        Ok(moe::shared_tp_split_requested() as u64)
    }),
    ("ATLAS_MOE_SHARED_REDUCE_OVERLAP", || {
        Ok(moe::shared_reduce_overlap_requested() as u64)
    }),
    ("ATLAS_GLM_K5_FUSED_TP_HC", || {
        Ok(layers::verify_fused_tp_hc_enabled() as u64)
    }),
    ("ATLAS_GLM_VERIFY_VOCAB_SPLIT", || {
        Ok(glm_vocab_split::enabled() as u64)
    }),
    // The lane a decode or verify step takes on both ranks.
    ("ATLAS_GLM_DFLASH", || {
        Ok(glm_repair_policy::dflash_enabled() as u64)
    }),
    ("ATLAS_GLM_DFLASH_PREFILL_VERIFY", || {
        Ok(glm_repair_policy::dflash_prefill_verify() as u64)
    }),
    ("ATLAS_GLM_MTP_REPAIR", || {
        Ok(glm_repair_policy::enabled() as u64)
    }),
    ("ATLAS_GLM_MTP_LONG_CONTEXT", || {
        Ok(glm_repair_policy::long_context_enabled() as u64)
    }),
    ("ATLAS_GLM_LONG_BATCH_VERIFY", || {
        Ok(glm_long_owner::enabled()? as u64)
    }),
    ("ATLAS_GLM_INDEPENDENT_DECODE", || {
        Ok(glm_independent::enabled(GLM)? as u64)
    }),
    // The rows of a multi-sequence decode step each layer runs together: one
    // collective over them, or one a sequence.
    ("ATLAS_GLM_KDA_MULTI_SEQ", || {
        Ok(layers::kda_multi_seq_enabled() as u64)
    }),
    ("ATLAS_GLM_MLA_MULTI_SEQ", || {
        Ok(glm_mla_multi_seq_enabled() as u64)
    }),
    ("ATLAS_GLM_KDA_BATCHED_FFN", || {
        Ok(layers::kda_batched_ffn_enabled() as u64)
    }),
    ("ATLAS_GLM_C2_COMPACT_MOE", || {
        Ok(moe::c2_compact_requested()? as u64)
    }),
    ("ATLAS_GLM_MULTI_SEQ_SPARSE", || {
        Ok(glm_multi_seq_sparse_enabled(GLM) as u64)
    }),
    ("ATLAS_GLM_MULTI_SEQ_SPARSE_GRAPHS", || {
        Ok(glm_multi_seq_sparse_graphs_enabled(GLM)? as u64)
    }),
    ("ATLAS_GLM_C4_DECODE", || Ok(glm_c4::enabled(GLM) as u64)),
    ("ATLAS_GLM_C4_SPARSE", || {
        Ok((glm_c4::enabled(GLM) && glm_c4::sparse_enabled(GLM)) as u64)
    }),
    ("ATLAS_GLM_C4_GROUPED_MOE", || {
        Ok((glm_c4::enabled(GLM) && moe::c4_grouped_requested()) as u64)
    }),
    // Models without the mHC highway: the MoE passes of a batched decode
    // step, so one EP reduce over its rows, one a pair or one a row.
    ("ATLAS_MOE_PAIRWISE_DECODE", || {
        Ok(pairwise_moe_decode_enabled() as u64)
    }),
    ("ATLAS_MOE_GROUPED_ROUTED_DECODE", || {
        Ok(grouped_routed_decode_enabled() as u64)
    }),
    ("ATLAS_MOE_GROUPED_ROUTED_DECODE_MIN", || {
        while_on(grouped_routed_decode_enabled(), grouped_routed_decode_min)
    }),
    ("ATLAS_MOE_LEGACY_PERTOKEN_DECODE", || {
        Ok(qwen3_ssm::moe_legacy_pertoken_decode() as u64)
    }),
    // The owner-batch verify: its FFN and its layers jointly or an owner at
    // a time, and the serial verifies its oracle adds.
    ("ATLAS_GLM_LONG_BATCH_FFN", || {
        while_on(glm_long_owner::enabled()?, || {
            glm_long_owner::ffn_mode() as usize
        })
    }),
    ("ATLAS_GLM_LONG_BATCH_SERIAL", || {
        while_on(glm_long_owner::enabled()?, || {
            2 * serial_diagnostic(true) as usize + serial_diagnostic(false) as usize
        })
    }),
    ("ATLAS_GLM_LONG_BATCH_ORACLE", || {
        Ok((glm_long_owner::enabled()? && oracle_enabled()) as u64)
    }),
    // Where a K=5 verify blends the shared expert: whole after the EP reduce,
    // or a half a rank before it, which also needs the tensor-core tiers.
    ("ATLAS_GLM_K5_GROUPED_MOE", || {
        Ok(moe::k5_grouped_moe_requested() as u64)
    }),
    ("ATLAS_GLM_K5_FUSED_MOE_HC", || {
        Ok((moe::k5_grouped_moe_requested() && moe::k5_fused_moe_hc_requested()) as u64)
    }),
    ("ATLAS_W4A16_TC", || {
        Ok(w4a16_gemv_tiers::tc_requested() as u64)
    }),
    // Whether a verify can roll back the PLE carry every rank replicates
    // (qwen4_exp): without its snapshots `commit_accepted_prefix` refuses a
    // partial accept, on that rank alone.
    ("ATLAS_PLE_VERIFY_SNAPSHOTS", || {
        Ok(layers::ple::verify_snapshots_enabled() as u64)
    }),
    // The GDN projection weights each rank loads (qwen4_exp): ranks on two
    // formats sum two different models' partial out_proj in one all-reduce.
    ("ATLAS_QWEN4EXP_FP8_GDN", || {
        Ok(GdnProjections::from_env()?.fp8_word())
    }),
    ("ATLAS_QWEN4EXP_BF16_GDN", || {
        Ok((GdnProjections::from_env()? == GdnProjections::Nvfp4) as u64)
    }),
    // The MTP body a rank loads.
    ("ATLAS_GLM_MTP_DISTRIBUTED", || {
        Ok(glm_repair_policy::mtp_distributed() as u64)
    }),
    // The drafter a worker loads and the swaps of a rank-split propose,
    // batched ones included (ATLAS_GLM_DRAFT_TP_BATCH, bit 2), and how the
    // worker reads its announce (ATLAS_GLM_DRAFT_TP_CTX, bit 3).
    ("ATLAS_GLM_DRAFT_TP", || {
        Ok(rank_split::parity_word(
            rank_split::requested()?,
            rank_split::batch_requested()?,
            rank_split::ctx_requested()?,
        ))
    }),
    // Whether a step runs as a CUDA graph: a capturing forward takes other
    // collective paths than an eager one (`graph_flags`).
    ("ATLAS_EP_GRAPHS", || Ok(graph_flags::ep_graphs() as u64)),
    ("ATLAS_GDN_DECODE_GRAPH", || {
        Ok(graph_flags::gdn_decode_graph() as u64)
    }),
    ("ATLAS_NO_DECODE_GRAPHS", || {
        Ok(graph_flags::no_decode_graphs() as u64)
    }),
    ("ATLAS_NO_DECODE_GRAPHS_MULTISEQ", || {
        Ok(!graph_flags::multiseq_graphs_enabled() as u64)
    }),
    ("ATLAS_GLM_TP_VERIFY_GRAPH", || {
        Ok(graph_flags::glm_tp_verify_graph() as u64)
    }),
    ("ATLAS_GLM_MTP1_VERIFY_GRAPH", || {
        Ok(graph_flags::glm_mtp1_verify_graph() as u64)
    }),
    ("ATLAS_DEBUG_NO_GRAPH", || {
        Ok(graph_flags::debug_no_graph() as u64)
    }),
    ("ATLAS_GLM_VERIFY_GRAPH", || {
        Ok(verify_pieces::requested() as u64)
    }),
    // Rank-safe either way (`decode_pieces` docs), carried like the GLM
    // pieces so both ranks run one graph plan.
    ("ATLAS_QWEN4EXP_DECODE_GRAPH", || {
        Ok(decode_pieces::requested() as u64)
    }),
    ("ATLAS_QWEN4EXP_DECODE_GRAPH_COLLECTIVES", || {
        Ok((decode_pieces::requested() && decode_pieces::collectives_requested()) as u64)
    }),
    ("ATLAS_QWEN4EXP_DECODE_GRAPH_WIDE", || {
        Ok((decode_pieces::requested() && decode_pieces::wide_requested()) as u64)
    }),
    // The share of the qwen4_exp LM head a rank projects, and the exchange
    // of the halves every decode/verify step adds.
    ("ATLAS_QWEN4EXP_LMHEAD_SPLIT", || {
        Ok(qwen4exp_lmhead_split::enabled() as u64)
    }),
    // The qwen4_exp TP2 QSA prefill attention arm: each rank's heads, so a
    // split pair would serve neither arm's numerics.
    ("ATLAS_QWEN4EXP_PREFILL_QSA_TC2R", || {
        Ok(ops::qwen4exp_prefill::qsa_tc2r_requested() as u64)
    }),
    // The qwen4_exp sequence-parallel prefill: a split rank exchanges rows
    // the other never sends.
    ("ATLAS_QWEN4EXP_PREFILL_SP", || {
        Ok(super::qwen4exp_prefill_sp::requested() as u64)
    }),
    // The qwen4_exp verify lane (per-row MoE all-reduces in the GDN layers'
    // verify) and the check's serial steps (their collectives).
    ("ATLAS_QWEN4EXP_EXACT_VERIFY", || {
        Ok(qwen4exp_exact_verify::requested() as u64)
    }),
    ("ATLAS_QWEN4EXP_EXACT_VERIFY_CHECK", || {
        Ok(qwen4exp_exact_verify::check_requested() as u64)
    }),
    // The qwen4_exp verify width ceiling: the K the head dispatches and the
    // worker mirrors.
    ("ATLAS_QWEN4EXP_MTP_DEPTH", || {
        Ok(qwen4exp_mtp_depth::requested() as u64)
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

/// This process's settings: [`SETTINGS`], then the `caller`'s.
fn settings(caller: &[Setting]) -> Result<Vec<Setting>> {
    SETTINGS
        .iter()
        .map(|&(name, read)| Ok((name, read()?)))
        .chain(caller.iter().copied().map(Ok))
        .collect()
}

/// Call on every rank right after a multi-rank communicator comes up, before
/// any other collective. Fails, on every rank, when a setting differs across
/// the ranks or a rank's parser refuses one, and logs why before returning:
/// a rank that fails is gone before its peer's teardown reaches its own
/// report. `caller` carries the settings resolved outside this crate: the
/// same names in the same order on every rank.
pub fn agree(comm: &dyn CommBackend, gpu: &dyn GpuBackend, caller: &[Setting]) -> Result<()> {
    agree_on(settings(caller), warn_only(), comm, gpu)
        .inspect_err(|why| tracing::error!("Startup settings agreement: {why:#}"))
}

/// What a rank whose parser refused a setting gathers in place of its
/// table's id.
const REFUSED: u64 = 0;

/// Names a table: FNV-1a over its setting names, in order. Never [`REFUSED`].
fn table_id(settings: &[Setting]) -> u64 {
    settings
        .iter()
        .flat_map(|s| s.0.bytes().chain([0]))
        .fold(0xcbf2_9ce4_8422_2325, |id, b| {
            (id ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
        | 1
}

/// Fails with `why`; `warn` (`ATLAS_STARTUP_PARITY=warn`) logs it instead.
fn fail(warn: bool, why: anyhow::Error) -> Result<()> {
    if !warn {
        return Err(why);
    }
    tracing::warn!("Startup settings agreement ignored (ATLAS_STARTUP_PARITY=warn): {why:#}");
    Ok(())
}

fn agree_on(
    ours: Result<Vec<Setting>>,
    warn: bool,
    comm: &dyn CommBackend,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let me = comm.rank();
    // The table is part of the build and the second gather is as long as the
    // table, so first compare the tables, in a gather of one word. Every rank
    // takes part, a rank without settings too, and every rank reads the same
    // words, so they all stop here or all go on.
    let table = ours.as_ref().map_or(REFUSED, |ours| table_id(ours));
    let tables = gather_words(comm, gpu, &[table])?;
    let ours = match (ours, tables.iter().position(|&t| t == REFUSED)) {
        (Err(why), _) => {
            return fail(
                warn,
                why.context(format!("rank {me} refuses one of its settings")),
            );
        }
        (Ok(_), Some(rank)) => {
            return fail(
                warn,
                anyhow!("rank {rank} refuses one of its settings: see its log"),
            );
        }
        (Ok(ours), None) => ours,
    };
    ensure!(
        tables.iter().all(|&t| t == table),
        "rank {me} compares other settings at startup than its peers: \
         the ranks run different builds"
    );
    let values: Vec<u64> = ours.iter().map(|s| s.1).collect();
    let all = gather_words(comm, gpu, &values)?;
    let differ: Vec<String> = all
        .chunks(ours.len())
        .enumerate()
        .flat_map(|(rank, theirs)| {
            ours.iter()
                .zip(theirs)
                .filter(|(ours, theirs)| ours.1 != **theirs)
                .map(move |((name, ours), theirs)| {
                    format!("{name}: rank {me} has {ours}, rank {rank} has {theirs}")
                })
        })
        .collect();
    if differ.is_empty() {
        return Ok(());
    }
    fail(
        warn,
        anyhow!(
            "every rank must run the same settings ({})",
            differ.join("; ")
        ),
    )
}

/// Every rank's `words`, in rank order: one all-gather of `8 * words.len()`
/// bytes a rank.
pub(crate) fn gather_words(
    comm: &dyn CommBackend,
    gpu: &dyn GpuBackend,
    words: &[u64],
) -> Result<Vec<u64>> {
    let ours: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut all = vec![0u8; ours.len() * comm.world_size()];
    let buf = gpu.alloc(ours.len() + all.len())?;
    let recv = buf.offset(ours.len());
    let gathered = gpu
        .copy_h2d(&ours, buf)
        .and_then(|()| comm.all_gather(buf.0, recv.0, ours.len()))
        .and_then(|()| gpu.copy_d2h(recv, &mut all));
    gpu.free(buf)?;
    gathered?;
    Ok(all
        .chunks_exact(8)
        .map(|w| u64::from_le_bytes(w.try_into().expect("8-byte words")))
        .collect())
}

#[cfg(test)]
#[path = "startup_parity_tests.rs"]
mod tests;
