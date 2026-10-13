// SPDX-License-Identifier: AGPL-3.0-only

//! The table [`super::agree`] compares, split from `startup_parity.rs` for
//! the 500-LoC cap. One entry per setting, valued by the feature's own
//! parser; what belongs here is in the parent module's docs.

use anyhow::Result;
use spark_runtime::radix_tree::{glm_pc_evict_enabled, snap_evict_alpha, snap_evict_legacy};

use crate::layer::glm_long_owner;
use crate::layers::dflash_head::rank_split;
use crate::layers::qwen3_attention::{
    glm_mla_multi_seq_enabled, glm_multi_seq_sparse_enabled, glm_multi_seq_sparse_graphs_enabled,
    grouped_routed_decode_enabled, grouped_routed_decode_min, index_split_words,
    pairwise_moe_decode_enabled, write_floor_legacy,
};
use crate::layers::{self, glm_kv_shard, glm_sp, moe, ops, qwen3_ssm, w4a16_gemv_tiers};
use crate::model::glm_long_verify::{oracle_enabled, serial_diagnostic};
use crate::model::trait_impl::finish_leaf;
use crate::model::trait_impl::prefill_b::pc_policy as pc;
use crate::model::trait_impl::prefill_b::qwen4exp_points as qp;
use crate::model::{
    decode_pieces, glm_c4, glm_independent, glm_vocab_split, graph_flags, mtp_carry,
    qwen4exp_batch_fast, qwen4exp_exact_verify, qwen4exp_lmhead_split, qwen4exp_mtp_depth,
    verify_pieces,
};
use crate::speculative::glm_repair_policy;
use crate::weight_loader::qwen4_exp::GdnProjections;

use super::while_on;

/// The model type the GLM parsers are asked about.
const GLM: &str = "glm5_next";

/// The settings read in this crate: name, and this process's value from the
/// feature's own parser. One entry per setting.
pub(super) const SETTINGS: &[(&str, fn() -> Result<u64>)] = &[
    // A rank that would only warn beside one that fails.
    (
        "ATLAS_STARTUP_PARITY=warn",
        || Ok(super::warn_only() as u64),
    ),
    // The head's command words.
    ("ATLAS_EP_PROTOCOL=v2", || {
        Ok(crate::model::ep_protocol_v2_requested() as u64)
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
    ("ATLAS_QWEN4EXP_DENSE_CKPT", || Ok(qp::dense_every() as u64)),
    ("ATLAS_QWEN4EXP_PC_BRANCH", || Ok(qp::branch_on() as u64)),
    ("ATLAS_QWEN4EXP_PC_BRANCH_MIN", || {
        while_on(qp::branch_on(), qp::branch_min)
    }),
    // In effect, that is with the preconditions `finish_leaf::flag` checks.
    ("ATLAS_GLM_PC_FINISH_LEAF", || {
        Ok(finish_leaf::enabled() as u64)
    }),
    ("ATLAS_GLM_PC_FINISH_LEAF_BLOCKS", || {
        let on = finish_leaf::enabled() || finish_leaf::qwen4exp_enabled();
        while_on(on, finish_leaf::span_blocks)
    }),
    // Bit 0: where qwen4_exp's in-pass tail checkpoint lands (the switch
    // alone); bit 1: its decode leaf (with the preconditions above).
    ("ATLAS_QWEN4EXP_FINISH_LEAF", || {
        let on = [
            finish_leaf::qwen4exp_requested(),
            finish_leaf::qwen4exp_enabled(),
        ];
        Ok(on.iter().enumerate().map(|(i, &b)| (b as u64) << i).sum())
    }),
    // Chunk shapes, the attention path and the restores a rank takes.
    ("ATLAS_QWEN4EXP_PREFILL_ROWINV", || {
        Ok(crate::layers::ops::qwen4exp_rowinv::on() as u64)
    }),
    // Who sends the prefill commands and in what shape.
    ("ATLAS_QWEN4EXP_PREFILL_MULTI", || {
        Ok(crate::model::trait_impl::prefill_b::multi_requested() as u64)
    }),
    ("ATLAS_QWEN4EXP_PREFILL_MULTI_CACHED", || {
        Ok(crate::model::trait_impl::prefill_b::multi_cached() as u64)
    }),
    ("ATLAS_QWEN4EXP_PREFILL_BF16_PROJ", || {
        Ok(crate::layers::ops::qwen4exp_rowinv::bf16_proj() as u64)
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
    // The draft head a qwen4_exp worker builds and the swaps of each batched
    // propose it serves (`--mtp-vocab` is agreed by the server).
    ("ATLAS_QWEN4EXP_MTP_DRAFT_TP", || {
        layers::qwen4exp_mtp::draft_tp::parity_word()
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
    // Where a batched verify votes on KV admission (`kv_lookahead`): a split
    // pair would vote at different steps and mispair the broadcasts.
    ("ATLAS_QWEN4EXP_KV_LOOKAHEAD", || {
        Ok(crate::model::kv_lookahead::lookahead_tokens() as u64)
    }),
    // The share of the qwen4_exp LM head a rank projects, and the exchange
    // of the halves every decode/verify step adds.
    ("ATLAS_QWEN4EXP_LMHEAD_SPLIT", || {
        Ok(qwen4exp_lmhead_split::enabled() as u64)
    }),
    // The rows the split head serves: a rank staged for the batched verify
    // exchanges where a narrower one projects the full head.
    ("ATLAS_QWEN4EXP_LMHEAD_SPLIT_VERIFY", || {
        Ok((qwen4exp_lmhead_split::enabled() && qwen4exp_lmhead_split::verify_requested()) as u64)
    }),
    // The qwen4_exp TP2 QSA prefill attention arm: each rank's heads, so a
    // split pair would serve neither arm's numerics.
    ("ATLAS_QWEN4EXP_PREFILL_QSA_TC2R", || {
        Ok(ops::qwen4exp_prefill::qsa_tc2r_requested() as u64)
    }),
    // The qwen4_exp sequence-parallel prefill: a split rank exchanges rows
    // the other never sends.
    ("ATLAS_QWEN4EXP_PREFILL_SP", || {
        Ok(crate::model::qwen4exp_prefill_sp::requested() as u64)
    }),
    // Its slab-pipelined exchanges (gathers; `_RS_PIPE`: the MoE reduce-scatter)
    // move the same rows in pieces: a piecewise rank never pairs a whole one.
    // `_MIDCHUNK_CKPT` runs a prompt's last chunk as one pass, not two;
    // `_QSA_SPLIT` / `_SP_ROUTE` swap QSA block lists / MoE routes.
    (
        "ATLAS_QWEN4EXP_PREFILL_SP_PIPE/_RS_PIPE/_MIDCHUNK_CKPT/_QSA_SPLIT/_SP_ROUTE",
        || {
            use crate::layers::moe::forward_prefill_route_sp as r;
            use crate::layers::{
                qsa::qsa_select_sp as q, qwen4exp_ckpt as c, qwen4exp_sp_pipe as p,
            };
            let bits = [
                p::requested(),
                p::rs_requested(),
                c::requested(),
                q::requested(),
                r::requested(),
            ];
            Ok(bits.iter().enumerate().map(|(i, &b)| (b as u64) << i).sum())
        },
    ),
    // The SP shared expert on this rank's rows only: the rows it gathers.
    ("ATLAS_QWEN4EXP_PREFILL_SP_SHARED", || {
        Ok(moe::sp_shared_requested() as u64)
    }),
    // The q38 prefill MoE arm (`_PREFILL_BF16_PROJ` implies it) routes under
    // `_SP_ROUTE`: a rank without it routes every row and never joins the
    // routes' all-gather. `_MOE_W2` also takes the router below 32 rows.
    ("ATLAS_QWEN4EXP_PREFILL_MOE", || {
        Ok(moe::q38_requested() as u64)
    }),
    ("ATLAS_QWEN4EXP_PREFILL_MOE_W2", || {
        Ok((moe::q38_requested() && moe::w2_requested()) as u64)
    }),
    // Same bytes either way, but not the same memory: a rank without it
    // keeps the routed K-major copy, and the two KV pools part.
    ("ATLAS_QWEN4EXP_PREFILL_MOE_NODUP", || {
        Ok((moe::q38_requested() && moe::nodup_requested()) as u64)
    }),
    // The routed arm and the shared-reduce overlap (off while profiling)
    // decide whether `_SP_RS_PIPE` pipes the MoE reduce-scatter.
    ("ATLAS_MOE_PREFILL_FP8_DOWN", || {
        while_on(crate::layers::qwen4exp_sp_pipe::rs_requested(), || {
            moe::prefill_fp8_down() as usize
        })
    }),
    (
        "ATLAS_MOE_GROUPED_CUTLASS/_HOLO_MOE_GROUPED_CUTLASS",
        || {
            while_on(crate::layers::qwen4exp_sp_pipe::rs_requested(), || {
                moe::grouped_cutlass_gate_up_enabled() as usize
            })
        },
    ),
    ("ATLAS_PROFILE_FIRST", || {
        Ok(graph_flags::profile_first() as u64)
    }),
    // Under ROWINV, host ids decide the text-only (plain RoPE) attention.
    ("ATLAS_QWEN4EXP_PREFILL_HOST_IDS", || {
        while_on(ops::qwen4exp_rowinv::on(), || {
            crate::model::trait_impl::prefill_b::embed_chunk::host_ids_requested() as usize
        })
    }),
    // The o_proj arm decides whether `_SP_RS_PIPE` pipes its reduce-scatter.
    ("ATLAS_ATTN_W4A4", || {
        Ok(layers::qwen3_attention::attn_w4a4_requested() as u64)
    }),
    ("ATLAS_CUTLASS_NVFP4_ATTN_O/_GEMM", || {
        Ok(ops::GemmDispatch::from_env().cutlass_nvfp4_attn_o as u64)
    }),
    // Dense prefill past the QSA bound skips `_QSA_SPLIT`'s exchanges.
    ("ATLAS_QSA_NO_PREFILL_SELECT", || {
        Ok(layers::qsa::no_prefill_select() as u64)
    }),
    // qwen4_exp decode numerics (exactness contract (b) baselines, and the
    // same-bytes A/B arms beside them): each rank's routed sums meet in the
    // EP all-reduce and both ranks run the mHC collapse, so a split pair
    // would serve neither arm's numerics.
    ("ATLAS_QWEN4EXP_MOE_TC", || {
        Ok(qwen4exp_batch_fast::tc_requested() as u64)
    }),
    ("ATLAS_QWEN4EXP_MOE_TC_V1", || {
        while_on(qwen4exp_batch_fast::tc_requested(), || {
            qwen4exp_batch_fast::tc_v1_requested() as usize
        })
    }),
    ("ATLAS_QWEN4EXP_MOE_TC_V2", || {
        while_on(qwen4exp_batch_fast::tc_requested(), || {
            qwen4exp_batch_fast::tc_v2_requested() as usize
        })
    }),
    ("ATLAS_QWEN4EXP_MOE_UNITS", || {
        Ok(qwen4exp_batch_fast::units_requested() as u64)
    }),
    ("ATLAS_QWEN4EXP_MOE_NO_CLAMP", || {
        Ok(qwen4exp_batch_fast::no_clamp_requested("qwen4_exp") as u64)
    }),
    ("ATLAS_QWEN4EXP_PREFILL_MOE_BF16", || {
        Ok(moe::tcp_requested() as u64)
    }),
    // Both apply only through the vectorized collapse (`_HC_FAST`).
    ("ATLAS_QWEN4EXP_HC_MMA", || {
        while_on(ops::hc_fast(), || ops::hc_mma() as usize)
    }),
    ("ATLAS_QWEN4EXP_HC_STAGE_FIT", || {
        while_on(ops::hc_fast(), || ops::hc_stage_fit() as usize)
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
