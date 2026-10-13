// SPDX-License-Identifier: AGPL-3.0-only

//! GPU init + pre-load reserve preflight + post-load OOM check.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

mod gpu_init;
mod independent;
mod ssm_h_fp16;
pub(crate) use gpu_init::{init_gpu_backend, post_load_memory_audit};
pub(crate) use independent::prepare_reserve;
use ssm_h_fp16::ssm_h_fp16_preconditions;

pub(crate) struct ReservePreflight {
    pub(crate) inference_reserve: usize,
    pub(crate) buffer_arena_bytes: usize,
    pub(crate) gdn_two_phase_bytes: usize,
    pub(crate) ssm_prefill_chunk: usize,
    pub(crate) max_batch_tokens_pre: usize,
    pub(crate) resolved_prefill: Option<super::PrefillBudget>,
}

fn glm5_dual_spark_parallelism(world: usize, tp: usize, ep: usize) -> bool {
    world == 2 && ep == 2 && matches!(tp, 1 | 2)
}

fn glm5_concurrency_supported(
    max_batch: usize,
    max_num_seqs: usize,
    ep_v2: bool,
    c4: bool,
) -> bool {
    (c4 && max_batch == 4 && max_num_seqs == 4 && ep_v2)
        || ((1..=3).contains(&max_batch)
            && (max_batch..=5).contains(&max_num_seqs)
            && (max_batch == 1 || ep_v2))
}

fn glm5_context_supported(max_seq_len: usize, max_prefill_tokens: usize, model_max: usize) -> bool {
    max_seq_len <= model_max && max_prefill_tokens > 0
}

fn glm5_long_context_concurrency_supported(
    max_seq_len: usize,
    index_topk: usize,
    max_batch: usize,
    max_num_seqs: usize,
    sparse_decode: bool,
    c4_sparse: bool,
) -> bool {
    max_seq_len <= index_topk
        || (max_batch == 1 && max_num_seqs == 1)
        || (sparse_decode && matches!(max_batch, 2 | 3) && (max_batch..=5).contains(&max_num_seqs))
        || (sparse_decode
            && c4_sparse
            && spark_model::model::glm_c4::validate_launch_limits_for_mode(
                max_batch,
                max_num_seqs,
                max_seq_len,
                true,
                true,
            )
            .is_ok())
}

pub(crate) fn preflight_reserve(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    free_mem: usize,
) -> Result<ReservePreflight> {
    // The GLM long-context lane: DFlash verification over the bounded sparse
    // domain, with the topology resolved before any reserve (`independent.rs`).
    let long_context = spark_model::speculative::glm_repair_policy::long_context_enabled();
    anyhow::ensure!(
        !long_context
            || (args.dflash && spark_model::speculative::glm_repair_policy::dflash_enabled()),
        "GLM long-context verification requires the GLM DFlash lane"
    );
    // The long-context lane retains TP-local topology and the exact row budget.
    let bounded = long_context;
    let c4_sparse = spark_model::model::glm_c4::validate_sparse_flag(
        &config.model_type,
        std::env::var("ATLAS_GLM_C4_SPARSE").ok().as_deref(),
    )?;
    if config.model_type == "glm5_next" {
        let sparse_decode =
            spark_model::layers::qwen3_attention::glm_multi_seq_sparse_enabled(&config.model_type);
        anyhow::ensure!(
            !sparse_decode
                || !(args.speculative || args.self_speculative || args.ngram_speculative),
            "ATLAS_GLM_MULTI_SEQ_SPARSE=1 supports non-speculative decode only"
        );
        anyhow::ensure!(
            glm5_context_supported(
                args.max_seq_len,
                args.max_prefill_tokens,
                config.max_position_embeddings,
            ),
            "GLM-5 requires --max-seq-len <= {} and a non-zero --max-prefill-tokens for bounded chunked prefill",
            config.max_position_embeddings,
        );
        let ep_v2 = spark_model::model::ep_protocol_v2_requested();
        anyhow::ensure!(
            !long_context || args.max_batch_size == 1 || ep_v2,
            "GLM concurrent long-context verification requires ATLAS_EP_PROTOCOL=v2 for owner slot identity"
        );
        let c4 = spark_model::model::glm_c4::enabled(&config.model_type);
        anyhow::ensure!(
            std::env::var("ATLAS_GLM_C4_GROUPED_MOE").as_deref() != Ok("1") || c4,
            "ATLAS_GLM_C4_GROUPED_MOE=1 requires ATLAS_GLM_C4_DECODE=1"
        );
        if !long_context && (c4 || c4_sparse || args.max_batch_size == 4) {
            spark_model::model::glm_c4::validate_prefill_budget(
                args.max_prefill_tokens,
                c4_sparse,
            )?;
            spark_model::model::glm_c4::policy_from_env(
                &config.model_type,
                args.world_size,
                args.tp_size,
                args.ep_size,
                ep_v2,
                !(args.speculative
                    || args.self_speculative
                    || args.ngram_speculative
                    || args.dflash),
            )
            .validate()?;
            spark_model::model::glm_c4::validate_launch_limits(
                args.max_batch_size,
                args.max_num_seqs,
                args.max_seq_len,
                args.kv_cache_dtype.as_deref() == Some("bf16"),
            )?;
        }
        anyhow::ensure!(
            long_context
                || glm5_concurrency_supported(args.max_batch_size, args.max_num_seqs, ep_v2, c4),
            "GLM-5 dual-Spark concurrency supports --max-batch-size 1..=3 and \
             --max-num-seqs max_batch..=5, or explicitly opted-in C4 with active/admitted4; \
             batches above one require ATLAS_EP_PROTOCOL=v2"
        );
        anyhow::ensure!(
            long_context
                || glm5_long_context_concurrency_supported(
                    args.max_seq_len,
                    config.index_topk,
                    args.max_batch_size,
                    args.max_num_seqs,
                    sparse_decode,
                    c4_sparse,
                ),
            "GLM-5 context above index_topk={} requires one active/admitted sequence, opt-in sparse C2/C3, or guarded eager ATLAS_GLM_C4_SPARSE=1",
            config.index_topk,
        );
        anyhow::ensure!(
            !(args.speculative || args.self_speculative || args.ngram_speculative),
            "GLM-5 speculative decoding is served through --dflash; its native MTP layer \
             and self/n-gram speculation are not supported"
        );
        anyhow::ensure!(
            glm5_dual_spark_parallelism(args.world_size, args.tp_size, args.ep_size),
            "GLM-5 dual-Spark support requires --world-size 2 --ep-size 2 and either --tp-size 1 (EP fallback) or --tp-size 2 (overlapping TP+EP)"
        );
    }
    let h_state_bytes = config.ssm_h_state_bytes();
    let conv_state_bytes = config.ssm_conv_state_bytes();
    // `--dflash` allocates the same verify pools, sized by its γ-derived
    // draft count (see `build_model`), not the MTP `--num-drafts`.
    let spec_on_pool =
        args.speculative || args.dflash || args.self_speculative || args.ngram_speculative;
    let pool_drafts = if args.dflash {
        super::build::checked_dflash_num_drafts(args.resolved_dflash_gamma())?
    } else {
        args.resolved_num_drafts()
    };
    ssm_h_fp16_preconditions(args, config)?;
    // SSM state pool = per-seq live state (max_batch blobs) + MTP verify
    // state (intermediates + checkpoint) for the slots spec dispatch can
    // actually reach. SSOT: `ssm_reserve::mtp_state_slots` — the SAME
    // number `SsmStatePool::new` allocates and the scheduler's spec
    // dispatch guard enforces. At bs<=32 this reproduces the historical
    // `max_batch × blob × (1 + (num_drafts+1) + 1)` byte-for-byte; above
    // 32 it stops reserving verify blobs for slots that can never verify
    // (25.4 GB at bs=64/K=4 on the 27B — the bs=64 preflight refusal).
    // Kill switch: ATLAS_MTP_POOL_FULL_WIDTH (presence) restores
    // full-width sizing on BOTH sides.
    let mtp_state_slots = spark_model::ssm_reserve::mtp_state_slots(args.max_batch_size);
    // Tiered verify slots (2026-08-16): the H-intermediate term is per-slot
    // (`verify_slot_h_intermediates`); DFlash pools are γ-sized and do not
    // follow the MTP ladder, so they reserve uniform full width — mirroring
    // `SsmStatePool::new`'s DFlash-active condition (`args.dflash` → the
    // target's `dflash_capture_layers`).
    // Stage-3 f16-SIZED pool: the FP32 prefill staging arena, ONE blob per
    // slot (shared across layers — see `ssm_h_prefill_stage_bytes`). A
    // separate term for the same reason the replay ring is: it is sized by a
    // SINGLE layer's h blob, not by the across-layers per-seq total every
    // other term here uses. Zero on an FP32-sized pool. `max_batch_size`, not
    // `+1`: this preflight has never counted the pools' dummy slot.
    let ssm_h_stage_bytes = spark_model::ssm_reserve::ssm_h_prefill_stage_bytes(
        args.max_batch_size,
        h_state_bytes,
        spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
    );
    // Selected FP32 pools allocate one additional live dummy slot.
    // It has no prefix/rollback snapshots; do not inflate those counts.
    let live_slots = args.max_batch_size + usize::from(bounded);
    let ssm_pool_bytes = spark_model::ssm_reserve::ssm_pool_reserve_bytes(
        live_slots,
        config.num_ssm_layers() * h_state_bytes,
        config.num_ssm_layers() * conv_state_bytes,
        spec_on_pool,
        pool_drafts,
        mtp_state_slots,
        args.dflash,
        // Stage-3 f16-SIZED pool: mirrors `SsmStatePool::new`'s narrowing.
        // Unreachable today (ssm_h_fp16_preconditions refuses the mode
        // above), wired so preflight and allocator cannot diverge when the
        // refusal lifts.
        spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
        // `--ssm-rollback-mode` (published by serve_flags before this runs).
        // Replay drops every per-token verify intermediate.
        spark_model::ssm_reserve::ssm_rollback_mode(),
    );
    // The replay-mode verify-window input ring is likewise allocated inside
    // `SsmStatePool::new` (pre-snapshot) — no reserve term.
    let spec_tokens_pre = if args.speculative || args.self_speculative || args.ngram_speculative {
        args.resolved_num_drafts() + 2
    } else {
        1
    };
    // B4 (chunked-prefill BF16 KV cliff): the prior `.min(8192)` cap forced
    // every prompt > 8 k to chunk, which compounds K-side BF16 rounding noise
    // at chunk boundaries (per the 4-agent audit 2026-05-27). When the user
    // explicitly passes `--max-prefill-tokens N` (anything other than the
    // default 8192), respect it — no hard cap. Otherwise default to 8192 to
    // bound GDN persistent-buffer reservation for unbounded `max_seq_len`.
    let ssm_prefill_chunk: usize = if config.num_ssm_layers() > 0 {
        if args.max_prefill_tokens != 8192 && args.max_prefill_tokens > 0 {
            args.max_seq_len.min(args.max_prefill_tokens)
        } else {
            args.max_seq_len.min(8192)
        }
    } else {
        0
    };
    let user_set_prefill_pre = args.max_prefill_tokens != 8192;
    let prefill_budget_pre = if user_set_prefill_pre && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    // Issue #15 auto-clamp removed (2026-07-02): snapshot reachability is
    // handled by the tail-checkpoint split in `prefill_chunk_dispatch`, so
    // the budget (and this arena-sizing mirror) stays at full chunk size.
    let resolved_prefill = bounded.then(|| super::resolve_prefill_budget(args, ssm_prefill_chunk));
    let max_batch_tokens_pre = resolved_prefill.as_ref().map_or_else(
        || {
            prefill_budget_pre
                .max(spec_tokens_pre)
                .max(args.max_batch_size)
        },
        |budget| budget.max_batch_tokens,
    );
    // The selected envelope is bounded before BufferSizes' unchecked products.
    anyhow::ensure!(
        !bounded || max_batch_tokens_pre <= 65535,
        "selected arena rows exceed the CUDA grid limit"
    );
    let buffer_arena_bytes = spark_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens_pre,
        args.max_seq_len,
        args.block_size,
        args.max_batch_size,
    )
    .total_bytes();
    // SSM snapshot pool = Marconi prefix-cache region + Phase-C
    // decode-rollback ring. The decode ring is sized per active
    // sequence (ring slots × `max_batch_size`) and only allocated for SSM
    // models. SSOT: `spark_model::ssm_reserve::decode_rollback_ring_slots`
    // makes the SAME decision (same env vars, same constant) the runtime
    // allocation in `TransformerModel::new` makes — including the skip under
    // `--speculative`/`--dflash` (the ring's save/rollback path only runs on
    // plain decode; the spec path rolls back through the verify snapshot).
    // Reserving the ring unconditionally while the runtime skipped it
    // stranded ~38 GB at bs32 on the 27B (75.2 GB SSM reserve vs an 85.2 GB
    // budget at util 0.70) and capped the native batch at ~20.
    // `use_speculative` here MUST mirror what `build_model` passes:
    // `args.speculative || args.dflash`.
    // Kill switch: `ATLAS_SSM_RESERVE_RING_FULL` present ⇒ restore the old
    // unconditional reservation (accounting-only, safe over-reserve;
    // presence-style — `=0` is NOT "off").
    let decode_ring_slots = if std::env::var("ATLAS_SSM_RESERVE_RING_FULL").is_ok() {
        if config.num_ssm_layers() > 0 {
            atlas_kernels::DECODE_ROLLBACK_RING_SLOTS
        } else {
            0
        }
    } else {
        spark_model::ssm_reserve::decode_rollback_ring_slots(
            config.num_ssm_layers(),
            args.speculative || args.dflash,
        )
        .slots
    };
    let ssm_snapshot_bytes = if bounded {
        args.ssm_cache_slots
            .checked_add(decode_ring_slots * args.max_batch_size)
            .and_then(|n| n.checked_mul(config.num_ssm_layers()))
            .and_then(|n| n.checked_mul(h_state_bytes + conv_state_bytes))
            .context("selected snapshot reserve overflow")?
    } else {
        (args.ssm_cache_slots + decode_ring_slots * args.max_batch_size)
            * config.num_ssm_layers()
            * (h_state_bytes + conv_state_bytes)
    };
    let cuda_headroom: usize =
        if args.speculative || args.self_speculative || args.ngram_speculative {
            4 * 1024 * 1024 * 1024
        } else {
            512 * 1024 * 1024
        };
    let gdn_two_phase_bytes: usize = {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        if conv_dim > 0 && config.num_ssm_layers() > 0 {
            let sl = if bounded {
                max_batch_tokens_pre.min(args.max_seq_len)
            } else {
                max_batch_tokens_pre
            };
            sl * conv_dim * 2 + sl * nv * 2 * 4 + sl * value_dim * 2 + sl * value_dim * 2
        } else {
            0
        }
    };
    // The SSM state/snapshot pool terms (ssm_pool_bytes, ssm_h_stage_bytes,
    // ssm_replay_ring, ssm_snapshot_bytes) are deliberately NOT in the
    // reserve: `build_model` allocates the real pools before the KV-budget
    // `free_memory()` snapshot, so they land in `used_so_far` — reserving
    // them again here double-charged ~20 GB and refused configs that
    // physically fit (the #61 fix turned a silent oversubscription into a
    // false refusal; seen on the post-merge bfcl leg at util 0.70).
    let inference_reserve: usize = gdn_two_phase_bytes + cuda_headroom;
    let total_reserve = inference_reserve + buffer_arena_bytes;
    if total_reserve > free_mem {
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gb = free_mem as f64 / (1024.0 * 1024.0 * 1024.0);
        let fixed = ssm_pool_bytes + ssm_h_stage_bytes + ssm_snapshot_bytes + cuda_headroom;
        let budget_for_seq_term = free_mem.saturating_sub(fixed) / 2;
        let per_tok_bytes = {
            let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
            let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
            let nv = config.linear_num_value_heads;
            let conv_dim = key_dim * 2 + value_dim;
            if conv_dim > 0 && config.num_ssm_layers() > 0 {
                (conv_dim * 2) + (nv * 2 * 4) + (value_dim * 2) + (value_dim * 2)
            } else {
                0
            }
        };
        let suggested = budget_for_seq_term
            .checked_div(per_tok_bytes)
            .map(|q| q.max(2048))
            .unwrap_or(0);
        let hint = if suggested > 0 && suggested < args.max_seq_len {
            format!(
                " Try --max-seq-len {} (or lower --max-batch-size / --num-drafts).",
                suggested
            )
        } else if args.max_batch_size > 1 {
            " Reduce --max-batch-size.".to_string()
        } else {
            " Use a smaller model or a GPU with more memory.".to_string()
        };
        anyhow::bail!(
            "Preflight failed: inference buffers alone need {:.2} GB but only {:.2} GB is free on the GPU \
             (before weights load). SSM pool + GDN chunked prefill scales with --max-seq-len={} × --max-batch-size={}.{}",
            need_gb,
            free_gb,
            args.max_seq_len,
            args.max_batch_size,
            hint,
        );
    }
    tracing::info!(
        "Preflight reserve: inference={} MB, buffer_arena={} MB (pre-load free: {:.1} GB)",
        inference_reserve / (1024 * 1024),
        buffer_arena_bytes / (1024 * 1024),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    // Q09: per-component breakdown so future MTP/spec-decode reserve
    // jumps are diagnosable from the log alone. Each line is dropped at
    // debug to avoid noise on hot startup paths; flip to info if you
    // need to trace a specific deployment's reserve.
    let spec_on = spec_on_pool;
    tracing::debug!(
        "Preflight reserve breakdown (ssm_pool/ssm_snapshot are estimates of \
         the pre-snapshot pool allocations — excluded from inference_reserve): \
         ssm_pool={} MB ({} max_batch blobs + {} MTP-covered slots × {} verify blobs, \
         {} ssm_layers × (h+conv)), \
         ssm_snapshot={} MB ({} slots), \
         gdn_two_phase={} MB ({} tokens), \
         cuda_headroom={} MB ({}), \
         spec_on={}, num_drafts={}",
        ssm_pool_bytes / (1024 * 1024),
        live_slots,
        if spec_on_pool { mtp_state_slots } else { 0 },
        if spec_on_pool {
            args.resolved_num_drafts() + 2
        } else {
            0
        },
        config.num_ssm_layers(),
        ssm_snapshot_bytes / (1024 * 1024),
        args.ssm_cache_slots,
        gdn_two_phase_bytes / (1024 * 1024),
        max_batch_tokens_pre,
        cuda_headroom / (1024 * 1024),
        if spec_on { "spec/MTP on" } else { "no spec" },
        spec_on,
        if spec_on {
            args.resolved_num_drafts() as i64
        } else {
            -1
        },
    );
    Ok(ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
        resolved_prefill,
    })
}

#[cfg(test)]
mod tests;
