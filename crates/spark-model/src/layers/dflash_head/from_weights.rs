// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter constructor + target-config validation.
//!
//! Split out of `dflash_head.rs` for file-size budget. Contains
//! [`BlockDiffusionDraftHead::from_weights`] (kernel resolution + KV
//! cache setup) and [`BlockDiffusionDraftHead::validate_against_target`].

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use super::{
    BlockDiffusionDraftHead, DflashKernels, DflashLane, DflashLayer, DflashQuantization,
    DflashScratch, DsparkStartupExecution,
};
use crate::weight_loader::DflashWeights;

impl BlockDiffusionDraftHead {
    pub fn from_weights(
        weights: DflashWeights,
        embed_tokens_shared: DevicePtr,
        lm_head_shared: DevicePtr,
        lm_head_nvfp4: Option<crate::weight_map::QuantizedWeight>,
        target_hidden_size: usize,
        gamma: Option<usize>,
        window_size: Option<usize>,
        gpu: &dyn GpuBackend,
        max_seq_len: usize,
        max_batch_size: usize,
        startup: DsparkStartupExecution,
    ) -> Result<Self> {
        let embed_tokens_shared = weights
            .embed_tokens
            .as_ref()
            .map(|e| e.weight)
            .unwrap_or(embed_tokens_shared);
        // Drafter's `fc` is `[draft_hidden, len(target_layer_ids) * target_hidden]`.
        // We rely on the drafter config's `hidden_size` and the parsed
        // `target_layer_ids` to derive the expected target_hidden, then
        // validate it matches what the caller provided.
        let target_layer_ids = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.target_layer_ids.clone())
            .unwrap_or_default();
        let mask_token_id = weights
            .config
            .dflash_config
            .as_ref()
            .map(|c| c.mask_token_id)
            .unwrap_or(0);
        let query_causal = weights
            .config
            .dflash_config
            .as_ref()
            .and_then(|c| c.causal)
            .unwrap_or(false);
        let window_size = window_size.or_else(|| {
            weights
                .config
                .dflash_config
                .as_ref()
                .and_then(|c| c.swa_window_size)
        });

        if target_layer_ids.is_empty() {
            anyhow::bail!(
                "DFlash drafter config.json has no `dflash_config.target_layer_ids` — \
                 cannot determine which target hidden states to capture"
            );
        }

        let _ = target_hidden_size;

        let num_layers = weights.config.num_hidden_layers;
        let hidden_size = weights.config.hidden_size;
        let intermediate_size = weights.config.intermediate_size;
        let num_q_heads = weights.config.num_attention_heads;
        let num_kv_heads = weights.config.num_key_value_heads;
        let head_dim = weights.config.head_dim;
        let vocab_size = weights.config.vocab_size;
        let gamma_val = gamma.unwrap_or(weights.config.block_size);

        // Allocate the drafter's paged FP8 KV cache. One multi-layer cache,
        // sized for `max_seq_len + γ + 1` positions (prompt + γ drafts +
        // 1 bonus). Block size 16 matches the rest of Atlas.
        let block_size = 16;
        let kv_config = KvCacheConfig {
            block_size,
            num_kv_heads,
            head_dim,
            num_layers,
            // Phase 2 (Option B): flip drafter KV cache to BF16. The BF16
            // paged-attn dispatcher `prefill_attention_paged_dflash` reads
            // contiguous BF16 K/V from the layer pool; FP8 here would force
            // either an FP8 attn kernel (acceptance collapses on SM12.x —
            // see dflash_head.rs:82–86) or a dtype-mismatched read. BF16
            // first to land correctness; FP8 KV is a follow-up once the
            // architecture is right.
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let per_seq_blocks = (window_size.unwrap_or(max_seq_len) + gamma_val + 1) / block_size + 1;
        let num_blocks = per_seq_blocks * max_batch_size.max(1);
        tracing::info!(
            "DFlash KV pool: {} blocks ({} per seq × batch {})",
            num_blocks,
            per_seq_blocks,
            max_batch_size.max(1)
        );
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;

        // Resolve kernel handles. All BF16 paths since drafter weights are
        // BF16 (DflashQuantization::Bf16); FP8 cache uses the FP8 reshape +
        // FP8-aware paged-attention kernel. Module/function names verified
        // against existing Atlas resolutions in `qwen3_attention/mod.rs` and
        // `mtp_head.rs` plus the `extern "C" __global__` declarations under
        // `kernels/gb10/common/`.
        let kernels = DflashKernels {
            // DFlash drafter uses HF's vanilla RMSNorm convention
            // (`out = x * w / RMS(x)`), NOT Atlas's default offset-from-1
            // form (`out = x * (1 + w) / RMS(x)`). Atlas's standard
            // `rms_norm` kernel includes the `+1` for Qwen3-Next-style
            // checkpoints; we must use `rms_norm_vanilla` for the drafter
            // to match the drafter's HF-trained weights exactly.
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            // `rms_norm_residual` lands the post-attn / post-MLP add+norm in
            // a single launch. Atlas exposes this as a separate kernel — see
            // `mtp_head.rs:469` for the established lookup.
            residual_rms_norm: gpu
                .kernel("norm", "rms_norm_residual")
                .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            w4a16_gemm: super::super::try_kernel(gpu, "w4a16", "w4a16_gemm"),
            dense_gemm_pipelined: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            // Qwen3.6-DFlash uses yarn RoPE — confirmed in the drafter
            // `config.json:rope_scaling.rope_type="yarn"`. Atlas's yarn
            // kernel is `rope::rope_forward_yarn`.
            rope_qwen3: gpu.kernel("rope", "rope_forward_yarn")?,
            // FP8 KV cache writeback. Module name is the .cu stem
            // `reshape_and_cache`, function is `reshape_and_cache_flash_fp8`
            // (qwen3_attention/mod.rs:377-378 uses the same path).
            reshape_cache_fp8: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")?,
            // BF16 KV writeback — same module as the FP8 variant, different
            // function symbol. Used by precompute_ctx_kv + the per-layer
            // γ-block cache write that feeds prefill_attention_paged_dflash.
            reshape_cache_bf16: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?,
            // The Phase-2 γ-block kernel — same module as the existing
            // FP8 paged-prefill kernel (we just pass `causal_mask_enabled=0`
            // via a different dispatcher).
            prefill_attn_dflash_fp8: gpu
                .kernel("prefill_paged_fp8", "inferspark_prefill_paged_fp8")?,
            // Phase 2 (Option B) BF16 γ-block paged-attention. Same kernel
            // module as the target's BF16 prefill (`prefill_paged`); the
            // Rust dispatcher `ops::prefill_attention_paged_dflash` passes
            // `causal_mask_enabled=0` for bidirectional γ-block attention.
            prefill_attn_dflash_bf16: gpu
                .kernel("prefill_paged_sink", "inferspark_prefill_paged_sink")?,
            // Phase 5 (CUDA graph): indirect-args BF16 paged dispatcher + sinks.
            prefill_attn_dflash_bf16_indirect: gpu.kernel(
                "prefill_paged_indirect_sink",
                "inferspark_prefill_paged_indirect_sink",
            )?,
            prefill_attn_dflash_bf16_batched_sink: gpu.kernel(
                "inferspark_prefill_paged_batched_sink",
                "inferspark_prefill_paged_batched_sink",
            )?,
            silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            argmax: gpu.kernel("argmax", "argmax_bf16")?,
            argmax_batch: gpu.kernel("argmax", "argmax_bf16_batch")?,
            batched_embed: gpu.kernel("embed_from_argmax", "batched_embed")?,
            batch_anchor_add: gpu.kernel("dflash_batch_anchor_add", "dflash_batch_anchor_add")?,
            batch_markov_add_bias: gpu
                .kernel("dflash_batch_markov", "dflash_batch_add_depth_bias")?,
            batch_markov_store_tokens: gpu
                .kernel("dflash_batch_markov", "dflash_batch_store_depth_tokens")?,
            // Phase 2 Option B: slot_mapping builder. Same kernel the
            // target model uses for its KV cache writeback (see
            // crates/spark-model/src/model/impl_a1.rs:92).
            fill_slots: gpu.kernel("metadata_fill", "fill_slots_from_block_table")?,
            // Drafter has head_dim=128, but the target's
            // `inferspark_prefill` is compiled with HDIM=256. Using that
            // kernel produces corrupted attn_out for the drafter (kernel
            // reads 256 elements per head when only 128 are valid →
            // garbage in the back half of SMEM tiles → per-head sign-flip
            // pattern across q-heads). The HDIM=128 specialization
            // `inferspark_prefill_h128.cu` lives in the shared kernel
            // dir (`kernels/<hw>/common/`) so every target gets it.
            prefill_attn: gpu
                .kernel("inferspark_prefill_h128", "inferspark_prefill_h128")
                .map_err(|e| {
                    anyhow::anyhow!(
                        "{e}\n\nDFlash needs the HDIM=128 prefill kernel \
                         (`inferspark_prefill_h128`) compiled for this target. \
                         The kernel source lives at \
                         `kernels/<hw>/common/inferspark_prefill_h128.cu`. \
                         If you've added a new hardware target, copy the \
                         .cu file there."
                    )
                })?,
            // Phase G — BF16→FP8 weight quant kernel. Already in tree via
            // dense_gemv_fp8w.cu:36 under namespace "gemv_fp8w". Used at
            // load time only.
            quantize_bf16_to_fp8: gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?,
            // Phase G — Row-scaled BF16 × FP8 → BF16 GEMM. Atlas custom
            // kernel `fp8_gemm_t_row_scaled` appended to w4a16_gemm.cu
            // for Phase G (module namespace "w4a16").
            // try_kernel: absent on targets whose w4a16 module predates
            // Phase G — the FP8 drafter path is then skipped at the
            // ATLAS_DFLASH_DRAFTER_FP8 gate below (BF16 fallback).
            fp8_gemm_n128_row_scaled: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "fp8_gemm_t_row_scaled",
            ),
            // Phase G — Row-scaled BF16 × FP8 → BF16 GEMV (M=1). Used
            // by the lm_head GEMM swap in a γ-loop, since the
            // fp8_gemm_n128 GEMM kernel wastes 75% of its M_TILE at
            // M=γ=16 against vocab=248320.
            dense_gemv_fp8w: gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?,
            // Phase G — Small-M (M≤16) row-scaled FP8 GEMM for lm_head.
            // Single warp per CTA, no M_TILE waste. Custom kernel in
            // w4a16_gemm.cu, module namespace "w4a16".
            fp8_gemm_n128_row_scaled_m16: crate::layers::try_kernel(
                gpu,
                "w4a16",
                "fp8_gemm_t_row_scaled_m16",
            ),
            w4a16_gemv_batch4: crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch4"),
        };

        // Per-step scratch buffers. BF16 = 2 bytes/element.
        //
        // Sized for `n_attn_slots = ctx_window + γ` rows in the attention
        // path. The first `ctx_window` slots hold projected target ctx
        // (K/V only — Q is zero-padded so its attention output is
        // discarded). The next γ slots hold the noise tokens. lm_head +
        // logits + argmax tail still operates on γ rows (offset past ctx).
        let bf16 = 2usize;
        let g = gamma_val;
        // Phase 2.5n: ctx_window controls how many captured target positions
        // the drafter attends to per step. The drafter was trained over the
        // FULL captured prefix (paper §A.1), but capping at γ=16 cripples it
        // on prompts past a tiny window — Atlas's 6-10% acceptance vs the
        // paper's 70% is dominated by this cap. Default raised 512 → 4096
        // (2026-07-08): long generations (MinHeap ~2.6k tok) blow past 512
        // captured rows → truncated prefix → accept collapse + droop.
        // ATLAS_DFLASH_CTX_WINDOW overrides at construction time.
        //
        // Memory cost: attention-path scratch scales linearly with
        // `n_attn = γ + cw`. At cw=4096: stream/norm/acc ≈ 16.8 MB each;
        // mlp_intermediate/mlp_up = 4112 × 6144 × 2 ≈ 50.5 MB each (must
        // stay n_attn: contiguous path runs MLP over all rows, and
        // precompute_ctx_kv borrows mlp_intermediate as all_k_stage
        // [L×n×kv_dim ≈ 21 MB]); fused_kv_out ≈ 42 MB. logits is γ-rows
        // only (see alloc below). Total scratch ≈ 250 MB per head.
        let ctx_window: usize = std::env::var("ATLAS_DFLASH_CTX_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        tracing::info!(
            "DFlash ctx_window = {} (set ATLAS_DFLASH_CTX_WINDOW to override; \
             drafter trained on full captured prefix — larger is better, \
             scratch grows linearly)",
            ctx_window
        );
        let n_attn = g + ctx_window; // total attention slots
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // One scratch set per propose lane: the piecewise graphs bake these
        // pointers at capture, so each lane needs its own. γ rows only for
        // logits (see the per-field note below).
        let make_scratch = |gpu: &dyn GpuBackend| -> Result<DflashScratch> {
            let s = DflashScratch {
                stream_buf: gpu.alloc(n_attn * hidden_size * bf16)?,
                norm_buf: gpu.alloc(n_attn * hidden_size * bf16)?,
                q_buf: gpu.alloc(n_attn * q_dim * bf16)?,
                k_buf: gpu.alloc(n_attn * kv_dim * bf16)?,
                v_buf: gpu.alloc(n_attn * kv_dim * bf16)?,
                attn_out: gpu.alloc(n_attn * q_dim * bf16)?,
                mlp_intermediate: gpu.alloc(n_attn * intermediate_size * bf16)?,
                mlp_up: gpu.alloc(n_attn * intermediate_size * bf16)?,
                stream_acc: gpu.alloc(n_attn * hidden_size * bf16)?,
                fc_proj: gpu.alloc(ctx_window * hidden_size * bf16)?,
                // Phase 2 (Option B) precompute scratch. Worst-case the
                // first propose runs precompute over the whole captured
                // prefix up to `ctx_window`, so size for that. Per row:
                // `L * 2 * kv_dim * bf16` bytes. At L=5, kv_dim=512,
                // ctx_window=512: 5·2·512·512·2 = 5.24 MB.
                fused_kv_out: gpu
                    .alloc(ctx_window * num_layers * 2 * num_kv_heads * head_dim * bf16)?,
                // i64 slot mapping for reshape_and_cache (kernel takes
                // `long long*`). One entry per new ctx row.
                slot_mapping_dev: gpu.alloc(ctx_window * 8)?,
                // 12 bytes of device memory holding the per-call triple
                // `[u32 kv_len, u32 q_offset, u32 q_rope_pos]` that the indirect
                // paged-attention kernel reads at entry. Host writes via H2D
                // BEFORE entering the captured region.
                option_b_indirect_args_dev: gpu.alloc(12)?,
                // Phase E.2: pinned host buffer + event for the per-propose
                // drafter D2H. Pinned memory lets cuMemcpyDtoHAsync issue a
                // true async DMA on the caller's stream (vs. the synchronous
                // staging fallback the driver picks for pageable destinations).
                // The event lets us wait on the *copy*, not the whole stream,
                // so target-model verify work issued on the same stream can
                // proceed in parallel.
                draft_tokens_host_pinned: std::sync::atomic::AtomicPtr::new(
                    gpu.alloc_host_pinned(gamma_val * 4)?,
                ),
                draft_tokens_event: gpu.create_event()?,
                // γ rows only — NOT n_attn. The lm_head GEMM writes M=γ rows,
                // argmax + BLOCK_DUMP read rows 0..γ, and no path indexes logits
                // by ctx offset. Sizing at n_attn×vocab would cost 2.04 GB at
                // cw=4096 for rows nothing ever touches (γ rows ≈ 8.4 MB).
                logits: gpu.alloc(g * vocab_size * bf16)?,
                draft_tokens_dev: gpu.alloc(n_attn * 4)?,
                markov_prev_dev: gpu.alloc(4)?,
                markov_prev_host_pinned: std::sync::atomic::AtomicPtr::new(
                    gpu.alloc_host_pinned(4)?,
                ),
                position_ids: gpu.alloc(n_attn * 4)?,
            };
            // C1 diagnostic: zero ALL device buffers so any uninitialized
            // read sees deterministic zeros instead of per-lane garbage.
            // (Alloc_host_pinned already zeroes; device allocs do not.)
            for (p, bytes) in [
                (s.stream_buf, n_attn * hidden_size * bf16),
                (s.norm_buf, n_attn * hidden_size * bf16),
                (s.q_buf, n_attn * q_dim * bf16),
                (s.k_buf, n_attn * kv_dim * bf16),
                (s.v_buf, n_attn * kv_dim * bf16),
                (s.attn_out, n_attn * q_dim * bf16),
                (s.mlp_intermediate, n_attn * intermediate_size * bf16),
                (s.mlp_up, n_attn * intermediate_size * bf16),
                (s.stream_acc, n_attn * hidden_size * bf16),
                (s.fc_proj, ctx_window * hidden_size * bf16),
                (
                    s.fused_kv_out,
                    ctx_window * num_layers * 2 * num_kv_heads * head_dim * bf16,
                ),
                (s.slot_mapping_dev, ctx_window * 8),
                (s.option_b_indirect_args_dev, 12),
                (s.logits, g * vocab_size * bf16),
                (s.draft_tokens_dev, n_attn * 4),
                (s.markov_prev_dev, 4),
                (s.position_ids, n_attn * 4),
            ] {
                gpu.memset(p, 0, bytes)?;
            }
            Ok(s)
        };
        let scratch = make_scratch(gpu)?;
        let batch_capacity = max_batch_size.max(1);
        let batch_rows = batch_capacity
            .checked_mul(gamma_val)
            .ok_or_else(|| anyhow::anyhow!("DFlash batch row capacity overflow"))?;
        let batch_query_ids_dev = gpu.alloc(batch_rows * 4)?;
        let batch_position_ids = gpu.alloc(batch_rows * 4)?;
        let batch_query_embed = gpu.alloc(batch_rows * hidden_size * bf16)?;
        let batch_target_width = target_layer_ids
            .len()
            .checked_mul(target_hidden_size)
            .ok_or_else(|| anyhow::anyhow!("DFlash batch target-hidden width overflow"))?;
        let batch_target_bytes = batch_capacity
            .checked_mul(batch_target_width)
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("DFlash batch target-hidden bytes overflow"))?;
        let batch_fc_bytes = batch_capacity
            .checked_mul(hidden_size)
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("DFlash batch fc bytes overflow"))?;
        let batch_target_hidden = gpu.alloc(batch_target_bytes)?;
        let batch_fc_proj = gpu.alloc(batch_fc_bytes)?;
        let batch_fc_norm = gpu.alloc(batch_fc_bytes)?;
        let batch_tensor_bytes = |width: usize, label: &str| -> Result<usize> {
            batch_rows
                .checked_mul(width)
                .and_then(|n| n.checked_mul(bf16))
                .ok_or_else(|| anyhow::anyhow!("DFlash {label} bytes overflow"))
        };
        let batch_norm_bytes = batch_tensor_bytes(hidden_size, "batch norm")?;
        let batch_q_bytes = batch_tensor_bytes(q_dim, "batch q")?;
        let batch_kv_bytes = batch_tensor_bytes(kv_dim, "batch kv")?;
        let batch_mlp_bytes = batch_tensor_bytes(intermediate_size, "batch mlp")?;
        let batch_logits_bytes = batch_tensor_bytes(vocab_size, "batch logits")?;
        let batch_norm = gpu.alloc(batch_norm_bytes)?;
        let batch_q = gpu.alloc(batch_q_bytes)?;
        let batch_k = gpu.alloc(batch_kv_bytes)?;
        let batch_v = gpu.alloc(batch_kv_bytes)?;
        let batch_block_table_ptrs = gpu.alloc(batch_capacity * 8)?;
        let batch_cu_seqlens = gpu.alloc((batch_capacity + 1) * 4)?;
        let batch_kv_lens = gpu.alloc(batch_capacity * 4)?;
        let batch_slot_mapping = gpu.alloc(batch_rows * 8)?;
        let batch_attn_out = gpu.alloc(batch_q_bytes)?;
        let batch_attn_proj = gpu.alloc(batch_norm_bytes)?;
        let batch_mlp_gate = gpu.alloc(batch_mlp_bytes)?;
        let batch_mlp_up = gpu.alloc(batch_mlp_bytes)?;
        let batch_mlp_down = gpu.alloc(batch_norm_bytes)?;
        let batch_logits = gpu.alloc(batch_logits_bytes)?;
        let batch_tokens = gpu.alloc(batch_rows * 4)?;
        let batch_markov_prev = gpu.alloc(batch_capacity * 4)?;
        let batch_markov_embed_bytes = batch_capacity
            .checked_mul(weights.markov_rank)
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("DFlash batch Markov embed bytes overflow"))?;
        let batch_markov_bias_bytes = batch_capacity
            .checked_mul(vocab_size)
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("DFlash batch Markov bias bytes overflow"))?;
        let batch_markov_embed = if weights.markov_rank > 0 {
            gpu.alloc(batch_markov_embed_bytes)?
        } else {
            DevicePtr::NULL
        };
        let batch_markov_bias = if weights.markov_rank > 0 {
            gpu.alloc(batch_markov_bias_bytes)?
        } else {
            DevicePtr::NULL
        };
        gpu.memset(batch_query_ids_dev, 0, batch_rows * 4)?;
        gpu.memset(batch_position_ids, 0, batch_rows * 4)?;
        gpu.memset(batch_query_embed, 0, batch_rows * hidden_size * bf16)?;
        gpu.memset(batch_target_hidden, 0, batch_target_bytes)?;
        gpu.memset(batch_fc_proj, 0, batch_fc_bytes)?;
        gpu.memset(batch_fc_norm, 0, batch_fc_bytes)?;
        gpu.memset(batch_norm, 0, batch_norm_bytes)?;
        gpu.memset(batch_q, 0, batch_q_bytes)?;
        gpu.memset(batch_k, 0, batch_kv_bytes)?;
        gpu.memset(batch_v, 0, batch_kv_bytes)?;
        gpu.memset(batch_block_table_ptrs, 0, batch_capacity * 8)?;
        gpu.memset(batch_cu_seqlens, 0, (batch_capacity + 1) * 4)?;
        gpu.memset(batch_kv_lens, 0, batch_capacity * 4)?;
        gpu.memset(batch_slot_mapping, 0, batch_rows * 8)?;
        gpu.memset(batch_attn_out, 0, batch_q_bytes)?;
        gpu.memset(batch_attn_proj, 0, batch_norm_bytes)?;
        gpu.memset(batch_mlp_gate, 0, batch_mlp_bytes)?;
        gpu.memset(batch_mlp_up, 0, batch_mlp_bytes)?;
        gpu.memset(batch_mlp_down, 0, batch_norm_bytes)?;
        gpu.memset(batch_logits, 0, batch_logits_bytes)?;
        gpu.memset(batch_tokens, 0, batch_rows * 4)?;
        gpu.memset(batch_markov_prev, 0, batch_capacity * 4)?;
        if weights.markov_rank > 0 {
            gpu.memset(batch_markov_embed, 0, batch_markov_embed_bytes)?;
            gpu.memset(batch_markov_bias, 0, batch_markov_bias_bytes)?;
        }

        // Extra propose lanes: `proposal_lane_count` total lanes
        // (default 1). Each extra lane gets its own stream, scratch set,
        // Markov scratch, and a done-event the default stream waits on
        // before verify. Lanes >1 only engaged by propose_batch. The count
        // comes from the frozen startup policy, never a per-build env read.
        let n_lanes: usize = startup.proposal_lane_count;
        let mut extra_lanes = Vec::with_capacity(n_lanes - 1);
        for _ in 1..n_lanes {
            let stream = gpu.create_stream()?;
            let me = if weights.markov_rank > 0 {
                gpu.alloc(weights.markov_rank * 2)?
            } else {
                DevicePtr::NULL
            };
            let mb = if weights.markov_rank > 0 {
                gpu.alloc(weights.config.vocab_size * 2)?
            } else {
                DevicePtr::NULL
            };
            // C1 diagnostic: markov scratch zeroed like the rest of the
            // scratch set (uninitialized-read bisect).
            if weights.markov_rank > 0 {
                gpu.memset(me, 0, weights.markov_rank * 2)?;
                gpu.memset(mb, 0, weights.config.vocab_size * 2)?;
            }
            extra_lanes.push(DflashLane {
                stream,
                scratch: make_scratch(gpu)?,
                markov_embed: me,
                markov_bias: mb,
                done_event: gpu.create_event()?,
            });
        }
        tracing::info!(
            "DFlash propose lanes: {} (extra scratch + streams)",
            n_lanes
        );

        // Pre-compute inv_freq table for drafter RoPE.
        //
        // The drafter's `config.json:rope_scaling` is the source of truth:
        //   * `None`  ⇒ plain RoPE (`inv_freq[j] = 1 / θ^(2j/dim)`). The
        //     v2 2026-04-27 Qwen3.6-DFlash drafter ships `rope_scaling: null`.
        //   * `Some(yarn)` ⇒ YaRN-scaled table (Mistral-Small-4 lineage).
        //
        // Historical bug (Friday/Avarok 2026-05): this loader unconditionally
        // applied YaRN with factor=64 / orig_max_pos=4096 hardcoded, which
        // mis-scaled every low-frequency RoPE pair (pairs 0..11 divided by
        // 64, pairs 11..26 ramped). Result: drafter Q/K rotations landed in
        // the wrong angular basis at every layer → 0% draft acceptance. Now
        // we read the drafter's own scaling block instead of guessing.
        let rope_theta = weights.config.rope_theta;
        let rotary_dim = head_dim; // Qwen3.6-DFlash applies rope to full head_dim
        let dim_f = rotary_dim as f32;
        let n_pairs = rotary_dim / 2;
        let mut inv_freq_table = vec![0.0f32; n_pairs];

        // Default = plain RoPE. Overwritten in the YaRN arm below.
        for j in 0..n_pairs {
            inv_freq_table[j] = 1.0 / rope_theta.powf((2 * j) as f32 / dim_f);
        }

        let rope_kind: &str;
        if let Some(scaling) = weights.config.rope_scaling.as_ref() {
            match scaling.rope_type.as_deref() {
                Some("yarn") => {
                    let factor = scaling.factor.unwrap_or(1.0);
                    let beta_fast = scaling.beta_fast.unwrap_or(32.0);
                    let beta_slow = scaling.beta_slow.unwrap_or(1.0);
                    let orig_max_pos = scaling.original_max_position_embeddings.unwrap_or(4096.0);
                    let find_correction_dim = |num_rot: f32| -> f32 {
                        (dim_f * (orig_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
                            / (2.0 * rope_theta.ln())
                    };
                    let low = find_correction_dim(beta_fast).floor().max(0.0);
                    let high = find_correction_dim(beta_slow)
                        .ceil()
                        .min((rotary_dim - 1) as f32);
                    let ramp_denom = if (high - low).abs() < 1e-6 {
                        high - low + 0.001
                    } else {
                        high - low
                    };
                    for j in 0..n_pairs {
                        let pos_freq = rope_theta.powf((2 * j) as f32 / dim_f);
                        let inv_freq_extrap = 1.0 / pos_freq;
                        let inv_freq_interp = 1.0 / (factor * pos_freq);
                        let ramp = ((j as f32 - low) / ramp_denom).clamp(0.0, 1.0);
                        let extrap_factor = 1.0 - ramp;
                        inv_freq_table[j] = inv_freq_interp * (1.0 - extrap_factor)
                            + inv_freq_extrap * extrap_factor;
                    }
                    tracing::info!(
                        "DFlash RoPE = YaRN: theta={rope_theta}, factor={factor}, \
                         beta_fast={beta_fast}, beta_slow={beta_slow}, \
                         max_pos={orig_max_pos}, low_dim={low:.1}, high_dim={high:.1}",
                    );
                    rope_kind = "yarn";
                }
                Some(other) => {
                    tracing::warn!(
                        "DFlash drafter config has rope_scaling.rope_type={other:?} which Atlas \
                         doesn't recognise — falling back to plain RoPE (theta={rope_theta})."
                    );
                    rope_kind = "plain (unknown rope_type)";
                }
                None => {
                    tracing::warn!(
                        "DFlash drafter config has rope_scaling without rope_type — \
                         falling back to plain RoPE (theta={rope_theta})."
                    );
                    rope_kind = "plain (no rope_type)";
                }
            }
        } else {
            tracing::info!(
                "DFlash RoPE = plain (no rope_scaling in drafter config), theta={rope_theta}, \
                 {n_pairs} pairs",
            );
            rope_kind = "plain";
        }
        let _ = rope_kind; // logged above, retained for future debug surfaces

        let inv_freq_bytes: Vec<u8> = inv_freq_table
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let yarn_inv_freq = gpu.alloc(inv_freq_bytes.len())?;
        gpu.copy_h2d(&inv_freq_bytes, yarn_inv_freq)?;

        // ── Phase 2 (Option B) fused KV weight build ──────────────
        //
        // Concatenate every drafter layer's `k_proj.weight` and
        // `v_proj.weight` into a single `[L * 2 * kv_dim, h]` BF16
        // tensor laid out as `[K_0, V_0, K_1, V_1, …, K_{L-1}, V_{L-1}]`.
        // Lets `precompute_ctx_kv` derive all layers' ctx K/V in one
        // fused `dense_gemm` instead of `2 * L` separate calls per
        // propose. Layout matches vLLM's `_fused_kv_weight` in
        // `qwen3_dflash.py:301-303`.
        //
        // Memory layout reasoning: per-layer K weight is `[kv_dim, h]`
        // BF16 = `kv_dim * h * 2` bytes; V weight is the same shape.
        // Fused buffer total = `L * 2 * kv_dim * h * 2` bytes.
        // At L=5, kv_dim=512, h=2048: 5·2·512·2048·2 = 20.97 MB.
        //
        // We rely on the existing per-layer `DenseWeight.weight`
        // device pointers — no host roundtrip, just GPU `copy_d2d`.
        let kv_dim_bytes = num_kv_heads * head_dim * hidden_size * bf16; // K or V per layer
        let fused_total_bytes = num_layers * 2 * kv_dim_bytes;
        let fused_kv_weight = gpu.alloc(fused_total_bytes)?;
        for (l, layer) in weights.layers.iter().enumerate() {
            let layer_base = l * 2 * kv_dim_bytes;
            // K slot for layer l.
            gpu.copy_d2d(
                layer.k_proj.weight,
                fused_kv_weight.offset(layer_base),
                kv_dim_bytes,
            )?;
            // V slot for layer l (immediately after K).
            gpu.copy_d2d(
                layer.v_proj.weight,
                fused_kv_weight.offset(layer_base + kv_dim_bytes),
                kv_dim_bytes,
            )?;
        }
        tracing::info!(
            "DFlash fused_kv_weight: {} bytes ({} layers × 2 × kv_dim × h × bf16), \
             layout [K0,V0,K1,V1,…] to match vLLM precompute_and_store_context_kv",
            fused_total_bytes,
            num_layers,
        );

        let mut head = Self {
            num_layers,
            hidden_size,
            intermediate_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            draft_vocab_size: weights.config.draft_vocab_size.unwrap_or(vocab_size),
            gamma: gamma_val,
            mask_token_id,
            window_size,
            query_causal,
            target_layer_ids,
            target_hidden_size,

            embed_tokens_shared,
            lm_head_shared,
            lm_head_nvfp4,
            lm_head_shared_fp8: None,
            hidden_norm: weights.hidden_norm,
            norm: weights.norm,
            fc: weights.fc,
            markov_w1: weights.markov_w1,
            markov_w2: weights.markov_w2,
            markov_rank: weights.markov_rank,
            lane0_markov_embed: if weights.markov_rank > 0 {
                gpu.alloc(weights.markov_rank * 2)?
            } else {
                DevicePtr::NULL
            },
            lane0_markov_bias: if weights.markov_rank > 0 {
                gpu.alloc(weights.config.vocab_size * 2)?
            } else {
                DevicePtr::NULL
            },
            draft_id_to_target_id: None,
            layers: weights
                .layers
                .into_iter()
                .map(|l| DflashLayer {
                    input_layernorm: l.input_layernorm,
                    post_attention_layernorm: l.post_attention_layernorm,
                    q_proj: l.q_proj,
                    k_proj: l.k_proj,
                    v_proj: l.v_proj,
                    o_proj: l.o_proj,
                    q_norm: l.q_norm,
                    k_norm: l.k_norm,
                    gate_proj: l.gate_proj,
                    up_proj: l.up_proj,
                    down_proj: l.down_proj,
                    attention_sink_bias: l.attention_sink_bias,
                    // Phase G — populated below if ATLAS_DFLASH_DRAFTER_FP8=1.
                    q_proj_fp8: None,
                    k_proj_fp8: None,
                    v_proj_fp8: None,
                    o_proj_fp8: None,
                    gate_proj_fp8: None,
                    up_proj_fp8: None,
                    down_proj_fp8: None,
                    q_proj_nvfp4: None,
                    k_proj_nvfp4: None,
                    v_proj_nvfp4: None,
                    o_proj_nvfp4: None,
                    gate_proj_nvfp4: None,
                    up_proj_nvfp4: None,
                    down_proj_nvfp4: None,
                })
                .collect(),
            // Phase 2 stage 2: fused KV weight built above by copy_d2d
            // from each layer's k_proj/v_proj. precompute_ctx_kv will
            // GEMM against it in stage 3 once we wire the call site.
            fused_kv_weight: Some(fused_kv_weight),
            kv_cache: Mutex::new(kv_cache),
            scratch,
            batch_capacity,
            batch_query_ids_dev,
            batch_position_ids,
            batch_query_embed,
            batch_target_hidden,
            batch_fc_proj,
            batch_fc_norm,
            batch_norm,
            batch_q,
            batch_k,
            batch_v,
            batch_block_table_ptrs,
            batch_cu_seqlens,
            batch_kv_lens,
            batch_slot_mapping,
            batch_attn_out,
            batch_attn_proj,
            batch_mlp_gate,
            batch_mlp_up,
            batch_mlp_down,
            batch_logits,
            batch_tokens,
            batch_markov_prev,
            batch_markov_embed,
            batch_markov_bias,
            extra_lanes,
            kernels,
            max_seq_len,
            yarn_inv_freq,
            rope_theta,
            rotary_dim,
            rms_norm_eps: 1e-6,
            ctx_window,
            // Phase F: per-subgraph graph state — empty until the first
            // capture pass lands. Layout: [pre_0, post_0, ..., tail].
            propose_graphs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            next_lane: std::sync::atomic::AtomicUsize::new(0),
            lanes_start_event: gpu.create_event()?,
            suppress_graphs: std::sync::atomic::AtomicBool::new(false),
            propose_warmup_count: std::sync::atomic::AtomicUsize::new(0),
            quant: DflashQuantization::Bf16,
            startup,
        };

        tracing::info!(
            "BlockDiffusionDraftHead loaded: {} layers, hidden={}, intermediate={}, \
             GQA {}/{}, head_dim={}, γ={}, vocab={}, mask_token_id={}, causal={}, window={:?}, sinks={}, target_layers={:?}",
            head.num_layers,
            head.hidden_size,
            head.intermediate_size,
            head.num_q_heads,
            head.num_kv_heads,
            head.head_dim,
            head.gamma,
            head.vocab_size,
            head.mask_token_id,
            head.query_causal,
            head.window_size,
            head.layers
                .iter()
                .filter(|l| l.attention_sink_bias.is_some())
                .count(),
            head.target_layer_ids,
        );

        // Phase G — opt-in drafter MLP FP8. Quantize the seven dense-GEMM
        // weights per layer (q/k/v/o/gate/up/down) BF16 → FP8 E4M3 with
        // per-row f32 scales. One-shot at model load; runtime hot path
        // consumes the Fp8DenseWeight via fp8_gemm_n128 in pre/post_attn
        // (wired in G.3). Default OFF — bit-identical to F.2 baseline.
        //
        // Acceptance gate (G.4 design doc §16.7): bench must hold
        // ≥43% accept (vs 44.9% BF16) AND ≥11.0 tok/s (vs 8.70). If hard
        // fail, layer-by-layer ablation; skip layer 0 first.
        let fp8_requested = std::env::var("ATLAS_DFLASH_DRAFTER_FP8").ok().as_deref() == Some("1");
        let fp8_kernels_present = head.kernels.fp8_gemm_n128_row_scaled.0 != 0
            && head.kernels.fp8_gemm_n128_row_scaled_m16.0 != 0;
        if fp8_requested && !fp8_kernels_present {
            tracing::warn!(
                "ATLAS_DFLASH_DRAFTER_FP8=1 but fp8_gemm_t_row_scaled(_m16) kernels are \
                 not in this target's w4a16 PTX module — staying on the BF16 drafter path. \
                 Port the Phase G kernels from kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu."
            );
        }
        if fp8_requested && fp8_kernels_present {
            tracing::info!(
                "DFlash Phase G: quantizing drafter weights to FP8 E4M3 ({} layers × 7 GEMMs)",
                head.num_layers
            );
            let stream = 0u64; // default stream — load-time, no concurrency
            let q_dim_local = q_dim;
            let kv_dim_local = kv_dim;
            let h = head.hidden_size;
            let inter = head.intermediate_size;
            let quant_k = head.kernels.quantize_bf16_to_fp8;
            for (layer_idx, layer) in head.layers.iter_mut().enumerate() {
                // Q proj: [q_dim, h]
                layer.q_proj_fp8 =
                    Some(
                        layer
                            .q_proj
                            .quantize_to_fp8(gpu, quant_k, q_dim_local, h, stream)?,
                    );
                // K proj: [kv_dim, h]
                layer.k_proj_fp8 =
                    Some(
                        layer
                            .k_proj
                            .quantize_to_fp8(gpu, quant_k, kv_dim_local, h, stream)?,
                    );
                // V proj: [kv_dim, h]
                layer.v_proj_fp8 =
                    Some(
                        layer
                            .v_proj
                            .quantize_to_fp8(gpu, quant_k, kv_dim_local, h, stream)?,
                    );
                // O proj: [h, q_dim]
                layer.o_proj_fp8 =
                    Some(
                        layer
                            .o_proj
                            .quantize_to_fp8(gpu, quant_k, h, q_dim_local, stream)?,
                    );
                // Gate proj: [inter, h]
                layer.gate_proj_fp8 = Some(
                    layer
                        .gate_proj
                        .quantize_to_fp8(gpu, quant_k, inter, h, stream)?,
                );
                // Up proj: [inter, h]
                layer.up_proj_fp8 = Some(
                    layer
                        .up_proj
                        .quantize_to_fp8(gpu, quant_k, inter, h, stream)?,
                );
                // Down proj: [h, inter]
                layer.down_proj_fp8 = Some(
                    layer
                        .down_proj
                        .quantize_to_fp8(gpu, quant_k, h, inter, stream)?,
                );
                tracing::debug!("DFlash Phase G: layer {} quantized", layer_idx);
            }
            // Phase G — also quantize the shared lm_head weight. It's the
            // largest GEMM in the drafter (vocab × hidden = 248320 × 5120 ≈
            // 1.27B weights, ~14× any per-layer GEMM). We allocate a SEPARATE
            // FP8 buffer so we don't mutate the target model's BF16 lm_head
            // (the BF16 ptr stays valid for the BF16 path).
            tracing::info!(
                "DFlash Phase G: quantizing shared lm_head [{} × {}]",
                head.vocab_size,
                head.hidden_size
            );
            let lm_head_bf16 = crate::weight_map::DenseWeight {
                weight: head.lm_head_shared,
            };
            head.lm_head_shared_fp8 = Some(lm_head_bf16.quantize_to_fp8(
                gpu,
                quant_k,
                head.vocab_size,
                head.hidden_size,
                stream,
            )?);
            head.quant = DflashQuantization::Fp8Weights;
            tracing::info!(
                "DFlash Phase G: drafter weights ready as FP8 (quant = Fp8Weights). \
                 Set ATLAS_DFLASH_DRAFTER_FP8=0 to revert to BF16."
            );
        }

        head.try_install_nvfp4(gpu)?;
        Ok(head)
    }

    /// Borrow-validate the drafter dimensions against the target's hidden_size
    /// at construction time. Mismatch is a hard error — the `fc` projection
    /// width is baked from `target_hidden_size` and a runtime mismatch would
    /// produce silent garbage (vLLM's loader hits this same check).
    pub fn validate_against_target(&self, target_hidden_size: usize) -> Result<()> {
        if self.target_hidden_size != target_hidden_size {
            anyhow::bail!(
                "DFlash drafter target_hidden_size mismatch: drafter expects {}, target is {}",
                self.target_hidden_size,
                target_hidden_size
            );
        }
        Ok(())
    }

    /// Lightning DSpark: `dflash_config.causal=true`. Qwen-DFlash: false.
    pub(super) fn attn_causal(&self) -> bool {
        self.query_causal
    }

    /// SWA window in tokens. 0 = no window (full context).
    pub(super) fn attn_sliding_window(&self) -> u32 {
        self.window_size.unwrap_or(0) as u32
    }
}
