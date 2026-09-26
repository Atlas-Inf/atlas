// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash block-diffusion draft head implementing [`DraftProposer`].
//!
//! Block-diffusion drafter (Z Lab, arXiv 2602.06036): a small Qwen3-architecture
//! transformer (8 layers, hidden=2048, GQA 32:4, head_dim=128) that emits γ=16
//! tokens **in a single forward pass** via bidirectional in-block attention.
//! Conditioned on five intermediate hidden states captured from the target
//! model at `target_layer_ids` (e.g., `[1, 10, 19, 28, 37]` for
//! Qwen3.6-35B-A3B-DFlash), projected through a single `fc` layer at model
//! entry — NOT per-layer KV injection (early plan was wrong; cf. vLLM
//! `qwen3_dflash.py`).
//!
//! Phase 1 deliverable: type + trait wiring. The actual γ-block forward kernel
//! (`inferspark_dflash_block_attn_fp8`) lands in Phase 2; until then `propose()`
//! returns the bonus token repeated `num_drafts` times so the verify path
//! degenerates to single-token decode (acceptance ~100% but no speedup).

use parking_lot::Mutex;
use std::any::Any;
use std::collections::HashMap;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{DenseWeight, QuantizedWeight};

pub mod conv;
mod levers;
mod paged_attn_modules;
pub mod product_policy;
pub mod selector;
mod startup_diagnostics;
pub use conv::Dflash2Conv;
pub use product_policy::{
    DsparkStartupExecution, LightningDsparkIdentityLatch, LightningDsparkPolicyError,
    LightningDsparkProductPolicy, LightningDsparkRuntimeToggles, LightningStructuralGraphState,
    enforce_lightning_structural_gate,
};
pub use selector::Dflash2CandidateSelector;
pub use startup_diagnostics::DsparkDiagnostics;
#[cfg(test)]
mod product_policy_tests;

/// Kernel handles for the DFlash γ-block forward chain. All resolved once
/// at `BlockDiffusionDraftHead::from_weights` against the active GPU backend
/// (which compiles target-specific PTX at startup); subsequent
/// `propose()` calls just `KernelLaunch::new(...).launch(stream)`.
pub struct DflashKernels {
    pub rms_norm: KernelHandle,
    pub residual_rms_norm: KernelHandle,
    pub dense_gemv: KernelHandle,
    pub dense_gemv_batchm: KernelHandle,
    pub dense_gemm: KernelHandle,
    pub dflash2_conv: Option<KernelHandle>,
    pub dflash2_candidate_selector: Option<KernelHandle>,
    /// NVFP4 GEMM for the final logits when the shared lm_head is NVFP4
    /// (e.g. Holo): a BF16 `dense_gemm` on NVFP4-packed bytes reads garbage
    /// (and ~4× OOB → CUDA-700). `.0 == 0` when the target lm_head is BF16.
    pub w4a16_gemm: KernelHandle,
    pub dense_gemm_pipelined: KernelHandle,
    pub rope_qwen3: KernelHandle,
    pub reshape_cache_fp8: KernelHandle,
    /// BF16 KV cache writeback. Used by Phase 2 `precompute_ctx_kv` and
    /// the per-layer γ-block `reshape_and_cache` call to populate the
    /// drafter's BF16 paged cache before each `prefill_attention_paged_dflash`.
    pub reshape_cache_bf16: KernelHandle,
    pub prefill_attn_dflash_fp8: KernelHandle,
    /// BF16 paged-attention dispatcher for the DFlash γ-block.
    /// Calls `inferspark_prefill_paged` with `causal_mask_enabled=0`,
    /// reading BF16 K/V from the per-layer paged cache pool. Phase 2
    /// (Option B) drafter attention runs through this kernel; the FP8
    /// variant above is retained for a future quality-validated FP8 KV
    /// path. See `ops::prefill_attention_paged_dflash`.
    pub prefill_attn_dflash_bf16: KernelHandle,
    /// Phase 5 (CUDA graph) variant of `prefill_attn_dflash_bf16` that reads
    /// `kv_len` and `q_offset` from device pointers instead of taking them as
    /// kernel scalar args. Used by the graph-captured forward_block path so a
    /// single graph instance can be replayed across steps with different
    /// dynamic values written to the indirect-args buffer pre-launch.
    /// Resolves to kernel `inferspark_prefill_paged_indirect`.
    pub prefill_attn_dflash_bf16_indirect: KernelHandle,
    pub prefill_attn_dflash_bf16_batched_sink: KernelHandle,
    pub silu_mul: KernelHandle,
    pub residual_add: KernelHandle,
    pub argmax: KernelHandle,
    pub argmax_batch: KernelHandle,
    pub batched_embed: KernelHandle,
    pub batch_anchor_add: KernelHandle,
    pub batch_markov_add_bias: KernelHandle,
    pub batch_markov_store_tokens: KernelHandle,
    /// Phase 2 Option B: builds `[count]` i32 slot indices on-device
    /// from a host-provided block_table. Used by propose.rs to populate
    /// the slot_mapping passed to reshape_and_cache and precompute_ctx_kv.
    pub fill_slots: KernelHandle,
    /// Non-paged prefill attention (used for the γ-block self-attention
    /// when there's no persistent K/V cache to walk).
    pub prefill_attn: KernelHandle,
    /// Phase G — BF16 → FP8 E4M3 per-row weight quantization. Used at
    /// model load time to convert the seven dense-GEMM drafter weights
    /// (q/k/v/o/gate/up/down) when `ATLAS_DFLASH_DRAFTER_FP8=1`. Never
    /// on the hot path.
    pub quantize_bf16_to_fp8: KernelHandle,
    /// Phase G — Row-scaled BF16 × FP8 → BF16 GEMM. Consumes the
    /// `Fp8DenseWeight` (FP8 weight + per-row f32 scale) produced at
    /// load time by `quantize_bf16_to_fp8`. Wraps
    /// `kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu fp8_gemm_t_row_scaled`.
    /// Replaces `dense_gemm_bf16` on the seven dense-GEMM call sites in
    /// `forward_block_layer_pre_attn` / `_post_attn` when
    /// `self.quant == DflashQuantization::Fp8Weights`.
    pub fp8_gemm_n128_row_scaled: KernelHandle,
    /// Phase G — Row-scaled BF16 × FP8 → BF16 GEMV (M=1) for the
    /// lm_head fall-back. At γ=16 vs vocab=248320 the row-scaled GEMM
    /// wastes 75% of its M_TILE; the GEMV in a γ-loop is faster.
    pub dense_gemv_fp8w: KernelHandle,
    /// Phase G — Small-M (M≤16) row-scaled FP8 GEMM. Drop-in replacement
    /// for `fp8_gemm_n128_row_scaled` when M=γ=16. Single warp per CTA,
    /// no wasted M_TILE rows. Used by the lm_head GEMM.
    pub fp8_gemm_n128_row_scaled_m16: KernelHandle,
    pub w4a16_gemv_batch4: KernelHandle,
    pub w4a16_gemv_batch8: KernelHandle,
    pub w4a16_gemv_batch16: KernelHandle,
    pub w4a16_gemv_batch32: KernelHandle,
}

/// Per-step scratch buffers for the γ-block forward.
///
/// Each propose lane (see [`DflashLane`]) owns one full copy: the piecewise
/// propose graphs bake these pointers at capture, so a lane must always
/// replay with the scratch it captured with.
pub struct DflashLane {
    /// CUDA stream for this lane (lane 0 = the backend default stream;
    /// lanes 1.. are non-blocking secondary streams).
    pub stream: u64,
    pub scratch: DflashScratch,
    /// Scratch `[rank]` BF16 for the previous-token Markov embed.
    pub markov_embed: DevicePtr,
    /// Scratch `[vocab]` BF16 Markov bias row.
    pub markov_bias: DevicePtr,
    /// Event recorded after the lane's propose work; the default stream
    /// waits on it before verify so cross-lane work is ordered.
    pub done_event: u64,
}

/// Per-step scratch buffers for the γ-block forward.
///
/// Sized for `n_attn_slots = ctx_window + γ` rows, where ctx_window is the
/// max number of past target positions the drafter attends to per step. The
/// first `ctx_window` slots hold post-`fc` projected target context (K/V
/// only — Q is zero-padded); the next γ slots hold the noise tokens.
///
/// At γ=16 and ctx_window=γ=16: 32 rows × 2048 BF16 × ~10 buffers = ~1.3 MB
/// per head. lm_head logits buffer is the largest single alloc:
/// 32 × 248320 × 2 = 15 MB.
pub struct DflashScratch {
    pub stream_buf: DevicePtr,
    pub norm_buf: DevicePtr,
    pub q_buf: DevicePtr,
    pub k_buf: DevicePtr,
    pub v_buf: DevicePtr,
    pub attn_out: DevicePtr,
    pub mlp_intermediate: DevicePtr,
    pub mlp_up: DevicePtr,
    pub stream_acc: DevicePtr,
    /// `[ctx_window, draft_hidden]` BF16 — fc-projected + hidden_norm'd
    /// ctx for the most recent `ctx_window` target positions.
    pub fc_proj: DevicePtr,
    /// Phase 2 (Option B) scratch for `precompute_ctx_kv`: fused KV
    /// GEMM output, shape `[max_new_ctx, L * 2 * kv_dim]` BF16.
    /// `max_new_ctx` = `ctx_window` (worst case: first propose runs
    /// precompute over the entire prefix).
    pub fused_kv_out: DevicePtr,
    /// Phase 2 scratch: i32 slot mapping for the per-layer
    /// `reshape_and_cache` calls. Sized `[ctx_window]`.
    pub slot_mapping_dev: DevicePtr,
    /// Phase 5 (CUDA graph) scratch: 8 bytes (`[u32 kv_len, u32 q_offset]`)
    /// holding the per-call dynamic values that the indirect paged-attention
    /// kernel reads at entry. Host writes via `copy_h2d` BEFORE entering the
    /// captured region so the graph itself sees a stable device pointer.
    pub option_b_indirect_args_dev: DevicePtr,
    /// Phase E.2: pinned host buffer (`γ × 4` bytes) for the per-propose
    /// draft-token D2H copy. Allocated once at construction via
    /// `gpu.alloc_host_pinned`; the async D2H lands here without touching
    /// the system pageable allocator each call.
    ///
    /// Wrapped in `AtomicPtr` to keep `DflashScratch: Send + Sync` (the
    /// proposer is stored as `Arc<dyn DraftProposer>` which requires both
    /// auto-traits). Reads via `Ordering::Relaxed` are safe: the pointer
    /// itself never changes after construction; we only need atomic
    /// access for the Send/Sync bound, not for any actual concurrency.
    pub draft_tokens_host_pinned: std::sync::atomic::AtomicPtr<u8>,
    /// Phase E.2: CUDA event recorded against the draft-tokens D2H so the
    /// host can block on completion just before reading the pinned buffer,
    /// without a full `cuStreamSynchronize`. Created once at construction.
    pub draft_tokens_event: u64,
    pub logits: DevicePtr,
    pub draft_tokens_dev: DevicePtr,
    /// 4-byte device slot holding the Markov prev token. Host writes it
    /// via pinned `markov_prev_host_pinned` BEFORE the captured tail
    /// graph; the graph only reads this pointer. Do not H2D last_token
    /// from a stack temporary inside the tail (replay would see garbage).
    pub markov_prev_dev: DevicePtr,
    /// Pinned 4-byte host source for `markov_prev_dev`. Stable address
    /// so a captured H2D (if any) would still be valid; we keep the
    /// H2D outside the graph anyway.
    pub markov_prev_host_pinned: std::sync::atomic::AtomicPtr<u8>,
    /// `[ctx_window + γ]` i32 positions. First ctx_window are
    /// historical target positions (decoded indices); last γ are
    /// the to-be-predicted noise positions.
    pub position_ids: DevicePtr,
    /// DFlash2 scratch: dynamic conv delta coefficients `[γ, 4 * num_groups]` BF16.
    pub dflash2_conv_delta: DevicePtr,
    /// DFlash2 scratch: convolution output `[γ, hidden_size]` BF16.
    pub dflash2_conv_out: DevicePtr,
    /// DFlash2 scratch: projected hidden states `[γ, selector_rank]` BF16.
    pub dflash2_projected_hidden: DevicePtr,
}

/// Drafter-side weight precision. Defaults to BF16. **Phase G (2026-05-28)**
/// adds `Fp8Weights`, gated by env var `ATLAS_DFLASH_DRAFTER_FP8`. The
/// historical SM12.x acceptance collapse note applied to drafter FP8 KV
/// cache (different concern — bidirectional attention math); Phase G
/// targets weight FP8 only, so the risk surface is dynamic-range loss
/// in MLP intermediate activations, which per-row scales mitigate.
/// `--mtp-quantization fp8` is still not honored for the DFlash drafter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DflashQuantization {
    Bf16,
    /// Weight-only FP8: q/k/v/o/gate/up/down BF16 → FP8 E4M3 with per-row
    /// f32 scales at model load. Activations stay BF16; KV cache stays
    /// BF16. GEMMs use `fp8_gemm_n128` (BF16 × FP8 → BF16).
    Fp8Weights,
    /// Weight-only NVFP4: same seven GEMMs quantized at load, consumed by
    /// `w4a16_gemv_batch4` at γ≤4 (small-M, no tile waste). Default on;
    /// `ATLAS_NO_DFLASH_DRAFTER_NVFP4` to keep BF16 pipelined GEMM.
    Nvfp4Weights,
}

/// Per-drafter-layer Qwen3-style weights. Phase 1 is BF16-only; **Phase G**
/// (2026-05-28) adds optional FP8 weight fields populated at model load
/// when `ATLAS_DFLASH_DRAFTER_FP8=1`. The BF16 fields are always present
/// (Fp8 path falls back to them for any GEMM whose Fp8 weight is None).
#[allow(dead_code)]
pub struct DflashLayer {
    // Norms
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    // Attention (Qwen3: per-head Q/K RMSNorm)
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    // MLP
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
    /// Per-q-head attention sink `[num_q_heads]` BF16. Lightning DSpark ships
    /// this; Qwen-DFlash does not.
    pub attention_sink_bias: Option<DenseWeight>,

    // Phase G — optional FP8 mirrors of the seven dense-GEMM weights.
    // Populated at load time when `ATLAS_DFLASH_DRAFTER_FP8=1`, consumed
    // by forward_block_layer_pre_attn / _post_attn when self.quant ==
    // DflashQuantization::Fp8Weights. None when BF16 path is active.
    pub q_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub k_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub v_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub o_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub gate_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub up_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub down_proj_fp8: Option<crate::weight_map::Fp8DenseWeight>,
    pub q_proj_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub k_proj_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub v_proj_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub o_proj_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub gate_proj_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub up_proj_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub down_proj_nvfp4: Option<crate::weight_map::QuantizedWeight>,
    pub attention_conv: Option<Dflash2Conv>,
    pub mlp_conv: Option<Dflash2Conv>,
}

/// Per-sequence DFlash drafter state. One paged KV cache per drafter layer
/// (8 typical), shared block table across layers since attention shape is
/// identical layer-to-layer for a vanilla Qwen3 architecture. Mirrors
/// `MtpProposerState` in spirit; the multi-layer cache keeps it distinct.
pub struct DflashProposerState {
    /// Block table for the drafter's KV cache (shared across all drafter layers).
    pub block_table: Vec<u32>,
    /// Current logical sequence length in the drafter's KV cache. Tracks how
    /// many target-aligned positions have been written via
    /// `precompute_and_store_context_kv`.
    pub seq_len: usize,
    /// Drafts produced in the last `propose()` call. `after_verify` consults
    /// this to know how many KV positions to roll back when the accept
    /// prefix is shorter than γ.
    pub last_num_drafted: usize,
    /// Whether the prompt-time `precompute_and_store_context_kv` has been
    /// called. The first `propose()` after model build needs to run prefill
    /// over the full prompt's captured hiddens; subsequent steps incrementally
    /// append the latest accepted tokens' projections.
    pub prefill_done: bool,
    /// Multi-token accumulator for captured target hidden states. Layout:
    /// `[max_ctx_len, 5 * target_hidden]` BF16 packed. The scheduler appends
    /// the model's `dflash_hidden_save` (latest decoded position's 5 hiddens)
    /// into slot `ctx_len` after each successful verify. `propose()` reads
    /// the full populated prefix and projects all positions through `fc`
    /// at forward time. Sized for `max_seq_len` total positions; not
    /// circular — fail-fast if exceeded (drafter can't handle longer
    /// context than allocated).
    pub ctx_hidden_acc: DevicePtr,
    /// Number of populated slots in `ctx_hidden_acc`. Capped at `max_ctx_len`.
    pub ctx_len: usize,
    /// Drafts accepted in the verify that immediately preceded this propose.
    /// Set by `after_verify` so propose can label row-0 with its TRUE position.
    pub last_num_accepted: usize,
    /// Effective γ for the NEXT propose. Initialised to the configured γ
    /// (`initial_gamma`), so lever-off is byte-identical to the legacy
    /// `self.gamma` path; `ATLAS_DFLASH_ADAPTIVE_GAMMA=1` starts it at the
    /// 8-row floor and `after_verify` moves it via `next_gamma`.
    pub propose_gamma: usize,
    /// EMA of accepted draft tokens per verify step (bonus excluded);
    /// drives `next_gamma`. Lever-off: stays 0 and is never read.
    pub accept_ema: f32,
    /// Times this sequence's `propose_gamma` changed (adaptive lever on
    /// only). Steps/accepted per γ live in the head-level
    /// `adaptive_stats` accumulator — the periodic INFO line.
    pub adaptive_switches: u64,
    /// The configured γ_max this state was allocated under — the
    /// `next_gamma` ceiling and the reclaim reset point.
    pub gamma_max: usize,
    /// Frozen lever copy so reclaim can reset γ without the head handle.
    pub adaptive_gamma_on: bool,
    /// EAGLE-fix one-shot: when set, the next `propose()` skips its internal
    /// decode-append because the verify step (K=2 accept) already appended
    /// row 0 + row 1 in EAGLE order before calling propose. Consumed (reset to
    /// false) by propose. Only set under ATLAS_DFLASH_EAGLE_FIX=1.
    pub skip_next_decode_append: bool,
    /// Allocation cap for `ctx_hidden_acc` (in slot count). Mirrors the
    /// `max_seq_len` build arg so we can clamp without re-fetching it.
    pub max_ctx_len: usize,
    /// Width (bytes) of one `ctx_hidden_acc` slot — `5 * target_hidden * bf16`.
    /// Stored to avoid re-deriving on every append.
    pub ctx_slot_bytes: usize,
    /// Propose lane this seq is pinned to for its lifetime. Assigned once
    /// round-robin at `alloc_state` and NEVER derived from the batch
    /// position: ramp/drain reorders the batch between steps, and a seq's
    /// captured propose graphs bake their lane's scratch pointers — moving
    /// lanes would replay against another lane's scratch (silent corruption).
    /// `usize::MAX` = pre-assignment sentinel (defensive only).
    pub lane_id: usize,
    /// Generation-stamped ownership descriptor. Bound by model sequence
    /// allocation before the state can propose or own CUDA graphs.
    pub(crate) lifecycle: Option<CaptureDescriptor>,

    // ─── Phase 2 Option B fields (paged KV cache for ctx) ───────────────
    /// Device-side block table for the drafter's paged KV cache. Allocated
    /// once at first propose with enough u32 slots to cover `max_seq_len`
    /// at block_size=16. Read by `prefill_attention_paged_dflash` to map
    /// logical block indices to physical pool block indices. Mirrors the
    /// host-side `block_table` Vec, copied to GPU after each `alloc_block`.
    pub block_table_dev: Option<DevicePtr>,
    /// Number of paged-cache slots populated with ctx K/V for this sequence.
    /// Distinct from `ctx_len` (which counts target_hidden_acc slots). The
    /// drafter writes one ctx K/V slot per accepted target token; the
    /// γ-block then attends over `[0..ctx_count_drafter+γ)`. Bumped by γ
    /// per propose (γ slots written for the noise rows) and trimmed in
    /// `after_verify` by `(γ - num_accepted)`.
    pub ctx_count_drafter: usize,
    /// Cap for `ctx_count_drafter`. Mirrors `block_table.len() * block_size`.
    pub max_ctx_count_drafter: usize,
    /// Phase I — incremental ctx precompute watermark. Number of ctx slots
    /// `[0..ctx_committed)` whose K/V is already valid in the paged cache
    /// from a prior propose. Each step we only precompute the new tail
    /// `[ctx_committed..ctx_len)` instead of rebuilding the whole prefix
    /// (the old O(ctx_len²) waste — see design doc §18). Reset to the
    /// current `ctx_len` on any rewind so stale slots can't be read.
    /// `0` forces a full rebuild (first propose, or the debug escape hatch).
    pub ctx_committed: usize,
    /// Phase I (v2) — per-slot TRUE absolute decoded position, stamped once
    /// when a ctx slot is appended and never recomputed. Indexed by ctx
    /// slot (parallel to `ctx_hidden_acc` slots, len == `ctx_len`). This is
    /// the vLLM convention: a cached token's rope position is fixed at
    /// insert time, so committed slots never go stale when later accepts
    /// shift the live `position`. Replaces the sliding `absolute_start_pos
    /// + i` formula in `precompute_ctx_kv`. Prefill positions are seeded
    /// `0..prompt_len` in `update_dflash_ctx_len_after_prefill`.
    pub ctx_positions: Vec<i32>,
}

impl DflashProposerState {
    /// Transactional reclaim when owner validation fails in `free_state`:
    /// retire the descriptor best-effort, return KV blocks, free the ctx
    /// accumulator and device block table, and reset the lazy-alloc
    /// watermarks — the error still propagates, but nothing owned by this
    /// state leaks.
    ///
    /// A backend `free` that itself fails is LOGGED and the pointer is
    /// RETAINED (not cleared), so a later cleanup retry can still release
    /// it — a failed free must not silently convert into an unrecoverable
    /// leak. Production-owned seam: directly unit-tested including the
    /// free-failure retention behavior.
    pub(crate) fn reclaim_on_owner_failure(
        &mut self,
        gpu: &dyn GpuBackend,
        kv_cache: &parking_lot::Mutex<spark_runtime::kv_cache::PagedKvCache>,
    ) {
        if let Some(lifecycle) = self.lifecycle.as_mut() {
            let _ = lifecycle.retire(lifecycle.owner());
        }
        if !self.block_table.is_empty() {
            kv_cache.lock().free_blocks(&self.block_table);
            self.block_table.clear();
        }
        if self.ctx_hidden_acc.0 != 0 {
            if let Err(error) = gpu.free(self.ctx_hidden_acc) {
                tracing::error!(
                    "DSpark reclaim: freeing ctx accumulator {:#x} failed ({error}); \
                     pointer retained for a later cleanup retry",
                    self.ctx_hidden_acc.0
                );
            } else {
                self.ctx_hidden_acc = DevicePtr(0);
            }
        }
        if let Some(bt) = self.block_table_dev.take() {
            match gpu.free(bt) {
                Ok(()) => {}
                Err(error) => {
                    tracing::error!(
                        "DSpark reclaim: freeing device block table {:#x} failed ({error}); \
                         handle restored for a later cleanup retry",
                        bt.0
                    );
                    // Restore the handle so a later cleanup retry can free it
                    // (a taken-and-dropped handle is an unrecoverable leak).
                    self.block_table_dev = Some(bt);
                }
            }
        }
        self.max_ctx_count_drafter = 0;
        self.ctx_count_drafter = 0;
        self.ctx_committed = 0;
        self.ctx_positions.clear();
        // γ/EMA back to the floor so a re-prepared state cannot carry a
        // stale adaptive level into a fresh sequence.
        self.propose_gamma = adaptive_gamma::initial_gamma(self.adaptive_gamma_on, self.gamma_max);
        self.accept_ema = 0.0;
        self.seq_len = 0;
        self.ctx_len = 0;
        self.prefill_done = false;
    }
}

impl ProposerState for DflashProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Block-diffusion draft head. Public API is the [`DraftProposer`] trait.
///
/// The drafter shares `embed_tokens` and `lm_head` with the target — these
/// are NOT in the drafter's safetensors checkpoint (verified against
/// `z-lab/Qwen3.6-35B-A3B-DFlash` commit 42d3b34). The constructor takes
/// the target's `embed_tokens_shared` and `lm_head_shared` device pointers
/// at build time and slots them in alongside the drafter's own `fc`,
/// `hidden_norm`, `norm`, and per-layer weights.
#[allow(dead_code)]
pub struct BlockDiffusionDraftHead {
    // Drafter-architecture config (mirrors the drafter's HF config.json).
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub draft_vocab_size: usize,
    pub gamma: usize,
    pub mask_token_id: u32,
    pub window_size: Option<usize>,
    /// Causal γ-block attention. `true` for Lightning DSpark
    /// (`dflash_config.causal`); `false` for Qwen-DFlash bidirectional.
    pub query_causal: bool,
    /// `target_layer_ids`. Same data as `TransformerModel::dflash_capture_layers`,
    /// repeated here so the loader is the single source of truth; the model
    /// reads these to size its capture buffer.
    pub target_layer_ids: Vec<usize>,
    /// Target-side hidden_size (used for the `fc` projection input width:
    /// `target_layer_ids.len() * target_hidden_size`).
    pub target_hidden_size: usize,

    // === Weights shared with the target ===
    /// Target's embed_tokens GPU pointer. The drafter's checkpoint has no
    /// own embeddings — both vocab and embedding dim must match the target
    /// (Qwen3.6-35B-A3B-DFlash: vocab=248320, hidden=2048 — same as target).
    pub embed_tokens_shared: DevicePtr,
    /// Target's lm_head GPU pointer. Used for the drafter's per-position
    /// argmax over `[γ, vocab]` logits. Valid only when the target lm_head is
    /// BF16; when `lm_head_nvfp4` is `Some`, the NVFP4 path is used instead.
    pub lm_head_shared: DevicePtr,
    /// Target's NVFP4 lm_head (packed + scales), shared with the drafter for
    /// the final logits GEMM. `Some` when the target ships an NVFP4 lm_head
    /// (e.g. Holo) — required because a BF16 `dense_gemm` on the NVFP4 buffer
    /// reads garbage and OOB. `None` → use the BF16 `lm_head_shared`.
    pub lm_head_nvfp4: Option<QuantizedWeight>,
    /// Phase G — optional FP8 mirror of the shared lm_head weight,
    /// `[vocab_size, hidden_size]` FP8 E4M3 + per-row f32 scales.
    /// Built at model load when `ATLAS_DFLASH_DRAFTER_FP8=1`. Owned by
    /// the drafter (separate allocation from the shared BF16 ptr) since
    /// it must not mutate the target model's lm_head. `None` on the
    /// BF16 path.
    pub lm_head_shared_fp8: Option<crate::weight_map::Fp8DenseWeight>,

    // === Weights from the drafter checkpoint ===
    /// Hidden-norm applied to the projected target context before mixing
    /// with the embedded tokens (Qwen3-DFlash convention; see vLLM
    /// `DFlashQwen3Model.hidden_norm`).
    pub hidden_norm: DenseWeight,
    /// Final RMSNorm before LM head.
    pub norm: DenseWeight,
    /// `fc` projection — `[draft_hidden, target_layer_ids.len() * target_hidden_size]`
    /// BF16. Maps the stack of captured target hiddens to drafter's input space
    /// once at model entry. Replaces the earlier (incorrect) "per-layer KV
    /// injection" design.
    pub fc: DenseWeight,
    /// DSpark Markov `w1` `[vocab, rank]` BF16. None for DFlash-only heads.
    pub markov_w1: Option<DenseWeight>,
    /// DSpark Markov `w2` `[vocab, rank]` BF16.
    pub markov_w2: Option<DenseWeight>,
    pub markov_rank: usize,
    /// Optional draft-vocab-id → target-vocab-id remap. `None` when the
    /// drafter shares vocab with the target (Qwen3.6-35B-A3B-DFlash case:
    /// vocab_size == draft_vocab_size == 248320).
    pub draft_id_to_target_id: Option<DevicePtr>,
    /// Drafter transformer layers (8 for Qwen3.6-35B-A3B-DFlash).
    pub layers: Vec<DflashLayer>,
    /// DFlash2 CandidateSelector bilinear path scorer (replaces greedy argmax).
    pub candidate_selector: Option<Dflash2CandidateSelector>,

    /// Phase 2 (Option B) fused K/V projection across all L drafter layers.
    /// Shape: `[L × 2 × kv_dim, h]` BF16 — concatenated `[K0; V0; K1; V1; …]`
    /// (per-layer K then V interleaved). Built once at construction by
    /// `copy_d2d`-stitching the per-layer `k_proj.weight` and `v_proj.weight`
    /// pointers from `layers[i]`. Lets `precompute_ctx_kv` derive every
    /// drafter layer's ctx K/V via a single `dense_gemm` of shape
    /// `[new_ctx_count, h] × [h, L·2·kv_dim]` instead of 2·L per-layer GEMMs.
    ///
    /// `None` until Phase 2 lands the build (stage 1: kernel/dispatcher
    /// scaffolding; stage 2: this allocation + the precompute_ctx_kv module;
    /// stage 3: pyref bit-exact diff). Layout (K then V per layer) chosen
    /// to match vLLM's `_fused_kv_weight` in `qwen3_dflash.py:381-389`.
    pub fused_kv_weight: Option<DevicePtr>,

    /// Paged FP8 KV cache. One cache holding all `num_layers` drafter layers,
    /// laid out the same way the target's KV cache is — block-table-keyed,
    /// `num_layers × num_kv_heads × head_dim` per slot. Allocating a single
    /// multi-layer cache (vs. one per drafter layer) matches Atlas's existing
    /// `PagedKvCache` ABI and lets us reuse the existing `reshape_and_cache`
    /// kernel without per-layer dispatch overhead.
    pub kv_cache: Mutex<PagedKvCache>,

    /// Per-step scratch buffers (allocated once at construction, reused).
    pub scratch: DflashScratch,
    /// Stable staging for the first native B×gamma operation. Capacity is the
    /// model's admitted max batch; rows are `[sequence][gamma]`.
    pub batch_capacity: usize,
    pub batch_query_ids_dev: DevicePtr,
    pub batch_position_ids: DevicePtr,
    pub batch_query_embed: DevicePtr,
    pub batch_target_hidden: DevicePtr,
    pub batch_fc_proj: DevicePtr,
    pub batch_fc_norm: DevicePtr,
    pub batch_norm: DevicePtr,
    pub batch_q: DevicePtr,
    pub batch_k: DevicePtr,
    pub batch_v: DevicePtr,
    pub batch_block_table_ptrs: DevicePtr,
    pub batch_cu_seqlens: DevicePtr,
    pub batch_kv_lens: DevicePtr,
    pub batch_attention_args: DevicePtr,
    pub batch_slot_mapping: DevicePtr,
    pub batch_attn_out: DevicePtr,
    pub batch_attn_proj: DevicePtr,
    pub batch_mlp_gate: DevicePtr,
    pub batch_mlp_up: DevicePtr,
    pub batch_mlp_down: DevicePtr,
    pub batch_logits: DevicePtr,
    pub batch_tokens: DevicePtr,
    pub batch_markov_prev: DevicePtr,
    pub batch_markov_embed: DevicePtr,
    pub batch_markov_bias: DevicePtr,
    /// DFlash2 selector projected-hidden scratch, `[B*gamma, rank]` BF16.
    /// NULL when the head has no candidate selector (Lightning DSpark).
    pub batch_dflash2_projected: DevicePtr,
    /// DFlash2 conv scratch: dynamic-delta rows `[B*gamma, 2*kernel_size*groups]`
    /// and conv output `[B*gamma, hidden]`. NULL when the drafter ships no conv.
    pub batch_conv_delta: DevicePtr,
    pub batch_conv_out: DevicePtr,

    /// Additional propose lanes (lane 0 IS `self.scratch` on the default
    /// stream). Sized `ATLAS_DFLASH_PROPOSE_LANES - 1` (default 1 lane).
    /// A sequence's lane is fixed for its lifetime (`slot % lanes`) so its
    /// captured graphs always replay against the scratch they captured with.
    pub extra_lanes: Vec<DflashLane>,

    /// Lane-0 Markov scratch mirrors (kept on the head for the single-lane
    /// path; lanes 1.. carry their own copies inside `extra_lanes`).
    pub lane0_markov_embed: DevicePtr,
    pub lane0_markov_bias: DevicePtr,

    /// All kernel handles needed by `propose()` and the eventual prefill
    /// projection (`precompute_and_store_context_kv`).
    pub kernels: DflashKernels,

    /// Per-sequence ctx accumulator capacity (mirrors model's `max_seq_len`).
    /// Used by `alloc_state` to size each new sequence's `ctx_hidden_acc`.
    pub max_seq_len: usize,

    /// Pre-computed yarn inv_freq table (`[head_dim/2]` f32 on GPU).
    /// Drafter rope_scaling: factor=64, beta_fast=32, beta_slow=1,
    /// original_max_position_embeddings=4096 (per drafter config.json).
    pub yarn_inv_freq: DevicePtr,

    /// rope_theta (10000000 for Qwen3.6-DFlash). Stored to pass into the
    /// rope_yarn kernel each step.
    pub rope_theta: f32,

    /// rotary_dim. Drafter uses full-rotation (rotary_dim = head_dim = 128).
    pub rotary_dim: usize,

    /// RMSNorm epsilon (drafter inherits Qwen3 default 1e-6).
    pub rms_norm_eps: f32,

    /// Max number of past target positions injected into the drafter's K/V
    /// per step. Default γ — drafter sees at most γ ctx + γ noise = 2γ
    /// attention positions per step. ctx_window=0 disables ctx conditioning
    /// (degraded quality, ablation only).
    pub ctx_window: usize,

    // === Phase D (CUDA graph capture) → Phase F (piecewise) ===
    /// Per-subgraph captured handles. `None` until warm-up completes and
    /// the first capture pass lands; on the capture pass we fill this
    /// `Vec` with `2 × num_layers + 1` handles laid out as
    /// `[pre_0, post_0, pre_1, post_1, ..., pre_{N-1}, post_{N-1}, tail]`.
    /// Slot index = `layer_idx * 2 + half` for the layer halves
    /// (half = 0 for pre_attn, 1 for post_attn) and `num_layers * 2` for
    /// the tail (final norm + lm_head + argmax). `GraphHandle(0)` is the
    /// "empty capture" sentinel and means that slot replays eager.
    ///
    /// Phase F.2 (2026-05-28): replaces the single full-region capture
    /// with one capture per subgraph. Attention is NEVER captured —
    /// it's the natural sync barrier between captured subgraphs
    /// (vLLM piecewise convention). See design doc §15.
    /// Piecewise propose graphs keyed by validated sequence generation and
    /// every captured pointer/lane identity. Pointer reuse cannot cross a
    /// retired generation. `GraphHandle(0)` remains the eager sentinel.
    pub propose_graphs: Mutex<HashMap<DflashGraphIdentity, Vec<spark_runtime::gpu::GraphHandle>>>,
    /// Round-robin counter handing out propose lanes at `alloc_state`.
    /// One extra-lane stream may serve several seqs (n > lanes); the lane
    /// itself never moves for a seq.
    pub next_lane: std::sync::atomic::AtomicUsize,
    /// Entry-ordering event: recorded on the default stream at the top of
    /// the multi-lane `propose_batch`; every extra lane waits on it so
    /// drafter-ctx precompute / `after_verify` writes enqueued on the
    /// default stream are visible before lanes read them.
    pub lanes_start_event: u64,
    /// When set, all `forward_block` calls run eagerly. Mirrors target-model
    /// `TransformerModel::suppress_graphs` so external code can disable
    /// graphs at runtime (e.g. while calibrating FP8 KV).
    pub suppress_graphs: std::sync::atomic::AtomicBool,
    /// Adaptive-γ counters (lever on only): steps/accepted per γ bucket,
    /// γ switches, total verify calls — the periodic `DFLASH ADAPTIVE`
    /// INFO line aggregates here so it reflects the whole batch, not one
    /// sequence. Lever-off: never touched.
    pub adaptive_stats: Mutex<adaptive_gamma::AdaptiveGammaStats>,
    /// How many eager warm-up calls we've executed against the graph path.
    /// Default warmup target is 2 (override via `ATLAS_DFLASH_PROPOSE_WARMUP_N`).
    /// Two eager passes warm the PTX→SASS cache, ramp GB10 clocks to steady
    /// state, and bring hot weight tiles into L2 before the capture freezes
    /// SASS variants the driver picks. Shared across all subgraphs — every
    /// subgraph captures on the same propose call after the warmup target
    /// is hit.
    pub propose_warmup_count: std::sync::atomic::AtomicUsize,

    // Quantization mode (BF16 only for Phase 1).
    pub quant: DflashQuantization,

    /// Startup-static execution values, resolved once at construction.
    /// `propose`/`forward_block`/lane build read this instead of the
    /// process environment. Product heads derive it from the validated
    /// Lightning policy; generic heads keep legacy lenient semantics.
    pub startup: DsparkStartupExecution,
}

mod contract;
pub use contract::{
    AttentionLayout, BonusLayout, CheckpointLayout, ConfidenceLayout, KvDtype, KvLayout,
    LIGHTNING_ALGORITHM, LIGHTNING_CHECKPOINT_BLOCK_SIZE, LIGHTNING_EP, LIGHTNING_MARKOV_RANK,
    LIGHTNING_MODEL_IDENTITY, LIGHTNING_NUM_DRAFTS, LIGHTNING_PHYSICAL_KV_PAGE_SIZE,
    LIGHTNING_SERVED_GAMMA, LIGHTNING_SWA_WINDOW, LIGHTNING_TAPS, LIGHTNING_TARGET_HIDDEN_SIZE,
    LIGHTNING_TARGET_MODEL_TYPE, LIGHTNING_TARGET_NUM_EXPERTS, LIGHTNING_TARGET_NUM_LAYERS,
    LIGHTNING_TARGET_TOP_K, LIGHTNING_TP, LightningDsparkContractError, LightningDsparkProfile,
    MarkovLayout, ParallelismLayout,
};
#[cfg(test)]
mod contract_tests;
mod row_contract;
pub use row_contract::{CommitProjection, DsparkProposal, DsparkRowError, LightningRowContract};
mod adaptive_gamma;
mod batch_inputs;
pub use batch_inputs::{DsparkBatchInput, DsparkBatchInputError, DsparkBatchSequence};
mod batch_attention;
mod batch_conv;
mod batch_execution;
#[cfg(test)]
mod batch_execution_tests;
mod batch_fallback;
mod batch_forward;
mod batch_forward_tail;
#[cfg(test)]
mod batch_inputs_tests;
mod batch_plan;
mod batch_projection;
mod batch_propose;
mod batch_tail_dflash2;
mod lifecycle;
#[cfg(test)]
mod row_contract_tests;
pub use lifecycle::{
    CaptureDescriptor, CaptureStatus, DflashGraphIdentity, DsparkLifecycleError, SequenceGeneration,
};
mod forward_block;
mod forward_block_layer;
mod forward_block_layer_paged;
#[cfg(test)]
mod free_state_tests;
mod from_weights;
#[cfg(test)]
mod lifecycle_tests;
mod markov;
mod nvfp4;
mod parity_report;
mod precompute_ctx_kv;
mod propose;
mod small_m_gemm;

impl BlockDiffusionDraftHead {
    /// Total propose lanes (lane 0 = default-stream scratch; the rest live
    /// in `extra_lanes`). `ATLAS_DFLASH_PROPOSE_LANES` overrides (default 1).
    pub fn lane_count(&self) -> usize {
        1 + self.extra_lanes.len()
    }

    /// Resolve a lane's mutable propose resources: (stream, scratch,
    /// markov_embed, markov_bias). Lane 0 is the head's own scratch on the
    /// backend default stream; lanes 1.. are independent copies.
    pub(super) fn lane(
        &self,
        lane: usize,
        default_stream: u64,
    ) -> (u64, &DflashScratch, DevicePtr, DevicePtr) {
        if lane == 0 || self.extra_lanes.is_empty() {
            (
                default_stream,
                &self.scratch,
                self.lane0_markov_embed,
                self.lane0_markov_bias,
            )
        } else {
            let l = &self.extra_lanes[(lane - 1).min(self.extra_lanes.len() - 1)];
            (l.stream, &l.scratch, l.markov_embed, l.markov_bias)
        }
    }
    fn validate_dflash_owner(
        &self,
        dstate: &DflashProposerState,
        expected_owner: Option<SequenceGeneration>,
    ) -> Result<SequenceGeneration> {
        let expected_owner = expected_owner
            .ok_or_else(|| anyhow::anyhow!("DFlash operation requires expected owner"))?;
        dstate
            .lifecycle
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash operation has no generation owner"))?
            .validate_access(expected_owner)?;
        Ok(expected_owner)
    }

    /// Terminal-path owner validation for `free_state`: ownership only, so
    /// a same-owner SECOND cleanup is an idempotent success (everything was
    /// reclaimed by the first pass) rather than a Retired error.
    fn validate_dflash_owner_terminal(
        &self,
        dstate: &DflashProposerState,
        expected_owner: Option<SequenceGeneration>,
    ) -> Result<SequenceGeneration> {
        let expected_owner =
            expected_owner.ok_or_else(|| anyhow::anyhow!("DFlash free requires expected owner"))?;
        dstate
            .lifecycle
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DFlash free has no generation owner"))?
            .validate_ownership(expected_owner)?;
        Ok(expected_owner)
    }
}

impl DraftProposer for BlockDiffusionDraftHead {
    fn startup_diagnostics(&self) -> Option<&DsparkDiagnostics> {
        Some(&self.startup.diagnostics)
    }

    fn propose_batch_max(
        &self,
        _buffers: &spark_runtime::buffers::BufferArena,
        _config: &atlas_core::config::ModelConfig,
    ) -> usize {
        batch_plan::propose_batch_width(
            self.startup.native_batch_authoritative,
            self.startup.diagnostics.batch_parity,
            self.startup.generic_batch_authoritative,
            // Generic multi-lane (ATLAS_DFLASH_PROPOSE_LANES > 1): each seq
            // proposes on its pinned lane stream, so the batched entry can
            // produce output at the full admission width even though the
            // Lightning Bxgamma seam (gamma == 4 contract) never applies.
            self.lane_count() > 1,
            self.batch_capacity,
        )
    }

    fn propose_batch_min(&self) -> usize {
        batch_plan::propose_batch_floor(
            self.startup.native_batch_authoritative,
            self.startup.diagnostics.batch_parity,
        )
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        // Per-seq ctx accumulator: `[max_seq_len, 5 * target_hidden] BF16`.
        // Sized once, re-used across the seq's lifetime; reset on
        // `free_state`. At max_seq_len=16384 and 5×2048 BF16: 320 MB per
        // seq — tolerable on a single Spark with max_batch_size=1; for
        // higher batch we may want to reduce to a smaller working window.
        let bf16 = 2usize;
        let ctx_slot_bytes = self.target_layer_ids.len() * self.target_hidden_size * bf16;
        let total = self.max_seq_len * ctx_slot_bytes;
        let ctx_hidden_acc = gpu.alloc(total)?;
        // Initialize to zero so stale data doesn't leak between sequences.
        // Transactional: a failed memset frees the accumulator instead of
        // leaking it for the server's lifetime; a failed FREE during that
        // cleanup is logged (the allocation is then backend-orphaned — the
        // pointer is already unreachable from any live state).
        if let Err(error) = gpu.memset(ctx_hidden_acc, 0, total) {
            if let Err(free_error) = gpu.free(ctx_hidden_acc) {
                tracing::error!(
                    "DSpark alloc_state: freeing failed-memset accumulator {:#x} failed \
                     ({free_error}); allocation orphaned on the backend",
                    ctx_hidden_acc.0
                );
            }
            return Err(error);
        }
        Ok(Box::new(DflashProposerState {
            block_table: Vec::with_capacity(64),
            seq_len: 0,
            last_num_drafted: 0,
            prefill_done: false,
            ctx_hidden_acc,
            ctx_len: 0,
            last_num_accepted: 0,
            propose_gamma: adaptive_gamma::initial_gamma(self.startup.adaptive_gamma, self.gamma),
            accept_ema: 0.0,
            adaptive_switches: 0,
            gamma_max: self.gamma,
            adaptive_gamma_on: self.startup.adaptive_gamma,
            skip_next_decode_append: false,
            max_ctx_len: self
                .window_size
                .unwrap_or(self.max_seq_len)
                .min(self.max_seq_len),
            ctx_slot_bytes,
            // Phase 2 Option B: lazily allocated on first propose when
            // Option B is on (the generic default; ATLAS_DFLASH_OPTION_B=0
            // restores the legacy contiguous path). None until then to
            // keep alloc_state
            // cheap for sequences that never use Option B.
            block_table_dev: None,
            ctx_count_drafter: 0,
            max_ctx_count_drafter: 0,
            ctx_committed: 0,
            ctx_positions: Vec::new(),
            // Propose lane: fixed for the seq lifetime (batch positions
            // reorder; captured graphs bake lane scratch pointers). Round-
            // robin keeps concurrent seqs spread across the lane streams.
            lane_id: self
                .next_lane
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                % self.lane_count(),
            lifecycle: None,
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: spark_runtime::gpu::DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        expected_owner: Option<SequenceGeneration>,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        draft_embed_target: Option<spark_runtime::gpu::DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<spark_runtime::gpu::DevicePtr>,
    ) -> Result<Vec<u32>> {
        self.propose_drafts(
            last_token,
            target_hidden,
            position,
            num_drafts,
            state,
            expected_owner,
            ctx,
            stream,
            draft_embed_target,
            grammar_bitmask,
            target_hidden_stack,
        )
    }

    fn propose_batch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[spark_runtime::gpu::DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn crate::speculative::ProposerState],
        expected_owners: Option<&[SequenceGeneration]>,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
        _out_conf: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let n = last_tokens.len();
        let expected_owners = expected_owners
            .ok_or_else(|| anyhow::anyhow!("DFlash batched propose requires expected owners"))?;
        // Preserve the historical n<2 fallback, but only after the complete
        // structural-length seam has run. No GPU work or state downcast occurs
        // before this check.
        batch_inputs::validate_batch_input_lengths(
            n,
            n,
            n,
            target_hiddens.len(),
            positions.len(),
            states.len(),
            expected_owners.len(),
        )?;
        let native_authoritative = self.startup.native_batch_authoritative;
        let generic_auth =
            self.startup.generic_batch_authoritative && !self.startup.diagnostics.batch_parity;
        // Generic DFlash keeps the historical n<2 fallback — also under the
        // authoritative batched mode (propose_batch_min=2 keeps the
        // scheduler off this entry for a lone pending sequence; a defensive
        // n==1 call still returns None so the caller keeps the serial path).
        // Lightning and explicit parity both enter the native B1 path.
        if n == 0 || (n == 1 && !(native_authoritative || self.startup.diagnostics.batch_parity)) {
            return Ok(None);
        }
        let consume_native =
            native_authoritative || self.startup.diagnostics.batch_parity || generic_auth;
        if !consume_native {
            // Generic DFlash never consumes the Lightning Bxgamma seam below:
            // its gamma == 4 contract rejects this head's row shape and the
            // staged native rows would be discarded anyway. Dispatch directly —
            // serial on the single lane, or the overlapped per-lane enqueue
            // when ATLAS_DFLASH_PROPOSE_LANES > 1.
            if self.lane_count() == 1 {
                let mut serial = Vec::with_capacity(n);
                for i in 0..n {
                    serial.push(self.propose_drafts(
                        last_tokens[i],
                        target_hiddens[i],
                        positions[i],
                        num_drafts,
                        states[i],
                        Some(expected_owners[i]),
                        ctx,
                        stream,
                        None,
                        None,
                        Some(target_hiddens[i]),
                    )?);
                }
                return Ok(Some(serial));
            }
            return self
                .propose_on_lanes(
                    last_tokens,
                    target_hiddens,
                    positions,
                    num_drafts,
                    states,
                    expected_owners,
                    ctx,
                )
                .map(Some);
        }
        if self.startup.diagnostics.batch_parity {
            tracing::info!("DFlash Bxgamma parity dispatch: batch={n}");
        }
        // Per-sequence propose γ: `propose_gamma` is the configured γ for
        // every sequence when `ATLAS_DFLASH_ADAPTIVE_GAMMA` is off, so the
        // group table below degenerates to the single-group shape today.
        let propose_gammas: Vec<usize> = states
            .iter()
            .map(|s| {
                s.as_any()
                    .downcast_ref::<DflashProposerState>()
                    .map(|d| d.propose_gamma)
                    .unwrap_or(self.gamma)
            })
            .collect();
        let (parity_oracle, parity_hidden_oracle) = if self.startup.diagnostics.batch_parity {
            let mut oracle = Vec::with_capacity(n);
            let mut hidden: Vec<Vec<u8>> = Vec::with_capacity(n);
            for i in 0..n {
                oracle.push(self.propose_drafts(
                    last_tokens[i],
                    target_hiddens[i],
                    positions[i],
                    num_drafts,
                    states[i],
                    Some(expected_owners[i]),
                    ctx,
                    stream,
                    None,
                    None,
                    Some(target_hiddens[i]),
                )?);
                ctx.gpu.synchronize(stream)?;
                let mut bytes = vec![0u8; propose_gammas[i] * self.hidden_size * 2];
                ctx.gpu.copy_d2h(self.scratch.stream_buf, &mut bytes)?;
                hidden.push(bytes);
            }
            (Some(oracle), Some(hidden))
        } else {
            (None, None)
        };
        if (native_authoritative || generic_auth) && parity_oracle.is_none() {
            // prepare_drafts_state advances the lifecycle and commits the
            // ctx-slot precompute; a sequence that was prepared must never
            // be re-prepared (double advance corrupts the drafter context),
            // so a mid-loop failure under generic authoritative falls back
            // to forward_prepared for the prepared prefix and serial
            // propose for the rest instead of erroring to the scheduler.
            let mut prepared = 0usize;
            for i in 0..n {
                if let Err(e) = self.prepare_drafts_state(
                    last_tokens[i],
                    target_hiddens[i],
                    positions[i],
                    num_drafts,
                    states[i],
                    expected_owners[i],
                    ctx,
                    stream,
                    target_hiddens[i],
                ) {
                    if !generic_auth {
                        return Err(e);
                    }
                    tracing::warn!(
                        "DFlash batched propose: prepare failed at sequence {i}/{n}; per-sequence fallback: {e:#}"
                    );
                    return self
                        .prepared_fallback_batch(
                            n,
                            prepared,
                            Some(i),
                            last_tokens,
                            target_hiddens,
                            positions,
                            num_drafts,
                            states,
                            expected_owners,
                            ctx,
                            stream,
                        )
                        .map(Some);
                }
                prepared = i + 1;
            }
        }
        // Same-γ grouping for the staged pass: mask-row count, attention
        // span, conv tail, and graph identity all key on γ, so a mixed-γ
        // batch cannot share one launch shape. At most two groups with the
        // shipped {8, 12} set; lever-off degenerates to a single group at
        // the configured γ — the exact pre-existing call.
        let groups: Vec<(usize, Vec<usize>)> = if self.startup.adaptive_gamma {
            adaptive_gamma::distinct_gammas(&propose_gammas)
                .into_iter()
                .map(|g| (g, (0..n).filter(|&i| propose_gammas[i] == g).collect()))
                .collect()
        } else {
            vec![(self.gamma, (0..n).collect())]
        };
        if groups.len() > 1 {
            let dist = groups
                .iter()
                .map(|(g, idxs)| format!("g{g}={}", idxs.len()))
                .collect::<Vec<_>>()
                .join(" ");
            tracing::info!("DFLASH BATCHED propose: {dist}");
        }
        let staged = (|| -> Result<Option<Vec<Vec<u32>>>> {
            if groups.len() == 1 {
                return self.propose_batch_staged_dispatch(
                    last_tokens,
                    target_hiddens,
                    positions,
                    num_drafts,
                    groups[0].0,
                    states,
                    expected_owners,
                    native_authoritative,
                    generic_auth,
                    parity_oracle,
                    parity_hidden_oracle.map(|h| h.concat()),
                    ctx,
                    stream,
                );
            }
            // Mixed γ: one staged pass per group on the γ_max-sized shared
            // scratch (a γ=8 group writes only its prefix), then reassemble
            // drafts in input order. `elems` unwraps each `&mut` once so a
            // group's disjoint index set can re-collect it without aliasing.
            let mut merged: Vec<Option<Vec<u32>>> =
                std::iter::repeat_with(|| None).take(n).collect();
            let mut elems: Vec<Option<&mut dyn crate::speculative::ProposerState>> =
                states.iter_mut().map(|st| Some(&mut **st)).collect();
            for (g, idxs) in &groups {
                let g_tokens: Vec<u32> = idxs.iter().map(|&i| last_tokens[i]).collect();
                let g_hiddens: Vec<spark_runtime::gpu::DevicePtr> =
                    idxs.iter().map(|&i| target_hiddens[i]).collect();
                let g_pos: Vec<usize> = idxs.iter().map(|&i| positions[i]).collect();
                let g_owners: Vec<SequenceGeneration> =
                    idxs.iter().map(|&i| expected_owners[i]).collect();
                let g_oracle: Option<Vec<Vec<u32>>> = parity_oracle
                    .as_ref()
                    .map(|o| idxs.iter().map(|&i| o[i].clone()).collect());
                let g_hidden: Option<Vec<u8>> = parity_hidden_oracle
                    .as_ref()
                    .map(|hs| idxs.iter().flat_map(|&i| hs[i].iter().copied()).collect());
                let mut g_states: Vec<&mut dyn crate::speculative::ProposerState> = idxs
                    .iter()
                    .map(|&i| {
                        elems[i]
                            .take()
                            .expect("disjoint γ groups share no sequence")
                    })
                    .collect();
                let out = self.propose_batch_staged_dispatch(
                    &g_tokens,
                    &g_hiddens,
                    &g_pos,
                    num_drafts,
                    *g,
                    &mut g_states,
                    &g_owners,
                    native_authoritative,
                    generic_auth,
                    g_oracle,
                    g_hidden,
                    ctx,
                    stream,
                )?;
                let out = out.unwrap_or_else(|| idxs.iter().map(|_| Vec::new()).collect());
                for (j, &i) in idxs.iter().enumerate() {
                    merged[i] = Some(out[j].clone());
                }
            }
            Ok(Some(
                merged.into_iter().map(|o| o.unwrap_or_default()).collect(),
            ))
        })();
        match staged {
            Err(e) if generic_auth => {
                // State for every sequence was already advanced — surface
                // the failure as a per-sequence forward fallback, never Err.
                tracing::warn!(
                    "DFlash batched propose failed after prepare; per-sequence fallback: {e:#}"
                );
                self.prepared_fallback_batch(
                    n,
                    n,
                    None,
                    last_tokens,
                    target_hiddens,
                    positions,
                    num_drafts,
                    states,
                    expected_owners,
                    ctx,
                    stream,
                )
                .map(Some)
            }
            other => other,
        }
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        expected_owner: Option<SequenceGeneration>,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let expected_owner = expected_owner
            .ok_or_else(|| anyhow::anyhow!("DFlash after_verify requires expected owner"))?;
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
        // Phase 1: no real KV trim because `propose()` is a stub. Phase 2
        // adds the rollback that drops `(last_num_drafted - num_accepted)`
        // tokens from each layer's paged cache.
        //
        // Phase I invariant: `ctx_committed` is the watermark of ctx slots
        // already precomputed into the paged cache. It is monotonic only as
        // long as `ctx_len` is monotonic (today it is — ctx is append-only
        // and never rewound here). IF a future rollback ever shrinks the
        // committed ctx (rewinds `ctx_len`), it MUST also reset
        // `dstate.ctx_committed = dstate.ctx_len` so the next propose
        // recomputes the rolled-back tail instead of reading stale K/V.
        // The `.min(ctx_len)` clamp in propose() is the defensive backstop.
        // The batched propose dispatcher reads this to select the
        // just-verified accepted row's 5-layer stack from the seq's
        // dflash_hidden_save region [i*kmax + acc). Row acc = the last
        // accepted position (0 = the greedy row when no draft matched).
        let lifecycle = dstate
            .lifecycle
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("DFlash after_verify: missing generation owner"))?;
        lifecycle.validate_access(expected_owner)?;
        let valid_rows = num_accepted
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("DFlash accepted-row count overflow"))?;
        lifecycle.advance(
            expected_owner,
            lifecycle.absolute_position(),
            valid_rows,
            lifecycle.row_stride_bytes(),
        )?;
        dstate.last_num_accepted = num_accepted;
        dstate.last_num_drafted = 0;
        // Adaptive γ (lever-gated; propose_gamma is otherwise pinned to the
        // configured γ and this block is a no-op read of the environment
        // resolved once at startup): feed the accepted-draft count into the
        // EMA and move the next propose's γ. `num_accepted` counts draft
        // tokens only — the bonus token is a verify output, not a draft.
        if self.startup.adaptive_gamma {
            let mut stats = self.adaptive_stats.lock();
            stats.record(dstate.propose_gamma, num_accepted, self.gamma);
            dstate.accept_ema = adaptive_gamma::update_ema(dstate.accept_ema, num_accepted);
            let (next, ema) = adaptive_gamma::next_gamma(
                dstate.propose_gamma,
                dstate.accept_ema,
                adaptive_gamma::ADAPTIVE_GAMMA_LO,
                self.gamma,
            );
            if next != dstate.propose_gamma {
                stats.switches += 1;
                dstate.adaptive_switches += 1;
                tracing::info!(
                    "DFlash adaptive γ: {} -> {} (ema={:.2} -> {:.2}, accepted={}, seq_switches={})",
                    dstate.propose_gamma,
                    next,
                    dstate.accept_ema,
                    ema,
                    num_accepted,
                    dstate.adaptive_switches,
                );
                dstate.propose_gamma = next;
                dstate.accept_ema = ema;
            }
            stats.log_periodic();
        }
        Ok(())
    }

    fn free_state(
        &self,
        gpu: &dyn GpuBackend,
        expected_owner: Option<SequenceGeneration>,
        state: &mut dyn ProposerState,
    ) -> Result<()> {
        // Phase 2 (Option B) reclaim: return the drafter's lazily-allocated
        // paged KV blocks to the pool on request completion. Without this the
        // ~257-block Option-B drafter cache (allocated in propose.rs when
        // block_table_dev.is_none()) is never freed, so the SECOND request to
        // a long-lived server starts with zero free drafter blocks and floods
        // "DFlash Option B: paged KV cache exhausted". Mirrors MtpHead::free_state.
        let dstate = match state.as_any_mut().downcast_mut::<DflashProposerState>() {
            Some(s) => s,
            // Phase 1 / non-DFlash proposer state: nothing allocated, nothing to free.
            None => return Ok(()),
        };
        // Transactional cleanup: owner validation runs FIRST, but a
        // validation failure still reclaims every resource below before
        // propagating — a mismatched owner must not leak graphs, KV blocks,
        // or the ctx accumulator. (Lifecycle gaps noted 2026-08-17.)
        let owner = match self.validate_dflash_owner_terminal(dstate, expected_owner) {
            Ok(owner) => owner,
            Err(error) => {
                // Transactional cleanup: reclaim everything reclaimable
                // before propagating the validation failure. Graphs are
                // owner-keyed and stay pooled (reclaimed on generation
                // turnover); blocks, accumulator, and the device block
                // table belong to THIS state and must not leak.
                dstate.reclaim_on_owner_failure(gpu, &self.kv_cache);
                return Err(error);
            }
        };
        dstate
            .lifecycle
            .as_mut()
            .expect("owner validated above")
            .retire(owner)?;
        let retired_graphs = {
            let mut graphs = self.propose_graphs.lock();
            lifecycle::take_owned_graphs(&mut graphs, owner)
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        };
        for graph in retired_graphs {
            if graph.0 != 0 {
                gpu.destroy_graph(graph)?;
            }
        }
        if !dstate.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&dstate.block_table);
            dstate.block_table.clear();
        }
        // Free the per-seq ctx accumulator — the dominant per-request
        // allocation (`max_seq_len × 5 × target_hidden` BF16; ~320 MB at
        // max_seq_len=16384). `DevicePtr` has no Drop, so without this every
        // finished sequence leaks it for the server's lifetime. Guarded on a
        // non-null pointer so a double free_state is a no-op. A failed free
        // is logged and the pointer RETAINED so a cleanup retry can release
        // it (a silently cleared pointer would leak unrecoverably).
        if dstate.ctx_hidden_acc.0 != 0 {
            if let Err(error) = gpu.free(dstate.ctx_hidden_acc) {
                tracing::error!(
                    "DSpark free_state: freeing ctx accumulator {:#x} failed ({error}); \
                     pointer retained for a later cleanup retry",
                    dstate.ctx_hidden_acc.0
                );
            } else {
                dstate.ctx_hidden_acc = DevicePtr(0);
            }
        }
        // Free the device-side block table (lazily allocated in propose.rs).
        // A failed free logs and RESTORES the handle so a later cleanup retry
        // can release it (propose gates re-alloc on block_table_dev.is_none(),
        // so a restored handle is retried, never re-allocated around).
        if let Some(bt) = dstate.block_table_dev.take()
            && let Err(error) = gpu.free(bt)
        {
            tracing::error!(
                "DSpark free_state: freeing device block table {:#x} failed ({error}); \
                 handle restored for a later cleanup retry",
                bt.0
            );
            dstate.block_table_dev = Some(bt);
        }
        // Reset the lazy-alloc guard + watermarks so the NEXT request's first
        // propose re-allocates fresh blocks and re-precomputes ctx from a clean
        // slate (propose.rs gates alloc on block_table_dev.is_none()).
        dstate.max_ctx_count_drafter = 0;
        dstate.ctx_count_drafter = 0;
        dstate.ctx_committed = 0;
        dstate.ctx_positions.clear();
        dstate.seq_len = 0;
        dstate.ctx_len = 0;
        dstate.prefill_done = false;
        dstate.last_num_drafted = 0;
        dstate.last_num_accepted = 0;
        dstate.skip_next_decode_append = false;
        Ok(())
    }
}
