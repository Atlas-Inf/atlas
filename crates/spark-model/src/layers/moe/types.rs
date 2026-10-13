// SPDX-License-Identifier: AGPL-3.0-only

//! `MoeLayer` struct definition; its behaviour lives in the sibling modules.

use super::*;

/// MoE feed-forward network component.
///
/// Not a `TransformerLayer` — used as a component inside layers
/// for the FFN/MoE block after post-attention norm.
#[allow(dead_code)]
pub struct MoeLayer {
    pub weights: MoeWeights,
    pub(super) btile_storage: gate_up_repack::Storage,
    /// Quant format of the ROUTED experts as landed in GPU memory. `Nvfp4`
    /// (default) = packed E2M1 + FP8-E4M3 per-16 block scales + f32 per-tensor
    /// global. Set to `Mxfp4E8m0` by the DeepSeek-V4 native-MXFP4 loader
    /// (transcode-free: E8M0 per-32 scales, no global) so the Phase-K E8M0
    /// GEMM variants dispatch on it instead of the NVFP4 kernels. Consumed at
    /// the grouped/decode GEMM call sites (assert via `WeightQuantFormat::expect`).
    // Written by the loader (Phase L); READ at the GEMM dispatch sites in Phase K.
    // Until Phase K wires the read, `deny(warnings)` would flag it never-read.
    #[allow(dead_code)]
    pub(crate) experts_scale_kind: crate::weight_map::WeightQuantFormat,
    /// Quant format of the SHARED expert (ARM-2 Phase-K RIDER A1). The native
    /// V4 ckpt is heterogeneous: routed experts `Mxfp4E8m0` but the shared
    /// expert is FP8→`Nvfp4`. Keyed off the weight tag (not `is_shared`
    /// positionality) so the dual-format decode kernel's `expect` net fires if
    /// a future ckpt ships a different shared format. Default `Nvfp4`.
    #[allow(dead_code)]
    pub(crate) shared_experts_scale_kind: crate::weight_map::WeightQuantFormat,
    // NVFP4-quantized gate weight (quarters bandwidth for routing)
    pub(super) gate_nvfp4: Option<QuantizedWeight>,
    /// Pre-expert norm: applied to input AFTER routing but BEFORE expert dispatch.
    /// Gemma-4 26B: router sees raw residual, experts see pre_feedforward_layernorm_2(residual).
    pub pre_expert_norm: Option<crate::weight_map::DenseWeight>,
    pub(super) pre_expert_norm_k: spark_runtime::gpu::KernelHandle,
    pub(super) dense_gemv: KernelHandle,
    /// `dense_gemv_bf16_batchm` or its load-ahead tier for the router logits
    /// (bit-identical per row to `dense_gemv`), or 0.
    pub(super) dense_gemv_batchm: KernelHandle,
    pub(super) w4a16_gemv: KernelHandle,
    /// Single-warp `w4a16_gemv_sw`. `KernelHandle(0)` on miss → base GEMV.
    pub(super) w4a16_gemv_sw: KernelHandle,
    pub(super) w4a16_gemm: KernelHandle,
    pub(super) dense_gemm: KernelHandle,
    /// Order-preserving register-blocked router GEMM (`dense_gemm_bf16_router`):
    /// bit-identical to the scalar `dense_gemm` (same per-output FP32 k-order,
    /// `--fmad=false` build) at ~2x speed. `KernelHandle(0)` on miss → the
    /// pinned scalar kernel. Used ONLY by `router_gate_gemm_dense`.
    pub(super) dense_gemm_router: KernelHandle,
    pub(super) router_prefill_bn32: KernelHandle,
    /// Exact-M=5, scalar-order router specialization for GLM verification.
    pub(super) dense_gemm_router_m5: KernelHandle,
    pub(super) dense_gemm_router_rows: KernelHandle,
    pub(super) dense_gemm_pipelined: KernelHandle,
    /// FP32-output router GEMM + FP32-input top-K for the ATLAS_FP32_GATE path.
    /// Zero (unresolved) when the kernels are absent; dispatch falls back to BF16.
    pub(super) dense_gemm_f32out: KernelHandle,
    /// FP32-in/FP32-out router GEMM for ATLAS_FP32_ROUTING (reads the FP32
    /// router_in from residual_add_rms_norm_gatef32). Zero if absent.
    pub(super) dense_gemm_f32in: KernelHandle,
    pub(super) moe_topk_f32: KernelHandle,
    pub(super) moe_expert_gate_up_shared: KernelHandle,
    pub(super) moe_expert_silu_down_shared: KernelHandle,
    pub(super) moe_topk: KernelHandle,
    pub(super) moe_weighted_sum_blend: KernelHandle,
    pub(super) residual_add: KernelHandle,
    pub(super) moe_topk_batched: KernelHandle,
    // K=2 fused MoE kernel handles
    pub(super) moe_expert_gate_up_shared_batch2: KernelHandle,
    pub(super) moe_expert_silu_down_shared_batch2: KernelHandle,
    pub(super) moe_weighted_sum_blend_batch2: KernelHandle,
    pub(super) w4a16_gemv_batch2: KernelHandle,
    // K=3 fused MoE kernel handles
    pub(super) moe_expert_gate_up_shared_batch3: KernelHandle,
    pub(super) moe_expert_silu_down_shared_batch3: KernelHandle,
    pub(super) moe_weighted_sum_blend_batch3: KernelHandle,
    pub(super) w4a16_gemv_batch3: KernelHandle,
    /// Exact-M GEMVs used to read GLM's retained shared expert once during
    /// K=4/K=5 verification (routed experts remain on fused K2/K3 kernels).
    pub(super) w4a16_batchm: W4a16BatchmTiers,
    /// Two-plane exact-M=5 shared-expert gate/up projection for GLM verify.
    pub(super) w4a16_batch5_dual_k: KernelHandle,
    // Generic token-major NVFP4 MoE kernels. Used as an opt-in decode
    // concurrency experiment for N>=4 without grouped-GEMM sorting.
    pub(super) moe_expert_gate_up_shared_token_major: KernelHandle,
    pub(super) moe_expert_silu_down_shared_token_major: KernelHandle,
    pub(super) moe_weighted_sum_blend_token_major: KernelHandle,
    pub(super) moe_decode_atomic_c4_silu_down_accum_k: KernelHandle,
    pub(super) moe_decode_atomic_c4_finalize_k: KernelHandle,
    // Sorted/grouped prefill path
    pub(super) moe_sort_by_expert: KernelHandle,
    pub(super) moe_sorted_gate_up: KernelHandle,
    pub(super) moe_sorted_silu_down: KernelHandle,
    pub(super) moe_grouped_gemm: KernelHandle,
    pub(super) moe_silu_mul: KernelHandle,
    /// Activation kernel for sorted/unfused path. SiLU by default, GeGLU for Gemma-4.
    pub(super) moe_act_mul: KernelHandle,
    /// When true, decode uses the sorted prefill path (avoids fused SiLU kernels).
    pub(super) gelu_activation: bool,
    pub(super) moe_unpermute_reduce: KernelHandle,
    pub(super) moe_unpermute_reduce_ep: KernelHandle,
    pub(super) moe_batched_blend: KernelHandle,
    /// Pointer tables for batched expert dispatch.
    pub(super) gate_ptrs: ExpertPtrTable,
    pub(super) up_ptrs: ExpertPtrTable,
    pub(super) down_ptrs: ExpertPtrTable,
    /// Transposed pointer tables for coalesced prefill GEMM.
    pub(super) gate_ptrs_t: Option<ExpertPtrTable>,
    pub(super) up_ptrs_t: Option<ExpertPtrTable>,
    pub(super) down_ptrs_t: Option<ExpertPtrTable>,
    /// CUTLASS grouped-NVFP4 host tables (`ATLAS_HOLO_MOE_GROUPED_CUTLASS`).
    /// Per-expert packed/SFB pointer values + scale2, snapshotted ONCE at load
    /// by `build_cutlass_grouped_sfb` (the SFB swizzle is built there from the
    /// `gate_ptrs_t`/`up_ptrs_t` `[K/16,N]` scales via `pack_weight_sfb`). The
    /// grouped C entry consumes these host-side, so the snapshot lives here —
    /// owned by the layer, dying with the model — rather than in any global
    /// cache keyed on device addresses, which a model swap's free/realloc
    /// would turn stale. `None` => the CUTLASS grouped path is unavailable.
    pub(super) cutlass_grouped_host: Option<ops::MoeCutlassHostTables>,
    /// Keeps the per-expert SFB buffers alive.
    pub(super) _cutlass_sfb_owned: Vec<DevicePtr>,
    /// Routed checkpoint scales were freed after the CUTLASS SFB swizzle; every
    /// routed-expert call must take the grouped CUTLASS path.
    pub(super) routed_scales_released: bool,
    /// Lazy down_proj transpose scratch — populated at the start of each
    /// prefill call when the persistent transpose pass couldn't fit
    /// down_proj. Decode keeps using `down_ptrs` (untransposed); prefill
    /// uses `down_ptrs_t` pointing into this scratch. Shared across all
    /// MoE layers (the same scratch is overwritten layer-by-layer during
    /// the sequential forward).
    ///
    /// `down_t_scratch_packed`: contiguous `[num_experts × N × K/2]` bytes.
    /// `down_t_scratch_scale`:  contiguous `[num_experts × N × K/16]` bytes.
    /// Both `None` when the persistent transpose pass already covered
    /// down (full-fits path) or when the layer doesn't need scratch
    /// transpose (FP8 experts, etc.).
    pub(super) down_t_scratch_packed: Option<DevicePtr>,
    pub(super) down_t_scratch_scale: Option<DevicePtr>,
    /// Kernel handle for the batched per-expert uint8 transpose.
    pub(super) moe_transpose_u8_batched_k: KernelHandle,
    // ── Phase 8a transposed-layout decode kernels (unified-layout MoE).
    // Loaded eagerly at construction. Currently NOT wired into the
    // dispatch — Phase 8a part 3/3 will route decode through these once
    // the weight loader produces transposed-only pointer tables.
    pub(super) moe_expert_gate_up_shared_t_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_t_k: KernelHandle,
    // ARM-2 Phase-K: native-MXFP4 (E8M0 routed / NVFP4 shared) dual-format
    // decode variants. KernelHandle(0) on models that don't ship them.
    pub(super) moe_expert_gate_up_shared_t_e8m0_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_t_e8m0_k: KernelHandle,
    // ── sqrtsoftplus routing (DeepSeek-V4) ──
    pub(super) moe_topk_sqrtsoftplus_k: KernelHandle,
    pub(super) moe_topk_sqrtsoftplus_batched_k: KernelHandle,
    // ── hash routing (DeepSeek-V4 first `num_hash_layers` MoE layers) ──
    pub(super) moe_hash_route_k: KernelHandle,
    pub(super) moe_hash_route_batched_k: KernelHandle,
    // ── LongCat softmax+bias routing with zero-computation experts ──
    /// Router logit width = num_experts + zero_expert_num. Equal to
    /// num_experts on every non-LongCat model (behavior-neutral).
    pub(crate) router_logits_n: u32,
    pub(super) moe_topk_softmax_bias_k: KernelHandle,
    pub(super) moe_topk_softmax_bias_batched_k: KernelHandle,
    pub(super) moe_zero_expert_add_k: KernelHandle,
    /// Per-token folded zero-expert weight (f32, written by the softmax+bias
    /// router kernels). Fixed-size allocation (16K tokens) — graph-safe.
    pub(super) zero_accum_dev: DevicePtr,
    /// Static `tid2eid` table [vocab_size, top_k] i64 — present ONLY for the
    /// hash-routed layers (the loader supplies it only for those). `Some`
    /// here is the SSOT that this layer routes via the static hash table
    /// instead of the learned gate's top-K.
    pub(super) tid2eid_dev: Option<DevicePtr>,
    pub(super) moe_expert_gate_up_shared_batch2_t_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_batch2_t_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_batch3_t_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_batch3_t_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_fp8_t_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_fp8_t_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_fp8_batch2_t_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_fp8_batch2_t_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_fp8_batch3_t_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_fp8_batch3_t_k: KernelHandle,
    /// `ATLAS_UNIFIED_MOE_LAYOUT=1` opts in to the unified-layout decode
    /// path: gate/up/down all use transposed `[K/2, N]` layout, decode
    /// dispatches to `moe_expert_*_shared_t` kernels. Default off — the
    /// dispatch falls through to the original `[N, K/2]` kernels.
    /// Resolved once at construction.
    pub(super) unified_layout: bool,
    /// Equal-size block_nvfp4 routed-weight replacement used by the grouped
    /// Blackwell MMQ prefill path and its decode-compatible GEMV kernels.
    pub(super) nvfp4_mmq_layout: bool,
    /// Slab allocations backing `gate_ptrs` / `up_ptrs` / `down_ptrs` while
    /// `nvfp4_mmq_layout` is active. Pointer tables do not own their targets.
    pub(super) _nvfp4_mmq_owned: Vec<DevicePtr>,
    /// `ATLAS_NVFP4_GATE_UP_M128=1` opts in to the M=128 fused gate+up
    /// kernel (Block D #3, Atlas tile-shape rewrite). Halves block count
    /// at large prefill — better SM amortization on GB10's 25-SM budget.
    /// Currently only minimax-m2-229b ships the kernel; other models keep
    /// `moe_fused_gate_up_t_k64_m128 == KernelHandle(0)` and dispatch
    /// falls through to the M=64 path even when the env var is set.
    pub(super) nvfp4_gate_up_m128: bool,
    /// `ATLAS_NVFP4_DOWN_M32=1` opts routed prefill down projection into a
    /// two-warp M=32 specialization. This targets sparse expert batches and
    /// is intentionally independent of gate/up while it is being measured.
    pub(super) nvfp4_down_m32: bool,
    /// Quantize routed activations once, then use native block-scaled FP4 MMA
    /// for gate/up/down without any persistent weight duplication.
    pub(super) nvfp4_prequant_moe: bool,
    /// Immutable, default-off independent C2 compact FFN experiment.
    pub(super) c2_compact_moe: bool,
    /// Vectorize NVFP4 activation/weight scale staging with cp.async.
    pub(super) nvfp4_vecscale: bool,
    /// Fuse DeepSeek/GLM SiLU·mul with activation NVFP4 quantization. The
    /// compact result is staged safely through down scratch before down GEMM.
    pub(super) nvfp4_fused_silu_quant: bool,
    /// `ATLAS_HOLO_MOE_GATEUP_FP4=1` opts the prefill fused gate_up onto the
    /// block-scaled FP4 kernel. Reads the SHARED FAST_MOE=full `gate_ptrs_t`/
    /// `up_ptrs_t` `[K/2,N]` tables (no extra MoE memory); dispatch also requires
    /// those tables present + the FP4 kernel handle != 0.
    pub(super) gateup_fp4: bool,
    /// `ATLAS_HOLO_MOE_DOWN_FP4=1` — same, for the prefill down projection over
    /// the shared `down_ptrs_t` table.
    pub(super) down_fp4: bool,
    /// `ATLAS_HYBRID_MOE_LAYOUT=1` opts in to the hybrid-layout path:
    /// keep BOTH original `[N, K/2]` weights (for decode + MTP verify) AND
    /// transposed `[K/2, N]` weights (for prefill). Doubles MoE-weight
    /// memory but recovers the ~15 % decode regression that pure unified
    /// layout suffers from. Resolved once at construction; mutually
    /// exclusive with `unified_layout` at the dispatch level (hybrid wins
    /// on decode paths since it preserves untransposed warp-reduction
    /// parallelism).
    pub(super) hybrid_layout: bool,
    /// Transposed shared expert weights for prefill.
    pub(super) shared_gate_t: Option<QuantizedWeight>,
    pub(super) shared_gate_up_receipt: Option<helpers_a::SharedGateUpReceipt>,
    pub(super) shared_up_t: Option<QuantizedWeight>,
    pub(super) shared_down_t: Option<QuantizedWeight>,
    pub(super) moe_grouped_gemm_t: KernelHandle,
    pub(super) moe_grouped_gemm_t_k64: KernelHandle,
    /// Optional M=32 NVFP4 twin of `moe_grouped_gemm_t_k64`.
    pub(super) moe_grouped_gemm_t_k64_m32: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64_vecscale: KernelHandle,
    /// Same prequant FP4 grouped GEMM with K128 stages and ldmatrix-fed MMAs
    /// (bitwise-identical outputs); `ATLAS_MOE_PREQUANT_K128=1`, else null.
    pub(super) moe_w4a4_prequant_t_k128: KernelHandle,
    /// 64x256-tile twin of `moe_w4a4_prequant_t_k128` launched over only the
    /// local experts' row tiles (`moe_mtile_prefix_k`), bitwise-identical
    /// outputs; null unless the K128 kernel is on and GLM
    /// (`ATLAS_MOE_PREQUANT_K128W=0` disables); with its `_persist` twin
    /// under `ATLAS_GLM_MOE_PREFILL_PERSIST=1`.
    pub(super) moe_w4a4_prequant_t_k128w: ops::K128wKernel,
    pub(super) moe_mtile_prefix_k: KernelHandle,
    /// K128W gate and up in one launch with `silu_mul_quant_nvfp4` applied in
    /// its epilogue (same bytes); loaded with `moe_w4a4_prequant_t_k128w`
    /// unless `ATLAS_MOE_GATE_UP_SILU=0`.
    pub(super) moe_w4a4_prequant_gate_up_silu: ops::K128wKernel,
    /// CTAs of the persistent K128W twins when `ATLAS_GLM_MOE_PREFILL_PERSIST=1`
    /// resolved every needed one above, else 0 (row-tile grid only).
    pub(super) k128w_persist_ctas: u32,
    /// Compact-worklist twins of the prequantized native-FP4 MoE kernel.
    /// Optional and used only for guarded K=5 gate/up verification.
    pub(super) moe_w4a4_prequant_t_k64_compact: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64_vecscale_compact: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64_compact_gate_up: KernelHandle,
    pub(super) moe_w4a4_prequant_t_k64_vecscale_compact_gate_up: KernelHandle,
    pub(super) m16_gate_up: gate_up_m16::M16GateUp,
    /// Verify-decode tile selection: the M16 twins of the K128W kernels
    /// (`ATLAS_GLM_MOE_DECODE_M16=1`, else null) and the K128W control
    /// (`ATLAS_GLM_MOE_DECODE_K128W=1`).
    pub(super) decode_m16: decode_m16::DecodeM16,
    pub(super) m5_projections: m5_projections::M5Projections,
    pub(super) moe_nvfp4_mmq_gate_up_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_down_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_quantize_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_repack_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_silu_scale2_k: KernelHandle,
    pub(super) moe_nvfp4_mmq_scale2_rows_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_mmq_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_mmq_k: KernelHandle,
    pub(super) quantize_nvfp4_k: KernelHandle,
    pub(super) silu_mul_quant_nvfp4_k: KernelHandle,
    pub(super) moe_fused_gate_up_t: KernelHandle,
    pub(super) moe_fused_gate_up_t_k64: KernelHandle,
    // ARM-2 Phase-K: native-MXFP4 (E8M0 per-32) prefill variants of the W4A16
    // routed-expert GEMMs. KernelHandle(0) on models that don't ship them
    // (only the deepseek-v4-flash target compiles the `_e8m0` entries).
    pub(super) moe_grouped_gemm_e8m0: KernelHandle,
    pub(super) moe_grouped_gemm_t_e8m0: KernelHandle,
    pub(super) moe_grouped_gemm_t_k64_e8m0: KernelHandle,
    pub(super) moe_fused_gate_up_t_e8m0: KernelHandle,
    pub(super) moe_fused_gate_up_t_k64_e8m0: KernelHandle,
    /// M=128 variant of the K64 fused gate+up kernel (Block D #3, Atlas
    /// tile-shape rewrite). Loaded with `try_kernel` — falls back to
    /// `KernelHandle(0)` on models that don't ship the kernel; dispatch
    /// gates on `nvfp4_gate_up_m128` AND handle non-zero.
    pub(super) moe_fused_gate_up_t_k64_m128: KernelHandle,
    /// FUSED FP4 (block-scaled e2m1) variant of the K64 fused gate+up kernel
    /// (`ATLAS_HOLO_MOE_GATEUP_FP4`). Same signature as `moe_fused_gate_up_t_k64`
    /// but runs one `mma.sync.kind::mxf4nvf4.scale_vec::4X.m16n8k64` per k64
    /// tile (vs 2× m16n8k32 e4m3). `try_kernel` — `KernelHandle(0)` on images
    /// lacking it; the dispatch in `forward_prefill_routed` only fires when this
    /// handle != 0, `gateup_fp4` is set, and the shared `gate_ptrs_t`/`up_ptrs_t`
    /// tables are present (FAST_MOE=full).
    pub(super) moe_fused_gate_up_t_k64_fp4: KernelHandle,
    pub(super) moe_fp8_grouped_gemm_t: KernelHandle,
    pub(super) w4a16_gemm_t: KernelHandle,
    pub(super) bf16_to_fp8_k: KernelHandle,
    /// Pre-dequanted FP8 weights for zero-overhead prefill GEMMs.
    pub(super) gate_fp8: Option<DevicePtr>,
    pub(super) shared_gate_fp8: Option<DevicePtr>,
    pub(super) shared_fp8_cache: shared_fp8_cache::SharedFp8CacheState,
    pub(super) shared_fp8_origins: Option<[shared_fp8_origin::SharedFp8Origin; 3]>,
    pub(super) shared_up_fp8: Option<DevicePtr>,
    pub(super) shared_down_fp8: Option<DevicePtr>,
    pub(super) fp8_gemm_k: KernelHandle,
    /// Secondary CUDA stream for overlapping shared expert with routed experts.
    pub(super) prefill_stream: u64,
    /// Event pair for stream synchronization (input_ready, shared_done).
    pub(super) event_a: u64,
    pub(super) event_b: u64,
    // ── Sigmoid + correction-bias routing (DeepSeek-V3 / MiniMax-M2 style) ──
    /// Device pointer to `[num_experts]` correction bias. Populated from
    /// `MoeWeights.correction_bias` in `new()` when the loader sets it.
    /// `None` = Atlas's default softmax path. When `Some`, every top-k
    /// dispatch site branches to `moe_topk_sigmoid` with this bias arg.
    pub(super) correction_bias_dev: Option<DevicePtr>,
    /// Handle to `moe_topk_sigmoid` kernel. Lazy-loaded in `new()` even
    /// when bias is `None` (harmless if kernel isn't used).
    pub(super) moe_topk_sigmoid_k: KernelHandle,
    /// Batched variant for prefill / MTP-verify (one block per token).
    /// Loaded via `try_kernel` — returns KernelHandle(0) on models whose
    /// KERNEL.toml doesn't register the sigmoid kernels (e.g. Mistral).
    /// Never dispatched on those paths because `correction_bias_dev` is
    /// `None` there.
    pub(super) moe_topk_sigmoid_batched_k: KernelHandle,
    // FP8 fused MoE kernels (used when experts are FP8)
    pub(super) moe_expert_gate_up_shared_fp8: KernelHandle,
    pub(super) moe_expert_silu_down_shared_fp8: KernelHandle,
    // FP8 batch2/3 fused MoE kernels (for MTP K=2/K=3 verify)
    pub(super) moe_expert_gate_up_shared_fp8_batch2: KernelHandle,
    pub(super) moe_expert_silu_down_shared_fp8_batch2: KernelHandle,
    pub(super) moe_weighted_sum_blend_fp8_batch2: KernelHandle,
    pub(super) moe_expert_gate_up_shared_fp8_batch3: KernelHandle,
    pub(super) moe_expert_silu_down_shared_fp8_batch3: KernelHandle,
    pub(super) moe_weighted_sum_blend_fp8_batch3: KernelHandle,
    // THE routed-expert FP8 grouped GEMM for sorted MoE prefill: grid-compaction
    // (persistent 96-CTA grid over a COMPACTED (expert, m_tile, n_tile) work-list
    // built by `moe_build_tile_worklist`). Handle may be 0 on images that don't
    // ship the kernel.
    pub(super) moe_fp8_grouped_gemm_k: KernelHandle,
    // Builds the grouped-GEMM work-list (moe_build_tile_worklist, module "moe").
    // Launched on the SAME stream as the grouped GEMM (read-after-write of
    // total_tiles). Handle may be 0 on older images.
    pub(super) moe_build_tile_worklist_k: KernelHandle,
    // W8A8 + FP32 epilogue MoE GEMM (vLLM-equivalent). Opt-in via
    // ATLAS_FP8_W8A8=1. Requires per-token-quanted A_fp8 + a_scale.
    pub(super) moe_w8a8_grouped_gemm_k: KernelHandle,
    // PM4-geometry W8A8 grouped GEMM over the compacted work-list (kernel
    // `moe_w8a8_grouped_gemm_pm4`, same module). Bit-identical numerics to
    // the dense kernel; preferred when present (gb10). Handle may be 0 on
    // targets/images that don't ship it — dispatch falls back to the dense
    // 3D-grid `moe_w8a8_grouped_gemm_k`.
    pub(super) moe_w8a8_grouped_gemm_pm4_k: KernelHandle,
    pub(super) per_token_group_quant_fp8_k: KernelHandle,
    /// Fused SiLU·mul + per-token-group FP8 quant (bit-identical replacement
    /// for the `silu_mul` → `per_token_group_quant_fp8` pair on the W8A8
    /// prefill down-path). Optional: handle 0 (e.g. a model shadowing
    /// moe_silu_mul.cu without this entry point) falls back to the pair.
    pub(super) silu_mul_quant_fp8_k: KernelHandle,
    // Dense W8A8 (same kernel used by attention QKV/O proj) for shared-expert path.
    pub(super) fp8_gemm_t_blockscaled_k: KernelHandle,
    // BF16 grouped GEMM — for FP8-source models dequanted to BF16 at load.
    // Activates the high-precision MoE path that closes the per-layer
    // 0.989 FP8 cosine ceiling. Handle may be 0 on images that don't ship
    // the kernel; dispatch site is gated on Some(bf16_*_weight_ptrs).
    pub(super) moe_bf16_grouped_gemm_k: KernelHandle,
    // Fused BF16 decode kernels (mirror moe_expert_*_shared_fp8 layout).
    pub(super) moe_expert_gate_up_shared_bf16_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_bf16_k: KernelHandle,
    // Fused BF16 K=2 batch kernels for MTP verify (mirror the FP8 batch2 layout).
    // Handle may be 0 on images that don't ship the kernel; the K=2 BF16
    // dispatch site is gated on this being non-null and falls back to the
    // per-token batched path otherwise.
    pub(super) moe_expert_gate_up_shared_bf16_batch2_k: KernelHandle,
    pub(super) moe_expert_silu_down_shared_bf16_batch2_k: KernelHandle,
    pub(super) w8a16_gemm_k: KernelHandle, // for shared expert FP8 prefill
    pub(super) w8a16_gemm_pipelined_k: KernelHandle, // ATLAS_W8A16_PIPELINED shared-expert variant
    // Fused gate GEMV + topK softmax (saves 1 kernel launch per layer)
    pub(super) moe_gate_topk_fused_k: KernelHandle,
    // FP8 expert pointer tables (None when experts are NVFP4)
    pub(super) fp8_gate_weight_ptrs: Option<Fp8ExpertPtrTable>,
    pub(super) fp8_up_weight_ptrs: Option<Fp8ExpertPtrTable>,
    pub(super) fp8_down_weight_ptrs: Option<Fp8ExpertPtrTable>,
    // BF16 expert pointer tables — populated by the FP8-dequant-on-load
    // path. When Some, the routed-expert dispatch in `forward_prefill_fp8`
    // routes through `moe_bf16_grouped_gemm` instead of the FP8 grouped
    // GEMM, eliminating the per-layer FP8 quantization ceiling.
    pub(super) bf16_gate_weight_ptrs: Option<DevicePtr>,
    pub(super) bf16_up_weight_ptrs: Option<DevicePtr>,
    pub(super) bf16_down_weight_ptrs: Option<DevicePtr>,
    // Checkpoint-native BF16 shared expert. Independent of routed-expert
    // precision so mixed NVFP4-routed/BF16-shared checkpoints stay faithful.
    pub(super) bf16_shared_expert: Option<Bf16SharedExpert>,
    // FP8 shared expert weights (None when shared expert is NVFP4)
    pub(super) fp8_shared_expert: Option<Fp8ExpertWeight>,
    /// FP4 down kernel handle (`moe_w4a16_down_t_k64_fp4`). `try_kernel` =>
    /// `KernelHandle(0)` on images lacking it; the FP4-down dispatch checks this
    /// handle != 0, `down_fp4` is set, and the shared `down_ptrs_t` table is present.
    pub(crate) moe_down_t_k64_fp4: KernelHandle,

    /// Non-zero when this model's `moe_w4a16_{grouped_gemm_ptrtable,fused_gate_up}
    /// _t_k64` kernels stride `blockIdx.y` over m-tiles instead of returning past
    /// the first. Only then may the prefill grid be sized by the AVERAGE expert
    /// (`ATLAS_MOE_PREFILL_PERSIST_TILES`) rather than the hottest one.
    pub(crate) moe_k64_strides_m_tiles: KernelHandle,
    /// `moe_permute_tokens` gather kernel — only needed by the FP4 escape-hatch
    /// (which consumes expert-sorted contiguous rows, unlike the FP8 fused
    /// kernel that gathers via `sorted_token_ids` internally). `try_kernel`
    /// (handle may be 0 on images lacking it). Now unused — the CUTLASS grouped
    /// path fuses the gather into its A-pack — kept for potential reuse.
    #[allow(dead_code)]
    pub(crate) moe_permute_tokens_k: KernelHandle,
    // Phase 2.7 Tier C — Frankenstein dispatch flag.
    // True when this layer's index is in `config.dflash_capture_layers`.
    // When the env var `ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL=1` is set,
    // `forward()` (single-token decode) will route through `forward_prefill`
    // (tensor-core grouped GEMM kernel) on this layer only, so the captured
    // hidden states use a different numerical recipe than the scalar GEMV
    // path. Used to test whether the kernel choice is the dominant cause
    // of low DFlash drafter acceptance on FP4/FP8 targets.
    pub is_dflash_capture_layer: bool,
    /// Feature-1 (MoE expert + router LoRA): this layer's installed router +
    /// routed-expert deltas + apply scratch. `None` = no adapter / feature off
    /// → the base MoE path is byte-identical. Set by
    /// [`MoeLayer::set_lora_weights`] (`moe/lora.rs`); applied in the prefill
    /// forward. Decode/verify paths are a phase-1 followup (they REFUSE rather
    /// than silently drop the delta — see `reject_decode_lora`).
    pub(crate) lora: Option<MoeLoraWeights>,
}
