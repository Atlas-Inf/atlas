// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::new constructor.

use super::*;

impl MoeLayer {
    pub fn new(
        weights: MoeWeights,
        num_experts: usize,
        gate_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        Self::new_with_hash(weights, num_experts, gate_nvfp4, None, gpu, config)
    }

    /// Like [`MoeLayer::new`] but with an optional DeepSeek-V4 hash-routing
    /// `tid2eid` table ([vocab_size, top_k] i64). `Some` marks this as a
    /// hash-routed layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_hash(
        weights: MoeWeights,
        num_experts: usize,
        gate_nvfp4: Option<QuantizedWeight>,
        tid2eid_dev: Option<DevicePtr>,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        // Sanity-check the routing config: top-k that exceeds the
        // expert count would index OOB in the topk kernel and produce
        // silent NaN routing. Catch the misconfiguration at load time.
        let c2_compact_moe = super::forward_c2::c2_compact_requested()?;
        anyhow::ensure!(
            config.num_experts_per_tok <= num_experts && num_experts > 0,
            "MoE config invalid: num_experts_per_tok={} must be in 1..={}",
            config.num_experts_per_tok,
            num_experts,
        );
        // The check above bounds top-k by the expert count, which is the OOB
        // the topk kernel can READ. It says nothing about the OOB the kernel
        // can WRITE: the sigmoid routing kernels stage their top-K in a
        // fixed-size shared array, and `top_k` was passed to them unbounded.
        anyhow::ensure!(
            config.num_experts_per_tok <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K
                && num_experts <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
            "MoE config exceeds the routing kernels' fixed shared-memory bounds: \
             num_experts_per_tok={} (max {}), num_experts={} (max {}). Raise \
             MAX_TOP_K / MAX_EXPERTS in kernels/gb10/common/moe_topk_sigmoid.cu \
             and their mirrors in layers::ops together.",
            config.num_experts_per_tok,
            crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K,
            num_experts,
            crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
        );
        let m16_gate_up = super::gate_up_m16::M16GateUp::new(gpu, config)?;
        let m5_projections = super::m5_projections::M5Projections::new(gpu, config)?;
        let gate_ptrs = build_ptr_table(&weights.experts, |e| &e.gate_proj, gpu)?;
        let up_ptrs = build_ptr_table(&weights.experts, |e| &e.up_proj, gpu)?;
        let down_ptrs = build_ptr_table(&weights.experts, |e| &e.down_proj, gpu)?;

        // Extract the optional correction-bias device pointer before the
        // struct literal below moves `weights`. `.map(|dw| dw.weight)` turns
        // an `Option<DenseWeight>` into an `Option<DevicePtr>` for the
        // `moe_topk_sigmoid` kernel's bias arg.
        let weights_correction_bias: Option<DevicePtr> =
            weights.correction_bias.map(|dw| dw.weight);

        let _ = num_experts;
        let optional = super::init_optional::OptionalKernels::resolve(gpu, config);
        let rms_norm_k = gpu.kernel("norm", "rms_norm")?;
        let mut layer = Self {
            weights,
            btile_storage: super::gate_up_repack::Storage::Legacy,
            // Default: standard NVFP4 (FP8-E4M3 per-16 + f32 global). The
            // DeepSeek-V4 native-MXFP4 loader overrides this to `Mxfp4E8m0`
            // after construction (see deepseek_v4/assemble.rs).
            experts_scale_kind: crate::weight_map::WeightQuantFormat::Nvfp4,
            shared_experts_scale_kind: crate::weight_map::WeightQuantFormat::Nvfp4,
            gate_nvfp4,
            pre_expert_norm: None,
            pre_expert_norm_k: rms_norm_k,
            dense_gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemv_batchm: super::init_optional::router_gemv_batchm(gpu, config),
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw: super::super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            dense_gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_router: super::super::try_kernel(gpu, "gemm", "dense_gemm_bf16_router"),
            router_prefill_bn32: router_prefill_bn32::resolve(config, gpu)?,
            dense_gemm_router_m5: super::super::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_router_m5",
            ),
            dense_gemm_router_rows: super::super::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_router_rows",
            ),
            dense_gemm_pipelined: super::super::try_kernel(
                gpu,
                "gemm",
                "dense_gemm_bf16_pipelined",
            ),
            // FP32 gate path (ATLAS_FP32_GATE) — optional; KernelHandle(0) if the
            // target's kernel set predates these symbols, dispatch then stays BF16.
            dense_gemm_f32out: super::super::try_kernel(gpu, "gemm", "dense_gemm_bf16_f32out"),
            dense_gemm_f32in: super::super::try_kernel(gpu, "gemm", "dense_gemm_f32in_f32out"),
            moe_topk_f32: super::super::try_kernel(gpu, "moe_topk", "moe_topk_softmax_f32"),
            moe_expert_gate_up_shared: gpu
                .kernel("moe_shared_expert_fused", "moe_expert_gate_up_shared")?,
            moe_expert_silu_down_shared: gpu
                .kernel("moe_shared_expert_fused", "moe_expert_silu_down_shared")?,
            moe_topk: gpu.kernel("moe_topk", "moe_topk_softmax")?,
            moe_weighted_sum_blend: gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            moe_topk_batched: gpu.kernel("moe_topk", "moe_topk_softmax_batched")?,
            moe_expert_gate_up_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_gate_up_shared_batch2")?,
            moe_expert_silu_down_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_silu_down_shared_batch2")?,
            moe_weighted_sum_blend_batch2: gpu
                .kernel("moe_fused_batch2", "moe_weighted_sum_blend_batch2")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            moe_expert_gate_up_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_gate_up_shared_batch3")?,
            moe_expert_silu_down_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_silu_down_shared_batch3")?,
            moe_weighted_sum_blend_batch3: gpu
                .kernel("moe_fused_batch3", "moe_weighted_sum_blend_batch3")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_batchm: W4a16BatchmTiers::resolve(gpu),
            w4a16_batch5_dual_k: super::super::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch5_dual",
            ),
            moe_expert_gate_up_shared_token_major: gpu
                .kernel("moe_prefill", "moe_expert_gate_up_shared_prefill")?,
            moe_expert_silu_down_shared_token_major: gpu
                .kernel("moe_prefill", "moe_expert_silu_down_shared_prefill")?,
            moe_weighted_sum_blend_token_major: gpu
                .kernel("moe_prefill", "moe_weighted_sum_blend_prefill")?,
            moe_decode_atomic_c4_silu_down_accum_k: super::super::try_kernel(
                gpu,
                "moe_decode_atomic_c4",
                "moe_decode_atomic_c4_silu_down_accum",
            ),
            moe_decode_atomic_c4_finalize_k: super::super::try_kernel(
                gpu,
                "moe_decode_atomic_c4",
                "moe_decode_atomic_c4_finalize",
            ),
            moe_sort_by_expert: ops::glm_decode_fuse::twin(
                gpu,
                ops::glm_decode_fuse::MOE_SORT,
                gpu.kernel("moe", "moe_sort_by_expert")?,
                ("moe", "moe_sort_by_expert_scan"),
            )?,
            moe_sorted_gate_up: gpu.kernel("moe_sorted", "moe_sorted_gate_up")?,
            moe_sorted_silu_down: gpu.kernel("moe_sorted", "moe_sorted_silu_down")?,
            moe_grouped_gemm: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")?,
            moe_grouped_gemm_t: gpu.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_t")?,
            moe_grouped_gemm_t_k64: gpu
                .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_t_k64")?,
            moe_grouped_gemm_t_k64_m32: optional.moe_grouped_gemm_t_k64_m32,
            moe_w4a4_prequant_t_k64: optional.moe_w4a4_prequant_t_k64,
            moe_w4a4_prequant_t_k64_vecscale: optional.moe_w4a4_prequant_t_k64_vecscale,
            moe_w4a4_prequant_t_k128: optional.moe_w4a4_prequant_t_k128,
            moe_w4a4_prequant_t_k128w: optional.moe_w4a4_prequant_t_k128w,
            moe_w4a4_prequant_gate_up_silu: optional.moe_w4a4_prequant_gate_up_silu,
            k128w_persist_ctas: optional.k128w_persist_ctas,
            moe_mtile_prefix_k: optional.moe_mtile_prefix_k,
            moe_w4a4_prequant_t_k64_compact: optional.moe_w4a4_prequant_t_k64_compact,
            moe_w4a4_prequant_t_k64_vecscale_compact: optional
                .moe_w4a4_prequant_t_k64_vecscale_compact,
            m16_gate_up,
            decode_m16: super::decode_m16::DecodeM16::new(gpu, config)?,
            m5_projections,
            moe_w4a4_prequant_t_k64_compact_gate_up: optional
                .moe_w4a4_prequant_t_k64_compact_gate_up,
            moe_w4a4_prequant_t_k64_vecscale_compact_gate_up: optional
                .moe_w4a4_prequant_t_k64_vecscale_compact_gate_up,
            moe_nvfp4_mmq_gate_up_k: optional.moe_nvfp4_mmq_gate_up_k,
            moe_nvfp4_mmq_down_k: optional.moe_nvfp4_mmq_down_k,
            moe_nvfp4_mmq_quantize_k: optional.moe_nvfp4_mmq_quantize_k,
            moe_nvfp4_mmq_repack_k: optional.moe_nvfp4_mmq_repack_k,
            moe_nvfp4_mmq_silu_scale2_k: optional.moe_nvfp4_mmq_silu_scale2_k,
            moe_nvfp4_mmq_scale2_rows_k: optional.moe_nvfp4_mmq_scale2_rows_k,
            moe_expert_gate_up_shared_mmq_k: optional.moe_expert_gate_up_shared_mmq_k,
            moe_expert_silu_down_shared_mmq_k: optional.moe_expert_silu_down_shared_mmq_k,
            quantize_nvfp4_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
            silu_mul_quant_nvfp4_k: optional.silu_mul_quant_nvfp4_k,
            moe_fused_gate_up_t: gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t")?,
            moe_fused_gate_up_t_k64: gpu.kernel("moe_w4a16", "moe_w4a16_fused_gate_up_t_k64")?,
            moe_grouped_gemm_e8m0: optional.moe_grouped_gemm_e8m0,
            moe_grouped_gemm_t_e8m0: optional.moe_grouped_gemm_t_e8m0,
            moe_grouped_gemm_t_k64_e8m0: optional.moe_grouped_gemm_t_k64_e8m0,
            moe_fused_gate_up_t_e8m0: optional.moe_fused_gate_up_t_e8m0,
            moe_fused_gate_up_t_k64_e8m0: optional.moe_fused_gate_up_t_k64_e8m0,
            moe_fused_gate_up_t_k64_m128: optional.moe_fused_gate_up_t_k64_m128,
            moe_fused_gate_up_t_k64_fp4: optional.moe_fused_gate_up_t_k64_fp4,
            moe_fp8_grouped_gemm_t: gpu.kernel("moe_w4a16", "moe_fp8_grouped_gemm_ptrtable_t")?,
            moe_fp8_grouped_gemm_k: optional.moe_fp8_grouped_gemm_k,
            moe_build_tile_worklist_k: ops::glm_decode_fuse::twin(
                gpu,
                ops::glm_decode_fuse::MOE_SORT,
                optional.moe_build_tile_worklist_k,
                ("moe", "moe_build_tile_worklist_scan"),
            )?,
            moe_w8a8_grouped_gemm_k: optional.moe_w8a8_grouped_gemm_k,
            moe_w8a8_grouped_gemm_pm4_k: optional.moe_w8a8_grouped_gemm_pm4_k,
            per_token_group_quant_fp8_k: optional.per_token_group_quant_fp8_k,
            silu_mul_quant_fp8_k: optional.silu_mul_quant_fp8_k,
            fp8_gemm_t_blockscaled_k: optional.fp8_gemm_t_blockscaled_k,
            moe_bf16_grouped_gemm_k: optional.moe_bf16_grouped_gemm_k,
            moe_expert_gate_up_shared_bf16_k: optional.moe_expert_gate_up_shared_bf16_k,
            moe_expert_silu_down_shared_bf16_k: optional.moe_expert_silu_down_shared_bf16_k,
            moe_expert_gate_up_shared_bf16_batch2_k: optional
                .moe_expert_gate_up_shared_bf16_batch2_k,
            moe_expert_silu_down_shared_bf16_batch2_k: optional
                .moe_expert_silu_down_shared_bf16_batch2_k,
            w8a16_gemm_k: optional.w8a16_gemm_k,
            w8a16_gemm_pipelined_k: optional.w8a16_gemm_pipelined_k,
            moe_gate_topk_fused_k: optional.moe_gate_topk_fused_k,
            w4a16_gemm_t: gpu.kernel("w4a16", "w4a16_gemm_t")?,
            bf16_to_fp8_k: gpu.kernel("w4a16", "bf16_to_fp8")?,
            fp8_gemm_k: gpu.kernel("w4a16", "fp8_gemm_t")?,
            moe_silu_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            moe_act_mul: gpu.kernel("moe_silu_mul", "moe_silu_mul")?, // default: SiLU
            gelu_activation: false,
            moe_unpermute_reduce: gpu.kernel("moe", "moe_unpermute_reduce_indexed")?,
            moe_unpermute_reduce_ep: super::ep_prefill::unpermute_ep_kernel(gpu, config)?,
            moe_batched_blend: gpu.kernel("moe", "moe_batched_blend")?,
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            gate_ptrs_t: None,
            up_ptrs_t: None,
            down_ptrs_t: None,
            cutlass_grouped_host: None,
            _cutlass_sfb_owned: Vec::new(),
            routed_scales_released: false,
            down_t_scratch_packed: None,
            down_t_scratch_scale: None,
            moe_transpose_u8_batched_k: gpu
                .kernel("moe_transpose_batched", "moe_transpose_u8_batched")?,
            // ── Phase 8a transposed-layout decode kernels ──
            // Module name = file stem (default convention in atlas-kernels).
            moe_expert_gate_up_shared_t_k: gpu
                .kernel("moe_shared_expert_fused_t", "moe_expert_gate_up_shared_t")?,
            moe_expert_silu_down_shared_t_k: gpu
                .kernel("moe_shared_expert_fused_t", "moe_expert_silu_down_shared_t")?,
            // ARM-2 Phase-K dual-format decode variants (E8M0 routed / NVFP4
            // shared). try_kernel — the entries are in the common .cu but load
            // by name; 0 where a target doesn't compile that module.
            moe_expert_gate_up_shared_t_e8m0_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_gate_up_shared_t_e8m0",
            ),
            moe_expert_silu_down_shared_t_e8m0_k: super::super::try_kernel(
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_silu_down_shared_t_e8m0",
            ),
            // sqrtsoftplus kernels: lazy-loaded via try_kernel so models that
            // don't register them (all except DeepSeek-V4) start fine.
            moe_topk_sqrtsoftplus_k: super::super::try_kernel(
                gpu,
                "moe_topk_sqrt",
                "moe_topk_sqrtsoftplus",
            ),
            moe_topk_sqrtsoftplus_batched_k: super::super::try_kernel(
                gpu,
                "moe_topk_sqrt",
                "moe_topk_sqrtsoftplus_batched",
            ),
            // Hash routing (DeepSeek-V4 hash_moe layers): lazy-loaded so other
            // models start fine. `tid2eid_dev` is the per-layer table (Some
            // only for hash layers).
            router_logits_n: (config.num_experts + config.zero_expert_num) as u32,
            moe_topk_softmax_bias_k: super::super::try_kernel(
                gpu,
                "moe_topk_softmax_bias",
                "moe_topk_softmax_bias",
            ),
            moe_topk_softmax_bias_batched_k: super::super::try_kernel(
                gpu,
                "moe_topk_softmax_bias",
                "moe_topk_softmax_bias_batched",
            ),
            moe_zero_expert_add_k: super::super::try_kernel(
                gpu,
                "moe_topk_softmax_bias",
                "moe_zero_expert_add",
            ),
            // 64 KB, unconditional: written by the softmax+bias router even
            // with zero_expert_num == 0 (always zeros then).
            zero_accum_dev: gpu.alloc(16384 * 4)?,
            moe_hash_route_k: super::super::try_kernel(gpu, "moe_hash_route", "moe_hash_route"),
            moe_hash_route_batched_k: super::super::try_kernel(
                gpu,
                "moe_hash_route",
                "moe_hash_route_batched",
            ),
            tid2eid_dev,
            moe_expert_gate_up_shared_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch2_t",
                "moe_expert_gate_up_shared_batch2_t",
            )?,
            moe_expert_silu_down_shared_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch2_t",
                "moe_expert_silu_down_shared_batch2_t",
            )?,
            moe_expert_gate_up_shared_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch3_t",
                "moe_expert_gate_up_shared_batch3_t",
            )?,
            moe_expert_silu_down_shared_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_batch3_t",
                "moe_expert_silu_down_shared_batch3_t",
            )?,
            qwen4exp_moe_fast: ops::Qwen4ExpMoeFast::resolve(gpu, config),
            qwen4exp_moe_rows: ops::Qwen4ExpMoeRows::resolve(gpu, config),
            moe_expert_gate_up_shared_fp8_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_t",
                "moe_expert_gate_up_shared_fp8_t",
            )?,
            moe_expert_silu_down_shared_fp8_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_t",
                "moe_expert_silu_down_shared_fp8_t",
            )?,
            moe_expert_gate_up_shared_fp8_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2_t",
                "moe_expert_gate_up_shared_fp8_batch2_t",
            )?,
            moe_expert_silu_down_shared_fp8_batch2_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2_t",
                "moe_expert_silu_down_shared_fp8_batch2_t",
            )?,
            moe_expert_gate_up_shared_fp8_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3_t",
                "moe_expert_gate_up_shared_fp8_batch3_t",
            )?,
            moe_expert_silu_down_shared_fp8_batch3_t_k: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3_t",
                "moe_expert_silu_down_shared_fp8_batch3_t",
            )?,
            unified_layout: std::env::var("ATLAS_UNIFIED_MOE_LAYOUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            nvfp4_mmq_layout: false,
            _nvfp4_mmq_owned: Vec::new(),
            hybrid_layout: std::env::var("ATLAS_HYBRID_MOE_LAYOUT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            nvfp4_gate_up_m128: std::env::var("ATLAS_NVFP4_GATE_UP_M128")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            nvfp4_down_m32: std::env::var("ATLAS_NVFP4_DOWN_M32")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            nvfp4_prequant_moe: std::env::var("ATLAS_NVFP4_PREQUANT_MOE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            c2_compact_moe,
            nvfp4_vecscale: std::env::var("ATLAS_NVFP4_VECSCALE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            nvfp4_fused_silu_quant: std::env::var("ATLAS_NVFP4_FUSED_SILU_QUANT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            // FP4 prefill MoE over the shared FAST_MOE=full [K/2,N] tables.
            gateup_fp4: std::env::var("ATLAS_HOLO_MOE_GATEUP_FP4")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            down_fp4: std::env::var("ATLAS_HOLO_MOE_DOWN_FP4")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            shared_gate_t: None,
            shared_gate_up_receipt: None,
            shared_up_t: None,
            shared_down_t: None,
            gate_fp8: None,
            shared_gate_fp8: None,
            shared_fp8_cache: Default::default(),
            shared_fp8_origins: None,
            shared_up_fp8: None,
            shared_down_fp8: None,
            prefill_stream: gpu.create_stream()?,
            event_a: gpu.create_event()?,
            event_b: gpu.create_event()?,
            moe_expert_gate_up_shared_fp8: gpu.kernel(
                "moe_shared_expert_fused_fp8",
                "moe_expert_gate_up_shared_fp8",
            )?,
            moe_expert_silu_down_shared_fp8: gpu.kernel(
                "moe_shared_expert_fused_fp8",
                "moe_expert_silu_down_shared_fp8",
            )?,
            // FP8 batch2/3 kernels for MTP verify
            moe_expert_gate_up_shared_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_expert_gate_up_shared_fp8_batch2",
            )?,
            moe_expert_silu_down_shared_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_expert_silu_down_shared_fp8_batch2",
            )?,
            moe_weighted_sum_blend_fp8_batch2: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch2",
                "moe_weighted_sum_blend_fp8_batch2",
            )?,
            moe_expert_gate_up_shared_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_expert_gate_up_shared_fp8_batch3",
            )?,
            moe_expert_silu_down_shared_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_expert_silu_down_shared_fp8_batch3",
            )?,
            moe_weighted_sum_blend_fp8_batch3: gpu.kernel(
                "moe_shared_expert_fused_fp8_batch3",
                "moe_weighted_sum_blend_fp8_batch3",
            )?,
            fp8_gate_weight_ptrs: None,
            fp8_up_weight_ptrs: None,
            fp8_down_weight_ptrs: None,
            bf16_gate_weight_ptrs: None,
            bf16_up_weight_ptrs: None,
            bf16_down_weight_ptrs: None,
            bf16_shared_expert: None,
            fp8_shared_expert: None,
            moe_down_t_k64_fp4: super::super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_down_t_k64_fp4",
            ),
            // Handle 0 unless this model ships `moe_persist_marker.cu`, i.e.
            // unless its k64 prefill grid has actually been measured short.
            // Deliberately NOT in the "moe_w4a16" module: that file is a symlink
            // shared by seven targets, so a marker there would speak for six
            // models nobody profiled. Gates ATLAS_MOE_PREFILL_PERSIST_TILES.
            moe_k64_strides_m_tiles: super::super::try_kernel(
                gpu,
                "moe_persist_marker",
                "moe_w4a16_k64_strides_m_tiles",
            ),
            moe_permute_tokens_k: super::super::try_kernel(gpu, "moe", "moe_permute_tokens"),
            // Phase 2.7 Tier C — set by loader after construction (qwen35.rs).
            is_dflash_capture_layer: false,
            lora: None,
            correction_bias_dev: weights_correction_bias,
            // `moe_topk_sig` is only registered for sigmoid-gated MoE models
            // (MiniMax-M2, Nemotron-Nano, Nemotron-Super). Softmax-gated MoEs
            // (Qwen3.5, Qwen3-Next, Gemma-4, Mistral) never hit the sigmoid
            // dispatch path, so a missing kernel is fine — fail at call time
            // via the KernelHandle(0) check in ops::moe_topk_sigmoid rather
            // than at MoeLayer::new(), which would otherwise block all
            // softmax-MoE model startup (observed on Qwen3.5-35B-A3B-FP8 in
            // alpha-2.43: "Module 'moe_topk_sig' not loaded" during model
            // build).
            moe_topk_sigmoid_k: super::super::try_kernel(gpu, "moe_topk_sig", "moe_topk_sigmoid"),
            moe_topk_sigmoid_batched_k: super::super::try_kernel(
                gpu,
                "moe_topk_sig",
                "moe_topk_sigmoid_batched",
            ),
        };
        if crate::model::glm_independent::enabled(&config.model_type)? {
            layer.nvfp4_prequant_moe = true;
            layer.nvfp4_fused_silu_quant = true;
            layer.validate_independent_handles()?;
        }
        Ok(layer)
    }
}
