// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H standalone MoE FFN layer.
//!
//! Supports two variants:
//!   - **Nano 30B**: Direct MoE — experts operate on full hidden_size.
//!   - **Super 120B**: LatentMoE — routed experts operate in latent space `[moe_latent_size]`,
//!     with fc1/fc2 latent projections bridging hidden↔latent.
//!
//! Forward: RMS norm → gate → sigmoid topK routing → (fc1_latent if latent) →
//!          batched up GEMV → fused relu²+down → weighted_sum → (fc2_latent if latent) →
//!          shared expert up+relu²+down → sum routed+shared → residual add.
//!
//! All expert dispatch is device-side (pointer tables) — zero D2H sync.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use crate::layers::{nemotron_decode_policy, ops};
use crate::weight_map::{DenseWeight, NemotronMoeWeights, QuantizedWeight};

/// Device-side pointer table for one projection across all experts.
struct ExpertPtrTable {
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
}

/// Nemotron-H standalone MoE FFN layer.
pub struct NemotronMoeLayer {
    weights: NemotronMoeWeights,
    input_norm: DenseWeight,
    /// LatentMoE dimension (0 = direct, >0 = latent).
    moe_latent_size: usize,
    /// Routed expert intermediate size for this layer (Puzzle: per-block).
    moe_inter: usize,
    /// Top-K experts activated per token for this layer (Puzzle: per-block).
    top_k: usize,
    // Kernel handles — decode (single token)
    rms_norm_residual_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    topk_sigmoid_k: KernelHandle,
    moe_expert_gemv_k: KernelHandle,
    moe_expert_gemv_wide_k: KernelHandle,
    moe_expert_gemv_wide_grouped_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemv_batch4_k: KernelHandle,
    w4a16_gemv_batch16_k: KernelHandle,
    /// Native-FP8 decode GEMV for the shared-expert up_proj (see
    /// `NemotronMoeWeights::shared_up_fp8`). 0 when unavailable.
    w8a16_gemv_k: KernelHandle,
    /// Native-FP8 prefill GEMM for the shared expert. 0 when unavailable.
    w8a16_gemm_k: KernelHandle,
    w8a16_gemm_pipelined_k: KernelHandle,
    relu2_down_shared_k: KernelHandle,
    relu2_down_wide_k: KernelHandle,
    weighted_sum_scale_k: KernelHandle,
    residual_add_k: KernelHandle,
    // Kernel handles — prefill (batched GEMM)
    dense_gemm_k: KernelHandle,
    /// Pipelined tensor-core BF16 GEMM (mma.sync.m16n8k16 + cp.async 2-stage,
    /// 128x128 tile). `dense_gemm_bf16` is a SCALAR 16x16 kernel — on the
    /// large-M prefill shapes it is ~40x slower, and the three dense GEMMs of a
    /// LatentMoE layer (gate, fc1_latent, fc2_latent) were the single largest
    /// prefill cost on Puzzle (34% of all GPU time). Same math (cosine=1.0).
    dense_gemm_pipelined_k: KernelHandle,
    w4a16_gemm_k: KernelHandle,
    // Batched N-token MoE prefill kernels
    topk_sigmoid_batched_k: KernelHandle,
    moe_up_prefill_k: KernelHandle,
    moe_relu2_down_prefill_k: KernelHandle,
    moe_weighted_sum_prefill_k: KernelHandle,
    // Sorted grouped GEMM (Qwen pattern — proven to work)
    moe_sort_k: KernelHandle,
    moe_grouped_gemm_k: KernelHandle,
    moe_relu2_elementwise_k: KernelHandle,
    moe_grouped_gemm_relu2_k: KernelHandle,
    moe_w4a4_grouped_k: KernelHandle,
    moe_unpermute_reduce_k: KernelHandle,
    moe_grouped_gemm_n128_k: KernelHandle,
    up_ptrs: ExpertPtrTable,
    down_ptrs: ExpertPtrTable,
    // Transposed expert pointer tables (for N128 grouped GEMM)
    up_ptrs_t: Option<ExpertPtrTable>,
    down_ptrs_t: Option<ExpertPtrTable>,
    // Transposed shared expert weights
    shared_up_t: Option<QuantizedWeight>,
    shared_down_t: Option<QuantizedWeight>,
    // Pre-dequantized FP8 E4M3 [N, K] copies of the shared-expert projections.
    // Consumed by `fp8_gemm_t_m128_mfast` (no dequant phase); see the SSM layer.
    shared_up_pd_fp8: Option<DevicePtr>,
    shared_down_pd_fp8: Option<DevicePtr>,
    // FP8 E4M3 copies of the BF16 latent projections, so prefill runs the tuned
    // FP8 GEMM instead of dense_gemm_bf16_pipelined (and halves their bytes).
    fc1_pd_fp8: Option<DevicePtr>,
    fc2_pd_fp8: Option<DevicePtr>,
    // Transposed SSM GEMM kernel handle (for shared expert)
    w4a16_gemm_t_k: KernelHandle,
    w4a16_gemm_t_m128_k: KernelHandle,
    fp8_gemm_m128_k: KernelHandle,
    w4a4_gemm_k: KernelHandle,
    quantize_nvfp4_k: KernelHandle,
    marlin: Option<marlin_sidecar::MarlinSidecar>,
}

impl NemotronMoeLayer {
    pub fn new(
        weights: NemotronMoeWeights,
        input_norm: DenseWeight,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        moe_inter: usize,
        top_k: usize,
    ) -> Result<Self> {
        let up_ptrs = build_ptr_table(&weights.experts, |e| &e.up_proj, gpu)?;
        let down_ptrs = build_ptr_table(&weights.experts, |e| &e.down_proj, gpu)?;
        let moe_inter = if moe_inter > 0 {
            moe_inter
        } else {
            config.moe_intermediate_size
        };
        let top_k = if top_k > 0 {
            top_k
        } else {
            config.num_experts_per_tok
        };
        // Nemotron's `top_k` is PER LAYER (`num_experts_per_tok_for`), so this
        // runs once per MoE layer and a single outlying block config cannot
        // slip through on the model-wide value. `MoeLayer::new` carries the
        // same pair of bounds; `NemotronMoeLayer` had no check at all and its
        // decode kernel is the one whose shadows were capped at 24.
        let num_experts = weights.experts.len();
        anyhow::ensure!(
            top_k > 0
                && top_k <= num_experts
                && top_k <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K
                && num_experts <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
            "Nemotron MoE config invalid: top_k={} must be in 1..={} and within \
             the routing kernels' bounds (top_k max {}, num_experts={} max {})",
            top_k,
            num_experts,
            crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K,
            num_experts,
            crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
        );

        let marlin = marlin_sidecar::MarlinSidecar::try_build(
            gpu,
            &weights.experts,
            moe_inter,
            config.hidden_size,
            config.hidden_size,
            moe_inter,
        )?;

        Ok(Self {
            weights,
            input_norm,
            moe_latent_size: config.moe_latent_size,
            moe_inter,
            top_k,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            topk_sigmoid_k: gpu.kernel("moe_topk_sig", "moe_topk_sigmoid")?,
            moe_expert_gemv_k: gpu.kernel("moe_expert_gemv", "moe_expert_gemv")?,
            moe_expert_gemv_wide_k: super::try_kernel(
                gpu,
                "moe_expert_gemv_wide",
                "moe_expert_gemv_wide",
            ),
            moe_expert_gemv_wide_grouped_k: super::try_kernel(
                gpu,
                "moe_expert_gemv_wide_grouped",
                "moe_expert_gemv_wide_grouped",
            ),
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_batch4_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch4"),
            w4a16_gemv_batch16_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch16"),
            w8a16_gemv_k: super::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv"),
            w8a16_gemm_k: super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            relu2_down_shared_k: gpu.kernel("moe_relu2_fused", "moe_expert_relu2_down_shared")?,
            relu2_down_wide_k: super::try_kernel(
                gpu,
                "moe_expert_relu2_down_wide",
                "moe_expert_relu2_down_wide",
            ),
            weighted_sum_scale_k: gpu.kernel("relu2", "moe_weighted_sum_scale")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_pipelined_k: super::try_kernel(gpu, "gemm", "dense_gemm_bf16_pipelined"),
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            topk_sigmoid_batched_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_topk_sigmoid_batched",
            ),
            moe_up_prefill_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_up_prefill",
            ),
            moe_relu2_down_prefill_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_relu2_down_prefill",
            ),
            moe_weighted_sum_prefill_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_weighted_sum_prefill",
            ),
            moe_sort_k: super::try_kernel(gpu, "moe", "moe_sort_by_expert"),
            moe_grouped_gemm_k: super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable",
            ),
            moe_relu2_elementwise_k: super::try_kernel(gpu, "relu2", "relu_squared_inplace"),
            moe_grouped_gemm_relu2_k: super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_relu2",
            ),
            moe_w4a4_grouped_k: super::try_kernel(gpu, "moe_w4a4", "moe_w4a4_grouped_gemm_relu2"),
            moe_unpermute_reduce_k: super::try_kernel(gpu, "moe", "moe_unpermute_reduce_indexed"),
            moe_grouped_gemm_n128_k: super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_t",
            ),
            up_ptrs,
            down_ptrs,
            up_ptrs_t: None,
            down_ptrs_t: None,
            shared_up_t: None,
            shared_down_t: None,
            shared_up_pd_fp8: None,
            shared_down_pd_fp8: None,
            fc1_pd_fp8: None,
            fc2_pd_fp8: None,
            w4a16_gemm_t_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t"),
            w4a16_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            fp8_gemm_m128_k: super::try_kernel(gpu, "w4a16", "fp8_gemm_t_m128_mfast"),
            w4a4_gemm_k: super::try_kernel(gpu, "w4a4", "w4a4_gemm_mfast"),
            quantize_nvfp4_k: super::try_kernel(gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4"),
            marlin,
        })
    }
}

mod decode_helpers;
mod decode_batched;
mod marlin_sidecar;
mod marlin_linear;
mod marlin_slots;
mod prefill_fallback;
mod prefill_marlin;
mod prefill_shared_up;
mod prefill_sorted;
mod prefill_weights;
mod ptr_tables;

use prefill_sorted::SortedPrefillCtx;
use ptr_tables::{build_ptr_table, build_ptr_table_from_weights};

impl TransformerLayer for NemotronMoeLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_inner(hidden, residual, ctx, stream)
    }

    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_batched_direct(hidden, residual, num_tokens, ctx, stream)
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // AR C>1: MoE is stateless. Serial decode is the default diagnostic
        // path. Set ATLAS_LIGHTNING_DECODE_MULTI=1 to opt into one
        // decode_batched over N tokens (vLLM/SGLang grouped-GEMM). The opt-in
        // path can differ from serial GEMV on near-ties due to accumulation.
        if nemotron_decode_policy::decode_multi_seq_batched(
            std::env::var("ATLAS_LIGHTNING_DECODE_MULTI")
                .ok()
                .as_deref(),
        ) {
            self.decode_batched_direct(hidden, residual, num_seqs, ctx, stream)
        } else {
            // Default serial diagnostic path: one per-sequence decode().
            let h = ctx.config.hidden_size;
            for i in 0..num_seqs {
                let offset = i * h * 2;
                let mut bt = _block_tables[i].clone();
                let mut stub_disk = Vec::<u32>::new();
                let mut stub_off = Vec::<u32>::new();
                self.decode(
                    hidden.offset(offset),
                    residual.offset(offset),
                    _states[i],
                    _kv_cache,
                    _seq_lens[i],
                    &mut bt,
                    &mut stub_disk,
                    &mut stub_off,
                    ctx,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    fn decode_verify_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        _states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _wy_tables: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(ks.len() == n_seqs, "decode_verify_multi: ks/n mismatch");
        // DSV4/Lightning: C=1 KEEP is n=k≤4 slot Marlin. One launch over
        // R=Σks (C=4 R=11) garbles — same class as fused-Mamba over R.
        // Loop per seq so each call is the proven n≤4 shape.
        if self.marlin.is_some() {
            let h = ctx.config.hidden_size;
            let mut off = 0usize;
            for &k in ks {
                self.decode_batched_direct(
                    hidden.offset(off),
                    residual.offset(off),
                    k,
                    ctx,
                    stream,
                )?;
                off += k * h * 2;
            }
            return Ok(());
        }
        let num_tokens: usize = ks.iter().sum();
        self.decode_batched_direct(hidden, residual, num_tokens, ctx, stream)
    }

    /// Batched MoE prefill: uses GEMM for gate/fc1/fc2/shared, per-token for routing + experts.
    ///
    /// For Super 120B with 40 MoE layers, this replaces O(N * 7 kernel_launches) decode calls
    /// with O(4 GEMMs + N * 3 kernel_launches), cutting TTFT by 30-50%.
    #[allow(clippy::overly_complex_bool_expr)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let inter = self.moe_inter as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = self.top_k as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let scale = ctx.config.routed_scaling_factor as f32;
        let n = num_tokens as u32;

        // ── 1. Batched RMS norm: [N, H] → normed[N, H] + residual update ──
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        // ── 2. Batched Gate GEMM: [N, H] x [H, num_experts]^T → [N, num_experts] ──
        let gate_logits = ctx.buffers.gate_logits();
        self.dense_gemm_prefill(
            ctx.gpu,
            normed,
            &self.weights.gate,
            gate_logits,
            n,
            num_experts,
            h as u32,
            stream,
        )?;

        // Check if batched MoE prefill kernels are available
        let has_batched = self.topk_sigmoid_batched_k.0 != 0
            && self.moe_up_prefill_k.0 != 0
            && self.moe_relu2_down_prefill_k.0 != 0
            && self.moe_weighted_sum_prefill_k.0 != 0;

        // ── 3. Shared expert UP ──
        // When batched MoE prefill is available, the shared expert UP is handled
        // inside the batched UP kernel (step 5b). We only pre-compute here for
        // the per-token fallback path or LatentMoE.
        let shared_up_out_base = ctx.buffers.ssm_qkvz();
        let use_batched_moe = has_batched && num_tokens > 1;
        // Always compute shared expert UP — even when batched path overwrites it later.
        // The batched UP kernel writes shared_up_out for shared blocks, but we need
        // this result for the per-token fallback path AND it's harmless to overwrite.
        // Arm selection (native FP8 → W4A4 → pre-dequant FP8 → transposed NVFP4
        // → plain W4A16) lives in `prefill_shared_up.rs` (500-LoC cap split).
        self.prefill_shared_up(normed, shared_up_out_base, n, h, shared_inter, ctx, stream)?;

        // ── 4. LatentMoE: batched fc1_latent GEMM [N, H] → [N, L] ──
        // Use attn_output as temp buffer (m*max_dim*2, large enough for [N, L]).
        // Cannot use ssm_ba (too small) or moe_output (used later for unpermute).
        let latent = self.moe_latent_size as u32;
        let latent_base = if latent > 0 {
            let latent_buf = ctx.buffers.attn_output();
            if let Some(w_fp8) = self.fc1_pd_fp8 {
                ops::fp8_gemm_m128_mfast(
                    ctx.gpu,
                    self.fp8_gemm_m128_k,
                    normed,
                    w_fp8,
                    latent_buf,
                    n,
                    latent,
                    h as u32,
                    stream,
                )?;
            } else {
                let fc1 = self.weights.fc1_latent_proj.as_ref().unwrap();
                self.dense_gemm_prefill(
                    ctx.gpu, normed, fc1, latent_buf, n, latent, h as u32, stream,
                )?;
            }
            Some(latent_buf)
        } else {
            None
        };

        // ── 5. Batched routing + expert dispatch (N tokens, 4 kernel launches) ──
        // When batched prefill kernels are available, replace the per-token loop
        // (N × 5 launches = 10k+ launches) with 4 batched launches.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(n as usize * top_k as usize * 4);

        // Sorted MoE prefill: sort tokens by expert, then grouped GEMM.
        // G9 FIXED (2026-08-15): the sorted chain was dispatched through the
        // n128 wrapper (grid.x=ceil(N/128)) while the nano/lightning target
        // compiles the COMMON kernel (N_TILE=64) — only ceil(N/128)*64 columns
        // were ever computed. Now uses `moe_w4a16_grouped_gemm_ptrtable_64`.
        // DEFAULT ON after the full battery: GSM8K 12/12, NIAH 2k/12k/44k
        // PASS, dist sweep 0-256 PASS, canary exact, prefill ~1.7k tok/s
        // (vs 175-184 on the GEMV fallback). ATLAS_NO_MOE_SORTED=1 reverts.
        let use_sorted = use_batched_moe
            && std::env::var("ATLAS_NO_MOE_SORTED").is_err()
            && self.moe_sort_k.0 != 0
            && self.moe_grouped_gemm_k.0 != 0
            && self.moe_unpermute_reduce_k.0 != 0;

        // Marlin MoE prefill (NVIDIA fused_marlin_moe family): sorted layout +
        // per-expert plain m8 Marlin launches. Opt-in until the full battery
        // passes; ATLAS_MOE_MARLIN=1 (sidecar) + ATLAS_MOE_MARLIN_PREFILL=1.
        let use_marlin_prefill = use_sorted
            && std::env::var("ATLAS_MOE_MARLIN_PREFILL").is_ok()
            && self
                .marlin
                .as_ref()
                .is_some_and(|m| m.lin_up_k.0 != 0 && m.lin_down_k.0 != 0 && m.pack_rows_k.0 != 0);

        let p = SortedPrefillCtx {
            n,
            num_tokens,
            h,
            inter,
            shared_inter,
            num_experts,
            top_k,
            scale,
            latent,
            gate_logits,
            indices_dev,
            weights_dev,
            normed,
            hidden,
            latent_base,
            shared_up_out_base,
        };
        if use_marlin_prefill {
            self.prefill_marlin_path(&p, ctx, stream)?;
        } else if use_sorted {
            self.prefill_sorted_path(&p, ctx, stream)?;
        } else {
            self.prefill_fallback_path(&p, ctx, stream)?;
        }

        Ok(())
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }
}
