// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::NemotronMoeLayer;
use super::nemotron_decode_policy;
use super::ops;
use super::prefill_sorted::SortedPrefillCtx;
use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
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
        // path. ATLAS_LIGHTNING_DECODE_MULTI=1 opts into both hybrid mixers;
        // ATLAS_LIGHTNING_MOE_MULTI is the component diagnostic override.
        if nemotron_decode_policy::decode_multi_seq_batched(
            std::env::var("ATLAS_LIGHTNING_DECODE_MULTI")
                .ok()
                .as_deref(),
            std::env::var("ATLAS_LIGHTNING_MOE_MULTI").ok().as_deref(),
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
        // Lightning's verified C=1 contract is one native K-row MoE launch per
        // sequence. Collapsing unrelated sequences into R=Σks changes the
        // decode dispatch width (and therefore the reduction order/kernel arm),
        // which is not greedy-equivalent: distinct C>1 prompts diverge from
        // their C1 controls even though attention, Mamba, and LM-head are exact.
        // Keep the weight-preserving batched verifier at the sequence boundary:
        // each sequence still verifies all of its K rows natively in one call,
        // but no call may mix another sequence's rows.
        let h = ctx.config.hidden_size;
        let mut off_bytes = 0usize;
        for &k in ks {
            self.decode_batched_direct(
                hidden.offset(off_bytes),
                residual.offset(off_bytes),
                k,
                ctx,
                stream,
            )?;
            off_bytes += k * h * 2;
        }
        Ok(())
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
