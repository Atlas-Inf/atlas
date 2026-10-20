// SPDX-License-Identifier: AGPL-3.0-only

//! Section 10 of `prefill_attention_paged`: O-projection GEMM
//! `[N, nq*hd] → [N, h]`. 6-way quantization dispatch (FP8 transposed,
//! FP8, FP8 col-scale, NVFP4 transposed, BF16 dense, NVFP4 default).
//! Extracted from `paged.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_paged_oproj(
        &self,
        attn_out: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        hd: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let o_out = ctx.buffers.norm_output();
        if self.oproj_reduce_scatter_piped(attn_out, o_out, [n, h, nq * hd], ctx, stream)? {
            return Ok(o_out);
        }
        // ATLAS_QWEN4EXP_PREFILL_BF16_PROJ: BF16 at decode's weight values,
        // on the pass's row-invariant k-chain.
        if crate::layers::ops::qwen4exp_rowinv::active()
            && let Some(w) = self.rowinv_bf16[3]
        {
            ops::bf16_gemm(attn_out, w.0, o_out, n, h, nq * hd, stream)?;
            return Ok(o_out);
        }
        // Keep-packed Q2_0 (Tier-1c): transient-dequant o_proj then dense GEMM.
        if let Some(r) =
            self.try_q2_prefill(ctx, self.o_weight.as_ref(), attn_out, o_out, n, stream)
        {
            r?;
            return Ok(o_out);
        }
        let force_w8a8 = ctx.dispatch.fp8_blockscaled_prefill;
        // Native FP4 o_proj: quantize attn_out to NVFP4 and consume o_proj in
        // its original NVFP4 form. OPT-IN ONLY -- see the QKV path's comment.
        let w4a4 = n >= 256
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (n as usize) * (nq as usize) * (hd as usize)
            && super::attn_w4a4_requested();
        if w4a4 {
            let kd = nq * hd;
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * (kd as usize) / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                attn_out,
                a4,
                a4_sf,
                n,
                kd,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.attn.o_proj,
                o_out,
                n,
                h,
                kd,
                stream,
            )?;
        } else if ctx.dispatch.cutlass_nvfp4_attn_o
            && let Some(ref nvfp4_t) = self.o_nvfp4_t
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "attn_o", n, h, nq * hd);
            ops::cutlass_nvfp4_proj(ctx, attn_out, nvfp4_t, o_out, n, h, nq * hd, stream)?;
        } else if ctx.dispatch.cutlass_nvfp4_attn_o
            && let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, "attn_o", n, h, nq * hd);
            ops::cutlass_nvfp4_proj_from_fp8(ctx, attn_out, fp8w, o_out, n, h, nq * hd, stream)?;
        } else if ctx.dispatch.cublas_gemm
            && let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
        {
            // cuBLASLt BF16 (3x the hand-written mma.sync GEMM on GB10).
            ops::cublas_bf16_proj(
                ctx.gpu,
                ctx.derived,
                attn_out,
                fp8w,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if force_w8a8
            && let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
            && self.per_token_group_quant_fp8_k.0 != 0
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            // o_proj GEMM: C[M, N] = A[M, K] @ B[N, K]
            //   A = attn_out  : [num_tokens, nq*hd]  (input,  K = nq*hd)
            //   B = o_weight  : [h, nq*hd]           (stored row-major, N = h)
            //   C = o_out     : [num_tokens, h]      (output, N = h)
            let m = n as usize;
            let k_dim = (nq * hd) as usize; // inner contract dim — input width of o_proj
            let n_out = h as usize; // output width of o_proj
            // Persistent arena scratch (no per-projection alloc/sync/free): the
            // quant→GEMM→next-layer chain is same-stream ordered, so the buffer
            // is safely reused without a host sync.
            let a_fp8_buf = ctx.buffers.fp8_act();
            let a_scale_buf = ctx.buffers.fp8_act_scale();
            debug_assert!(m * k_dim <= ctx.buffers.fp8_act_bytes());
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                attn_out,
                a_fp8_buf,
                a_scale_buf,
                n,
                nq * hd,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                o_out,
                n,
                n_out as u32,
                k_dim as u32,
                stream,
            )?;
        } else if let Some(ref fp8t) = self.o_fp8w_t
            && self.w8a16_gemm_t_pipelined_k.0 != 0
        {
            // o_proj via the byte-identical ~4.2x faster pipelined transposed
            // tensor-core kernel where available (NVIDIA). gfx1151/HIP has no
            // cp.async -> pipelined absent -> non-pipelined w8a16_gemm_t below.
            ops::w8a16_gemm_t_pipelined(
                ctx.gpu,
                self.w8a16_gemm_t_pipelined_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
            && fp8w.scale_format == crate::weight_map::WeightQuantFormat::Fp8BlockScaled
            && (n > 128 || self.o_fp8w_t.is_none())
            && self.w8a16_gemm_n_m128_k.0 != 0
        {
            // gfx1151 NON-transposed FP8 m128: native B[N,K] k-contiguous +
            // block_scale[N/128,K/128] — contiguous smem stores, no strided
            // bank-conflicting writes. Preferred over the transposed m128 arm.
            ops::w8a16_gemm_n_m128(
                ctx.gpu,
                self.w8a16_gemm_n_m128_k,
                attn_out,
                fp8w.weight,
                fp8w.row_scale,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(ref fp8t) = self.o_fp8w_t
            && self.w8a16_gemm_t_m128_k.0 != 0
        {
            // gfx1151/HIP: no cp.async, so the pipelined kernel is absent — the
            // 128x128-tile `w8a16_gemm_t_m128` is the fast transposed fallback
            // (halves B re-reads vs the 64x64 base tile). Same B_t/scale layout.
            ops::w8a16_gemm_n128_m128(
                ctx.gpu,
                self.w8a16_gemm_t_m128_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(ref fp8t) = self.o_fp8w_t
            && self.w8a16_gemm_t_k.0 != 0
        {
            // cp.async-free fallback (gfx1151/HIP): non-pipelined transposed W8A16.
            ops::w8a16_gemm_t(
                ctx.gpu,
                self.w8a16_gemm_t_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if self.o_weight.as_ref().and_then(|w| w.as_fp8()).is_some()
            && self.w8a16_gemm_k.0 != 0
        {
            let fp8w = self.o_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                attn_out,
                fp8w.weight,
                fp8w.row_scale,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(fp8) = self.o_fp8 {
            // ATLAS_QWEN4EXP_PREFILL_ROWINV: one arm at every width.
            if n > 128 || crate::layers::ops::qwen4exp_rowinv::active() {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4_t) = self.o_nvfp4_t {
            if n > 128 {
                self.w4a16_gemm_m128_dispatch(
                    ctx.gpu,
                    ctx.dispatch,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            // BF16 dense fallback (Gemma-4 dense per Nvidia ModelOpt's
            // ignore list — all self_attn projections must stay BF16).
            // Tensor-core pipelined GEMM (~40× scalar on large-M prefill).
            if ctx.dispatch.cublas_gemm && n > 1 {
                ops::cublas_bf16_proj_dense(attn_out, o_bf16.weight, o_out, n, h, nq * hd, stream)?;
            } else if self.dense_gemm_pipelined_k.0 != 0 {
                ops::dense_gemm_bf16_pipelined(
                    ctx.gpu,
                    self.dense_gemm_pipelined_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        }
        // ── LoRA delta on o_proj: o_out[n,h] += scale·(attn_out[n,nq*hd]@Aᵀ)@Bᵀ.
        // attn_out is the exact o_proj input (post-attention), matching HF.
        // Before the op_dump so dumps show the adapted output.
        if let Some(ref lw) = self.lora
            && let Some(ref pair) = lw.o
        {
            debug_assert_eq!(pair.k_in, nq * hd);
            debug_assert_eq!(pair.n_out, h);
            // #30 routed-prefill precision (see paged_qkv.rs): a routed (non-active)
            // prefill selects the REQUEST slot's O pair and folds it through the SAME
            // dense `apply_lora_delta` (dense_gemm_tc) the active adapter uses. Indexed
            // by `lw.layer_idx` (GLOBAL layer index, not attn_layer_idx). MUST win over
            // the bgmv branch (a routed prefill satisfies both). `None` → bgmv/installed.
            let routed_pair = ctx.routed_lora_layers.and_then(|ls| {
                crate::lora::select_routed_pair(ls, lw.layer_idx, crate::lora::LoraModule::OProj)
            });
            // Request-scoped routing (see paged_qkv.rs). attn_out is contiguous
            // [n, nq*hd] and o_out is contiguous [n, h], so the routed bgmv is
            // byte-identical to `n` single-row `apply_lora_delta`. No pool / no
            // route → installed-active-pair path (pre-M2 behaviour).
            let seq_slot = ctx
                .attn_metadata
                .map(|m| m.seq_slot)
                .unwrap_or(DevicePtr(0));
            if let Some(routed_pair) = routed_pair {
                debug_assert_eq!(routed_pair.k_in, nq * hd);
                debug_assert_eq!(routed_pair.n_out, h);
                ops::lora_delta::apply_lora_delta(
                    ctx.gpu,
                    &lw.kernels,
                    routed_pair,
                    attn_out,
                    o_out,
                    n,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            } else if seq_slot.0 != 0
                && let Some(ref route) = lw.o_route
            {
                ops::lora_delta::apply_lora_bgmv(
                    ctx.gpu,
                    &lw.kernels,
                    route,
                    attn_out,
                    o_out,
                    seq_slot,
                    n,
                    pair.k_in,
                    pair.n_out,
                    ctx.buffers.lora_xa(),
                    stream,
                )?;
            } else {
                ops::lora_delta::apply_lora_delta(
                    ctx.gpu,
                    &lw.kernels,
                    pair,
                    attn_out,
                    o_out,
                    n,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            }
        }
        // ATLAS_OP_DUMP hook: post-O-projection — this is the FULL attention
        // block output (Q*K^T*V * O_proj). Compares 1:1 against the HF
        // module hooked on `full_attention.o_proj.forward` for the last
        // token. Use `n` as token-count so we slice the last token.
        let bf16 = 2usize;
        let num_tokens = n as usize;
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                o_out,
                (num_tokens - 1) * h as usize * bf16,
                h as usize,
                self.attn_layer_idx,
                "o_proj",
                stream,
            )?;
        }
        Ok(o_out)
    }
}

impl Qwen3AttentionLayer {
    /// The o_proj by row ranges with the SP reduce-scatter pipelined behind
    /// it (`ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE`), when the caller offered its
    /// reduce-scatter (`qwen4exp_sp_pipe::RsOffer`). Only on the `o_fp8` arm
    /// (`fp8_gemm_t_m128`, row independent at any row ranges:
    /// `fp8_gemm_m128_rows_are_independent`) and only when the dispatch
    /// above would take it, so every byte is the unsplit o_proj + reduce-
    /// scatter's. `Ok(false)`: not taken, nothing launched.
    fn oproj_reduce_scatter_piped(
        &self,
        attn_out: DevicePtr,
        o_out: DevicePtr,
        [n, h, kd]: [u32; 3],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        use crate::layers::qwen4exp_sp_pipe as pipe;
        let Some(sp) = crate::layers::glm_sp::current().filter(|sp| sp.total() == n as usize)
        else {
            return Ok(false);
        };
        let earlier_arms_skip = self
            .o_weight
            .as_ref()
            .is_none_or(|w| w.as_packed_q2().is_none() && w.as_fp8().is_none())
            && self.o_fp8w_t.is_none()
            && !ctx.dispatch.cutlass_nvfp4_attn_o
            && !super::attn_w4a4_requested();
        // ATLAS_QWEN4EXP_PREFILL_BF16_PROJ: the BF16 o_proj, row-invariant too.
        let rowinv = self.rowinv_bf16[3].filter(|_| crate::layers::ops::qwen4exp_rowinv::active());
        let fp8 = self.o_fp8.filter(|_| n > 128 && earlier_arms_skip);
        if rowinv.is_none() && fp8.is_none() {
            return Ok(false);
        }
        if !pipe::rs_offered() {
            return Ok(false);
        }
        let (row_in, row_out) = (kd as usize * 2, h as usize * 2);
        let done = pipe::compute_and_reduce_scatter(
            sp,
            o_out,
            h as usize,
            ctx,
            stream,
            |r0, rows, s| match (rowinv, fp8) {
                (Some(w), _) => ops::bf16_gemm(
                    attn_out.offset(r0 * row_in),
                    w.0,
                    o_out.offset(r0 * row_out),
                    rows as u32,
                    h,
                    kd,
                    s,
                ),
                (None, Some(fp8)) => ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    attn_out.offset(r0 * row_in),
                    fp8,
                    o_out.offset(r0 * row_out),
                    rows as u32,
                    h,
                    kd,
                    s,
                ),
                (None, None) => unreachable!("checked above"),
            },
            |_| Ok(()),
        )?;
        if done {
            pipe::rs_took();
        }
        Ok(done)
    }
}
