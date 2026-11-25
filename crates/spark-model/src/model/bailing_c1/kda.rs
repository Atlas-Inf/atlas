// SPDX-License-Identifier: AGPL-3.0-only

//! Ling KDA decoder layer (35 of the 42): short-conv + bounded-decay
//! deltanet recurrence per `kernels/gb10/ling-3.2-air/nvfp4/kda.cu`.
//!
//! Decode: `kda_recurrent_bf16` advances the sequence's per-layer FP32
//! state one token. Prefill: the same kernel over each sub-chunk's rows
//! (it walks tokens serially inside the launch — the regresident pair is
//! the follow-up). The slot-pool twin `ling_kda_recurrent_indexed` is the
//! multi-seq/graph seam, deliberately unwired in this PR (see
//! `model/bailing_c1.rs`). Multi-row MTP VERIFY: `decode_batched` runs the
//! GEMM-batched projections of the prefill arm plus the verify twin
//! `kda_recurrent_bf16_verify_snap` (PR #110), which walks the K rows in
//! one launch, leaves the live state after the last row (a full accept is
//! already committed) and writes the per-row rollback snapshots
//! `rollback_ssm_states` indexes on a partial accept (`k - 1` H slabs into
//! the contiguous `h_state_intermediates` region, conv snapshots after
//! each row's per-row conv launch). One NEXTN draft = K = 2 rows.

use std::sync::Arc;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::spec::{KDA_HEAD_DIM, KDA_HEADS};
use super::{LingFfn, LingKernels, LingScratch};
use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// KDA projection widths at the supported geometry: 32 heads x 128.
const T: usize = KDA_HEADS * KDA_HEAD_DIM;

pub(crate) struct LingKdaWeights {
    /// q/k/v_proj `[T, hidden]`; b_proj `[32, hidden]`; f/g_proj `[T, hidden]`;
    /// o_proj `[hidden, T]`; conv `[3T, conv_k]` (q|k|v filters back to back);
    /// A_log `[32]` FP32; dt_bias `[T]` FP32; o_norm `[128]` BF16.
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub b_proj: DenseWeight,
    pub f_proj: DenseWeight,
    pub g_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub conv: DenseWeight,
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    pub o_norm: DenseWeight,
}

pub(crate) struct LingKdaLayer {
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    weights: LingKdaWeights,
    ffn: LingFfn,
    kernels: Arc<LingKernels>,
    scratch: Arc<LingScratch>,
    lower_bound: f32,
    eps: f32,
}

fn state_of<'a>(state: &'a mut dyn LayerState) -> Result<&'a mut SsmLayerState> {
    state
        .as_any_mut()
        .downcast_mut::<SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("ling KDA: expected SsmLayerState"))
}

impl LingKdaLayer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_norm: DenseWeight,
        post_attn_norm: DenseWeight,
        weights: LingKdaWeights,
        ffn: LingFfn,
        kernels: Arc<LingKernels>,
        scratch: Arc<LingScratch>,
        lower_bound: f32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            input_norm,
            post_attn_norm,
            weights,
            ffn,
            kernels,
            scratch,
            lower_bound,
            eps,
        })
    }

    /// The attention half of `rows` rows: reads `hidden` (rows x hidden),
    /// adds the attention output into it, returns nothing (state advanced).
    pub(crate) fn attention_half(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &mut SsmLayerState,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let w = &self.weights;
        let hidden_w = self.scratch.hidden;
        ops::rms_norm(
            gpu,
            k.rms_norm_k,
            hidden,
            &self.input_norm,
            s.normed,
            rows,
            hidden_w as u32,
            self.eps,
            stream,
        )?;
        let decode = rows == 1;
        if decode {
            for (plane, proj) in [&w.q_proj, &w.k_proj, &w.v_proj].iter().enumerate() {
                ops::dense_gemv(
                    gpu,
                    k.gemv_k,
                    s.normed,
                    proj,
                    s.kda_qkv.offset(plane * T * 2),
                    T as u32,
                    hidden_w as u32,
                    stream,
                )?;
            }
            // One conv-update pass over the three planes (channels 3*T,
            // state [3T, conv_k] FP32): in-place on the packed rows.
            ops::conv1d_update(
                gpu,
                k.conv_update_k,
                state.conv_state,
                s.kda_qkv,
                &w.conv,
                s.kda_qkv,
                (3 * T) as u32,
                4,
                1,
                stream,
            )?;
        } else {
            // Projections GEMM into the three contiguous `[rows, T]`
            // plane regions of the staging buffer, then three strided
            // prefill-conv launches write the packed q|k|v rows.
            for (plane, proj) in [&w.q_proj, &w.k_proj, &w.v_proj].iter().enumerate() {
                ops::dense_gemm(
                    gpu,
                    k.gemm_k,
                    s.normed,
                    proj,
                    s.kda_preconv.offset(rows as usize * (plane * T) * 2),
                    rows,
                    T as u32,
                    hidden_w as u32,
                    stream,
                )?;
            }
            for plane in 0..3 {
                ops::conv1d_update_prefill(
                    gpu,
                    k.conv_prefill_k,
                    spark_runtime::gpu::KernelHandle(0),
                    state.conv_state.offset(plane * T * 4 * 4),
                    s.kda_preconv.offset(rows as usize * (plane * T) * 2),
                    // The conv blob is [3T channels][4 taps] bf16: each
                    // plane's taps start at channel plane*T, i.e. byte
                    // offset plane*T*4*2 — matching the conv_state slice
                    // above (plane*T*4 fp32 elements).
                    &DenseWeight {
                        weight: w.conv.weight.offset((plane * T) * 4 * 2),
                    },
                    DevicePtr::NULL,
                    s.kda_qkv.offset((plane * T) * 2),
                    T as u32,
                    4,
                    rows,
                    T as u32,
                    (3 * T) as u32,
                    stream,
                )?;
            }
        }
        // Side projections: b (beta, 32 wide), f (decay gate), g (o_norm gate).
        if decode {
            ops::dense_gemv(
                gpu,
                k.gemv_k,
                s.normed,
                &w.b_proj,
                s.kda_beta,
                32,
                hidden_w as u32,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                k.gemv_k,
                s.normed,
                &w.f_proj,
                s.kda_gate,
                T as u32,
                hidden_w as u32,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                k.gemv_k,
                s.normed,
                &w.g_proj,
                s.kda_o_g,
                T as u32,
                hidden_w as u32,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                gpu,
                k.gemm_k,
                s.normed,
                &w.b_proj,
                s.kda_beta,
                rows,
                32,
                hidden_w as u32,
                stream,
            )?;
            ops::dense_gemm(
                gpu,
                k.gemm_k,
                s.normed,
                &w.f_proj,
                s.kda_gate,
                rows,
                T as u32,
                hidden_w as u32,
                stream,
            )?;
            ops::dense_gemm(
                gpu,
                k.gemm_k,
                s.normed,
                &w.g_proj,
                s.kda_o_g,
                rows,
                T as u32,
                hidden_w as u32,
                stream,
            )?;
        }
        ops::kda_recurrent(
            gpu,
            k.kda_recurrent_k,
            s.kda_qkv,
            s.kda_gate,
            s.kda_beta,
            w.a_log,
            w.dt_bias,
            state.h_state,
            s.kda_out,
            rows,
            KDA_HEADS as u32,
            KDA_HEAD_DIM as u32,
            self.lower_bound,
            stream,
        )?;
        ops::kda_sigmoid_gated_norm(
            gpu,
            k.kda_gated_norm_k,
            s.kda_out,
            s.kda_o_g,
            w.o_norm.weight,
            s.kda_o,
            rows,
            KDA_HEADS as u32,
            KDA_HEAD_DIM as u32,
            self.eps,
            stream,
        )?;
        // o_proj: T -> hidden, then hidden += attn_out.
        if decode {
            ops::dense_gemv(
                gpu,
                k.gemv_k,
                s.kda_o,
                &w.o_proj,
                s.attn_out,
                hidden_w as u32,
                T as u32,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                gpu,
                k.gemm_k,
                s.kda_o,
                &w.o_proj,
                s.attn_out,
                rows,
                hidden_w as u32,
                T as u32,
                stream,
            )?;
        }
        ops::residual_add(
            gpu,
            k.residual_add_k,
            hidden,
            s.attn_out,
            rows as u32 * hidden_w as u32,
            stream,
        )
    }

    /// The FFN half: post norm -> FFN -> add.
    fn ffn_half(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm_k,
            hidden,
            &self.post_attn_norm,
            s.ffn_in,
            rows,
            self.scratch.hidden as u32,
            self.eps,
            stream,
        )?;
        self.ffn
            .forward(gpu, s.ffn_in, s.ffn_out, rows as usize, stream)?;
        ops::residual_add(
            gpu,
            self.kernels.residual_add_k,
            hidden,
            s.ffn_out,
            rows * self.scratch.hidden as u32,
            stream,
        )
    }

    fn forward_rows(
        &self,
        hidden: DevicePtr,
        state: &mut SsmLayerState,
        rows: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.attention_half(ctx.gpu, hidden, state, rows, stream)?;
        self.ffn_half(ctx.gpu, hidden, rows, stream)
    }

    /// The per-sequence verify-rollback region this layer's
    /// [`Self::decode_batched`] needs: `k - 1` contiguous H slabs (the
    /// `kda_recurrent_bf16_verify_snap` snapshot stride is the H bytes) and
    /// as many conv slabs, allocated lazily on the FIRST verify of the
    /// sequence (per-sequence, driver-owned — this family's KDA states are
    /// `alloc_state` allocations, not pool slots). Widened in place if a
    /// later verify arrives with more rows: the region is fully rewritten
    /// each verify, so re-init is safe.
    fn ensure_verify_slabs(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut SsmLayerState,
        inter: usize,
    ) -> Result<()> {
        let h_bytes = KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4;
        let conv_bytes = 3 * T * 4 * 4;
        if state.h_state_intermediates.len() < inter {
            let slab = gpu.alloc(inter * h_bytes)?;
            state.h_state_intermediates =
                (0..inter).map(|t| slab.offset(t * h_bytes)).collect();
        }
        anyhow::ensure!(
            state.h_state_intermediates.windows(2).all(|w| {
                w[1].0.saturating_sub(w[0].0) == h_bytes as u64
            }),
            "ling KDA: verify H snapshots must be contiguous with the H stride \
             (kda_recurrent_bf16_verify_snap state_inter layout)"
        );
        if state.conv_state_intermediates.len() < inter {
            let slab = gpu.alloc(inter * conv_bytes)?;
            state.conv_state_intermediates =
                (0..inter).map(|t| slab.offset(t * conv_bytes)).collect();
        }
        Ok(())
    }

    /// The K-token verify forward of ONE sequence (`decode_batched`, the
    /// scheduler's K = drafts + 1 in-place verify). Same projections as the
    /// prefill arm (GEMM over all rows), the verify twin of the recurrence
    /// (live state advanced through every row, snapshots after rows
    /// 0..k-2), and per-row conv launches so the conv state can be
    /// snapshotted between rows.
    fn verify_rows(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &mut SsmLayerState,
        rows: usize,
        _ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let w = &self.weights;
        let hidden_w = s.hidden as u32;
        let rows_u32 = rows as u32;
        let inter = rows - 1;
        self.ensure_verify_slabs(gpu, state, inter)?;
        // Attention half. Projections: GEMM into the planar staging (the
        // prefill arm), GEMM gates, then per-row conv launches — the
        // multi-row prefill conv folds all rows at once and gives no
        // mid-window conv state.
        ops::rms_norm(
            gpu,
            k.rms_norm_k,
            hidden,
            &self.input_norm,
            s.normed,
            rows_u32,
            hidden_w,
            self.eps,
            stream,
        )?;
        // At the K=2 verify width the projections are pure weight
        // streaming: the batch2 GEMV reads each weight once and is
        // bit-identical to the two decode GEMVs, where the M=2 GEMM tiles
        // re-read the weight ~2.7x (nsys: 292us vs 109us on the q_proj).
        // Wider rows keep the GEMM arm.
        let batch2 = rows == 2;
        for (plane, proj) in [&w.q_proj, &w.k_proj, &w.v_proj].iter().enumerate() {
            let dst = s.kda_preconv.offset(rows as usize * (plane * T) * 2);
            if batch2 {
                ops::dense_gemv_batch2(
                    gpu,
                    k.gemv_batch2_k,
                    s.normed,
                    proj,
                    dst,
                    T as u32,
                    hidden_w,
                    T as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    gpu,
                    k.gemm_k,
                    s.normed,
                    proj,
                    dst,
                    rows_u32,
                    T as u32,
                    hidden_w,
                    stream,
                )?;
            }
        }
        for (plane, proj) in [&w.b_proj, &w.f_proj, &w.g_proj].iter().enumerate() {
            let (dst, width) = match plane
            {
                0 => (s.kda_beta, 32),
                1 => (s.kda_gate, T),
                _ => (s.kda_o_g, T),
            };
            if batch2 {
                ops::dense_gemv_batch2(
                    gpu,
                    k.gemv_batch2_k,
                    s.normed,
                    proj,
                    dst,
                    width as u32,
                    hidden_w,
                    width as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    gpu,
                    k.gemm_k,
                    s.normed,
                    proj,
                    dst,
                    rows_u32,
                    width as u32,
                    hidden_w,
                    stream,
                )?;
            }
        }
        let h_bytes = KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4;
        let conv_bytes = 3 * T * 4 * 4;
        for t in 0..rows {
            for plane in 0..3 {
                ops::conv1d_update_prefill(
                    gpu,
                    k.conv_prefill_k,
                    spark_runtime::gpu::KernelHandle(0),
                    state.conv_state.offset(plane * T * 4 * 4),
                    s.kda_preconv
                        .offset((rows * (plane * T) + t * T) * 2),
                    &DenseWeight {
                        weight: w.conv.weight.offset((plane * T) * 4 * 2),
                    },
                    DevicePtr::NULL,
                    s.kda_qkv.offset((t * 3 * T + plane * T) * 2),
                    T as u32,
                    4,
                    1,
                    T as u32,
                    (3 * T) as u32,
                    stream,
                )?;
            }
            // The conv snapshot after row `t` is the reject-rollback state
            // `rollback_ssm_states(num_accepted = t + 1)` restores; the
            // last row's conv needs no snapshot (a full accept keeps the
            // live state).
            if t + 1 < rows {
                gpu.copy_d2d_async(
                    state.conv_state,
                    state.conv_state_intermediates[t],
                    conv_bytes,
                    stream,
                )?;
            }
        }
        ops::kda_recurrent_verify_snap(
            gpu,
            k.kda_verify_snap_k,
            s.kda_qkv,
            s.kda_gate,
            s.kda_beta,
            w.a_log,
            w.dt_bias,
            state.h_state,
            s.kda_out,
            state.h_state_intermediates[0],
            h_bytes,
            rows_u32,
            KDA_HEADS as u32,
            KDA_HEAD_DIM as u32,
            self.lower_bound,
            stream,
        )?;
        ops::kda_sigmoid_gated_norm(
            gpu,
            k.kda_gated_norm_k,
            s.kda_out,
            s.kda_o_g,
            w.o_norm.weight,
            s.kda_o,
            rows_u32,
            KDA_HEADS as u32,
            KDA_HEAD_DIM as u32,
            self.eps,
            stream,
        )?;
        if batch2 {
            ops::dense_gemv_batch2(
                gpu,
                k.gemv_batch2_k,
                s.kda_o,
                &w.o_proj,
                s.attn_out,
                hidden_w,
                T as u32,
                hidden_w,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                gpu,
                k.gemm_k,
                s.kda_o,
                &w.o_proj,
                s.attn_out,
                rows_u32,
                hidden_w,
                T as u32,
                stream,
            )?;
        }
        ops::residual_add(
            gpu,
            k.residual_add_k,
            hidden,
            s.attn_out,
            rows_u32 * hidden_w,
            stream,
        )?;
        self.ffn_half(gpu, hidden, rows_u32, stream)
    }
}

impl TransformerLayer for LingKdaLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state_of(state)?;
        self.forward_rows(hidden, state, 1, ctx, stream)
    }

    /// The MTP/K-token verify of one sequence: rows `0..num_tokens` of
    /// `hidden` are consecutive tokens whose state folds live, with the
    /// `k - 1` per-row snapshots the accept/reject commit indexes. A single
    /// row runs the plain decode path unchanged.
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if num_tokens <= 1 {
            let state = state_of(state)?;
            return self.forward_rows(hidden, state, 1, ctx, stream);
        }
        anyhow::ensure!(
            num_tokens <= self.scratch.rows,
            "ling KDA: verify {num_tokens} rows exceeds the scratch cap {}",
            self.scratch.rows
        );
        let state = state_of(state)?;
        self.verify_rows(ctx.gpu, hidden, state, num_tokens, ctx, stream)
    }

    /// One NEXTN draft: this layer serves the K = 2 verify and nothing
    /// wider in this slice (the verify snapshot region is `k - 1` slabs,
    /// and the scheduler clamps every step's draft count to this).
    fn verify_max_drafts(&self) -> Option<usize> {
        Some(1)
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state_of(state)?;
        for (n, chunk) in (0..num_tokens)
            .step_by(self.scratch.rows)
            .map(|start| (num_tokens - start).min(self.scratch.rows))
            .enumerate()
        {
            self.forward_rows(
                hidden.offset(n * self.scratch.rows * self.scratch.hidden * 2),
                state,
                chunk as u32,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_bytes = KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4;
        let conv_bytes = 3 * T * 4 * 4;
        let h = gpu.alloc(h_bytes)?;
        gpu.memset(h, 0, h_bytes)?;
        let conv = gpu.alloc(conv_bytes)?;
        gpu.memset(conv, 0, conv_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state: h,
            conv_state: conv,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
            gdn_commit_qkv: DevicePtr::NULL,
            gdn_commit_gb: DevicePtr::NULL,
            gdn_commit_pending: false,
            gdn_fuse_n: DevicePtr::NULL,
            kda_records: DevicePtr::NULL,
            h_is_f16: false,
            h_prefill_stage: None,
            ple: None,
        }))
    }

    /// Eager only: the per-seq state path is not pool-slot-stable yet (the
    /// `ling_kda_recurrent_indexed` seam above).
    fn decode_graph_unsupported(&self) -> bool {
        true
    }
}
