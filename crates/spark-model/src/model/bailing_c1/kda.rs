// SPDX-License-Identifier: AGPL-3.0-only

//! Ling KDA decoder layer (35 of the 42): short-conv + bounded-decay
//! deltanet recurrence per `kernels/gb10/ling-3.2-air/nvfp4/kda.cu`.
//!
//! Decode: `kda_recurrent_bf16` advances the sequence's per-layer FP32
//! state one token. Prefill chunks (rows > 1, the default) run the
//! register-resident pair instead: `kda_preprocess_regresident` computes
//! the row-invariant q/k L2 norms, decay and beta once per token/head, and
//! `kda_recurrent_bf16_regresident` walks the chunk with each warp owning
//! a state column in registers — the FP32 state loads and stores once per
//! chunk instead of streaming 2 MiB per token
//! (`ATLAS_LING_KDA_REGRESIDENT_PREFILL=0` restores the token-serial walk).
//! Multi-row MTP VERIFY: `decode_batched` runs the GEMM-batched
//! projections of the prefill arm plus the verify twin
//! `kda_recurrent_bf16_verify_snap` (PR #110), which walks the K rows in
//! one launch, leaves the live state after the last row (a full accept is
//! already committed) and writes the per-row rollback snapshots
//! `rollback_ssm_states` indexes on a partial accept (`k - 1` H slabs into
//! the contiguous `h_state_intermediates` region, conv snapshots after
//! each row's per-row conv launch). One NEXTN draft = K = 2 rows.
//!
//! MULTI-SEQUENCE DECODE: `ling_kda_conv_indexed` +
//! `ling_kda_recurrent_indexed` address the house `SsmStatePool` through a
//! device-resident slot-id stream (`SsmBatchView` built by
//! `ssm_indexed_decode::prepare_runtime`), so every kernel argument of the
//! decode lane is a fixed address and the only per-step state is the
//! i32 slot stream the scheduler refreshes before replay — the contract a
//! captured decode graph needs. The lane runs when `ctx.ssm_batch` was
//! prepared (2..=8 unlatched rows on a single rank); otherwise
//! `decode_multi_seq` keeps the per-sequence loop over the pool-backed
//! per-slot states (the sequences' h/conv pointers are the pool addresses
//! `alloc_sequence` fixed at slot claim — stable across every decode
//! step of that sequence, just not shared across steps like the indexed
//! pair is).

use std::sync::Arc;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::spec::{KDA_HEAD_DIM, KDA_HEADS};
use super::{LingFfn, LingKernels, LingScratch};
use crate::layer::ssm_batch::SsmBatchLayer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::ops;
use crate::layers::ops::{KdaBuffer, KdaIndexedConv, KdaIndexedRecurrent, KdaIndexedShape};
use crate::weight_map::DenseWeight;

/// KDA projection widths at the supported geometry: 32 heads x 128.
const T: usize = KDA_HEADS * KDA_HEAD_DIM;

/// `ATLAS_LING_KDA_REGRESIDENT_PREFILL`: unset or any value other than
/// "0" prefills chunks on the register-resident pair
/// (`kda_preprocess_regresident` + `kda_recurrent_bf16_regresident`);
/// "0" restores the token-serial `kda_recurrent_bf16` walk. Read once.
fn regresident_prefill() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_LING_KDA_REGRESIDENT_PREFILL").as_deref() != Ok("0"))
}

/// `ATLAS_LING_PROJ_GEMM_MIN_ROWS`: the smallest chunk the prefill
/// projections take the BF16 tensor-core lane for. Default 16; `0` restores
/// the scalar-only floor. The lane never touches cuBLASLt's heuristic
/// first pick — the split-K it picked for these small-M shapes corrupted
/// every projection from a serve's second request on (PR #131) — it takes
/// the heuristic's in-order k-chain kernel instead, byte-identical to the
/// scalar path at any row count, and falls back to the scalar floor when
/// the runtime offers no k-chain for the shape or the library is absent.
/// Read once.
fn proj_gemm_min_rows() -> u32 {
    static MIN: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        proj_gemm_min_rows_from(
            std::env::var("ATLAS_LING_PROJ_GEMM_MIN_ROWS")
                .ok()
                .as_deref(),
        )
    })
}

/// The threshold resolution, split out for the unit test. `None` = env
/// absent (default 16); `Some("0")` = the scalar-only floor (u32::MAX);
/// `Some(other)` = that row count, clamped to >= 1.
fn proj_gemm_min_rows_from(v: Option<&str>) -> u32 {
    match v {
        None => 16,
        Some("0") => u32::MAX,
        Some(s) => s.parse::<u32>().ok().unwrap_or(16).max(1),
    }
}

/// One prefill-shaped projection: the BF16 tensor-core lane from the M
/// threshold up, the scalar GEMM below it.
#[allow(clippy::too_many_arguments)]
fn proj_gemm(
    gpu: &dyn GpuBackend,
    k: &LingKernels,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k_dim: u32,
    stream: u64,
) -> Result<()> {
    if rows >= proj_gemm_min_rows() {
        // The in-order k-chain (`_any`: the heuristic's first in-order
        // m16n8k16 kernel, tiles 256x128 / 128x256 / 128x64 / 64x256 / 64x64).
        // Byte-identical to the scalar path over the same operands and
        // never a split-K. A declined or unavailable lane falls back to the
        // scalar floor below — the heuristic's first pick is not a valid
        // option for these shapes (PR #131).
        let kchain_ok = spark_runtime::cublaslt::bf16_gemm_act_weight_t_kchain_any(
            input.0,
            weight.weight.0,
            output.0,
            [rows, n, k_dim],
            stream,
        )
        .unwrap_or_else(|e| {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "ling prefill TC lane unavailable ({e}); projections fall back to the scalar path"
                );
            }
            false
        });
        if kchain_ok {
            return Ok(());
        }
    }
    ops::dense_gemm(gpu, k.gemm_k, input, weight, output, rows, n, k_dim, stream)
}

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
    /// This layer's index among the SSM (`LinearAttention`) layers — the
    /// `SsmStatePool` ordinal both the pool state construction and the
    /// indexed decode lane address the layer by.
    ssm_ordinal: usize,
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
        ssm_ordinal: usize,
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
            ssm_ordinal,
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
                proj_gemm(
                    gpu,
                    k,
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
            proj_gemm(
                gpu,
                k,
                s.normed,
                &w.b_proj,
                s.kda_beta,
                rows,
                32,
                hidden_w as u32,
                stream,
            )?;
            proj_gemm(
                gpu,
                k,
                s.normed,
                &w.f_proj,
                s.kda_gate,
                rows,
                T as u32,
                hidden_w as u32,
                stream,
            )?;
            proj_gemm(
                gpu,
                k,
                s.normed,
                &w.g_proj,
                s.kda_o_g,
                rows,
                T as u32,
                hidden_w as u32,
                stream,
            )?;
        }
        if rows > 1 && regresident_prefill() {
            // Register-resident prefill recurrence: the FP32 planes sit in
            // lanes idle inside a KDA attention half (the MLA rows are MITM
            // between layer kinds and the MoE expert lane is only written by
            // this layer's FFN half, which is about to overwrite it anyway).
            // No persistent allocation is added.
            let plane_bytes = rows as usize * T * 4;
            let q_norm = s.mla_latent;
            let k_norm = s.mla_q_final;
            let decay = s.expert_out;
            let beta = decay.offset(plane_bytes);
            ops::kda_preprocess_regresident(
                gpu,
                k.kda_preproc_reg_k,
                s.kda_qkv,
                s.kda_gate,
                s.kda_beta,
                w.a_log,
                w.dt_bias,
                q_norm,
                k_norm,
                decay,
                beta,
                rows,
                KDA_HEADS as u32,
                KDA_HEAD_DIM as u32,
                self.lower_bound,
                stream,
            )?;
            ops::kda_recurrent_regresident(
                gpu,
                k.kda_reg_k,
                s.kda_qkv,
                q_norm,
                k_norm,
                decay,
                beta,
                state.h_state,
                s.kda_out,
                rows,
                KDA_HEADS as u32,
                KDA_HEAD_DIM as u32,
                stream,
            )?;
        } else {
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
        }
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
            proj_gemm(
                gpu,
                k,
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
            state.h_state_intermediates = (0..inter).map(|t| slab.offset(t * h_bytes)).collect();
        }
        anyhow::ensure!(
            state
                .h_state_intermediates
                .windows(2)
                .all(|w| { w[1].0.saturating_sub(w[0].0) == h_bytes as u64 }),
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
    /// The shared multi-row projection head: RMS-norm of `rows` hidden rows
    /// into `scratch.normed`, planar q|k|v projections into the
    /// `kda_preconv` plane regions, and the b/f/g gate rows into
    /// `kda_beta` / `kda_gate` / `kda_o_g` (`[rows, width]` row-major).
    /// Both K=2 verify rows and the multi-sequence decode lane start here;
    /// at rows == 2 the batch2 GEMV is bit-identical to two decode GEMVs,
    /// wider rows take the GEMM arm.
    fn project_rows(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let w = &self.weights;
        let hidden_w = s.hidden as u32;
        let rows_u32 = rows as u32;
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
            let dst = s.kda_preconv.offset(rows * (plane * T) * 2);
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
                    gpu, k.gemm_k, s.normed, proj, dst, rows_u32, T as u32, hidden_w, stream,
                )?;
            }
        }
        for (plane, proj) in [&w.b_proj, &w.f_proj, &w.g_proj].iter().enumerate() {
            let (dst, width) = match plane {
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
        Ok(())
    }

    /// The pool-indexed multi-sequence decode lane of this layer, if this
    /// step prepared a `SsmBatchView` whose width matches the dispatch —
    /// a whole-batch attention half whose conv and recurrence address the
    /// slot pool through the device slot-id stream (`ling_kda_conv_indexed`
    /// + `ling_kda_recurrent_indexed`). Returns `None` when the step must
    /// fall back to the per-sequence loop (the prepared view's layer row
    /// count disagreeing with the dispatch is a hard error, not a silent
    /// fallback: the owner of `ctx.ssm_batch` — `prepare_runtime` — and the
    /// decoder agree on eligibility before either runs).
    fn indexed_decode_lane(
        &self,
        num_seqs: usize,
        active_seqs: usize,
        ctx: &ForwardContext,
    ) -> Result<Option<(KdaIndexedShape, SsmBatchLayer)>> {
        let Some(batch) = ctx.ssm_batch else {
            return Ok(None);
        };
        let layer = batch
            .layer(self.ssm_ordinal)
            .map_err(|e| anyhow::anyhow!("ling KDA layer {} slot view: {e:#}", self.ssm_ordinal))?;
        if layer.rows() as usize != num_seqs {
            anyhow::bail!(
                "ling KDA: prepared SSM batch rows {} disagree with a \
                 {num_seqs}-row decode dispatch",
                layer.rows()
            );
        }
        if active_seqs != num_seqs || !(2..=8).contains(&num_seqs) {
            return Ok(None);
        }
        let Some(shape) = KdaIndexedShape::select(num_seqs, KDA_HEADS, KDA_HEAD_DIM, 4) else {
            return Ok(None);
        };
        static LANE_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !LANE_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                rows = num_seqs,
                ordinal = self.ssm_ordinal,
                "ling KDA pool-indexed decode lane selected (device slot-id \
                 addressing; per-sequence rows are the fallback)"
            );
        }
        Ok(Some((shape, layer)))
    }

    /// The attention half of the pool-indexed lane: the `project_rows`
    /// head over all rows, `kda_pack_qkv` staging into token-major q|k|v
    /// rows, the conv + recurrent indexed pair (pool slots addressed by
    /// the device slot-id stream — every kernel argument is a fixed
    /// address), then gated norm, o_proj and the residual add for all
    /// rows.
    #[allow(clippy::too_many_arguments)]
    fn attention_half_indexed(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        rows: usize,
        shape: KdaIndexedShape,
        layer: SsmBatchLayer,
        stream: u64,
    ) -> Result<()> {
        let s = &self.scratch;
        let k = &self.kernels;
        let w = &self.weights;
        let hidden_w = s.hidden as u32;
        let rows_u32 = rows as u32;
        self.project_rows(gpu, hidden, rows, stream)?;
        ops::kda_pack_qkv(
            gpu,
            k.kda_pack_qkv_k,
            s.kda_preconv,
            s.kda_qkv,
            rows_u32,
            T as u32,
            stream,
        )?;
        let qkv_bytes = rows * 3 * T * 2;
        let conv = KdaIndexedConv {
            input: KdaBuffer {
                ptr: s.kda_qkv,
                bytes: qkv_bytes,
            },
            weight: KdaBuffer {
                ptr: w.conv.weight,
                bytes: 3 * T * 4 * 2,
            },
            bias: None,
            output: KdaBuffer {
                ptr: s.kda_qkv,
                bytes: qkv_bytes,
            },
            input_stride: 3 * T,
            output_stride: 3 * T,
        };
        let recurrent = KdaIndexedRecurrent {
            qkv: KdaBuffer {
                ptr: s.kda_qkv,
                bytes: qkv_bytes,
            },
            gate: KdaBuffer {
                ptr: s.kda_gate,
                bytes: rows * T * 2,
            },
            beta: KdaBuffer {
                ptr: s.kda_beta,
                bytes: rows * 32 * 2,
            },
            a_log: KdaBuffer {
                ptr: w.a_log,
                bytes: 32 * 4,
            },
            dt_bias: KdaBuffer {
                ptr: w.dt_bias,
                bytes: T * 4,
            },
            output: KdaBuffer {
                ptr: s.kda_out,
                bytes: rows * T * 2,
            },
            lower_bound: self.lower_bound,
        };
        // The pair could validate once per layer construction; it is
        // cheap, host-only, and guarantees a mid-capture failure cannot
        // come from a drifted descriptor, so it stays per-step like the
        // GLM lane's `prepare_indexed_core`.
        ops::validate_kda_indexed_pair(
            shape,
            layer,
            k.kda_conv_indexed_k,
            k.kda_recurrent_indexed_k,
            conv,
            recurrent,
        )?;
        ops::kda_conv_indexed(gpu, k.kda_conv_indexed_k, shape, layer, conv, stream)?;
        ops::kda_recurrent_indexed(
            gpu,
            k.kda_recurrent_indexed_k,
            shape,
            layer,
            recurrent,
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
        // o_proj: T -> hidden, then hidden += attn_out (all rows).
        if rows == 2 {
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
                gpu, k.gemm_k, s.kda_o, &w.o_proj, s.attn_out, rows_u32, hidden_w, T as u32, stream,
            )?;
        }
        ops::residual_add(
            gpu,
            k.residual_add_k,
            hidden,
            s.attn_out,
            rows_u32 * hidden_w,
            stream,
        )
    }

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
        // Attention half. Projections via the shared multi-row staging
        // (normed rows + planar q|k|v + b/f/g gates), then per-row conv
        // launches — the multi-row prefill conv folds all rows at once and
        // gives no mid-window conv state.
        self.project_rows(gpu, hidden, rows, stream)?;
        // The K=2 verify keeps the batch2 GEMV for the o_proj as well
        // (bit-identical to the two decode GEMVs; see `project_rows`).
        let batch2 = rows == 2;
        let h_bytes = KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4;
        let conv_bytes = 3 * T * 4 * 4;
        for t in 0..rows {
            for plane in 0..3 {
                ops::conv1d_update_prefill(
                    gpu,
                    k.conv_prefill_k,
                    spark_runtime::gpu::KernelHandle(0),
                    state.conv_state.offset(plane * T * 4 * 4),
                    s.kda_preconv.offset((rows * (plane * T) + t * T) * 2),
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
                gpu, k.gemm_k, s.kda_o, &w.o_proj, s.attn_out, rows_u32, hidden_w, T as u32, stream,
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

    /// Whole-batch decode of one layer: when the dispatcher prepared an
    /// `SsmBatchView` (see `indexed_decode_lane`), every row's conv +
    /// recurrence run through the two pool-indexed kernels in one launch
    /// pair, addressing slots on device — no per-sequence pointers reach
    /// kernel arguments. Otherwise the per-sequence loop over the
    /// pool-backed `SsmLayerState`s (the previous behavior, bit-identical)
    /// runs instead; the loop is bounded by `active_seqs`, so padding rows
    /// never touch KDA state (their hidden rows are zeroed and discarded,
    /// and the dummy `alloc_state` regions some dispatch paths create for
    /// them stay unread).
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        active_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            states.len() >= num_seqs,
            "ling KDA: state batch is truncated ({}/{} rows)",
            states.len(),
            num_seqs
        );
        if let Some((shape, batch_layer)) = self.indexed_decode_lane(num_seqs, active_seqs, ctx)? {
            let rows = num_seqs;
            self.attention_half_indexed(ctx.gpu, hidden, rows, shape, batch_layer, stream)?;
            self.ffn_half(ctx.gpu, hidden, rows as u32, stream)?;
            return Ok(());
        }
        // Per-sequence fallback (the `TransformerLayer` default loop,
        // bounded by the active rows; padding rows skip this layer).
        let row_bytes = self.scratch.hidden * 2;
        for i in 0..active_seqs.min(num_seqs) {
            let mut block_table = block_tables.get(i).cloned().unwrap_or_default();
            // Phase 6.1: per-seq disk_block_ids aren't threaded through
            // this fallback (same stubs as the trait default).
            let mut stub_disk = Vec::<u32>::new();
            let mut stub_last_offloaded = Vec::<u32>::new();
            self.decode(
                hidden.offset(i * row_bytes),
                residual.offset(i * row_bytes),
                states[i],
                kv_cache,
                seq_lens.get(i).copied().unwrap_or(0),
                &mut block_table,
                &mut stub_disk,
                &mut stub_last_offloaded,
                ctx,
                stream,
            )?;
        }
        Ok(())
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

    /// Graphable under the family switch
    /// (`crate::model::graph_flags::ling_decode_graph`, the graph slice's
    /// flip): every state address this layer's decode touches is a fixed
    /// `SsmStatePool` function of `(ssm_ordinal, slot)` — the indexed lane
    /// takes the slot-id stream refreshed before any replay, and the
    /// per-seq lane's h/conv pointers were fixed at slot claim — and the
    /// MTP verify's rollback snapshots attach at claim
    /// (`attach_verify_rollback`), so no decode or verify step allocates
    /// (the family doc's capture contract, `model/bailing_c1.rs`). With the
    /// switch off the family stays eager, as before.
    fn decode_graph_unsupported(&self) -> bool {
        !crate::model::graph_flags::ling_decode_graph()
    }
}

#[cfg(test)]
mod proj_gemm_threshold_tests {
    use super::proj_gemm_min_rows_from;

    #[test]
    fn default_is_16_and_zero_restores_the_floor() {
        assert_eq!(proj_gemm_min_rows_from(None), 16);
        assert_eq!(proj_gemm_min_rows_from(Some("0")), u32::MAX);
    }

    #[test]
    fn explicit_values_engage_at_their_row_count() {
        assert_eq!(proj_gemm_min_rows_from(Some("16")), 16);
        assert_eq!(proj_gemm_min_rows_from(Some("3")), 3);
        assert_eq!(proj_gemm_min_rows_from(Some("8192")), 8192);
    }

    #[test]
    fn unparsable_falls_back_to_default() {
        assert_eq!(proj_gemm_min_rows_from(Some("not-a-number")), 16);
    }

    /// The lane must never route back through the heuristic first pick
    /// (the split-K path that corrupted small-M projections, PR #131).
    #[test]
    fn lane_is_kchain_never_the_heuristic_first_pick() {
        let src = include_str!("kda.rs");
        assert!(src.contains("bf16_gemm_act_weight_t_kchain_any"));
        let body = src
            .splitn(2, "fn proj_gemm(")
            .nth(1)
            .unwrap()
            .split("\n}\n")
            .next()
            .unwrap();
        assert!(
            !body.contains("ops::bf16_gemm("),
            "proj_gemm must never call the heuristic first-pick lane"
        );
    }
}
