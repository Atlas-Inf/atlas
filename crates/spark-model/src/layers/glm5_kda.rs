// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 KDA recurrent block for the conservative GB10 bring-up path.
mod forward_attention;
mod forward_attention_entry;
mod forward_attention_qkv;
mod forward_ffn;
mod forward_recurrent;
mod forward_recurrent_owners;
mod hc;
mod indexed_core;
mod init;
mod multi_seq;
mod profile;
mod projection;
mod recurrent;
mod shared_cache;

use std::mem::size_of;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::FfnComponent;
use crate::layers::ops;
use crate::layers::qwen3_attention::HcWeights;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::DenseWeight;

pub(crate) use multi_seq::enabled as kda_multi_seq_enabled;
pub use projection::{Glm5KdaWeights, Glm5Projection};

/// `ATLAS_GLM_KDA_BATCHED_FFN=1`: one FFN pass over a verify block's or a
/// decode batch's rows, so one EP reduce of every row instead of one a row.
/// Both ranks must run the same value (`model::startup_parity`).
pub(crate) fn kda_batched_ffn_enabled() -> bool {
    std::env::var("ATLAS_GLM_KDA_BATCHED_FFN").ok().as_deref() == Some("1")
}

fn verify_batched_conv_snapshot_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_BATCHED_CONV_SNAPSHOT")
        .ok()
        .as_deref()
        == Some("1")
}

fn verify_batched_recurrent_snapshot_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_BATCHED_RECURRENT_SNAPSHOT")
        .ok()
        .as_deref()
        == Some("1")
}

fn verify_fused_qkv_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_FUSED_QKV").ok().as_deref() == Some("1")
}

fn verify_fused_dense_pairs_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_FUSED_DENSE_PAIRS")
        .ok()
        .as_deref()
        == Some("1")
}

fn verify_fused_dense_triple_enabled() -> bool {
    std::env::var("ATLAS_GLM_K5_FUSED_DENSE_TRIPLE")
        .ok()
        .as_deref()
        == Some("1")
}

/// Exact K=5-only seam: exchange the remote row-parallel KDA O projection and
/// combine it inside mHC post-mixing. The launcher retains a fallback switch.
/// Both ranks must run the same value (`model::startup_parity`): the exchange
/// stands in for the TP all-reduce.
pub(crate) fn verify_fused_tp_hc_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_K5_FUSED_TP_HC").ok().as_deref() == Some("1"))
}

fn verify_fused_moe_hc_check_once() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    *ENABLED.get_or_init(|| {
        std::env::var("ATLAS_GLM_K5_FUSED_MOE_HC_CHECK")
            .ok()
            .as_deref()
            == Some("1")
    }) && !CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed)
}

pub struct Glm5KdaLayer {
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    weights: Glm5KdaWeights,
    ffn: FfnComponent,
    hc: HcWeights,
    layer_idx: usize,
    ssm_ordinal: usize,
    indexed_trace_logged: std::sync::atomic::AtomicBool,
    hidden_size: usize,
    heads: usize,
    dim: usize,
    conv_width: usize,
    lower_bound: f32,
    h_state_bytes: usize,
    conv_state_bytes: usize,
    rms_norm_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    dense_gemv_batchm_k: KernelHandle,
    dense_gemv_batch5_k: KernelHandle,
    dense_gemv_batch5_dual_k: KernelHandle,
    dense_gemv_batch5_triple_n_k: KernelHandle,
    dense_gemv_batchm_dual_k: KernelHandle,
    dense_gemv_batchm_triple_n_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemv_sw_k: KernelHandle,
    w4a16_gemv_batch2_k: KernelHandle,
    w4a16_gemv_batch3_k: KernelHandle,
    w4a16_gemv_batch5_qkv_k: KernelHandle,
    w4a16_gemv_batchm: W4a16BatchmTiers,
    w4a16_gemm_k: KernelHandle,
    w4a16_gemm_t_m128_k: KernelHandle,
    dense_gemm_k: KernelHandle,
    dense_gemm_pipelined_k: KernelHandle,
    /// `dense_gemm_bf16_pipelined_triple_n` (prefill beta | f_a | g_a). On by
    /// default, unlike the opt-in `ATLAS_GLM_K5_FUSED_*` verify flags: zero
    /// only when the kernel is absent or `ATLAS_GLM_KDA_FUSED_SMALL_PREFILL`
    /// is exactly `0` (a kill switch, as `ATLAS_KDA_VERIFY_OWNERS`).
    dense_gemm_pipelined_triple_n_k: KernelHandle,
    conv_prefill_k: KernelHandle,
    conv_prefill_tp_k: KernelHandle,
    conv_prefill_tp_snap_k: KernelHandle,
    conv_indexed_k: KernelHandle,
    pack_k: KernelHandle,
    recurrent_k: KernelHandle,
    recurrent_indexed_k: KernelHandle,
    recurrent_verify_snap_k: KernelHandle,
    /// Owner-batched, register-resident twin of `recurrent_verify_snap_k`
    /// (bit-identical); `ATLAS_KDA_VERIFY_OWNERS=0` disables.
    recurrent_verify_owners_k: KernelHandle,
    recurrent_verify_rec_k: KernelHandle,
    preprocess_regresident_k: KernelHandle,
    recurrent_regresident_k: KernelHandle,
    register_resident_prefill: bool,
    gated_norm_k: KernelHandle,
    hc_expand_k: KernelHandle,
    hc_pre_k: KernelHandle,
    hc_pre_from_raw_mix_k: KernelHandle,
    hc_pre_mix_k: KernelHandle,
    hc_post_k: KernelHandle,
    hc_post_bf16_add_k: KernelHandle,
    hc_post_moe_blend_k: KernelHandle,
    hc_contract_k: KernelHandle,
}

/// Non-owning continuation for an immediate FFN call on the same layer/context/stream.
///
/// This carries the original profiling timer, not a new phase timer. It is not an
/// owner-safe snapshot: `normed` aliases singleton `norm_output`, and FFN/post-mHC
/// still reads `hc_streams`, `hc_post`, and `hc_comb` from `ctx.buffers`.
/// Another owner's attention phase may overwrite all four. Pair traversal must
/// provide disjoint checked views or preserve/restore their contents first.
struct FfnPhase {
    hidden: DevicePtr,
    normed: DevicePtr,
    tokens: usize,
    decode: bool,
    capture_verify_intermediates: bool,
    profile_timer: Option<std::time::Instant>,
}

impl Glm5KdaLayer {
    fn forward_inner(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        tokens: usize,
        decode: bool,
        capture_verify_intermediates: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let phase = self.forward_attention(
            hidden,
            state,
            tokens,
            decode,
            capture_verify_intermediates,
            ctx,
            stream,
        )?;
        self.forward_ffn(phase, ctx, stream)
    }
}

impl TransformerLayer for Glm5KdaLayer {
    fn decode_glm_long_owners(
        &self,
        owners: &mut [crate::layer::glm_long_owner::GlmLongOwner<'_>],
        _cache: &mut PagedKvCache,
        stage: &crate::layer::glm_long_owner::GlmLongStage,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::layer::glm_long_owner as owner;
        // Every row-local stage (mHC, norms, projections, TP reduce) runs once
        // over all owners' rows; only the recurrence is per owner.
        let rows = owner::owner_rows(owners)?;
        let owners_n = owners.len();
        let phase = self.forward_attention_rows(
            ctx.buffers.hidden_states(),
            owners_n * rows,
            false,
            true,
            ctx,
            stream,
            &mut |projected, g1, beta| {
                self.forward_recurrent_owners(projected, g1, beta, owners, rows, ctx, stream)
            },
        )?;
        let ffn_out = owner::ffn_per_owner(&self.ffn, owners_n, rows, stage, ctx, stream)?;
        self.forward_ffn_post(phase, ffn_out, None, ctx, stream)
    }

    fn prefill_with_glm_passengers(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _seq_len_start: usize,
        passengers: &mut [crate::layer::glm_long_owner::GlmLongOwner<'_>],
        _cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rows = crate::layer::glm_long_owner::owner_rows(passengers)?;
        ensure!(
            ctx.midchunk_capture.is_none(),
            "GLM fused prefill + verify does not split the chunk recurrence"
        );
        let state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("GLM-5 KDA expected SsmLayerState"))?;
        ensure!(!state.h_is_f16, "GLM-5 KDA requires FP32 recurrent state");
        // Prefill kernels over every row; only the recurrences split.
        let phase = self.forward_attention_rows(
            hidden,
            num_tokens + passengers.len() * rows,
            false,
            false,
            ctx,
            stream,
            &mut |projected, g1, beta| {
                self.forward_recurrent_passengers(
                    projected, g1, beta, state, num_tokens, passengers, rows, ctx, stream,
                )
            },
        )?;
        self.forward_ffn(phase, ctx, stream)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

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
        self.forward_inner(hidden, state, 1, true, false, ctx, stream)
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
        self.forward_inner(hidden, state, num_tokens, false, false, ctx, stream)
    }

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
        // Speculative verify rows are one temporal sequence, not independent
        // requests. Batch the stateless projections/mHC work, but advance the
        // KDA recurrent state row-by-row and capture rollback intermediates.
        self.forward_inner(hidden, state, num_tokens, false, true, ctx, stream)
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        _active_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let independent = crate::model::glm_independent::selected(ctx, num_seqs)?;
        if independent
            || crate::model::glm_c4::batched_kda_rows(
                num_seqs,
                multi_seq::enabled(),
                crate::model::glm_c4::enabled(&ctx.config.model_type),
            )?
        {
            if independent {
                crate::model::glm_independent::validate_positions(
                    seq_lens.iter().copied(),
                    num_seqs,
                    ctx.levers.max_decode_seqs as usize,
                )?;
            } else if num_seqs == 4 {
                crate::model::glm_c4::validate_positions(seq_lens.iter().copied(), 4)?;
            }
            self.decode_multi_seq_inner(hidden, num_seqs, states, ctx, stream)
        } else {
            let h = ctx.config.hidden_size;
            for i in 0..num_seqs {
                let mut bt = block_tables[i].clone();
                let mut disk = Vec::new();
                let mut last_offloaded = Vec::new();
                self.decode(
                    hidden.offset(i * h * 2),
                    residual.offset(i * h * 2),
                    states[i],
                    kv_cache,
                    seq_lens[i],
                    &mut bt,
                    &mut disk,
                    &mut last_offloaded,
                    ctx,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_state = gpu.alloc(self.h_state_bytes)?;
        gpu.memset(h_state, 0, self.h_state_bytes)?;
        let conv_state = gpu.alloc(self.conv_state_bytes)?;
        gpu.memset(conv_state, 0, self.conv_state_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            gdn_commit_qkv: spark_runtime::gpu::DevicePtr::NULL,
            gdn_commit_gb: spark_runtime::gpu::DevicePtr::NULL,
            gdn_commit_pending: false,
            gdn_fuse_n: spark_runtime::gpu::DevicePtr::NULL,
            kda_records: spark_runtime::gpu::DevicePtr::NULL,
            conv_state_intermediates: Vec::new(),
            h_is_f16: false,
            h_prefill_stage: None,
            ple: None,
        }))
    }
}
