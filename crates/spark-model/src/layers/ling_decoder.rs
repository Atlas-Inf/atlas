// SPDX-License-Identifier: AGPL-3.0-only

//! The ordinary one-token Ling decoder block.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState, SsmLayerState};
use crate::layers::FfnComponent;
use crate::layers::ling_kda::{
    LING_KDA_HEADS, LING_KDA_HIDDEN_SIZE, LING_KDA_PROJECTION_SIZE, LingKdaLayer,
    LingKdaPrefillScratch, LingKdaScratch, LingKdaState,
};
use crate::layers::ling_mla::{
    LingMlaConfig, LingMlaKernels, LingMlaScratch, LingMlaWeights, MLA_TILE_ROWS,
};
use crate::layers::ling_projection::{LingProjection, LingProjectionScratch};
use crate::layers::ops;

struct LingDenseFfn {
    gate: LingProjection,
    up: LingProjection,
    down: LingProjection,
    projection: LingProjectionScratch,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    output: DevicePtr,
    intermediate: usize,
    silu_k: spark_runtime::gpu::KernelHandle,
}

impl LingDenseFfn {
    fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.gate
            .forward(ctx.gpu, input, self.gate_out, 1, self.projection, stream)?;
        self.up
            .forward(ctx.gpu, input, self.up_out, 1, self.projection, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.silu_k,
            self.gate_out,
            self.up_out,
            self.gate_out,
            self.intermediate as u32,
            stream,
        )?;
        self.down.forward(
            ctx.gpu,
            self.gate_out,
            self.output,
            1,
            self.projection,
            stream,
        )?;
        Ok(self.output)
    }

    /// The shared arena supplies batched activation and scale storage for both formats.
    fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            num_tokens > 0,
            "Ling dense FFN prefill requires at least one token"
        );
        let sizes = ctx.buffers.sizes();
        let projection = LingProjectionScratch::new(
            ctx.buffers.fp8_act(),
            ctx.buffers.fp8_act_bytes(),
            ctx.buffers.fp8_act_scale(),
            sizes.fp8_act_scale,
        );
        let bf16 = std::mem::size_of::<u16>();
        let intermediate_bytes = num_tokens
            .checked_mul(self.intermediate)
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("Ling dense FFN intermediate size overflow"))?;
        let hidden_bytes = num_tokens
            .checked_mul(ctx.config.hidden_size)
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("Ling dense FFN hidden size overflow"))?;
        ensure!(
            intermediate_bytes <= sizes.expert_gate_out
                && intermediate_bytes <= sizes.expert_up_out,
            "Ling dense FFN prefill arena is too small for {num_tokens} rows of {} channels",
            self.intermediate
        );
        ensure!(
            hidden_bytes <= sizes.moe_output,
            "Ling dense FFN output arena is too small for {num_tokens} rows"
        );
        let rows = u32::try_from(num_tokens)
            .map_err(|_| anyhow::anyhow!("Ling dense FFN row count exceeds CUDA launch range"))?;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        self.gate
            .forward(ctx.gpu, input, gate_out, rows, projection, stream)?;
        self.up
            .forward(ctx.gpu, input, up_out, rows, projection, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.silu_k,
            gate_out,
            up_out,
            gate_out,
            rows.checked_mul(self.intermediate as u32)
                .ok_or_else(|| anyhow::anyhow!("Ling dense FFN activation size overflow"))?,
            stream,
        )?;
        self.down.forward(
            ctx.gpu,
            gate_out,
            ctx.buffers.moe_output(),
            rows,
            projection,
            stream,
        )?;
        Ok(())
    }
}

enum LingAttention {
    Kda {
        layer: LingKdaLayer,
        scratch: LingKdaScratch,
        projection: LingProjectionScratch,
    },
    Mla {
        weights: LingMlaWeights,
        config: LingMlaConfig,
        kernels: LingMlaKernels,
        scratch: LingMlaScratch,
    },
}

/// One normal Ling decoder block: input norm, attention, post-attention norm,
/// and one routed or dense FFN.  Scratch is layer-owned and therefore remains
/// at stable addresses during CUDA graph capture; sequence state is not owned
/// here and continues to live in Atlas's slot pool.
pub struct LingDecoderLayer {
    input_norm: crate::weight_map::DenseWeight,
    post_attn_norm: crate::weight_map::DenseWeight,
    attention: LingAttention,
    ffn: FfnComponent,
    dense_ffn: Option<LingDenseFfn>,
    rms_norm_residual_k: spark_runtime::gpu::KernelHandle,
    residual_add_rms_norm_k: spark_runtime::gpu::KernelHandle,
    residual_add_k: spark_runtime::gpu::KernelHandle,
    hidden_size: usize,
    h_state_bytes: usize,
    conv_state_bytes: usize,
    /// Compact index into the paged KV cache's full-attention pools.  Ling's
    /// transformer indices include 35 KDA layers, while the cache allocates
    /// only the seven MLA layers; keeping this separate prevents a layer 5,
    /// 11, ... transformer index from indexing past the seven cache pools.
    attention_layer_idx: Option<usize>,
}

impl LingDecoderLayer {
    fn alloc_projection_scratch(
        gpu: &dyn GpuBackend,
        max_input_width: usize,
    ) -> Result<LingProjectionScratch> {
        let input_width = max_input_width.max(LING_KDA_PROJECTION_SIZE).div_ceil(128) * 128;
        let scale_bytes = input_width.div_ceil(128) * 4;
        let act = gpu.alloc(input_width)?;
        let scales = gpu.alloc(scale_bytes)?;
        Ok(LingProjectionScratch::new(
            act,
            input_width,
            scales,
            scale_bytes,
        ))
    }

    fn alloc_kda_scratch(gpu: &dyn GpuBackend) -> Result<LingKdaScratch> {
        let bf16 = |elements: usize| gpu.alloc(elements * 2);
        Ok(LingKdaScratch::new(
            bf16(3 * LING_KDA_PROJECTION_SIZE)?,
            bf16(3 * LING_KDA_PROJECTION_SIZE)?,
            bf16(LING_KDA_PROJECTION_SIZE)?,
            bf16(LING_KDA_HEADS)?,
            bf16(LING_KDA_PROJECTION_SIZE)?,
            bf16(LING_KDA_PROJECTION_SIZE)?,
            bf16(LING_KDA_PROJECTION_SIZE)?,
            bf16(LING_KDA_HIDDEN_SIZE)?,
        ))
    }

    fn alloc_mla_scratch(
        gpu: &dyn GpuBackend,
        config: LingMlaConfig,
        projection: LingProjectionScratch,
    ) -> Result<LingMlaScratch> {
        let bf16 = |elements: usize| gpu.alloc(MLA_TILE_ROWS * elements * 2);
        let qk = config.qk_head_dim();
        let q = config.q_proj_rows();
        let kv_a = config.kv_lora_rank + config.qk_rope_head_dim;
        let kv_b = config.kv_b_rows();
        let compact_v = config.num_heads * config.v_head_dim;
        let padded = config.num_heads * qk;
        Ok(LingMlaScratch {
            projection,
            q_full: bf16(q)?,
            kv_a: bf16(kv_a)?,
            kv_norm: bf16(config.kv_lora_rank)?,
            q_rope: bf16(config.num_heads * config.qk_rope_head_dim)?,
            k_rope: bf16(config.qk_rope_head_dim)?,
            kv_b: bf16(kv_b)?,
            k_token: bf16(padded)?,
            v_compact: bf16(compact_v)?,
            v_padded: bf16(padded)?,
            attn_padded: bf16(padded)?,
            attn_compact: bf16(compact_v)?,
            attn_gated: bf16(compact_v)?,
            gate: bf16(config.num_heads)?,
        })
    }

    fn alloc_kda_state(
        gpu: &dyn GpuBackend,
        h_bytes: usize,
        conv_bytes: usize,
    ) -> Result<Box<dyn LayerState>> {
        let expected_h = LingKdaState::h_state_bytes();
        let expected_conv = 3 * LingKdaState::conv_state_bytes();
        ensure!(
            h_bytes == expected_h && conv_bytes == expected_conv,
            "Ling KDA SSM pool geometry is h={h_bytes}, conv={conv_bytes}; expected h={expected_h}, conv={expected_conv}"
        );
        let h_state = gpu.alloc(h_bytes)?;
        let conv_state = gpu.alloc(conv_bytes)?;
        gpu.memset(h_state, 0, h_bytes)?;
        gpu.memset(conv_state, 0, conv_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state,
            conv_state,
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

    fn state_kda(&self, state: &mut dyn LayerState, config: &ModelConfig) -> Result<LingKdaState> {
        let state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Ling KDA requires Atlas SsmLayerState"))?;
        ensure!(!state.h_is_f16, "Ling KDA requires an FP32 recurrent state");
        ensure!(
            state.h_prefill_stage.is_none(),
            "Ling KDA does not support an f16-sized staged SSM pool"
        );
        let plane = LingKdaState::conv_state_bytes();
        ensure!(
            config.ssm_conv_state_bytes() >= 3 * plane,
            "Ling KDA conv state is smaller than q/k/v planes"
        );
        Ok(LingKdaState {
            q_conv: state.conv_state,
            k_conv: state.conv_state.offset(plane),
            v_conv: state.conv_state.offset(2 * plane),
            h: state.h_state,
        })
    }

    fn metadata_row(meta: AttnMetadataDev, row: usize) -> AttnMetadataDev {
        let blocks = row * meta.max_blocks_per_seq as usize * 4;
        AttnMetadataDev {
            positions: meta.positions.offset(row * 4),
            positions_h: meta.positions_h.offset(row * 4),
            positions_w: meta.positions_w.offset(row * 4),
            slot: meta.slot.offset(row * 8),
            seq_len: meta.seq_len.offset(row * 4),
            block_table: meta.block_table.offset(blocks),
            max_blocks_per_seq: meta.max_blocks_per_seq,
            num_seqs: 1,
            seq_slot: if meta.seq_slot.is_null() {
                DevicePtr::NULL
            } else {
                meta.seq_slot.offset(row * 4)
            },
            moe_row_adapter: if meta.moe_row_adapter.is_null() {
                DevicePtr::NULL
            } else {
                meta.moe_row_adapter.offset(row * 4)
            },
        }
    }

    fn run_one(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        write_kv: bool,
        metadata: Option<AttnMetadataDev>,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h,
            eps,
            stream,
        )?;

        let attn = match &self.attention {
            LingAttention::Kda {
                layer,
                scratch,
                projection,
            } => {
                let kda_state = self.state_kda(state, ctx.config)?;
                let out = layer.forward_attention(
                    ctx.gpu,
                    normed,
                    kda_state,
                    *scratch,
                    *projection,
                    *projection,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    out,
                    ctx.buffers.attn_output(),
                    self.hidden_size * 2,
                    stream,
                )?;
                ctx.buffers.attn_output()
            }
            LingAttention::Mla {
                weights,
                config,
                kernels,
                scratch,
                ..
            } => {
                let metadata = metadata.ok_or_else(|| {
                    anyhow::anyhow!("Ling MLA decode requires uploaded AttnMetadataDev")
                })?;
                weights.forward_rows(
                    *config,
                    *kernels,
                    ctx.gpu,
                    kv_cache,
                    self.attention_layer_idx
                        .expect("Ling MLA layer must have a compact KV-cache index"),
                    metadata,
                    normed,
                    ctx.buffers.attn_output(),
                    *scratch,
                    1,
                    u32::from(!write_kv),
                    None,
                    stream,
                )?;
                ctx.buffers.attn_output()
            }
        };

        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            attn,
            &self.post_attn_norm,
            normed2,
            residual,
            1,
            h,
            eps,
            stream,
        )?;
        let ffn_out = match &self.dense_ffn {
            Some(ffn) => ffn.forward(normed2, ctx, stream)?,
            None => self.ffn.forward(normed2, ctx, stream)?,
        };
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, ffn_out, h, stream)
    }

    fn run_kda_rows(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        states: &[LingKdaState],
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        let plane_bytes = num_tokens * LING_KDA_PROJECTION_SIZE * 2;
        let sizes = ctx.buffers.sizes();
        ensure!(
            4 * plane_bytes <= sizes.ssm_qkvz && 4 * plane_bytes <= sizes.ssm_deinterleaved,
            "Ling KDA prefill arena is too small for {num_tokens} rows"
        );
        ensure!(
            num_tokens * LING_KDA_HEADS * 2 <= sizes.ssm_ba,
            "Ling KDA beta arena is too small for {num_tokens} rows"
        );
        ensure!(
            num_tokens * self.hidden_size * 2 <= sizes.attn_output,
            "Ling KDA output arena is too small for {num_tokens} rows"
        );
        let scratch = LingKdaPrefillScratch {
            qkv_planes: ctx.buffers.ssm_qkvz(),
            qkv: ctx.buffers.ssm_qkvz(),
            conv_planes: ctx.buffers.ssm_deinterleaved(),
            raw_gate: ctx.buffers.ssm_qkvz().offset(3 * plane_bytes),
            raw_beta: ctx.buffers.ssm_ba(),
            gate: ctx.buffers.ssm_deinterleaved().offset(3 * plane_bytes),
            recurrent: ctx.buffers.ssm_deinterleaved(),
            gated: ctx.buffers.ssm_qkvz(),
            output: ctx.buffers.attn_output(),
        };
        let projection = LingProjectionScratch::new(
            ctx.buffers.fp8_act(),
            ctx.buffers.fp8_act_bytes(),
            ctx.buffers.fp8_act_scale(),
            sizes.fp8_act_scale,
        );
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            num_tokens as u32,
            self.hidden_size as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        // The A16 prompt path can retain KDA state in registers through a tile.
        // The original FP8 checkpoint keeps its scalar reduction order.
        let a16 = ctx
            .config
            .quantization_config
            .as_ref()
            .is_some_and(|q| q.quant_method != "fp8");
        let fp32_workspace = (a16
            && states.len() == 1
            && num_tokens >= 32
            && sizes.expert_down_out >= 4 * plane_bytes + num_tokens * LING_KDA_HEADS * 4)
            .then(|| ctx.buffers.expert_down_out());
        let attn = match &self.attention {
            LingAttention::Kda { layer, .. } => layer.forward_prefill(
                ctx.gpu,
                normed,
                num_tokens,
                states,
                fp32_workspace,
                scratch,
                projection,
                stream,
            )?,
            LingAttention::Mla { .. } => unreachable!("KDA prefill called on MLA layer"),
        };
        self.finish_prefill_ffn(hidden, residual, attn, num_tokens, ctx, stream)
    }

    fn finish_prefill_ffn(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        attn: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        let h_bytes = self.hidden_size * 2;
        // The attention result is complete before the post-attention norm, so
        // the whole token matrix can be normalized in one launch.  This also
        // makes the arena's `norm_output` a stable input for the batched FFN.
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            attn,
            &self.post_attn_norm,
            ctx.buffers.norm_output(),
            residual,
            num_tokens as u32,
            self.hidden_size as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;

        let batched = if let Some(ffn) = &self.dense_ffn {
            ffn.forward_batched(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
            true
        } else if let FfnComponent::LingMoe(ffn) = &self.ffn {
            if ffn.supports_batched_prefill() {
                self.ffn
                    .forward_batched(ctx.buffers.norm_output(), num_tokens, ctx, stream)?;
                true
            } else {
                false
            }
        } else {
            false
        };
        if batched {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                ctx.buffers.moe_output(),
                (num_tokens * self.hidden_size) as u32,
                stream,
            )?;
            return Ok(());
        }

        // Strict FP8 dense and the original paged FP8 Ling path retain their
        // one-token ordering/host staging fallback, but consume the already
        // computed row from the matrix norm output.
        for token in 0..num_tokens {
            let hidden_t = hidden.offset(token * h_bytes);
            let ffn_input = ctx.buffers.norm_output().offset(token * h_bytes);
            let ffn_out = match &self.dense_ffn {
                Some(ffn) => ffn.forward(ffn_input, ctx, stream)?,
                None => self.ffn.forward(ffn_input, ctx, stream)?,
            };
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden_t,
                ffn_out,
                self.hidden_size as u32,
                stream,
            )?;
        }
        Ok(())
    }

    fn run_mla_prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        kv_write_start: usize,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        let meta = ctx
            .attn_metadata
            .ok_or_else(|| anyhow::anyhow!("Ling MLA prefill requires AttnMetadataDev"))?;
        ensure!(meta.num_seqs == 1, "Ling MLA prefill expects one sequence");
        ensure!(
            kv_write_start <= num_tokens,
            "Ling MLA prefill write floor {kv_write_start} exceeds {num_tokens} rows"
        );
        let LingAttention::Mla {
            weights,
            config,
            kernels,
            scratch,
            ..
        } = &self.attention
        else {
            unreachable!("MLA prefill called on KDA layer");
        };
        let mut scratch = *scratch;
        scratch.projection = LingProjectionScratch::new(
            ctx.buffers.fp8_act(),
            ctx.buffers.fp8_act_bytes(),
            ctx.buffers.fp8_act_scale(),
            ctx.buffers.sizes().fp8_act_scale,
        );
        let normed = ctx.buffers.norm_output();
        let attn = ctx.buffers.attn_output();
        let h_bytes = self.hidden_size * 2;
        ensure!(
            num_tokens * h_bytes <= ctx.buffers.sizes().attn_output,
            "Ling MLA prefill exceeds attention arena"
        );
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            num_tokens as u32,
            self.hidden_size as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        for first in (0..num_tokens).step_by(MLA_TILE_ROWS) {
            let rows = MLA_TILE_ROWS.min(num_tokens - first);
            let metadata = AttnMetadataDev {
                positions: meta.positions.offset(first * 4),
                positions_h: meta.positions_h.offset(first * 4),
                positions_w: meta.positions_w.offset(first * 4),
                slot: meta.slot.offset(first * 8),
                ..meta
            };
            weights.forward_rows(
                *config,
                *kernels,
                ctx.gpu,
                kv_cache,
                self.attention_layer_idx.expect("Ling MLA needs a KV layer"),
                metadata,
                normed.offset(first * h_bytes),
                attn.offset(first * h_bytes),
                scratch,
                rows as u32,
                kv_write_start.saturating_sub(first).min(rows) as u32,
                Some((seq_len_start + first) as u32),
                stream,
            )?;
        }
        self.finish_prefill_ffn(hidden, residual, attn, num_tokens, ctx, stream)
    }
}

#[path = "ling_decoder_trait.rs"]
mod ling_decoder_trait;

#[path = "ling_decoder_load.rs"]
mod ling_decoder_load;
pub use ling_decoder_load::load_ling_layers;
