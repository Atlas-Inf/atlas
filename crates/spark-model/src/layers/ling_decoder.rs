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
use crate::layers::ling_mla::{LingMlaConfig, LingMlaKernels, LingMlaScratch, LingMlaWeights};
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
    layer_idx: usize,
}

impl LingDecoderLayer {
    fn alloc_projection_scratch(gpu: &dyn GpuBackend) -> Result<LingProjectionScratch> {
        // The largest Ling projection contraction is the 4096-wide KDA/MLA
        // output.  Keep one reusable activation and scale buffer per layer.
        let act = gpu.alloc(LING_KDA_PROJECTION_SIZE)?;
        let scales = gpu.alloc((LING_KDA_PROJECTION_SIZE / 128) * 4)?;
        Ok(LingProjectionScratch::new(
            act,
            LING_KDA_PROJECTION_SIZE,
            scales,
            (LING_KDA_PROJECTION_SIZE / 128) * 4,
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
        let bf16 = |elements: usize| gpu.alloc(elements * 2);
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
                // KDA scratch is layer-owned; the shared attention buffer is
                // the FFN seam used by LingMoe's weighted residual fold.
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
                weights.forward_decode(
                    *config,
                    *kernels,
                    ctx.gpu,
                    kv_cache,
                    self.layer_idx,
                    metadata,
                    normed,
                    ctx.buffers.attn_output(),
                    *scratch,
                    write_kv,
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

    fn run_kda_prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
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
        let kda_state = self.state_kda(state, ctx.config)?;
        let attn = match &self.attention {
            LingAttention::Kda { layer, .. } => layer.forward_prefill(
                ctx.gpu, normed, num_tokens, kda_state, scratch, projection, stream,
            )?,
            LingAttention::Mla { .. } => unreachable!("KDA prefill called on MLA layer"),
        };
        let h_bytes = self.hidden_size * 2;
        for token in 0..num_tokens {
            let hidden_t = hidden.offset(token * h_bytes);
            let residual_t = residual.offset(token * h_bytes);
            let attn_t = attn.offset(token * h_bytes);
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden_t,
                attn_t,
                &self.post_attn_norm,
                ctx.buffers.norm_output(),
                residual_t,
                1,
                self.hidden_size as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;
            // LingMoe folds the attention branch from the shared base pointer;
            // preserve the row-local branch before its one-token FFN call.
            ctx.gpu.copy_d2d_async(
                attn_t,
                ctx.buffers.attn_output(),
                self.hidden_size * 2,
                stream,
            )?;
            let ffn_out = match &self.dense_ffn {
                Some(ffn) => ffn.forward(ctx.buffers.norm_output(), ctx, stream)?,
                None => self.ffn.forward(ctx.buffers.norm_output(), ctx, stream)?,
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
        state: &mut dyn LayerState,
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
        ensure!(
            ctx.buffers.scratch_bytes() >= 4,
            "Ling MLA prefill needs a scalar scratch word"
        );
        let seq_len_word = ctx.buffers.scratch();
        let h_bytes = self.hidden_size * 2;
        for token in 0..num_tokens {
            let seq_len = (seq_len_start + token + 1) as u32;
            ctx.gpu
                .copy_h2d_async(&seq_len.to_ne_bytes(), seq_len_word, stream)?;
            let token_meta = AttnMetadataDev {
                positions: meta.positions.offset(token * 4),
                positions_h: meta.positions_h.offset(token * 4),
                positions_w: meta.positions_w.offset(token * 4),
                slot: meta.slot.offset(token * 8),
                seq_len: seq_len_word,
                block_table: meta.block_table,
                max_blocks_per_seq: meta.max_blocks_per_seq,
                num_seqs: 1,
                seq_slot: DevicePtr::NULL,
                moe_row_adapter: DevicePtr::NULL,
            };
            self.run_one(
                hidden.offset(token * h_bytes),
                residual.offset(token * h_bytes),
                state,
                kv_cache,
                token >= kv_write_start,
                Some(token_meta),
                ctx,
                stream,
            )?;
        }
        Ok(())
    }
}

#[path = "ling_decoder_trait.rs"]
mod ling_decoder_trait;

#[path = "ling_decoder_load.rs"]
mod ling_decoder_load;
pub use ling_decoder_load::load_ling_layers;
