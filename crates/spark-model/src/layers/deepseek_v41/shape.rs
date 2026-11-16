// SPDX-License-Identifier: AGPL-3.0-only

//! The dimensions, kernels and working buffers of a DeepSeek-V4.1 block.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::cache_recipe::WINDOW;
use super::hc::{HcKernels, HcShape};
use super::index::{INDEX_DIM, IndexKernels};
use super::moe::MoeKernels;
use super::ops::{HEAD_DIM, Kernels, MAX_KEYS};
use super::side::SideKernels;

/// The dimensions and constants of a block.
#[derive(Clone, Copy)]
pub struct Shape {
    pub hidden: usize,
    pub heads: usize,
    pub q_rank: usize,
    pub o_groups: usize,
    pub o_rank: usize,
    pub rope: usize,
    pub inter: usize,
    pub experts: usize,
    pub top_k: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub norm_eps: f32,
    pub index_topk: usize,
    pub index_heads: usize,
    /// The two-level top-k keeps this many blocks of this many compressed
    /// positions.
    pub candidate_blocks: usize,
    pub candidate_block_size: usize,
    pub hc: HcShape,
}

/// The reference clamps SwiGLU at 10; the release's config says so and the
/// shared parser does not carry it.
const SWIGLU_LIMIT: f32 = 10.0;

impl Shape {
    pub fn from_config(c: &ModelConfig) -> Result<Self> {
        ensure!(
            c.head_dim == HEAD_DIM && c.sliding_window as usize == WINDOW,
            "deepseek_v41: head_dim {} and window {} are not the 512 and 128 the kernels are built for",
            c.head_dim,
            c.sliding_window
        );
        ensure!(
            c.index_head_dim == INDEX_DIM,
            "deepseek_v41: index_head_dim {} is not the {INDEX_DIM} the kernels are built for",
            c.index_head_dim
        );
        // Below `index_topk` positions every one is chosen and no candidates
        // are looked at, which holds only if candidates cover that many.
        ensure!(
            c.candidate_source_layer_id.is_none()
                || c.candidate_topk_blocks * c.candidate_block_size >= c.index_topk,
            "deepseek_v41: {} candidate blocks of {} hold fewer than index_topk {}",
            c.candidate_topk_blocks,
            c.candidate_block_size,
            c.index_topk
        );
        Ok(Self {
            hidden: c.hidden_size,
            heads: c.num_attention_heads,
            q_rank: c.q_lora_rank,
            o_groups: c.o_groups,
            o_rank: c.o_lora_rank,
            rope: c.qk_rope_head_dim,
            inter: c.moe_intermediate_size,
            experts: c.num_experts,
            top_k: c.num_experts_per_tok,
            route_scale: c.routed_scaling_factor as f32,
            swiglu_limit: SWIGLU_LIMIT,
            norm_eps: c.rms_norm_eps as f32,
            index_topk: c.index_topk,
            index_heads: c.index_n_heads,
            candidate_blocks: c.candidate_topk_blocks,
            candidate_block_size: c.candidate_block_size,
            hc: HcShape {
                hidden: c.hidden_size,
                copies: c.hc_mult,
                sinkhorn_iters: c.hc_sinkhorn_iters,
                norm_eps: c.rms_norm_eps as f32,
                hc_eps: c.hc_eps,
            },
        })
    }
}

/// Every kernel a block launches.
#[derive(Clone, Copy)]
pub struct BlockKernels {
    pub cache: Kernels,
    pub hc: HcKernels,
    pub index: IndexKernels,
    pub moe: MoeKernels,
    pub side: SideKernels,
    pub tiers: [KernelHandle; 3],
    pub rms_norm: KernelHandle,
}

impl BlockKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            cache: Kernels::load(gpu)?,
            hc: HcKernels::load(gpu)?,
            index: IndexKernels::load(gpu)?,
            moe: MoeKernels::load(gpu)?,
            side: SideKernels::load(gpu)?,
            tiers: [
                gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc8")?,
                gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc16")?,
                gpu.kernel("mxfp8_gemv", "mxfp8_gemv_tc32")?,
            ],
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
        })
    }
}

/// Working buffers for one token, shared by every block of a model: blocks
/// run one after another.
pub struct Scratch {
    pub(super) x: DevicePtr,
    pub(super) normed: DevicePtr,
    pub(super) quantized: DevicePtr,
    pub(super) q_low: DevicePtr,
    pub(super) q_low_normed: DevicePtr,
    pub(super) q_low_quantized: DevicePtr,
    pub(super) q: DevicePtr,
    pub(super) key: DevicePtr,
    pub(super) idx: DevicePtr,
    pub(super) position: DevicePtr,
    pub(super) group_position: DevicePtr,
    pub(super) attended: DevicePtr,
    pub(super) o_low: DevicePtr,
    pub(super) o_low_quantized: DevicePtr,
    pub(super) sublayer_out: DevicePtr,
    pub(super) gate: DevicePtr,
    pub(super) up: DevicePtr,
    pub(super) inner: DevicePtr,
    pub(super) down: DevicePtr,
    /// `[top_k, hidden]` BF16: each chosen expert's output, in expert order.
    pub(super) expert_out: DevicePtr,
    pub(super) acc: DevicePtr,
    pub(super) chosen: DevicePtr,
    pub(super) weights: DevicePtr,
    pub(super) pooled: DevicePtr,
    pub(super) site_pre: DevicePtr,
    pub(super) next_pre: DevicePtr,
    pub(super) post: DevicePtr,
    pub(super) comb: DevicePtr,
    pub(super) index_q: DevicePtr,
    pub(super) index_weights_f32: DevicePtr,
    pub(super) index_weights: DevicePtr,
    pub(super) index_key_f32: DevicePtr,
    pub(super) index_key: DevicePtr,
    pub(super) scores_f32: DevicePtr,
    pub(super) scores: DevicePtr,
    /// The compressed positions `scores` has room for.
    pub(super) max_groups: usize,
}

impl Scratch {
    /// `max_groups` bounds the compressed positions one query scores.
    pub fn alloc(gpu: &dyn GpuBackend, s: &Shape, max_groups: usize) -> Result<Self> {
        let bf16 = |values: usize| gpu.alloc(values * 2);
        let f32s = |values: usize| gpu.alloc(values * 4);
        let copies = s.hc.copies;
        Ok(Self {
            x: bf16(s.hidden)?,
            normed: bf16(s.hidden)?,
            quantized: bf16(s.hidden)?,
            q_low: bf16(s.q_rank)?,
            q_low_normed: bf16(s.q_rank)?,
            q_low_quantized: bf16(s.q_rank)?,
            q: bf16(s.heads * HEAD_DIM)?,
            key: bf16(HEAD_DIM)?,
            idx: f32s(MAX_KEYS)?,
            position: f32s(1)?,
            group_position: f32s(1)?,
            attended: bf16(s.heads * HEAD_DIM)?,
            o_low: bf16(s.o_groups * s.o_rank)?,
            o_low_quantized: bf16(s.o_groups * s.o_rank)?,
            sublayer_out: bf16(s.hidden)?,
            gate: bf16(s.inter)?,
            up: bf16(s.inter)?,
            inner: bf16(s.inter)?,
            down: bf16(s.hidden)?,
            expert_out: bf16(s.top_k * s.hidden)?,
            acc: f32s(s.hidden)?,
            chosen: f32s(s.top_k)?,
            weights: f32s(s.top_k)?,
            pooled: bf16(HEAD_DIM)?,
            site_pre: f32s(copies)?,
            next_pre: f32s(copies)?,
            post: f32s(copies)?,
            comb: f32s(copies * copies)?,
            index_q: bf16(s.index_heads * INDEX_DIM)?,
            index_weights_f32: f32s(s.index_heads)?,
            index_weights: bf16(s.index_heads)?,
            index_key_f32: f32s(INDEX_DIM)?,
            index_key: bf16(INDEX_DIM)?,
            scores_f32: f32s(max_groups.max(1))?,
            scores: bf16(max_groups.max(1))?,
            max_groups,
        })
    }
}
