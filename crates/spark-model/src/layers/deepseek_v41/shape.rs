// SPDX-License-Identifier: AGPL-3.0-only

//! The dimensions, kernels and working buffers of a DeepSeek-V4.1 block.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::cache_recipe::{WINDOW, rope_freqs};
use super::gemm::GemmScratch;
use super::hc::{HcKernels, HcShape};
use super::index::{INDEX_DIM, IndexKernels};
use super::moe::MoeKernels;
use super::moe_gemm::{ExpertGemm, expert_gemm_rows};
use super::moe_grouped::GroupedScratch;
use super::ops::{HEAD_DIM, Kernels, MAX_KEYS, attn_records};
use super::parts::{Held, PARTIAL_ROWS, PARTS, PartKernels};
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
            c.num_attention_heads.is_multiple_of(PARTS)
                && c.o_groups.is_multiple_of(PARTS)
                && c.moe_intermediate_size.is_multiple_of(PARTS * 32),
            "deepseek_v41: {} heads, {} output groups and {} expert rows do not split in {PARTS} parts",
            c.num_attention_heads,
            c.o_groups,
            c.moe_intermediate_size
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
    pub parts: PartKernels,
    pub side: SideKernels,
    pub rms_norm: KernelHandle,
}

impl BlockKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            cache: Kernels::load(gpu)?,
            hc: HcKernels::load(gpu)?,
            index: IndexKernels::load(gpu)?,
            moe: MoeKernels::load(gpu)?,
            parts: PartKernels::load(gpu)?,
            side: SideKernels::load(gpu)?,
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
        })
    }
}

/// Rows of a sequence a block runs together unless `ATLAS_DSV41_ROWS` says
/// otherwise: a decode step is one, a prefill chunk goes through in turns of
/// this many. The more rows, the more of them choose the same expert and
/// share one read of its weights: at 512 an expert has eight rows on average.
pub const ROWS: usize = 512;

/// The batch a model's blocks are given room for.
pub fn batch_rows() -> usize {
    std::env::var("ATLAS_DSV41_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&rows| rows > 0)
        .unwrap_or(ROWS)
}

/// The most tokens a compressor pools into one key that there is room for.
pub const MAX_RATIO: usize = 8;

/// Scores (query, compressed position) one launch of the indexer may leave:
/// the queries of a batch are scored this many at a time, a sixth of a
/// gigabyte of scratch.
const SCORES: usize = 1 << 25;

/// Working buffers for a batch of up to [`ROWS`] rows, shared by every block
/// of a model: blocks run one after another. Unless it says otherwise a
/// buffer is `[rows, ..]`, and those of the parts `[PARTS, rows, ..]`.
pub struct Scratch {
    /// Rows there is room for.
    pub(super) rows: usize,
    /// A device word for the ranks to agree on (`Layer::agree_on_rows`).
    pub(super) word: DevicePtr,
    /// The routed experts of a batch, grouped; and the shared expert's
    /// `[rows, 2, inter / PARTS]` gate and up rows.
    pub(super) grouped: GroupedScratch,
    pub(super) shared_gate_up: DevicePtr,
    /// A batch's activation as a GEMM reads it, and `[rows, hidden / PARTS]`
    /// BF16: one part of the shared expert's output from one.
    pub(super) gemm: GemmScratch,
    pub(super) shared_out: DevicePtr,
    /// The routed experts of a large batch as grouped GEMMs, where batches
    /// are that large.
    pub(super) expert_gemm: Option<ExpertGemm>,
    /// `[128 + rows, 512]` BF16: the window's keys in position order and
    /// then the batch's own, when a batch is more than one row.
    pub(super) chunk_keys: DevicePtr,
    pub(super) x: DevicePtr,
    pub(super) normed: DevicePtr,
    pub(super) quantized: DevicePtr,
    pub(super) q_low: DevicePtr,
    /// The low-rank query, normed: what the heads' projection and the
    /// indexer's read, rounding it to FP8 on the way in.
    pub(super) q_low_normed: DevicePtr,
    pub(super) q: DevicePtr,
    pub(super) key: DevicePtr,
    pub(super) idx: DevicePtr,
    /// `[max_positions]` u32, each its own index: the word a rotation reads
    /// for a position, with nothing to upload.
    pub(super) positions: DevicePtr,
    pub(super) attended: DevicePtr,
    /// What the attention keeps of one part's heads between its launches.
    pub(super) attn_records: DevicePtr,
    pub(super) o_low: DevicePtr,
    /// `[top_k + 1, 2, inter / PARTS]` BF16: one part's gate and up rows of
    /// each chosen expert, then of the shared expert.
    pub(super) gate_up: DevicePtr,
    /// `[PARTS, rows * (top_k + 1), inter / PARTS]` BF16: the parts of the
    /// experts' hidden activations, in FP8. A token alone has its chosen
    /// experts' and then the shared expert's; a batch has its (row, expert)
    /// pairs' in expert order and then every row's shared expert's. Like
    /// every buffer in parts, a part follows the last with no room between
    /// for the rows a batch does not have.
    pub(super) inner_parts: DevicePtr,
    /// `[rows, top_k + 1, hidden / PARTS]` BF16: one part of each expert's
    /// output.
    pub(super) slot_out: DevicePtr,
    /// `[PARTS, rows, hidden / PARTS]` BF16 each: the parts of the
    /// attention's output and of the MoE's.
    pub(super) attn_out: DevicePtr,
    pub(super) moe_out: DevicePtr,
    /// `[PARTS, rows, hidden]` FP32 for [`PARTIAL_ROWS`] rows at most: each
    /// part's partial of a few rows' attention output (`block_out.rs`).
    pub(super) attn_partial: DevicePtr,
    /// Scratch as large as the largest gather, for one that is not in place.
    pub(super) gather_recv: DevicePtr,
    /// `[experts]` FP32: the routing score of every expert.
    pub(super) route_scores: DevicePtr,
    pub(super) chosen: DevicePtr,
    pub(super) weights: DevicePtr,
    /// `[rows + MAX_RATIO, 512]` FP32 each: a compressor's key and gate
    /// projections of the tokens a group was left open with and then of the
    /// batch's; `[rows, 512]` BF16, the keys pooled from them; and `[rows]`
    /// u32, the positions the groups are turned to.
    pub(super) pool_kv: DevicePtr,
    pub(super) pool_score: DevicePtr,
    pub(super) pooled: DevicePtr,
    pub(super) group_positions: DevicePtr,
    pub(super) site_pre: DevicePtr,
    /// The mix's partial sums, a block of the token each; and `[rows, 3 *
    /// mix rows]` FP32, a batch's from its GEMM.
    pub(super) hc_partials: DevicePtr,
    pub(super) hc_sums: DevicePtr,
    pub(super) post: DevicePtr,
    pub(super) comb: DevicePtr,
    /// The indexer's query heads and head weights, `[score_rows, ..]`, and
    /// the index keys of a batch's groups, `[rows, 128]`.
    pub(super) index_q: DevicePtr,
    pub(super) index_weights_f32: DevicePtr,
    pub(super) index_weights: DevicePtr,
    pub(super) index_key_f32: DevicePtr,
    pub(super) index_key: DevicePtr,
    /// `[score_rows, max_groups]`: a row the scores of one query.
    pub(super) scores_f32: DevicePtr,
    pub(super) scores: DevicePtr,
    /// The compressed positions a row of `scores` has room for, and the
    /// queries one launch scores.
    pub(super) max_groups: usize,
    pub(super) score_rows: usize,
}

impl Scratch {
    /// `max_positions` bounds a sequence, and so the compressed positions one
    /// query scores.
    pub fn alloc(gpu: &dyn GpuBackend, s: &Shape, max_positions: usize) -> Result<Self> {
        Self::alloc_rows(gpu, s, max_positions, ROWS, Held::ALL)
    }

    /// The same with room for batches of `rows` rows, on a rank that holds
    /// `held` of the parts.
    pub fn alloc_rows(
        gpu: &dyn GpuBackend,
        s: &Shape,
        max_positions: usize,
        rows: usize,
        held: Held,
    ) -> Result<Self> {
        ensure!(rows > 0, "deepseek_v41: batches of no rows");
        let max_groups = max_positions.max(1);
        let score_rows = (SCORES / max_groups).clamp(1, rows);
        let bf16 = |values: usize| gpu.alloc(values * 2);
        let f32s = |values: usize| gpu.alloc(values * 4);
        let copies = s.hc.copies;
        Ok(Self {
            rows,
            word: gpu.alloc(64)?,
            grouped: GroupedScratch::alloc(gpu, rows, s.top_k, s.experts, s.inter / PARTS)?,
            shared_gate_up: bf16(rows * 2 * s.inter / PARTS)?,
            gemm: GemmScratch::alloc(gpu, rows, s.hidden.max(s.o_groups * s.o_rank))?,
            shared_out: bf16(rows * s.hidden / PARTS)?,
            expert_gemm: match expert_gemm_rows() {
                from if from > 0 && rows >= from => Some(ExpertGemm::alloc(gpu, s, rows)?),
                _ => None,
            },
            chunk_keys: bf16((WINDOW + rows) * HEAD_DIM)?,
            x: bf16(rows * (s.hidden))?,
            normed: bf16(rows * (s.hidden))?,
            quantized: bf16(rows * (s.hidden))?,
            q_low: bf16(rows * (s.q_rank))?,
            q_low_normed: bf16(rows * (s.q_rank))?,
            // The heads of the parts held, the first of them first.
            q: bf16(rows * (s.heads / PARTS * held.count * HEAD_DIM))?,
            key: bf16(rows * (HEAD_DIM))?,
            idx: f32s(rows * (MAX_KEYS))?,
            positions: {
                let table = f32s(max_positions.max(1))?;
                let words: Vec<u8> = (0..max_positions as u32)
                    .flat_map(u32::to_le_bytes)
                    .collect();
                gpu.copy_h2d(&words, table)?;
                table
            },
            attended: bf16(rows * (s.heads / PARTS * held.count * HEAD_DIM))?,
            attn_records: gpu.alloc(attn_records(rows, s.heads / PARTS))?,
            o_low: bf16(rows * (s.o_groups * s.o_rank))?,
            gate_up: bf16((s.top_k + 1) * 2 * s.inter / PARTS)?,
            inner_parts: bf16(rows * (s.top_k + 1) * s.inter)?,
            slot_out: bf16(rows * (s.top_k + 1) * s.hidden / PARTS)?,
            attn_out: bf16(rows * s.hidden)?,
            moe_out: bf16(rows * s.hidden)?,
            attn_partial: f32s(PARTS * rows.min(PARTIAL_ROWS) * s.hidden)?,
            gather_recv: bf16(
                (rows * ((s.top_k + 1) * s.inter).max(s.o_groups * s.o_rank))
                    .max(rows.min(PARTIAL_ROWS) * PARTS * s.hidden * 2),
            )?,
            route_scores: f32s(rows * (s.experts))?,
            chosen: f32s(rows * (s.top_k))?,
            weights: f32s(rows * (s.top_k))?,
            pool_kv: f32s((rows + MAX_RATIO) * HEAD_DIM)?,
            pool_score: f32s((rows + MAX_RATIO) * HEAD_DIM)?,
            pooled: bf16(rows * HEAD_DIM)?,
            group_positions: f32s(rows)?,
            site_pre: f32s(rows * (copies))?,
            hc_partials: gpu.alloc(s.hc.partials(rows))?,
            hc_sums: f32s(rows * 3 * super::hc::mix_rows(copies))?,
            post: f32s(rows * (copies))?,
            comb: f32s(rows * (copies * copies))?,
            index_q: bf16(score_rows * s.index_heads * INDEX_DIM)?,
            index_weights_f32: f32s(score_rows * s.index_heads)?,
            index_weights: bf16(score_rows * s.index_heads)?,
            index_key_f32: f32s(rows * INDEX_DIM)?,
            index_key: bf16(rows * INDEX_DIM)?,
            scores_f32: f32s(score_rows * max_groups)?,
            scores: bf16(score_rows * max_groups)?,
            max_groups,
            score_rows,
        })
    }
}

/// A rotary ladder on the device: `[rope / 2]` FP32 radians per position,
/// YaRN-scaled from `original` positions when that is not 0.
pub fn ladder(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    rope: usize,
    original: usize,
    theta: f64,
) -> Result<DevicePtr> {
    let freqs = rope_freqs(
        rope,
        original,
        theta as f32,
        config.yarn_factor,
        config.yarn_beta_fast,
        config.yarn_beta_slow,
    );
    let bytes: Vec<u8> = freqs.iter().flat_map(|f| f.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}
