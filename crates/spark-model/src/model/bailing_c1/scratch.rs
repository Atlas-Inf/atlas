// SPDX-License-Identifier: AGPL-3.0-only

//! Shared device scratch for the Ling family driver. One allocation set
//! per MODEL (layers run sequentially over the same stream), sized for the
//! prefill sub-chunk row cap — the conservative decode/prefill budget the
//! campaign's ISL-4096 envelope picks (`ATLAS_LING_PREFILL_ROWS`, default
//! 512 rows; see `spec::LingSpec::from_config` ownership).

use std::sync::Arc;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Query rows the paged attention kernel accepts per sequence (8 warps of
/// the absorbed kernel; see `LING_MLA_ATTN_WARPS` in gated_mla.cu).
pub(crate) const MLA_MAX_ROWS_PAGED: usize = 8;

/// Shared scratch. Row-sized buffers hold `rows` rows.
pub(crate) struct LingScratch {
    pub rows: usize,
    /// Model hidden width (row stride of the hidden-state buffers).
    pub hidden: usize,
    /// Capacity (entries) of the MLA block-table staging buffer.
    pub max_blocks: usize,
    /// `[rows, hidden]` BF16 — the attention-normed copy of `hidden`.
    pub normed: DevicePtr,
    /// `[rows, hidden]` BF16 — attention output before the residual add.
    pub attn_out: DevicePtr,
    /// `[rows, hidden]` BF16 — FFN output before the residual add.
    pub ffn_out: DevicePtr,
    // KDA half ([rows, dim_t] where dim_t = heads*dim):
    /// `[rows, 3 * 4096]` BF16 — q|k|v planes, projected then convolved.
    pub kda_qkv: DevicePtr,
    /// `[rows, 3 * 4096]` BF16 — prefill projection staging: three
    /// contiguous `[rows, 4096]` plane regions the strided conv reads.
    pub kda_preconv: DevicePtr,
    /// `[rows, 4096]` BF16 — f_proj (decay gate) rows.
    pub kda_gate: DevicePtr,
    /// `[rows, 4096]` BF16 — g_proj (output gate) rows.
    pub kda_o_g: DevicePtr,
    /// `[rows, 32]` BF16 — b_proj (beta) rows.
    pub kda_beta: DevicePtr,
    /// `[rows, 4096]` BF16 — recurrent output.
    pub kda_out: DevicePtr,
    /// `[rows, 4096]` BF16 — o_normed output.
    pub kda_o: DevicePtr,
    // MLA half:
    /// `[rows, 32 * 192]` BF16 — q_proj rows.
    pub mla_q: DevicePtr,
    /// `[rows, 576]` BF16 — kv_a rows.
    pub mla_kv_a: DevicePtr,
    /// `[rows, 32]` BF16 — g_proj rows.
    pub mla_g: DevicePtr,
    /// `[rows, 32, 576]` BF16 — absorbed + roped q.
    pub mla_q_final: DevicePtr,
    /// `[rows, 32, 512]` BF16 — attention latent output.
    pub mla_latent: DevicePtr,
    /// `[rows, 576]` BF16 — prefill contiguous K rows ([latent|rope]).
    pub mla_k_contig: DevicePtr,
    /// `[rows, 512]` BF16 — prefill contiguous V rows (normed latent).
    pub mla_v_contig: DevicePtr,
    /// `[rows]` u32 — rope positions upload.
    pub mla_pos: DevicePtr,
    /// `[rows]` i64 — paged-cache slots upload (-1 skips the paged write).
    pub mla_slots: DevicePtr,
    /// `[rows]` u64 — scratch zero-mix (unused; keeps neg slots simple).
    pub mla_base_len: DevicePtr,
    /// Block tables upload: `[max_blocks]` i32 for one sequence.
    pub mla_block_table: DevicePtr,
    // FFN half (rows-wide; MoE sub-batches reuse it):
    /// `[rows, hidden]` BF16 — FFN input staging (MLA/KDA may hand the
    /// normed buffer directly; dense FFN and MoE use this base).
    pub ffn_in: DevicePtr,
    /// MoE: `[rows * top_k]` u32 topk ids, `[..]` f32 weights, `[..]` i32
    /// sort/perm/offsets scratch, worklist, done word.
    pub topk_ids: DevicePtr,
    pub topk_weights: DevicePtr,
    pub sorted_tokens: DevicePtr,
    pub token_to_perm: DevicePtr,
    pub expert_offsets: DevicePtr,
    pub worklist: DevicePtr,
    pub moe_done: DevicePtr,
    /// MoE: `[rows, 512]` f32 router logits.
    pub moe_logits: DevicePtr,
    /// MoE: `[rows, hidden / 2]` u8 quantized activation + `[.., hidden/16]`
    /// scale.
    pub a_packed: DevicePtr,
    pub a_scale: DevicePtr,
    /// MoE: `[rows * top_k, moe_inter / 2]` u8 gate/up output + scales.
    pub gu_packed: DevicePtr,
    pub gu_scale: DevicePtr,
    /// MoE: `[rows * top_k, hidden]` BF16 — down-projection rows.
    pub expert_out: DevicePtr,
    /// Shared expert / dense FFN: `[rows, inter]` BF16 gate/up H rows and
    /// `[rows, hidden]` down rows.
    pub shared_h: DevicePtr,
    pub shared_y: DevicePtr,
}

impl LingScratch {
    fn alloc(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
        let p = gpu.alloc(bytes.max(16))?;
        gpu.memset(p, 0, bytes.max(16))?;
        Ok(p)
    }

    fn bf16(gpu: &dyn GpuBackend, elems: usize) -> Result<DevicePtr> {
        Self::alloc(gpu, elems * 2)
    }

    /// Allocate for `rows` prompt rows, `hidden` model dim, `experts` /
    /// `top_k` / `inter` (the WIDEST FFN intermediate — the dense-MLP
    /// layers 0-1 share the shared-expert scratch lanes).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        gpu: &dyn GpuBackend,
        rows: usize,
        hidden: usize,
        experts: usize,
        top_k: usize,
        inter: usize,
        max_blocks: usize,
    ) -> Result<Self> {
        let kda_t = 32 * 128; // heads * head_dim
        let self_ = Self {
            rows,
            hidden,
            max_blocks,
            normed: Self::bf16(gpu, rows * hidden)?,
            attn_out: Self::bf16(gpu, rows * hidden)?,
            ffn_out: Self::bf16(gpu, rows * hidden)?,
            kda_qkv: Self::bf16(gpu, rows * 3 * kda_t)?,
            kda_preconv: Self::bf16(gpu, rows * 3 * kda_t)?,
            kda_gate: Self::bf16(gpu, rows * kda_t)?,
            kda_o_g: Self::bf16(gpu, rows * kda_t)?,
            kda_beta: Self::bf16(gpu, rows * 32)?,
            kda_out: Self::bf16(gpu, rows * kda_t)?,
            kda_o: Self::bf16(gpu, rows * kda_t)?,
            mla_q: Self::bf16(gpu, rows * 32 * 192)?,
            mla_kv_a: Self::bf16(gpu, rows * 576)?,
            mla_g: Self::bf16(gpu, rows * 32)?,
            mla_q_final: Self::bf16(gpu, rows * 32 * 576)?,
            mla_latent: Self::bf16(gpu, rows * 32 * 512)?,
            mla_k_contig: Self::bf16(gpu, rows * 576)?,
            mla_v_contig: Self::bf16(gpu, rows * 512)?,
            mla_pos: Self::alloc(gpu, rows * 4)?,
            mla_slots: Self::alloc(gpu, rows * 8)?,
            mla_base_len: Self::alloc(gpu, rows * 4)?,
            mla_block_table: Self::alloc(gpu, max_blocks * 4)?,
            ffn_in: Self::bf16(gpu, rows * hidden)?,
            topk_ids: Self::alloc(gpu, rows * top_k * 4)?,
            topk_weights: Self::alloc(gpu, rows * top_k * 4)?,
            sorted_tokens: Self::alloc(gpu, rows * top_k * 4)?,
            token_to_perm: Self::alloc(gpu, rows * top_k * 4)?,
            expert_offsets: Self::alloc(gpu, (experts + 1) * 4)?,
            // Worst-case item bound: every expert at least one tile, plus
            // token tiles (see moe_decode.cu's worklist contract).
            worklist: Self::alloc(gpu, (1 + 2 * (experts + (rows * top_k) / 16)) * 4)?,
            moe_done: Self::alloc(gpu, 4)?,
            moe_logits: Self::alloc(gpu, rows * experts * 4)?,
            a_packed: Self::alloc(gpu, rows * (hidden / 2))?,
            a_scale: Self::alloc(gpu, rows * (hidden / 16))?,
            gu_packed: Self::alloc(gpu, rows * top_k * (inter / 2))?,
            gu_scale: Self::alloc(gpu, rows * top_k * (inter / 16))?,
            expert_out: Self::bf16(gpu, rows * top_k * hidden)?,
            shared_h: Self::bf16(gpu, rows * inter)?,
            shared_y: Self::bf16(gpu, rows * hidden)?,
        };
        Ok(self_)
    }

    /// Max blocks a block-table upload buffer must hold (native context):
    /// ceil(max_position_embeddings / block_size) + 1.
    pub(crate) fn max_blocks(max_positions: usize, block_size: usize) -> usize {
        max_positions / block_size.max(1) + 2
    }

    pub(crate) fn arc(
        gpu: &dyn GpuBackend,
        rows: usize,
        hidden: usize,
        experts: usize,
        top_k: usize,
        inter: usize,
        max_blocks: usize,
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::new(
            gpu, rows, hidden, experts, top_k, inter, max_blocks,
        )?))
    }
}
