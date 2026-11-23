// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2 attention launchers: split-key decode attention with its combine,
//! and the tensor-core prefill attention with the relative-bias table.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::ops::{Inkling2Kernels, MAX_D_REL, MAX_REL_EXTENT, u32_of};

/// Paged decode attention with relative bias and per-token sinks.
pub(crate) struct DecodeAttn {
    pub q: DevicePtr,
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    /// per (row, head, split) max, sum and unnormalised output, fp32
    pub partials: DevicePtr,
    pub out: DevicePtr,
    pub block_tables: DevicePtr,
    /// `[num_seqs]` query positions; a row attends keys 0 ..= its position
    pub positions: DevicePtr,
    pub rel: DevicePtr,
    pub rel_proj: DevicePtr,
    pub sinks: DevicePtr,
    pub max_blocks_per_seq: usize,
    pub num_seqs: usize,
    /// key splits per row
    pub splits: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub q_stride: usize,
    pub sliding_window: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
}

pub(crate) fn decode_attn(
    gpu: &dyn GpuBackend,
    kernels: &Inkling2Kernels,
    a: &DecodeAttn,
    stream: u64,
) -> Result<()> {
    ensure!(
        a.head_dim == 128,
        "inkling2 decode attention is built for head_dim 128"
    );
    ensure!(
        a.d_rel <= MAX_D_REL && a.rel_extent <= MAX_REL_EXTENT,
        "inkling2 rel bias d_rel {} / extent {} over the kernel limits",
        a.d_rel,
        a.rel_extent
    );
    ensure!(
        a.kv_heads > 0
            && a.heads.is_multiple_of(a.kv_heads)
            && a.heads / a.kv_heads <= MAX_GQA_GROUP,
        "inkling2 GQA heads"
    );
    ensure!(
        (1..=MAX_ATTN_SPLITS).contains(&a.splits),
        "inkling2 attention splits {}",
        a.splits
    );
    let rows = u32_of(a.num_seqs, "seqs")?;
    KernelLaunch::new(gpu, kernels.decode_attn)
        .grid([u32_of(a.kv_heads, "kv heads")?, a.splits as u32, rows])
        .block([256, 1, 1])
        .arg_ptr(a.q)
        .arg_ptr(a.k_cache)
        .arg_ptr(a.v_cache)
        .arg_ptr(a.partials)
        .arg_ptr(a.block_tables)
        .arg_ptr(a.positions)
        .arg_ptr(a.rel)
        .arg_ptr(a.rel_proj)
        .arg_u32(u32_of(a.max_blocks_per_seq, "max blocks")?)
        .arg_u32(u32_of(a.heads, "heads")?)
        .arg_u32(u32_of(a.kv_heads, "kv heads")?)
        .arg_u32(u32_of(a.block_size, "block size")?)
        .arg_f32((a.head_dim as f32).powf(-0.5))
        .arg_u32(u32_of(a.q_stride, "q stride")?)
        .arg_u32(u32_of(a.sliding_window, "window")?)
        .arg_u32(u32_of(a.d_rel, "d_rel")?)
        .arg_u32(u32_of(a.rel_extent, "rel extent")?)
        .launch(stream)?;
    KernelLaunch::new(gpu, kernels.decode_attn_combine)
        .grid([u32_of(a.heads, "heads")?, rows, 1])
        .block([a.head_dim as u32, 1, 1])
        .arg_ptr(a.partials)
        .arg_ptr(a.sinks)
        .arg_ptr(a.out)
        .arg_u32(u32_of(a.heads, "heads")?)
        .arg_u32(a.splits as u32)
        .launch(stream)
}

/// Key splits per decode row (prefill rows run one each).
pub(crate) const DECODE_ATTN_SPLITS: usize = 32;
const MAX_ATTN_SPLITS: usize = DECODE_ATTN_SPLITS;
const MAX_GQA_GROUP: usize = 4;

/// The attention body a forward's rows take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttnBody {
    /// The tensor-core paged prefill body: consecutive rows of one sequence.
    Prefill,
    /// The decode body, each row's keys in this many splits.
    Decode { splits: usize },
}

/// Which body `seqs` sequences of `rows_per_seq` rows take, with scratch for
/// `slots` (row, split) pairs.
///
/// A decode row splits its keys [`DECODE_ATTN_SPLITS`] ways. So does every
/// row of a verify run (up to `verify_rows` rows a sequence): the verified
/// token must be the one plain decode picks, and the prefill body sums a
/// row's keys in another order, enough to turn a tie between two logits one
/// BF16 step apart (greedy text with drafts parted from text without at such
/// ties). Longer runs are prefill: one sequence on the tensor cores, several
/// on the decode body unsplit (they have rows enough to fill the GPU).
pub(crate) fn attn_body(
    seqs: usize,
    rows_per_seq: usize,
    verify_rows: usize,
    slots: usize,
) -> AttnBody {
    let rows = seqs * rows_per_seq;
    if rows_per_seq <= verify_rows && rows * DECODE_ATTN_SPLITS <= slots {
        AttnBody::Decode {
            splits: DECODE_ATTN_SPLITS,
        }
    } else if seqs == 1 {
        AttnBody::Prefill
    } else {
        AttnBody::Decode { splits: 1 }
    }
}

/// (row, split) pairs the partials scratch holds for `max_rows` rows a
/// forward and `max_seqs` sequences ([`attn_partials_len`]).
pub(crate) fn attn_slots(max_rows: usize, max_seqs: usize) -> usize {
    max_rows.max(max_seqs * DECODE_ATTN_SPLITS)
}

/// fp32 elements of attention partials for `slots` (row, split) pairs.
pub(crate) fn attn_partials_len(slots: usize, heads: usize, head_dim: usize) -> usize {
    slots * heads * (head_dim + 2)
}

/// Prefill attention for `q_len` consecutive rows of one sequence at
/// positions `q_offset ..`: the rows' relative-bias table first, then Atlas's
/// tensor-core paged prefill body with the bias and per-row sinks
/// (`inkling2_prefill_attn.cu`).
pub(crate) struct PrefillAttn {
    pub q: DevicePtr,
    pub k_cache: DevicePtr,
    pub v_cache: DevicePtr,
    pub out: DevicePtr,
    pub block_table: DevicePtr,
    pub rel: DevicePtr,
    pub rel_proj: DevicePtr,
    /// `rel_proj` as F32 `[rel_extent, d_rel]`
    pub rel_proj_t: DevicePtr,
    /// `[q_len, heads, rel_extent]` BF16 scratch (the per-head kernel's table)
    pub bias: DevicePtr,
    pub sinks: DevicePtr,
    pub q_len: usize,
    pub q_offset: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub sliding_window: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
}

/// Rows from which a prefill run takes `inkling2_flash_prefill` (64-token
/// tiles; a verify block of a few rows keeps the per-head kernel, whose grid
/// covers the GPU at that size).
const FLASH_MIN_ROWS: usize = 64;
/// `inkling2_flash_prefill.cu` FP_GROUP: query heads per KV head; FP_D_REL.
const FLASH_GROUP: usize = 3;
const FLASH_D_REL: usize = 16;

pub(crate) fn prefill_attn(
    gpu: &dyn GpuBackend,
    k: &Inkling2Kernels,
    a: &PrefillAttn,
    stream: u64,
) -> Result<()> {
    ensure!(
        a.head_dim == 128 && a.d_rel <= MAX_D_REL && a.rel_extent <= MAX_REL_EXTENT,
        "inkling2 prefill attention limits"
    );
    let (heads, extent) = (
        u32_of(a.heads, "heads")?,
        u32_of(a.rel_extent, "rel extent")?,
    );
    let q_len = u32_of(a.q_len, "q_len")?;
    let kv_heads = u32_of(a.kv_heads, "kv heads")?;
    if a.q_len >= FLASH_MIN_ROWS && a.heads == FLASH_GROUP * a.kv_heads && a.d_rel == FLASH_D_REL {
        // a prefill chunk: one CTA per (KV head, 64 tokens) shares each K/V tile
        // with the group's query heads, computing the bias where it adds it
        return KernelLaunch::new(gpu, k.flash_prefill)
            .grid([kv_heads, q_len.div_ceil(64), 1])
            .block([384, 1, 1])
            .arg_ptr(a.q)
            .arg_ptr(a.k_cache)
            .arg_ptr(a.v_cache)
            .arg_ptr(a.out)
            .arg_ptr(a.block_table)
            .arg_ptr(a.rel)
            .arg_ptr(a.rel_proj_t)
            .arg_ptr(a.sinks)
            .arg_u32(q_len)
            .arg_u32(u32_of(a.q_offset, "q_offset")?)
            .arg_u32(heads)
            .arg_u32(kv_heads)
            .arg_u32(u32_of(a.block_size, "block size")?)
            .arg_u32(u32_of(a.sliding_window, "window")?)
            .arg_u32(extent)
            .arg_f32((a.head_dim as f32).powf(-0.5))
            .launch(stream);
    }
    KernelLaunch::new(gpu, k.rel_bias_table)
        .grid([
            extent.div_ceil(256),
            u32_of(a.q_len * a.heads, "bias rows")?,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a.rel)
        .arg_ptr(a.rel_proj)
        .arg_ptr(a.bias)
        .arg_u32(u32_of(a.d_rel, "d_rel")?)
        .arg_u32(extent)
        .launch(stream)?;
    KernelLaunch::new(gpu, k.prefill_attn)
        .grid([heads, q_len.div_ceil(32), 1])
        .block([128, 1, 1])
        .arg_ptr(a.q)
        .arg_ptr(a.k_cache)
        .arg_ptr(a.v_cache)
        .arg_ptr(a.out)
        .arg_ptr(a.block_table)
        .arg_u32(q_len)
        .arg_u32(u32_of(a.q_offset + a.q_len, "kv_len")?)
        .arg_u32(u32_of(a.q_offset, "q_offset")?)
        .arg_u32(heads)
        .arg_u32(u32_of(a.kv_heads, "kv heads")?)
        .arg_u32(u32_of(a.head_dim, "head dim")?)
        .arg_u32(u32_of(a.block_size, "block size")?)
        .arg_u32(u32_of(a.sliding_window, "window")?)
        .arg_u32(1)
        .arg_f32((a.head_dim as f32).powf(-0.5))
        .arg_ptr(a.sinks)
        .arg_ptr(a.bias)
        .arg_u32(extent)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::{AttnBody, DECODE_ATTN_SPLITS, attn_body, attn_slots};

    const DECODE: AttnBody = AttnBody::Decode {
        splits: DECODE_ATTN_SPLITS,
    };

    #[test]
    fn a_verify_run_attends_as_decode_rows_do() {
        let slots = attn_slots(4096, 8);
        // one token of one sequence, and of eight
        assert_eq!(attn_body(1, 1, 16, slots), DECODE);
        assert_eq!(attn_body(8, 1, 16, slots), DECODE);
        // a verify run: eight rows of one sequence, three rows each of two
        assert_eq!(attn_body(1, 8, 16, slots), DECODE);
        assert_eq!(attn_body(2, 3, 16, slots), DECODE);
        assert_eq!(attn_body(1, 16, 16, slots), DECODE);
        // a prefill chunk of one sequence, and of several
        assert_eq!(attn_body(1, 17, 16, slots), AttnBody::Prefill);
        assert_eq!(attn_body(1, 4096, 16, slots), AttnBody::Prefill);
        assert_eq!(attn_body(4, 512, 16, slots), AttnBody::Decode { splits: 1 });
    }

    #[test]
    fn a_run_the_scratch_cannot_split_keeps_its_old_body() {
        // scratch for eight split rows: a ninth row does not fit split
        let slots = attn_slots(64, 8);
        assert_eq!(slots, 8 * DECODE_ATTN_SPLITS);
        assert_eq!(attn_body(1, 8, 16, slots), DECODE);
        assert_eq!(attn_body(1, 9, 16, slots), AttnBody::Prefill);
        assert_eq!(attn_body(3, 3, 16, slots), AttnBody::Decode { splits: 1 });
    }
}
