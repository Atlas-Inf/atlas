// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for GLM-5's pooled semantic-index primitives.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

#[allow(clippy::too_many_arguments)]
pub fn glm_index_layernorm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    values: DevicePtr,
    weight: DevicePtr,
    bias: DevicePtr,
    rows: u32,
    dim: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([128, 1, 1])
        .shared_mem(128 * std::mem::size_of::<f32>() as u32)
        .arg_ptr(values)
        .arg_ptr(weight)
        .arg_ptr(bias)
        .arg_u32(rows)
        .arg_u32(dim)
        .arg_f32(eps)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_tail_write(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    keys: DevicePtr,
    gates: DevicePtr,
    tail: DevicePtr,
    tail_map: DevicePtr,
    slots: DevicePtr,
    num_tokens: u32,
    block_size: u32,
    pool_size: u32,
    head_dim: u32,
    tail_block_stride_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(keys)
        .arg_ptr(gates)
        .arg_ptr(tail)
        .arg_ptr(tail_map)
        .arg_ptr(slots)
        .arg_u32(num_tokens)
        .arg_u32(block_size)
        .arg_u32(pool_size)
        .arg_u32(head_dim)
        .arg_u64(tail_block_stride_bytes)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_kpool_finalize(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    tail: DevicePtr,
    tail_map: DevicePtr,
    ape: DevicePtr,
    cache: DevicePtr,
    slots: DevicePtr,
    num_tokens: u32,
    block_size: u32,
    pool_size: u32,
    head_dim: u32,
    tail_block_stride_bytes: u64,
    values_block_stride_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(tail)
        .arg_ptr(tail_map)
        .arg_ptr(ape)
        .arg_ptr(cache)
        .arg_ptr(slots)
        .arg_u32(num_tokens)
        .arg_u32(block_size)
        .arg_u32(pool_size)
        .arg_u32(head_dim)
        .arg_u64(tail_block_stride_bytes)
        .arg_u64(values_block_stride_bytes)
        .launch(stream)
}

pub fn glm_index_fill_causal(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    rows: u32,
    seq_len_start: u32,
    width: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, div_ceil(width, 256), 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(seq_len_start)
        .arg_u32(width)
        .launch(stream)
}

/// [`glm_index_fill_causal`] for one decode row whose length is read on the
/// device (`kv_len`: the i32 sequence length including the new token).
pub fn glm_index_fill_causal_dev(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    kv_len: DevicePtr,
    width: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(width, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(kv_len)
        .arg_u32(width)
        .launch(stream)
}

/// `pools_per_cta` of `glm_index_logits_bf16_mma_v2`: each CTA walks a
/// contiguous run of 32-pool chunks whose length the launch picks.
pub const GLM_INDEX_LOGITS_V2_POOLS: u32 = u32::MAX;
/// v2's rows per CTA, one per warp: `kIndexV2Warps` in glm_indexer_wmma.cu.
pub const GLM_INDEX_LOGITS_V2_ROWS: u32 = 4;
/// v2's pools per staged key chunk: `kIndexV2Pools` in glm_indexer_wmma.cu.
const V2_CHUNK_POOLS: u32 = 32;

#[derive(Clone, Copy, Debug)]
struct IndexLogitsLaunch {
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
}

/// Validate the kernel contract before constructing or submitting a launch.
#[allow(clippy::too_many_arguments)]
fn index_logits_launch(
    query_address: u64,
    rows: u32,
    seq_len_start: u32,
    logits_stride: u32,
    index_heads: u32,
    head_dim: u32,
    pool_size: u32,
    cache_block_size: u32,
    rows_per_cta: u32,
    pools_per_cta: u32,
) -> Result<IndexLogitsLaunch> {
    anyhow::ensure!(
        rows > 0 && logits_stride > 0 && index_heads > 0 && head_dim > 0,
        "GLM semantic scorer dimensions must be nonzero"
    );
    anyhow::ensure!(
        pool_size > 0 && cache_block_size > 0 && cache_block_size.is_multiple_of(pool_size),
        "GLM semantic scorer cache blocks must contain whole nonempty pools"
    );
    anyhow::ensure!(
        seq_len_start.checked_add(rows).is_some() && head_dim.checked_mul(index_heads).is_some(),
        "GLM semantic scorer sequence or head dimensions overflow u32"
    );
    let v2 = pools_per_cta == GLM_INDEX_LOGITS_V2_POOLS;
    anyhow::ensure!(
        (pools_per_cta == 8 && matches!(rows_per_cta, 1 | 8))
            || (((pools_per_cta, rows_per_cta) == (32, 8)
                || (v2 && rows_per_cta == GLM_INDEX_LOGITS_V2_ROWS))
                && index_heads == 32
                && head_dim == 128
                && pool_size == 4
                && query_address.is_multiple_of(32)),
        "unsupported GLM semantic scorer launch geometry or query alignment"
    );
    let shared_mem = if v2 {
        // index_v2_smem_bytes: two BF16 key stages plus one 32-pool x 32-head
        // FP32 product tile per warp.
        2 * V2_CHUNK_POOLS * 256 + GLM_INDEX_LOGITS_V2_ROWS * 32 * 32 * 4
    } else if rows_per_cta == 8 && pools_per_cta == 8 {
        head_dim
            .checked_mul(8 * std::mem::size_of::<u16>() as u32)
            .ok_or_else(|| anyhow::anyhow!("GLM semantic scorer shared memory overflows u32"))?
    } else {
        0
    };
    anyhow::ensure!(
        shared_mem <= 48 * 1024,
        "GLM semantic scorer exceeds default shared memory limit"
    );
    let row_tiles = rows.div_ceil(rows_per_cta);
    let threads = if v2 {
        GLM_INDEX_LOGITS_V2_ROWS * 32
    } else {
        256
    };
    let grid_x = if v2 {
        // About eight waves of three CTAs on GB10's 48 SMs, but at least 16
        // chunks per CTA (so query rows are reloaded rarely) unless that
        // leaves less than one wave.
        let chunks = logits_stride.div_ceil(V2_CHUNK_POOLS);
        let cap = (chunks / 16).max(144u32.div_ceil(row_tiles));
        1152u32.div_ceil(row_tiles).min(cap).min(chunks).max(1)
    } else {
        logits_stride.div_ceil(pools_per_cta)
    };
    Ok(IndexLogitsLaunch {
        grid: [grid_x, row_tiles, 1],
        block: [threads, 1, 1],
        shared_mem,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_logits(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    query: DevicePtr,
    weights: DevicePtr,
    index_cache: DevicePtr,
    logits: DevicePtr,
    block_table: DevicePtr,
    rows: u32,
    seq_len_start: u32,
    logits_stride: u32,
    index_heads: u32,
    head_dim: u32,
    pool_size: u32,
    cache_block_size: u32,
    index_block_stride_bytes: u64,
    rows_per_cta: u32,
    pools_per_cta: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        pools_per_cta != GLM_INDEX_LOGITS_V2_POOLS
            || (index_cache.0.is_multiple_of(16) && index_block_stride_bytes.is_multiple_of(16)),
        "GLM semantic scorer v2 stages pooled keys with 16-byte copies"
    );
    let launch = index_logits_launch(
        query.0,
        rows,
        seq_len_start,
        logits_stride,
        index_heads,
        head_dim,
        pool_size,
        cache_block_size,
        rows_per_cta,
        pools_per_cta,
    )?;
    KernelLaunch::new(gpu, kernel)
        .grid(launch.grid)
        .block(launch.block)
        .shared_mem(launch.shared_mem)
        .arg_ptr(query)
        .arg_ptr(weights)
        .arg_ptr(index_cache)
        .arg_ptr(logits)
        .arg_ptr(block_table)
        .arg_u32(rows)
        .arg_u32(seq_len_start)
        .arg_u32(logits_stride)
        .arg_u32(index_heads)
        .arg_u32(head_dim)
        .arg_u32(pool_size)
        .arg_u32(cache_block_size)
        .arg_u64(index_block_stride_bytes)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_index_topk_expand(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    output: DevicePtr,
    rows: u32,
    seq_len_start: u32,
    logits_stride: u32,
    topk_tokens: u32,
    pool_size: u32,
    output_width: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .shared_mem(4 * std::mem::size_of::<u32>() as u32)
        .arg_ptr(logits)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(seq_len_start)
        .arg_u32(logits_stride)
        .arg_u32(topk_tokens)
        .arg_u32(pool_size)
        .arg_u32(output_width)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn glm_sparse_mla_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    query: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    token_indices: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    rows: u32,
    num_heads: u32,
    head_dim: u32,
    index_width: u32,
    cache_block_size: u32,
    heads_per_cta: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    let heads_per_cta = if heads_per_cta == 8 { 8 } else { 1 };
    // head8 uses scores[8][8], betas[8][8], and three per-head state arrays.
    let shared_floats = if heads_per_cta == 8 { 152 } else { 19 };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_heads, heads_per_cta), rows, 1])
        .block([256, 1, 1])
        .shared_mem(shared_floats * std::mem::size_of::<f32>() as u32)
        .arg_ptr(query)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(token_indices)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(rows)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .arg_u32(index_width)
        .arg_u32(cache_block_size)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(rows_per_cta: u32, pools_per_cta: u32) -> Result<IndexLogitsLaunch> {
        index_logits_launch(
            256,
            9,
            4096,
            33,
            32,
            128,
            4,
            32,
            rows_per_cta,
            pools_per_cta,
        )
    }

    #[test]
    fn scorer_tiles_cover_partial_rows_and_pools() {
        let scalar = geometry(1, 8).unwrap();
        let row8 = geometry(8, 8).unwrap();
        let wmma = geometry(8, 32).unwrap();
        assert_eq!(scalar.grid, [5, 9, 1]);
        assert_eq!(row8.grid, [5, 2, 1]);
        assert_eq!(wmma.grid, [2, 2, 1]);
        for launch in [scalar, row8, wmma] {
            assert_eq!(launch.block, [256, 1, 1]);
        }
    }

    #[test]
    fn only_scalar_row8_requests_dynamic_shared_memory() {
        assert_eq!(geometry(1, 8).unwrap().shared_mem, 0);
        assert_eq!(geometry(8, 8).unwrap().shared_mem, 2048);
        // WMMA owns 25,600 static bytes; adding those again as dynamic memory
        // would exceed its intended per-CTA footprint and hurt occupancy.
        assert_eq!(geometry(8, 32).unwrap().shared_mem, 0);
    }

    #[test]
    fn single_row_wmma_and_exact_tiles_use_nonzero_grids() {
        let single = index_logits_launch(256, 1, 4096, 1, 32, 128, 4, 32, 8, 32).unwrap();
        let exact = index_logits_launch(256, 16, 4096, 64, 32, 128, 4, 32, 8, 32).unwrap();
        assert_eq!(single.grid, [1, 1, 1]);
        assert_eq!(exact.grid, [2, 2, 1]);
    }

    #[test]
    fn unsupported_tiles_are_rejected() {
        let v2 = GLM_INDEX_LOGITS_V2_POOLS;
        for (rows, pools) in [(0, 8), (4, 8), (1, 32), (8, v2), (8, 0), (8, 16), (8, 64)] {
            assert!(
                geometry(rows, pools).is_err(),
                "accepted tile {rows}x{pools}"
            );
        }
    }

    #[test]
    fn wmma_requires_exact_shape_and_query_alignment() {
        for (query, heads, dim, pool) in [
            (258, 32, 128, 4),
            (256, 16, 128, 4),
            (256, 32, 64, 4),
            (256, 32, 128, 2),
        ] {
            for (rows, pools) in [(8, 32), (4, GLM_INDEX_LOGITS_V2_POOLS)] {
                let launch =
                    index_logits_launch(query, 9, 4096, 33, heads, dim, pool, 32, rows, pools);
                assert!(launch.is_err());
            }
        }
        // WMMA's stricter alignment and specialization must not accidentally
        // remove shapes supported by the two scalar implementations.
        for rows_per_cta in [1, 8] {
            assert!(index_logits_launch(258, 9, 4096, 33, 16, 64, 2, 32, rows_per_cta, 8).is_ok());
        }
    }

    #[test]
    fn empty_dimensions_and_incomplete_cache_pools_are_rejected() {
        for (rows_per_cta, pools_per_cta) in
            [(1, 8), (8, 8), (8, 32), (4, GLM_INDEX_LOGITS_V2_POOLS)]
        {
            for (rows, stride, heads, dim, pool, block) in [
                (0, 33, 32, 128, 4, 32),
                (9, 0, 32, 128, 4, 32),
                (9, 33, 0, 128, 4, 32),
                (9, 33, 32, 0, 4, 32),
                (9, 33, 32, 128, 0, 32),
                (9, 33, 32, 128, 4, 0),
                (9, 33, 32, 128, 4, 2),
                (9, 33, 32, 128, 4, 34),
            ] {
                assert!(
                    index_logits_launch(
                        256,
                        rows,
                        4096,
                        stride,
                        heads,
                        dim,
                        pool,
                        block,
                        rows_per_cta,
                        pools_per_cta
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn arithmetic_and_shared_memory_limits_are_checked() {
        assert!(index_logits_launch(256, 9, u32::MAX, 33, 32, 128, 4, 32, 8, 32).is_err());
        assert!(index_logits_launch(256, 9, 4096, 33, u32::MAX, 128, 4, 32, 1, 8).is_err());
        assert!(index_logits_launch(256, 9, 4096, 33, 1, 4096, 4, 32, 8, 8).is_err());
        assert!(index_logits_launch(256, 9, 4096, 33, 1, u32::MAX, 4, 32, 8, 8).is_err());
    }
}

#[cfg(test)]
#[path = "glm_indexer_v2_tests.rs"]
mod v2_tests;
