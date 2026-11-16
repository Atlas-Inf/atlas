// SPDX-License-Identifier: AGPL-3.0-only

//! The indexer's scoring kernel (`dsv41_index.cu`): what each compressed
//! position scores for one query.

use anyhow::{Result, ensure};
use atlas_core::numeric::f32_to_bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Values in one index key, and in one index head's query.
pub const INDEX_DIM: usize = 128;

const THREADS: u32 = 256;
const MAX_Q: usize = 4096;
const MAX_HEADS: usize = 64;

#[derive(Clone, Copy)]
pub struct IndexKernels {
    pub score: KernelHandle,
}

impl IndexKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            score: gpu.kernel("dsv41_index", "dsv41_index_score")?,
        })
    }
}

/// What the reference multiplies the head weights by: `128^-0.5 *
/// heads^-0.5`, as the BF16 scalar it becomes there.
pub fn head_weight_scale(heads: usize) -> f32 {
    let scale = ((INDEX_DIM * heads) as f64).powf(-0.5) as f32;
    f32::from_bits(u32::from(f32_to_bf16(scale)) << 16)
}

/// `partial` (`[groups]` FP32) = each index key's score for one query, summed
/// over `heads` heads and not yet rounded: with the heads split over ranks,
/// the ranks' parts are added first. `q` is `[heads, 128]` BF16, rotated and
/// FP4-rounded; `keys` `[groups, 128]` BF16; `weights` `[heads]` BF16, each
/// multiplied by `weight_scale` and rounded to BF16.
#[allow(clippy::too_many_arguments)]
pub fn score(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    keys: DevicePtr,
    weights: DevicePtr,
    partial: DevicePtr,
    heads: usize,
    groups: usize,
    weight_scale: f32,
    stream: u64,
) -> Result<()> {
    if groups == 0 {
        return Ok(());
    }
    // One query that sees `groups` keys.
    let rows = Queries {
        count: 1,
        position: groups - 1,
        ratio: 1,
        stride: groups,
    };
    score_rows(
        gpu,
        kernel,
        q,
        keys,
        weights,
        partial,
        heads,
        rows,
        weight_scale,
        stream,
    )
}

/// The queries of one launch of [`score_rows`]: `count` tokens from
/// `position` on a layer of compression `ratio`, each seeing the keys of the
/// groups complete at its position; `stride` values of `partial` a query.
#[derive(Clone, Copy)]
pub struct Queries {
    pub count: usize,
    pub position: usize,
    pub ratio: usize,
    pub stride: usize,
}

/// [`score`] for a row of queries: `q` `[count, heads, 128]`, `weights`
/// `[count, heads]`, `partial` `[count, stride]`.
#[allow(clippy::too_many_arguments)]
pub fn score_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    keys: DevicePtr,
    weights: DevicePtr,
    partial: DevicePtr,
    heads: usize,
    rows: Queries,
    weight_scale: f32,
    stream: u64,
) -> Result<()> {
    ensure!(
        heads <= MAX_HEADS && heads * INDEX_DIM <= MAX_Q,
        "dsv41_index_score: {heads} heads"
    );
    if rows.count == 0 || rows.stride == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(rows.stride as u32, THREADS), rows.count as u32, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(q)
        .arg_ptr(keys)
        .arg_ptr(weights)
        .arg_ptr(partial)
        .arg_u32(heads as u32)
        .arg_u32(INDEX_DIM as u32)
        .arg_u32(rows.stride as u32)
        .arg_f32(weight_scale)
        .arg_u32(rows.position as u32)
        .arg_u32(rows.ratio as u32)
        .launch(stream)
}

#[cfg(test)]
#[path = "index_test_gpu.rs"]
mod test_gpu;
