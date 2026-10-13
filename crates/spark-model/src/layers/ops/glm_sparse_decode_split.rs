// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed-S8 repaired K3 attention. Reuses dead per-pass expert scratch.
use super::*;
use spark_runtime::gpu::KernelHandle;

const FLAG: &str = "ATLAS_GLM_SPARSE_DECODE_SPLIT";
const SPLITS: u32 = 8;
const PART_O_BYTES: usize = 8 * 32 * 512 * 4;
const PART_LSE_BYTES: usize = 8 * 32 * 4;
const SCRATCH_BYTES: usize = PART_O_BYTES + PART_LSE_BYTES + 32 * 4;

pub fn glm_sparse_decode_split_enabled(model: &str) -> Result<bool> {
    enabled(model, FLAG)
}

fn extent(p: DevicePtr, bytes: usize) -> Result<(u64, u64)> {
    ensure!(p.0 != 0 && bytes != 0, "GLM split has missing storage");
    let end =
        p.0.checked_add(bytes as u64)
            .ok_or_else(|| anyhow::anyhow!("GLM split pointer extent overflow"))?;
    Ok((p.0, end))
}

fn validate_scratch(p: DevicePtr, bytes: usize, live: &[(DevicePtr, usize)]) -> Result<()> {
    ensure!(
        bytes >= SCRATCH_BYTES && p.0.is_multiple_of(16),
        "GLM S8 split needs aligned 525440-byte scratch"
    );
    let (start, end) = extent(p, SCRATCH_BYTES)?;
    for &(other, n) in live {
        if n == 0 {
            continue;
        }
        let (lo, hi) = extent(other, n)?;
        ensure!(
            end <= lo || hi <= start,
            "GLM split scratch overlaps live storage"
        );
    }
    Ok(())
}

/// Run before any causal cache writes. Caller supplies actual arena/cache spans.
pub fn validate_glm_sparse_decode_split_scratch(
    model: &str,
    p: DevicePtr,
    bytes: usize,
    live: &[(DevicePtr, usize)],
) -> Result<()> {
    if !glm_sparse_decode_split_enabled(model)? {
        return Ok(());
    }
    ensure!(
        glm_sparse_decode_tc_enabled(model)?,
        "GLM split requires ATLAS_GLM_SPARSE_DECODE_TC=1"
    );
    validate_scratch(p, bytes, live)
}

pub fn try_glm_sparse_decode_split(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    scratch: DevicePtr,
    scratch_bytes: usize,
    stream: u64,
) -> Result<bool> {
    dispatch(
        gpu,
        a,
        scratch,
        scratch_bytes,
        stream,
        glm_sparse_decode_split_enabled(&a.config.model_type)?,
        glm_sparse_decode_tc_enabled(&a.config.model_type)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    scratch: DevicePtr,
    scratch_bytes: usize,
    stream: u64,
    on: bool,
    tc: bool,
) -> Result<bool> {
    // The split kernel reads BF16 latents; an fp8_g128 cache declines to the
    // TC kernel, which has its reader. Other dtypes still fail validation.
    if !on || a.dtype == spark_runtime::kv_cache::KvCacheDtype::Fp8G128 {
        return Ok(false);
    }
    ensure!(tc, "GLM split requires ATLAS_GLM_SPARSE_DECODE_TC=1");
    validate_geometry(a)?;
    ensure!(
        a.rows == 1 && a.identical_kv_latent,
        "GLM split requires one repaired row and identical K/V"
    );
    validate_storage(a)?;
    ensure!(
        [a.query, a.k_cache, a.v_cache]
            .iter()
            .all(|p| p.0.is_multiple_of(16))
            && [a.indices, a.output, a.block_table]
                .iter()
                .all(|p| p.0.is_multiple_of(4)),
        "GLM split vector/storage alignment"
    );
    let input = [
        (a.query, 32 * 512 * 2),
        (a.k_cache, 1),
        (a.v_cache, 1),
        (a.indices, 2051 * 4),
        (a.block_table, 4),
    ];
    validate_scratch(scratch, scratch_bytes, &input)?;
    validate_scratch(scratch, scratch_bytes, &[(a.output, 32 * 512 * 2)])?;
    let (lo, hi) = extent(a.output, 32 * 512 * 2)?;
    for (p, n) in input {
        let (a, b) = extent(p, n)?;
        ensure!(hi <= a || b <= lo, "GLM split output aliases input");
    }
    let (split, merge) = kernels(gpu)?;
    launch_split(gpu, split, a, scratch, scratch.offset(PART_O_BYTES), stream)?;
    launch_merge(
        gpu,
        merge,
        scratch,
        scratch.offset(PART_O_BYTES),
        a.output,
        scratch.offset(PART_O_BYTES + PART_LSE_BYTES),
        1,
        32,
        SPLITS,
        stream,
    )?;
    Ok(true)
}

pub(super) fn merge_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    let merge = gpu.op_cache().kernel(
        gpu,
        "glm_sparse_decode_split_merge",
        "glm_sparse_decode_split_merge",
    )?;
    ensure!(merge.0 != 0, "GLM split merge kernel unavailable");
    Ok(merge)
}

fn kernels(gpu: &dyn GpuBackend) -> Result<(KernelHandle, KernelHandle)> {
    // Handles belong to this backend/context, never a process-global static.
    let split =
        gpu.op_cache()
            .kernel(gpu, "glm_sparse_decode_split", "atlas_sparse_decode_split")?;
    ensure!(split.0 != 0, "GLM split kernels unavailable");
    Ok((split, merge_kernel(gpu)?))
}

fn launch_split(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: &GlmSparsePrefillTc<'_>,
    out: DevicePtr,
    lse: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([1, 1, SPLITS])
        .block([256, 1, 1])
        .shared_mem(69376)
        .arg_ptr(a.query)
        .arg_ptr(a.k_cache)
        .arg_ptr(a.v_cache)
        .arg_ptr(a.indices)
        .arg_ptr(out)
        .arg_ptr(a.block_table)
        .arg_u32(a.rows)
        .arg_u32(a.heads)
        .arg_u32(a.head_dim)
        .arg_u32(a.index_width)
        .arg_u32(a.block_size)
        .arg_f32(a.scale)
        .arg_ptr(lse)
        .arg_u32(SPLITS)
        .launch(stream)
}

/// `glm_sparse_decode_split_merge` over `grid` = rows x 32 heads.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_merge(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    part: DevicePtr,
    plse: DevicePtr,
    out: DevicePtr,
    lse: DevicePtr,
    rows: u32,
    grid: u32,
    splits: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(part)
        .arg_ptr(plse)
        .arg_ptr(out)
        .arg_ptr(lse)
        .arg_u32(rows)
        .arg_u32(32)
        .arg_u32(512)
        .arg_u32(splits)
        .launch(stream)
}

pub fn initialize_glm_sparse_decode_split(gpu: &dyn GpuBackend, c: &ModelConfig) -> Result<()> {
    initialize(
        gpu,
        c,
        glm_sparse_decode_split_enabled(&c.model_type)?,
        glm_sparse_decode_tc_enabled(&c.model_type)?,
    )
}

fn initialize(gpu: &dyn GpuBackend, c: &ModelConfig, on: bool, tc: bool) -> Result<()> {
    if !on {
        return Ok(());
    }
    ensure!(tc, "GLM split requires ATLAS_GLM_SPARSE_DECODE_TC=1");
    validate_config(c)?;
    let (split, merge) = kernels(gpu)?;
    let empty = GlmSparsePrefillTc {
        config: c,
        dtype: KvCacheDtype::Bf16,
        identical_kv_latent: true,
        query: DevicePtr::NULL,
        k_cache: DevicePtr::NULL,
        v_cache: DevicePtr::NULL,
        indices: DevicePtr::NULL,
        output: DevicePtr::NULL,
        block_table: DevicePtr::NULL,
        rows: 0,
        heads: 32,
        head_dim: 512,
        index_width: 2051,
        block_size: 16,
        scale: 0.0625,
    };
    let stream = gpu.default_stream();
    // Init-only zero rows uniformly return before any pointer access in BOTH
    // kernels. Real launches force lazy code/shared opt-in before KV sizing.
    launch_split(gpu, split, &empty, DevicePtr::NULL, DevicePtr::NULL, stream)?;
    launch_merge(
        gpu,
        merge,
        DevicePtr::NULL,
        DevicePtr::NULL,
        DevicePtr::NULL,
        DevicePtr::NULL,
        0,
        1,
        SPLITS,
        stream,
    )?;
    gpu.synchronize(stream)
}

#[cfg(test)]
#[path = "glm_sparse_decode_split_tests.rs"]
mod tests;
