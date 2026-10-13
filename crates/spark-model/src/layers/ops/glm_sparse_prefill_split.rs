// SPDX-License-Identifier: AGPL-3.0-only
//! Few-row GLM sparse attention split over the selected IDs, and its count.
use super::*;

/// Splits of the selected IDs for `rows` query rows: the count (at most 16,
/// the merge's cap) minimizing waves x 32-key tiles per CTA, with one CTA
/// per SM (69 KB of shared memory) on GB10's 48 SMs. 1 = the unsplit kernel.
pub(crate) fn sparse_split_count(rows: u32, heads: u32, index_width: u32) -> u32 {
    let ctas = rows * heads.div_ceil(32);
    let tiles = index_width.div_ceil(32);
    let cost = |s: u32| (ctas * s).div_ceil(48) * tiles.div_ceil(s);
    (1..=16u32).min_by_key(|&s| (cost(s), s)).unwrap_or(1)
}

/// `ATLAS_GLM_SPARSE_VERIFY_SPLIT_PIN=1`: see [`sparse_owner_splits`].
pub(super) const SPLIT_PIN: &str = "ATLAS_GLM_SPARSE_VERIFY_SPLIT_PIN";

/// Splits for an owner of `rows` rows. The split-merge is not associative, so
/// a count that follows the launch gives one position different bits at each
/// verify width, and the width follows the co-batched owners and the adaptive
/// width. Pinned (like `split_ref_seqs` for decode), an owner no wider than a
/// DFlash verify block takes the widest block's count: that block keeps its
/// bits and its one wave of CTAs, narrower owners change. Wider owners
/// (prefill pieces) are not pinned; a prefill piece of under eight rows is.
pub(crate) fn sparse_owner_splits(rows: u32, heads: u32, index_width: u32, pin: bool) -> u32 {
    let verify_rows = crate::speculative::glm_repair_policy::MAX_DFLASH_VERIFY_ROWS as u32;
    let ref_rows = if pin { rows.max(verify_rows) } else { rows };
    sparse_split_count(ref_rows, heads, index_width)
}

/// Bytes of split scratch: FP32 partial outputs, their LSEs, merged LSEs.
pub(crate) fn sparse_split_scratch_bytes(
    splits: u32,
    rows: u32,
    heads: u32,
    head_dim: u32,
) -> usize {
    let rh = rows as usize * heads as usize;
    splits as usize * rh * (head_dim as usize + 1) * 4 + rh * 4
}

/// The switches one split dispatch reads: `ATLAS_GLM_SPARSE_VERIFY_SPLIT`
/// (`on`), `_PREFILL_TC`, `_PREFILL_KV_REUSE`, `_PREFILL_PIPE` and
/// `_VERIFY_SPLIT_PIN`.
#[derive(Clone, Copy)]
pub(super) struct SplitFlags {
    pub on: bool,
    pub tc: bool,
    pub kv_reuse: bool,
    pub pipe: bool,
    pub pin: bool,
}

/// `try_glm_sparse_prefill_tc` for few-row callers (verify owners): split
/// over the selected IDs (`*_split` kernel + `glm_sparse_decode_split_merge`)
/// when `sparse_split_count` > 1 and `scratch` fits, so the rows fill the GPU
/// instead of one CTA each. Partial sums in a different order than the
/// unsplit kernel. `ATLAS_GLM_SPARSE_VERIFY_SPLIT=0` disables;
/// `ATLAS_GLM_SPARSE_VERIFY_SPLIT_PIN=1` pins the count of verify-sized owners.
pub fn try_glm_sparse_prefill_tc_split(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    scratch: DevicePtr,
    scratch_bytes: usize,
    stream: u64,
) -> Result<bool> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on =
        *ON.get_or_init(|| std::env::var("ATLAS_GLM_SPARSE_VERIFY_SPLIT").as_deref() != Ok("0"));
    let model = &a.config.model_type;
    let flags = SplitFlags {
        on,
        tc: enabled(model, "ATLAS_GLM_SPARSE_PREFILL_TC")?,
        kv_reuse: enabled(model, "ATLAS_GLM_SPARSE_PREFILL_KV_REUSE")?,
        pipe: enabled(model, PIPE)?,
        pin: enabled(model, SPLIT_PIN)?,
    };
    dispatch_split(gpu, a, scratch, scratch_bytes, stream, flags)
}

pub(super) fn dispatch_split(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    scratch: DevicePtr,
    scratch_bytes: usize,
    stream: u64,
    f: SplitFlags,
) -> Result<bool> {
    let splits = sparse_owner_splits(a.rows, a.heads, a.index_width, f.pin);
    let need = sparse_split_scratch_bytes(splits, a.rows, a.heads, a.head_dim);
    if !f.on
        || !f.tc
        || !f.kv_reuse
        || !a.identical_kv_latent
        || splits <= 1
        || scratch.0 == 0
        || !scratch.0.is_multiple_of(16)
        || scratch_bytes < need
    {
        return dispatch(gpu, a, stream, f.tc, f.kv_reuse, f.pipe);
    }
    if f.pin {
        log_pin(gpu, a, splits);
    }
    let rh = a.rows as usize * a.heads as usize;
    let part_lse = scratch.offset(splits as usize * rh * a.head_dim as usize * 4);
    let out_lse = part_lse.offset(splits as usize * rh * 4);
    launch_sparse_partials(gpu, a, splits, None, scratch, part_lse, stream)?;
    decode_split::launch_merge(
        gpu,
        decode_split::merge_kernel(gpu)?,
        scratch,
        part_lse,
        a.output,
        out_lse,
        a.rows,
        a.rows * a.heads,
        splits,
        stream,
    )?;
    Ok(true)
}

/// The `*_split` kernel alone: `splits` normalized FP32 partial outputs
/// `[splits, rows, heads, 512]` and their natural LSEs `[splits, rows, heads]`
/// (`-inf` for a partition with no valid ID) for a caller-side LSE merge.
/// With `row_counts` (`u32[rows]`) row `r` selects only its first
/// `row_counts[r]` IDs and the splits partition that prefix.
pub(crate) fn launch_sparse_partials(
    gpu: &dyn GpuBackend,
    a: &GlmSparsePrefillTc<'_>,
    splits: u32,
    row_counts: Option<DevicePtr>,
    part_o: DevicePtr,
    part_lse: DevicePtr,
    stream: u64,
) -> Result<()> {
    validate_geometry(a)?;
    validate_storage(a)?;
    ensure!(
        (1..=16).contains(&splits) && a.rows > 0 && a.identical_kv_latent,
        "GLM sparse partials need 1..=16 splits, rows and identical K/V"
    );
    let (module, _, shared_mem) = kernel_spec(true, a.dtype, false);
    let symbol = match (a.dtype == KvCacheDtype::Fp8G128, row_counts.is_some()) {
        (true, false) => "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split",
        (false, false) => "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split",
        (true, true) => "glm_sparse_mla_prefill_fp8g128_head32_tc_kv_pad_split_counted",
        (false, true) => "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad_split_counted",
    };
    // The counted entry points are the shard's (`glm_kv_shard.cu`).
    let module = row_counts.map_or(module, |_| super::super::glm_kv_shard::MODULE);
    let kernel = gpu.op_cache().kernel(gpu, module, symbol)?;
    ensure!(kernel.0 != 0, "GLM sparse split kernel is unavailable");
    let launch = KernelLaunch::new(gpu, kernel)
        .grid([a.heads.div_ceil(32), a.rows, splits])
        .block([256, 1, 1])
        .shared_mem(shared_mem)
        .arg_ptr(a.query)
        .arg_ptr(a.k_cache)
        .arg_ptr(a.v_cache)
        .arg_ptr(a.indices)
        .arg_ptr(a.output)
        .arg_ptr(a.block_table)
        .arg_u32(a.rows)
        .arg_u32(a.heads)
        .arg_u32(a.head_dim)
        .arg_u32(a.index_width)
        .arg_u32(a.block_size)
        .arg_f32(a.scale)
        .arg_ptr(part_o)
        .arg_ptr(part_lse);
    match row_counts {
        Some(counts) => launch.arg_ptr(counts),
        None => launch,
    }
    .launch(stream)
}

/// A pinned split launch, once per backend: that the pin is live, then each
/// row count whose bits it changes. Only a launch that really splits logs.
fn log_pin(gpu: &dyn GpuBackend, a: &GlmSparsePrefillTc<'_>, splits: u32) {
    let cache = gpu.op_cache();
    if cache.once("glm:sparse_split_pin") {
        let rows = crate::speculative::glm_repair_policy::MAX_DFLASH_VERIFY_ROWS;
        let pinned = sparse_owner_splits(1, a.heads, a.index_width, true);
        tracing::info!(
            "{SPLIT_PIN}=1: active, sparse owners of 1..={rows} rows use {pinned} splits"
        );
    }
    let unpinned = sparse_split_count(a.rows, a.heads, a.index_width);
    if splits != unpinned && cache.first_shape("glm:sparse_split_pin", a.rows, 0, 0) {
        tracing::info!(
            "{SPLIT_PIN}=1: rows={} splits {unpinned} -> {splits}",
            a.rows
        );
    }
}
