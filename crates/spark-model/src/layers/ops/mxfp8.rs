// SPDX-License-Identifier: AGPL-3.0-only
//! MXFP8 (E4M3 + E8M0 per 32 values) weights for small-row projections.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::Mxfp8Weight;

/// Values per E8M0 scale block.
pub const MXFP8_BLOCK: usize = 32;

/// Quantize a BF16 `[n, k]` weight into E4M3 `data` (`n*k` bytes) and E8M0
/// `scales` (`n*k/32` bytes).
pub fn mxfp8_quantize(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    weight: DevicePtr,
    data: DevicePtr,
    scales: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(MXFP8_BLOCK),
        "MXFP8 needs K % 32 == 0 (K={k})"
    );
    let blocks = (n * k / MXFP8_BLOCK) as u64;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(blocks as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight)
        .arg_ptr(data)
        .arg_ptr(scales)
        .arg_u64(blocks)
        .launch(stream)
}

/// `C[m, n] = A[m, k] · dequant(W)[n, k]ᵀ` on the tensor cores; `kernel` is
/// `mxfp8_gemv_tc8/16/32` for up to 8/16/32 rows (their `_tiled` twins on a
/// tiled twin: [`mxfp8_kernel`]).
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    data: DevicePtr,
    scales: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=32).contains(&m) && (k as usize).is_multiple_of(MXFP8_BLOCK),
        "mxfp8_gemv: m={m} k={k} unsupported"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 16), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(data)
        .arg_ptr(scales)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// Rows the widest plain tier (`mxfp8_gemv_tc32`) serves in one launch.
pub const MXFP8_MAX_ROWS: usize = 32;

/// [`mxfp8_gemv`] over any number of rows: `rows` rows of `input` (`[rows,
/// k]`, contiguous) through twin `w` (`[n, k]`) into `output` (row `r` at
/// `output + r * out_stride` elements), in launches of up to
/// [`MXFP8_MAX_ROWS`] rows, each on
/// [`mxfp8_kernel`]'s tier for its width (`row_major` holds
/// `mxfp8_gemv_tc{8,16,32}`).
///
/// A row's output bytes do not depend on how many rows ride with it: every
/// tier keeps each 8-row activation tile in its own MMA accumulators and sums
/// the eight K-split warps in one fixed order (`mxfp8_tiled_test_gpu.rs`
/// checks rows at `m` against the same rows at 32). So one copy can serve a
/// 1-row decode and a k-row verify or multi-sequence batch and they agree bit
/// for bit. Chunking past 32 rows keeps that; handing wide batches to a BF16
/// GEMM instead would not.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_gemv_rows(
    gpu: &dyn GpuBackend,
    row_major: &[KernelHandle; 3],
    w: &Mxfp8Weight,
    input: DevicePtr,
    output: DevicePtr,
    rows: usize,
    n: usize,
    k: usize,
    out_stride: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        out_stride >= n,
        "mxfp8_gemv_rows: out_stride {out_stride} < n {n}"
    );
    for first in (0..rows).step_by(MXFP8_MAX_ROWS) {
        let m = (rows - first).min(MXFP8_MAX_ROWS) as u32;
        let kernel = mxfp8_kernel(gpu, w, row_major, m);
        ensure!(
            kernel.0 != 0,
            "mxfp8_gemv_rows: no kernel for {m} rows (tiled={})",
            w.tiled
        );
        mxfp8_gemv(
            gpu,
            kernel,
            input.offset(first * k * 2),
            w.data,
            w.scales,
            output.offset(first * out_stride * 2),
            m,
            n as u32,
            k as u32,
            out_stride as u32,
            stream,
        )?;
    }
    Ok(())
}

/// The [`mxfp8_gemv`] tier for `m` rows: 0 (`tc8`, up to 8), 1 (`tc16`) or
/// 2 (`tc32`).
pub fn mxfp8_tier(m: u32) -> usize {
    match m {
        0..=8 => 0,
        9..=16 => 1,
        _ => 2,
    }
}

/// Outputs a `mxfp8_gemv_tc{8,16,32}_swiglu` CTA writes.
const SWIGLU_CTA_COLS: u32 = 8;

/// `SiLU(gate) * up` of `rows` rows of `input` (`[rows, k]`, contiguous) into
/// `output` (`[rows, inter]`, contiguous) through the MXFP8 `w` = `[gate;
/// up]` (`[2 * inter, k]`), each projection value the tensor-core tiers' sum
/// rounded to BF16 and the activation rounded as vLLM's `silu_and_mul`
/// rounds it (`mx_swiglu` in `mxfp8_gemv.cu`). `swiglu` holds
/// `mxfp8_gemv_tc{8,16,32}_swiglu`; launches of up to 32 rows, each row's
/// bytes independent of the rows beside it.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_swiglu_rows(
    gpu: &dyn GpuBackend,
    swiglu: &[KernelHandle; 3],
    w: &crate::weight_map::Mxfp8Weight,
    input: DevicePtr,
    output: DevicePtr,
    rows: usize,
    inter: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(MXFP8_BLOCK) && inter > 0,
        "mxfp8_swiglu_rows: inter={inter} k={k} (need k % 32 == 0)"
    );
    for first in (0..rows).step_by(32) {
        let m = (rows - first).min(32) as u32;
        let kernel = swiglu[mxfp8_tier(m)];
        ensure!(kernel.0 != 0, "mxfp8_swiglu_rows: no kernel for {m} rows");
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(inter as u32, SWIGLU_CTA_COLS), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input.offset(first * k * 2))
            .arg_ptr(w.data)
            .arg_ptr(w.scales)
            .arg_ptr(output.offset(first * inter * 2))
            .arg_u32(m)
            .arg_u32(inter as u32)
            .arg_u32(k as u32)
            .arg_u32(inter as u32)
            .launch(stream)?;
    }
    Ok(())
}

fn parse_tiled(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => anyhow::bail!("ATLAS_GLM_MXFP8_TILED must be 0 or 1, got {other:?}"),
    }
}

/// `ATLAS_GLM_MXFP8_TILED` as set (read once): the GLM MLA q_b and o twins
/// and the split verify head's shard take the `mxfp8_tile16` kernel's layout at load
/// and the `mxfp8_gemv_tc{8,16,32}_tiled` tiers. Every output bit is the
/// row-major tiers' (the same values into the same MMAs in the same order);
/// each warp's 64-value chunk of 16 rows is one contiguous KiB instead of
/// sixteen 64-byte pieces K bytes apart.
pub fn mxfp8_tiled_requested() -> Result<bool> {
    static ON: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    ON.get_or_init(|| {
        parse_tiled(std::env::var("ATLAS_GLM_MXFP8_TILED").ok().as_deref())
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// `mxfp8_gemv_tc{8,16,32}_tiled` and `mxfp8_tile16`, resolved once (zero
/// handles when the target lacks them).
fn tiled_kernels(gpu: &dyn GpuBackend) -> ([KernelHandle; 3], KernelHandle) {
    static K: std::sync::OnceLock<([KernelHandle; 3], KernelHandle)> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        let k = |name: &str| crate::layers::try_kernel(gpu, "mxfp8_gemv", name);
        (
            [
                "mxfp8_gemv_tc8_tiled",
                "mxfp8_gemv_tc16_tiled",
                "mxfp8_gemv_tc32_tiled",
            ]
            .map(k),
            k("mxfp8_tile16"),
        )
    })
}

/// The kernel serving `m` rows of twin `w`: its `_tiled` tier when `w` is
/// tiled, else `row_major[mxfp8_tier(m)]`.
pub fn mxfp8_kernel(
    gpu: &dyn GpuBackend,
    w: &Mxfp8Weight,
    row_major: &[KernelHandle; 3],
    m: u32,
) -> KernelHandle {
    let tiers = if w.tiled {
        tiled_kernels(gpu).0
    } else {
        *row_major
    };
    tiers[mxfp8_tier(m)]
}

/// Row-major bytes staged per `mxfp8_tile16` launch of [`mxfp8_tile_in_place`].
const TILE_STAGE_BYTES: usize = 32 << 20;

/// Rewrite the row-major twin `w` (`[n, k]`, `n % 16 == 0`, `k % 64 == 0`) in
/// place into `mxfp8_tile16`'s chunk-tiled layout and mark it tiled. Each
/// 16-row tile keeps its byte range, so the twin is rewritten a staged run of
/// whole tiles at a time. Load time only: synchronizes `stream`.
pub fn mxfp8_tile_in_place(
    gpu: &dyn GpuBackend,
    w: &mut Mxfp8Weight,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    ensure!(!w.tiled, "mxfp8_tile16: the twin is already tiled");
    tile_in_place(gpu, w.data, Some(w.scales), n, k, stream)?;
    w.tiled = true;
    Ok(())
}

/// [`mxfp8_tile_in_place`] of the values alone (an FP8 block twin, whose f32
/// grid the `_tiled` FP8 block tiers read row-major).
pub fn mx_tile_values_in_place(
    gpu: &dyn GpuBackend,
    data: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    tile_in_place(gpu, data, None, n, k, stream)
}

fn tile_in_place(
    gpu: &dyn GpuBackend,
    data: DevicePtr,
    scales: Option<DevicePtr>,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let kernel = tiled_kernels(gpu).1;
    ensure!(
        n > 0 && n.is_multiple_of(16) && k > 0 && k.is_multiple_of(64),
        "mxfp8_tile16: n={n} k={k}"
    );
    ensure!(
        kernel.0 != 0,
        "mxfp8_tile16 is missing from the kernel target"
    );
    let (tile_w, tile_s) = (16 * k, 16 * k / MXFP8_BLOCK);
    let tiles = n / 16;
    let batch = (TILE_STAGE_BYTES / tile_w).clamp(1, tiles);
    let stage_w = gpu.alloc(batch * tile_w)?;
    let stage_s = match scales {
        Some(_) => match gpu.alloc(batch * tile_s) {
            Ok(p) => p,
            Err(e) => {
                gpu.free(stage_w)?;
                return Err(e);
            }
        },
        None => DevicePtr::NULL,
    };
    let run = || -> Result<()> {
        for first in (0..tiles).step_by(batch) {
            let count = batch.min(tiles - first);
            let dw = data.offset(first * tile_w);
            gpu.copy_d2d_async(dw, stage_w, count * tile_w, stream)?;
            let ds = match scales {
                Some(s) => {
                    let ds = s.offset(first * tile_s);
                    gpu.copy_d2d_async(ds, stage_s, count * tile_s, stream)?;
                    ds
                }
                None => DevicePtr::NULL,
            };
            let threads = (count * (k / 64) * 32) as u32;
            KernelLaunch::new(gpu, kernel)
                .grid([div_ceil(threads, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(stage_w)
                .arg_ptr(stage_s)
                .arg_ptr(dw)
                .arg_ptr(ds)
                .arg_u32((count * 16) as u32)
                .arg_u32(k as u32)
                .launch(stream)?;
            // The next run overwrites the stage.
            gpu.synchronize(stream)?;
        }
        Ok(())
    };
    let done = run();
    gpu.free(stage_w)?;
    if scales.is_some() {
        gpu.free(stage_s)?;
    }
    done
}

/// [`mxfp8_tile_in_place`] under `ATLAS_GLM_MXFP8_TILED=1` for a twin served
/// only by [`mxfp8_gemv`] through [`mxfp8_kernel`]; a no-op otherwise, or
/// when the shape has a partial tile or chunk (the row-major tiers serve it).
pub fn mxfp8_tile_if_requested(
    gpu: &dyn GpuBackend,
    w: &mut Mxfp8Weight,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    if !mxfp8_tiled_requested()? || !n.is_multiple_of(16) || !k.is_multiple_of(64) {
        return Ok(());
    }
    ensure!(
        tiled_kernels(gpu).0.iter().all(|t| t.0 != 0),
        "ATLAS_GLM_MXFP8_TILED=1 but the mxfp8_gemv_tc*_tiled tiers are missing"
    );
    mxfp8_tile_in_place(gpu, w, n, k, stream)
}

/// Widest row count [`mxfp8_gemv_grouped`] serves.
pub const MXFP8_GROUPED_MAX_M: u32 = 16;

/// Head-grouped [`mxfp8_gemv`] for per-head weights: for each of `g` heads,
/// `C[:, h*n..][m, n] = A[:, h*k..][m, k] · dequant(W_h)[n, k]ᵀ`, with all
/// heads' `data` / `scales` contiguous (`[g, n, k]` / `[g, n, k/32]`) and
/// token-row strides `a_stride` / `c_stride`. `tiers` holds
/// `mxfp8_gemv_tc{8,16}_grouped`.
#[allow(clippy::too_many_arguments)]
pub fn mxfp8_gemv_grouped(
    gpu: &dyn GpuBackend,
    tiers: &[KernelHandle; 2],
    input: DevicePtr,
    data: DevicePtr,
    scales: DevicePtr,
    output: DevicePtr,
    m: u32,
    g: u32,
    k: u32,
    n: u32,
    a_stride: u32,
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (1..=MXFP8_GROUPED_MAX_M).contains(&m) && (k as usize).is_multiple_of(MXFP8_BLOCK),
        "mxfp8_gemv_grouped: m={m} k={k} unsupported"
    );
    // Rows are read as 16-byte vectors and must not overlap across tokens.
    let (gk, gn) = (u64::from(g) * u64::from(k), u64::from(g) * u64::from(n));
    ensure!(
        g > 0
            && u64::from(a_stride) >= gk
            && u64::from(c_stride) >= gn
            && a_stride.is_multiple_of(8),
        "mxfp8_gemv_grouped: g={g} n={n} k={k} a_stride={a_stride} c_stride={c_stride}"
    );
    KernelLaunch::new(gpu, tiers[usize::from(m > 8)])
        .grid([div_ceil(n, 16), 1, g])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(data)
        .arg_ptr(scales)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(a_stride)
        .arg_u32(c_stride)
        .launch(stream)
}

#[cfg(test)]
#[path = "mxfp8_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "mxfp8_tiled_test_gpu.rs"]
mod tiled_gpu_tests;
