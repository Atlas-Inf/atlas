// SPDX-License-Identifier: AGPL-3.0-only
//! Which cuBLASLt kernel runs an MXFP8 W8A8 GEMM
//! ([`super::fp8_gemm_act_weight_t_mx`]): per `(N, K, row bucket)`, the
//! fastest of the heuristic's in-order candidates, timed once and cached.
//!
//! Why not the heuristic's first pick: it is 2-3x slower than the best
//! offered kernel at some (shape, M) points of FP4-Local's prefill
//! projections, and which points lose moves with M and with the library
//! (GB10, the TP2 GDN qkvz 8192x2560 at 16384 rows: 7.89 vs 4.18 ms on
//! cuBLASLt 13.1.1; 13.0.0, the runtime image's, picks well there). So the
//! choice is per bucket of M (powers of two, 64 .. 16384), not per shape.
//!
//! Why only in-order candidates (no split-K, no reduction scheme): under one
//! such kernel every output is one FP32 accumulation of the block-scaled k32
//! MMAs over K in order, whatever the tile, the row count or the row's
//! position in the pass, so they should all give the same bytes, and a
//! bucket may then take any of them without breaking the row invariance the
//! qwen4_exp prefill keeps (`ATLAS_QWEN4EXP_PREFILL_ROWINV`: a token's output
//! independent of the pass's height and of where it starts). The heuristic's
//! split-K candidates (splitK -2, reduction 2) are not: they differed from
//! the in-order bytes in 1-17 outputs at many (shape, M) points, and are
//! sometimes its FIRST pick (13.0.0 offers only a split-K kernel for the
//! TP2 indexer 320x2560 at 64 and 100 rows unless asked for in-order ones).
//!
//! "Should" is checked, not assumed: `mxfp8_gemm_test_gpu.rs` (spark-model)
//! runs every candidate at M = 1 .. 16384 and odd pass offsets on each shape
//! of [`super::mx_sweep::MX_SWEEP`] and compares byte for byte. A GEMM takes
//! a kernel per bucket only where that sweep passed for its `(N, K)` on the
//! linked cuBLASLt ([`super::mx_sweep::verified`]); every other GEMM runs ONE
//! in-order kernel for all row counts, tuned once at the largest bucket it is
//! prepared for, which is row-invariant by itself (`fixed_algo`'s argument).
//!
//! Timing runs on the caller's own operands (the output is rewritten by the
//! real call that follows) and synchronizes `stream`, once per key; never
//! under graph capture. [`fp8_gemm_mx_prepare`] does it at load instead.

use std::sync::Mutex;

use super::fixed_algo::{Operands, Problem, cublasLtMatmulAlgoCheck};
use super::kchain_pin::{RESULT_BYTES, attr, describe};
use super::*;

/// Heuristic results asked for per selection.
pub const MX_CANDIDATES: usize = 16;
/// `sizeof(cublasLtMatmulAlgo_t)`.
const ALGO_BYTES: usize = 64;
/// Row buckets: powers of two in `MIN_BUCKET ..= MAX_BUCKET`.
const MIN_BUCKET: u32 = 64;
const MAX_BUCKET: u32 = 16384;
/// Timed launches per candidate, after `WARM` untimed ones.
const WARM: usize = 2;
const TIMED: usize = 10;

pub(super) type Algo = [u8; ALGO_BYTES];
type Key = (u32, u32, u32);
static CACHE: Mutex<Vec<(Key, Algo)>> = Mutex::new(Vec::new());

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
    fn cuMemsetD8Async(dst: u64, value: u8, count: usize, stream: u64) -> i32;
}

/// The row bucket `m` rows are served from.
pub(super) fn bucket(m: u32) -> u32 {
    m.max(1)
        .checked_next_power_of_two()
        .unwrap_or(MAX_BUCKET)
        .clamp(MIN_BUCKET, MAX_BUCKET)
}

/// The cache key `m` rows of `n x k` are served from on cuBLASLt `version`:
/// their row bucket where the sweep verified the shape on that version, else
/// 0, one entry for every row count.
fn key(version: usize, n: u32, k: u32, m: u32) -> Key {
    let per_bucket = super::mx_sweep::verified(version, n, k);
    (n, k, if per_bucket { bucket(m) } else { 0 })
}

/// [`key`] on the linked cuBLASLt.
fn key_here(n: u32, k: u32, m: u32) -> Key {
    key(super::version(), n, k, m)
}

/// Whether `n x k` takes a kernel per row bucket on the linked cuBLASLt
/// (else one kernel serves every row count).
#[cfg(any(test, feature = "test-utils"))]
pub fn mx_per_bucket(n: u32, k: u32) -> bool {
    super::mx_sweep::verified(super::version(), n, k)
}

/// `CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK`.
const PREF_REDUCTION_SCHEME_MASK: u32 = 3;

/// The heuristic's algorithms for `p`, best first; `in_order_only` asks for
/// reduction scheme NONE only (a mask of 0), i.e. no split-K. On cuBLASLt
/// 13.0.0 the unmasked heuristic offers ONLY a split-K kernel for some
/// narrow shapes at small M (the TP2 indexer 320x2560 at 64 and 100 rows);
/// masked, it offers in-order ones there too.
pub(super) fn candidates(ctx: &Ctx, p: &Problem, in_order_only: bool) -> Result<Vec<Algo>> {
    let mut results = vec![0u8; RESULT_BYTES * MX_CANDIDATES];
    let mut returned = 0i32;
    // SAFETY: `results` holds MX_CANDIDATES heuristic results; the
    // preference is created and destroyed here.
    unsafe {
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws = ctx.ws_size;
        let mut st = cublasLtMatmulPreferenceSetAttribute(
            pref,
            PREF_MAX_WORKSPACE_BYTES,
            &ws as *const usize as *const c_void,
            std::mem::size_of::<usize>(),
        );
        let none = 0u32;
        if st == 0 && in_order_only {
            st = cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_REDUCTION_SCHEME_MASK,
                &none as *const u32 as *const c_void,
                4,
            );
        }
        if st == 0 {
            st = cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                p.desc,
                p.la,
                p.lb,
                p.ld,
                p.ld,
                pref,
                MX_CANDIDATES as i32,
                results.as_mut_ptr() as *mut c_void,
                &mut returned,
            );
        }
        cublasLtMatmulPreferenceDestroy(pref);
        chk(st, "AlgoGetHeuristic")?;
    }
    let found = (returned.max(0) as usize).min(MX_CANDIDATES);
    Ok((0..found)
        .map(|i| {
            let mut a = [0u8; ALGO_BYTES];
            a.copy_from_slice(&results[i * RESULT_BYTES..i * RESULT_BYTES + ALGO_BYTES]);
            a
        })
        .collect())
}

/// No split-K and no reduction scheme (see the module docs).
pub(super) fn in_order(algo: &Algo) -> bool {
    super::kchain_pin::in_order(algo.as_ptr() as *const c_void)
}

/// `id=.. tile=.. splitK=.. reduction=.. stages=.. swizzle=.. custom=..
/// cluster=..` of `algo` (the heuristic's MXFP8 candidates on GB10 share one
/// id and tile and differ in the rest).
pub(super) fn description(algo: &Algo) -> String {
    let a = algo.as_ptr() as *const c_void;
    let v = |w| attr(a, w).map_or_else(|| "?".to_string(), |v| v.to_string());
    format!(
        "{} stages={} swizzle={} custom={} cluster={}",
        describe(a),
        v(CONFIG_STAGES_ID),
        v(CONFIG_CTA_SWIZZLING),
        v(CONFIG_CUSTOM_OPTION),
        v(CONFIG_CLUSTER_SHAPE_ID)
    )
}

/// `CUBLASLT_ALGO_CONFIG_*` attributes [`description`] adds.
const CONFIG_CTA_SWIZZLING: u32 = 4;
const CONFIG_CUSTOM_OPTION: u32 = 5;
const CONFIG_STAGES_ID: u32 = 6;
const CONFIG_CLUSTER_SHAPE_ID: u32 = 8;

/// Whether `algo` takes problem `p`.
pub(super) fn takes(ctx: &Ctx, p: &Problem, algo: &Algo) -> bool {
    let mut check = [0u8; 128];
    // SAFETY: `algo` is a heuristic result's algo field; `check` holds one
    // heuristic result.
    unsafe {
        cublasLtMatmulAlgoCheck(
            ctx.handle,
            p.desc,
            p.la,
            p.lb,
            p.ld,
            p.ld,
            algo.as_ptr() as *const c_void,
            check.as_mut_ptr() as *mut c_void,
        ) == 0
    }
}

/// `out = act @ weight^T` on `algo`; `[weight, act, out]` device pointers.
pub(super) fn matmul(
    ctx: &Ctx,
    p: &Problem,
    algo: &Algo,
    [weight, act, out]: [u64; 3],
    stream: u64,
) -> Result<()> {
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: the problem's handles outlive the call; the pointers are the
    // caller's device buffers of the problem's shapes.
    let st = unsafe {
        cublasLtMatmul(
            ctx.handle,
            p.desc,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            p.la,
            act as *const c_void,
            p.lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            p.ld,
            out as *mut c_void,
            p.ld,
            algo.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        )
    };
    chk(st, "Matmul")
}

/// The algorithm for `p` (`m` rows of `n x k`): the cached pick of its key,
/// tuned now on `ptrs` when there is none or when it refuses `m` rows.
pub(super) fn select(
    ctx: &Ctx,
    p: &Problem,
    [m, n, k]: [u32; 3],
    ptrs: [u64; 3],
    stream: u64,
) -> Result<Algo> {
    let key = key_here(n, k, m);
    let cached = CACHE
        .lock()
        .unwrap()
        .iter()
        .find(|e| e.0 == key)
        .map(|e| e.1);
    match cached {
        Some(a) if takes(ctx, p, &a) => return Ok(a),
        // Only past what the key was prepared for (prepare tunes a shared
        // key at its largest pass). Any in-order kernel serves a swept
        // shape's bucket alike; a shared key's rows run the new kernel from
        // here on.
        Some(a) => tracing::warn!(
            "cuBLASLt MXFP8 {n}x{k}: the kernel cached for {} ({}) refuses {m} rows; \
             retuning it at {m} rows",
            serves(key),
            description(&a)
        ),
        None => {}
    }
    let a = tune(ctx, p, [m, n, k], ptrs, stream)?;
    Ok(store(&mut CACHE.lock().unwrap(), key, a, |c| {
        takes(ctx, p, c)
    }))
}

#[path = "mx_pinned.rs"]
mod mx_pinned;
pub use mx_pinned::fp8_gemm_mx_pinned_prepare;
pub(super) use mx_pinned::pinned;

/// Cache `tuned` under `key` and return it, unless the entry already there
/// (another thread tuned the same key meanwhile) also serves the problem:
/// then that entry, so every caller of a key runs the one kernel it holds.
fn store(
    cache: &mut Vec<(Key, Algo)>,
    key: Key,
    tuned: Algo,
    serves: impl Fn(&Algo) -> bool,
) -> Algo {
    match cache.iter_mut().find(|e| e.0 == key) {
        Some(e) if serves(&e.1) => e.1,
        Some(e) => {
            e.1 = tuned;
            tuned
        }
        None => {
            cache.push((key, tuned));
            tuned
        }
    }
}

/// What `key`'s kernel serves, for logs.
fn serves(key: Key) -> String {
    match key.2 {
        0 => format!(
            "every row count (one kernel: {}x{} is not swept on cuBLASLt {})",
            key.0,
            key.1,
            super::version()
        ),
        b => format!("the {b}-row bucket"),
    }
}

/// Milliseconds per launch of `algo` on `p` (median-free: one timed run of
/// TIMED launches after WARM).
fn time(ctx: &Ctx, p: &Problem, algo: &Algo, ptrs: [u64; 3], stream: u64) -> Result<f32> {
    let (mut e0, mut e1) = (0u64, 0u64);
    // SAFETY: plain event create / record / sync / destroy on the caller's
    // stream.
    unsafe {
        chk(cuEventCreate(&mut e0, 0), "EventCreate")?;
        chk(cuEventCreate(&mut e1, 0), "EventCreate")?;
    }
    let run = || -> Result<f32> {
        for _ in 0..WARM {
            matmul(ctx, p, algo, ptrs, stream)?;
        }
        // SAFETY: as above.
        unsafe { chk(cuEventRecord(e0, stream), "EventRecord")? };
        for _ in 0..TIMED {
            matmul(ctx, p, algo, ptrs, stream)?;
        }
        let mut ms = 0f32;
        // SAFETY: as above; `ms` is a 4-byte out-parameter.
        unsafe {
            chk(cuEventRecord(e1, stream), "EventRecord")?;
            chk(cuEventSynchronize(e1), "EventSynchronize")?;
            chk(cuEventElapsedTime(&mut ms, e0, e1), "EventElapsedTime")?;
        }
        Ok(ms / TIMED as f32)
    };
    let ms = run();
    // SAFETY: both events were created above.
    unsafe {
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    ms
}

/// The fastest in-order candidate for `p`, timed on `ptrs`.
fn tune(ctx: &Ctx, p: &Problem, [m, n, k]: [u32; 3], ptrs: [u64; 3], stream: u64) -> Result<Algo> {
    let all = candidates(ctx, p, true)?;
    let offered = all.len();
    let fit: Vec<Algo> = all
        .into_iter()
        .filter(|a| in_order(a) && takes(ctx, p, a))
        .collect();
    anyhow::ensure!(
        !fit.is_empty(),
        "cuBLASLt MXFP8: no in-order kernel among {offered} offered for {m}x{n}x{k}"
    );
    let mut times = Vec::with_capacity(fit.len());
    for a in &fit {
        times.push(if fit.len() == 1 {
            0.0
        } else {
            time(ctx, p, a, ptrs, stream)?
        });
    }
    let best = (0..fit.len())
        .min_by(|&i, &j| times[i].total_cmp(&times[j]))
        .expect("non-empty");
    tracing::info!(
        "cuBLASLt MXFP8 {n}x{k}, {}, tuned at {m} rows: {} at {:.3} ms \
         ({} in-order of {offered} offered; the first ran {:.3} ms)",
        serves(key_here(n, k, m)),
        description(&fit[best]),
        times[best],
        fit.len(),
        times[0]
    );
    Ok(fit[best])
}

/// The cached algorithm for `m` rows of `n x k`, described (`None` before
/// its key's first GEMM).
#[cfg(any(test, feature = "test-utils"))]
pub fn mx_algo_description(n: u32, k: u32, m: u32) -> Option<String> {
    let cache = CACHE.lock().unwrap();
    let a = cache.iter().find(|e| e.0 == key_here(n, k, m))?.1;
    Some(description(&a))
}

/// Every heuristic candidate for `m` rows of `n x k` on these operands
/// (`in_order_only`: as [`candidates`]): `(description, in_order, ms per
/// launch)` by CUDA events (benches; the output is overwritten). Candidates
/// that refuse the problem are skipped.
#[cfg(any(test, feature = "test-utils"))]
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_mx_candidate_times(
    act_fp8: u64,
    act_sf: u64,
    weight_fp8: u64,
    weight_sf: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    in_order_only: bool,
    stream: u64,
) -> Result<Vec<(String, bool, f32)>> {
    let ctx = ctx()?;
    let p = Problem::with(m, n, k, Operands::Mxfp8 { weight_sf, act_sf })?;
    let mut times = Vec::new();
    for a in candidates(ctx, &p, in_order_only)? {
        if takes(ctx, &p, &a) {
            let ms = time(ctx, &p, &a, [weight_fp8, act_fp8, out], stream)?;
            times.push((description(&a), in_order(&a), ms));
        }
    }
    Ok(times)
}

/// Tune every row bucket up to `m_max` rows of an `n x k` MXFP8 GEMM now
/// (load time), on scratch operands, so serving never times a candidate.
/// Synchronizes `stream`.
pub fn fp8_gemm_mx_prepare(n: u32, k: u32, m_max: u32, stream: u64) -> Result<()> {
    let ctx = ctx()?;
    let top = bucket(m_max);
    let (rows, kpad) = (
        top.div_ceil(128) as usize * 128,
        (k as usize / 32).div_ceil(4) * 4,
    );
    let n_pad = (n as usize).div_ceil(128) * 128;
    let sizes = [
        (top as usize * k as usize, 0u8),     // act
        (rows * kpad, 127u8),                 // act scales (2^0)
        (n as usize * k as usize, 0u8),       // weight
        (n_pad * kpad, 127u8),                // weight scales
        (top as usize * n as usize * 2, 0u8), // out
    ];
    let mut bufs = [0u64; 5];
    let result = (|| -> Result<()> {
        for (b, &(bytes, fill)) in bufs.iter_mut().zip(&sizes) {
            // SAFETY: allocation and fill of a fresh device buffer.
            unsafe {
                chk(cuMemAlloc_v2(b, bytes), "prepare alloc")?;
                chk(cuMemsetD8Async(*b, fill, bytes, stream), "prepare fill")?;
            }
        }
        let [act, act_sf, weight, weight_sf, out] = bufs;
        let per_bucket = super::mx_sweep::verified(super::version(), n, k);
        for m in prepare_rows(per_bucket, top) {
            let p = Problem::with(m, n, k, Operands::Mxfp8 { weight_sf, act_sf })?;
            select(ctx, &p, [m, n, k], [weight, act, out], stream)?;
        }
        // SAFETY: plain stream sync before the buffers are freed.
        unsafe { chk(cuStreamSynchronize(stream), "prepare sync") }
    })();
    for b in bufs.into_iter().filter(|&b| b != 0) {
        // SAFETY: allocated above, no longer in use (synchronized or failed).
        unsafe {
            cuMemFree_v2(b);
        }
    }
    result
}

/// The row counts [`fp8_gemm_mx_prepare`] tunes at, up to the bucket `top`:
/// every bucket when each has its own kernel, else `top` alone, so the one
/// shared kernel is the largest pass's pick and takes every pass up to it.
fn prepare_rows(per_bucket: bool, top: u32) -> Vec<u32> {
    if !per_bucket {
        return vec![top];
    }
    std::iter::successors(Some(MIN_BUCKET), |m| Some(m * 2))
        .take_while(|&m| m <= top)
        .collect()
}

#[cfg(test)]
#[path = "mx_algo_tests.rs"]
mod tests;
