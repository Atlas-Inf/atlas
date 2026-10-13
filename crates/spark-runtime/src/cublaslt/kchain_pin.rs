// SPDX-License-Identifier: AGPL-3.0-only
//! The k-chain pin (`ATLAS_LT_KCHAIN_PIN=1`, default off): a faster cuBLASLt
//! kernel for a large BF16 projection, chosen only where it is byte-identical
//! to the one the heuristic picks.
//!
//! On the runtime image's cuBLASLt (CUDA 13.0, 130000) the heuristic's first
//! choice for a large prefill projection is often the 64x64-tile CUTLASS sm80
//! kernel (algo 21, tile 15) at ~57 TFLOP/s, while the same algorithm with a
//! 256x128 tile (tile 24), offered further down the same heuristic list, runs
//! the same products at ~91: the qwen4_exp TP2 GDN in_proj at 16016 rows is
//! 11.7 -> 7.4 ms. Both are algo 21 without split-K: every output is one FP32
//! accumulator over the m16n8k16 MMAs in increasing k, so the tile only
//! changes which CTA computes it -- byte-identical results
//! (`scripts/dev/qwen4exp_lt_algo_bench.cu pin N K M0 M1 STEP`, inside
//! atlas-release-builder: 0 differing bytes at every M swept for N x K =
//! 8192x2560, 6144x2560, 12288x2560, 4096x2560, 1024x2560, 512x2560,
//! 2560x3072, M = 1024..16384).
//!
//! The rule, applied only to `[N,K]` weights (opT) with M and N >= 2048: if
//! the heuristic's first result is algo 21 with split-K 1 and no reduction,
//! take the first offered algo-21, split-K-1, no-reduction result whose tile
//! is 24 (256x128), else 23 (128x256); otherwise keep the first result. The
//! pick depends only on the library's answer for the shape, so both TP ranks
//! (same library, same GPU) pick the same kernel.

use std::ffi::c_void;

/// Heuristic results requested when the pin may apply.
pub(super) const CANDIDATES: usize = 8;
/// `sizeof(cublasLtMatmulHeuristicResult_t)` (algo[64] + workspaceSize +
/// state + wavesCount + reserved[4]) on CUDA 12/13.
pub(super) const RESULT_BYTES: usize = 96;

const CONFIG_ID: u32 = 0;
const CONFIG_TILE_ID: u32 = 1;
const CONFIG_SPLITK_NUM: u32 = 2;
const CONFIG_REDUCTION_SCHEME: u32 = 3;
/// The CUTLASS sm80 tensor-op GEMM.
const ALGO_SM80: i32 = 21;
/// 256x128, then 128x256.
const PREFERRED_TILES: [i32; 2] = [24, 23];

unsafe extern "C" {
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const c_void,
        attr: u32,
        buf: *mut c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// `ATLAS_LT_KCHAIN_PIN=1`.
fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_LT_KCHAIN_PIN").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Whether a GEMM of this shape and weight layout asks for candidates.
pub(super) fn applies(m: u32, n: u32, weight_nk: bool) -> bool {
    requested() && weight_nk && m >= 2048 && n >= 2048 && super::version() >= 130000
}

fn attr(algo: *const c_void, which: u32) -> Option<i32> {
    let mut v = 0i32;
    let mut written = 0usize;
    // SAFETY: `algo` points at a heuristic result's algo field (64 bytes,
    // written by cublasLtMatmulAlgoGetHeuristic); `v` is a 4-byte buffer.
    let st = unsafe {
        cublasLtMatmulAlgoConfigGetAttribute(
            algo,
            which,
            &mut v as *mut i32 as *mut c_void,
            4,
            &mut written,
        )
    };
    (st == 0).then_some(v)
}

/// `id=.. tile=.. splitK=.. reduction=..` of a heuristic result's algorithm.
pub(super) fn describe(algo: *const c_void) -> String {
    let a = |w| attr(algo, w).map_or_else(|| "?".to_string(), |v| v.to_string());
    format!(
        "id={} tile={} splitK={} reduction={}",
        a(CONFIG_ID),
        a(CONFIG_TILE_ID),
        a(CONFIG_SPLITK_NUM),
        a(CONFIG_REDUCTION_SCHEME)
    )
}

/// Algo 21 without split-K or a reduction: the in-order k-chain.
fn kchain(algo: *const c_void) -> bool {
    attr(algo, CONFIG_ID) == Some(ALGO_SM80)
        && attr(algo, CONFIG_SPLITK_NUM).is_some_and(|s| s <= 1)
        && attr(algo, CONFIG_REDUCTION_SCHEME) == Some(0)
}

/// How `gemm_bf16_pick` chooses among the heuristic's results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pick {
    /// The first result (the default).
    First,
    /// The pin: a preferred-tile k-chain kernel when the first result is one.
    Pin,
    /// A k-chain kernel, preferring the fast tiles, or nothing -- to replace
    /// a k-chain GEMM of another library ([`bf16_gemm_act_weight_t_kchain`]).
    Force,
}

impl Pick {
    /// Heuristic results to ask for.
    pub(super) fn requested(self) -> i32 {
        match self {
            Pick::First => 1,
            Pick::Pin | Pick::Force => CANDIDATES as i32,
        }
    }

    /// Byte offset in `results` of the algorithm to run; `None` (Force only)
    /// when no k-chain kernel was offered.
    pub(super) fn offset(self, results: &[u8], returned: i32) -> Option<usize> {
        let found = (returned.max(0) as usize).min(results.len() / RESULT_BYTES);
        let at = |i: usize| results[i * RESULT_BYTES..].as_ptr() as *const c_void;
        let i = match self {
            Pick::First => 0,
            Pick::Pin => pick(results, found),
            Pick::Force => FORCE_TILES.iter().find_map(|&tile| {
                (0..found).find(|&i| kchain(at(i)) && attr(at(i), CONFIG_TILE_ID) == Some(tile))
            })?,
        };
        Some(i * RESULT_BYTES)
    }
}

/// Force's tile preference: 256x128, 128x256, 128x64, 64x256, 64x64.
const FORCE_TILES: [i32; 5] = [24, 23, 18, 19, 15];

/// `out[M,N] = act[M,K] @ weight[N,K]^T` (BF16) on a cuBLASLt k-chain kernel,
/// under `ATLAS_LT_KCHAIN_PIN` -- byte-identical to any other in-order
/// m16n8k16 k-chain over the same operands, such as the Atlas tile kernel
/// `dense_gemm_bf16_pipelined`. `Ok(false)`: off, unavailable, or no k-chain
/// kernel offered for the shape; nothing was launched.
pub fn bf16_gemm_act_weight_t_kchain(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> anyhow::Result<bool> {
    if !requested() || super::version() < 130000 || !super::available() {
        return Ok(false);
    }
    super::gemm_bf16::gemm_bf16_pick(
        act,
        weight,
        out,
        [m, n, k],
        super::CUBLAS_OP_T,
        Pick::Force,
        stream,
    )
}

/// [`bf16_gemm_act_weight_t_kchain`] without the `ATLAS_LT_KCHAIN_PIN`
/// switch, for the row-invariant qwen4_exp prefill
/// (`ATLAS_QWEN4EXP_PREFILL_ROWINV`): every k-chain kernel gives the same
/// bytes at any row count, so a wide projection may take the fastest one
/// offered for each `m`. `Ok(false)`: none offered, nothing launched.
pub fn bf16_gemm_act_weight_t_kchain_any(
    act: u64,
    weight: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    stream: u64,
) -> anyhow::Result<bool> {
    if super::version() < 130000 || !super::available() {
        return Ok(false);
    }
    super::gemm_bf16::gemm_bf16_pick(
        act,
        weight,
        out,
        [m, n, k],
        super::CUBLAS_OP_T,
        Pick::Force,
        stream,
    )
}

/// The pin's index among `found` heuristic results packed at `RESULT_BYTES`
/// in `results`.
fn pick(results: &[u8], found: usize) -> usize {
    let at = |i: usize| results[i * RESULT_BYTES..].as_ptr() as *const c_void;
    if found == 0 || !kchain(at(0)) {
        return 0;
    }
    PREFERRED_TILES
        .iter()
        .find_map(|&tile| {
            (0..found).find(|&i| kchain(at(i)) && attr(at(i), CONFIG_TILE_ID) == Some(tile))
        })
        .unwrap_or(0)
}
