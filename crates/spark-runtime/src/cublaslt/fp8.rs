// SPDX-License-Identifier: AGPL-3.0-only

//! Native FP8 (E4M3) cuBLASLt GEMM paths (row-wise + 128-block scaled).

use anyhow::{Result, bail};
use std::ffi::c_void;

use super::*;

/// Native FP8 (E4M3) `out[M,N] = act[M,K] @ weight[N,K]ᵀ` → BF16 with ROW-WISE
/// scaling (OUTER_VEC): per-output-row weight scale `weight_scale[N]` and
/// per-token activation scale `act_scale[M]`. This is the fp8 path GB10/sm_121
/// actually supports (128-block fp8 is B200-only). ~1.8× the bf16 path.
/// cuBLAS folds `A_scale[i]·B_scale[j]` into the FP32 epilogue; with D=`[N,M]`,
/// i indexes weight rows (N) and j indexes tokens (M) — exactly row-wise.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_rowwise(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        let set = |attr: u32, val: *const c_void, sz: usize, what: &str| -> Result<()> {
            chk(cublasLtMatmulDescSetAttribute(desc, attr, val, sz), what)
        };
        set(DESC_TRANSA, &ta as *const i32 as *const c_void, 4, "TRANSA")?;
        set(DESC_TRANSB, &tb as *const i32 as *const c_void, 4, "TRANSB")?;
        let mode = SCALE_MODE_OUTER_VEC_32F;
        set(
            DESC_A_SCALE_MODE,
            &mode as *const i32 as *const c_void,
            4,
            "A_SCALE_MODE",
        )?;
        set(
            DESC_B_SCALE_MODE,
            &mode as *const i32 as *const c_void,
            4,
            "B_SCALE_MODE",
        )?;
        set(
            DESC_A_SCALE_POINTER,
            &weight_scale as *const u64 as *const c_void,
            8,
            "A_SCALE_POINTER",
        )?;
        set(
            DESC_B_SCALE_POINTER,
            &act_scale as *const u64 as *const c_void,
            8,
            "B_SCALE_POINTER",
        )?;

        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_8F_E4M3, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_8F_E4M3, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc,
                la,
                lb,
                ld_,
                ld_,
                pref,
                1,
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt fp8 rowwise: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc,
            &alpha as *const f32 as *const c_void,
            weight_fp8 as *const c_void,
            la,
            act_fp8 as *const c_void,
            lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_,
            out as *mut c_void,
            ld_,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}

/// Native FP8 (E4M3) `out[M,N] = act[M,K] @ weight[N,K]ᵀ` → BF16, with the
/// weight per-128×128-block FP32-scaled (matches Atlas's `Fp8Weight.row_scale`
/// layout exactly) and the activation cast at unit scale (post-RMSNorm acts sit
/// in fp8's range). ~1.8× the bf16 path (152 vs 85 TFLOPS on GB10).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_blkscaled(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_block_scale: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = CUBLAS_OP_T;
        let tb = CUBLAS_OP_N;
        let set = |attr: u32, val: *const c_void, sz: usize, what: &str| -> Result<()> {
            chk(cublasLtMatmulDescSetAttribute(desc, attr, val, sz), what)
        };
        set(DESC_TRANSA, &ta as *const i32 as *const c_void, 4, "TRANSA")?;
        set(DESC_TRANSB, &tb as *const i32 as *const c_void, 4, "TRANSB")?;
        // FP8 block scaling requires BOTH operands use a 128-block mode (SCALAR
        // is rejected → status 7). Weight = per-128×128 block, activation =
        // per-[token,128-of-K] VEC128 (DeepSeek block-fp8 scheme).
        let a_mode = SCALE_MODE_BLK128X128_32F;
        let b_mode = SCALE_MODE_VEC128_32F;
        set(
            DESC_A_SCALE_MODE,
            &a_mode as *const i32 as *const c_void,
            4,
            "A_SCALE_MODE",
        )?;
        set(
            DESC_B_SCALE_MODE,
            &b_mode as *const i32 as *const c_void,
            4,
            "B_SCALE_MODE",
        )?;
        set(
            DESC_A_SCALE_POINTER,
            &weight_block_scale as *const u64 as *const c_void,
            8,
            "A_SCALE_POINTER",
        )?;
        set(
            DESC_B_SCALE_POINTER,
            &act_scale as *const u64 as *const c_void,
            8,
            "B_SCALE_POINTER",
        )?;

        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_8F_E4M3, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_8F_E4M3, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut ld_, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws_size = ctx.ws_size;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                pref,
                PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            ),
            "PrefWorkspace",
        )?;
        let mut result = [0u8; 128];
        let mut returned: i32 = 0;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                desc,
                la,
                lb,
                ld_,
                ld_,
                pref,
                1,
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt fp8: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc,
            &alpha as *const f32 as *const c_void,
            weight_fp8 as *const c_void,
            la,
            act_fp8 as *const c_void,
            lb,
            &beta as *const f32 as *const c_void,
            out as *const c_void,
            ld_,
            out as *mut c_void,
            ld_,
            result.as_ptr() as *const c_void,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            stream as *mut c_void,
        );
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(ld_);
        cublasLtMatmulDescDestroy(desc);
        chk(status, "Matmul")?;
    }
    Ok(())
}

/// MXFP8 W8A8 `out[M,N] = act[M,K] @ weight[N,K]ᵀ` → BF16 (FP32 accumulate),
/// both operands E4M3 with one UE8M0 scale per 32 K values
/// (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` on A and B; TN only on GB10).
/// GB10, FP4-Local's TP2 prefill shapes (`mxfp8_gemm_bench_gpu.rs` in
/// spark-model, 2026-10-15): ~150-172 TFLOPS, against 57-60 for the BF16
/// cuBLASLt heuristic on 13.0.0 (the runtime image's) and 80-94 on 13.1.1.
///
/// Layouts (row-major, 16-byte aligned):
/// * `weight_fp8` `[N, K]`, `act_fp8` `[M, K]` E4M3; `out` `[M, N]` BF16.
/// * `weight_sf`, `act_sf`: cuBLASLt's tiled scale layout for `[N, K/32]` /
///   `[M, K/32]` (`kernels/gb10/common/mxfp8_act_quant.cuh`), `round_up(rows,
///   128) * round_up(K/32, 4)` bytes, columns past K/32 zero.
///
/// `K % 32 == 0`. The kernel is a tuned in-order one (`mx_algo`), per row
/// bucket for a shape the invariance sweep passed on this cuBLASLt, else one
/// for every row count: the first call of a key (or one its kernel refuses)
/// times the candidates on these operands and synchronizes `stream` (or call
/// [`fp8_gemm_mx_prepare`](super::fp8_gemm_mx_prepare) at load).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_mx(
    act_fp8: u64,
    act_sf: u64,
    weight_fp8: u64,
    weight_sf: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        m > 0 && n > 0 && k > 0 && k.is_multiple_of(32),
        "cuBLASLt MXFP8: {m}x{n}x{k} (K % 32 != 0 or an empty dimension)"
    );
    let ctx = ctx()?;
    let p = super::fixed_algo::Problem::with(
        m,
        n,
        k,
        super::fixed_algo::Operands::Mxfp8 { weight_sf, act_sf },
    )?;
    let ptrs = [weight_fp8, act_fp8, out];
    let algo = super::mx_algo::select(ctx, &p, [m, n, k], ptrs, stream)?;
    super::mx_algo::matmul(ctx, &p, &algo, ptrs, stream)
}

/// [`fp8_gemm_act_weight_t_mx`] on ONE kernel for every row count, chosen
/// once without timing (`mx_algo::pinned`): what a packed GLM prefill runs,
/// so a prompt's rows get the same bytes alone and packed.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_mx_pinned(
    act_fp8: u64,
    act_sf: u64,
    weight_fp8: u64,
    weight_sf: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        m > 0 && n > 0 && k > 0 && k.is_multiple_of(32),
        "cuBLASLt MXFP8: {m}x{n}x{k} (K % 32 != 0 or an empty dimension)"
    );
    let ctx = ctx()?;
    let ops = super::fixed_algo::Operands::Mxfp8 { weight_sf, act_sf };
    let p = super::fixed_algo::Problem::with(m, n, k, ops)?;
    let algo = super::mx_algo::pinned(ctx, n, k, ops)?;
    anyhow::ensure!(
        super::mx_algo::takes(ctx, &p, &algo),
        "cuBLASLt MXFP8 {m}x{n}x{k}: the pinned kernel refuses {m} rows"
    );
    super::mx_algo::matmul(ctx, &p, &algo, [weight_fp8, act_fp8, out], stream)
}

/// Choose (once) the pinned kernel [`fp8_gemm_act_weight_t_blkscaled_pinned`]
/// would run for `n x k`, without launching anything: `Err` when this
/// cuBLASLt cannot run block FP8 on this GPU (e.g. cuBLASLt 13.0 on sm_121
/// answers the VEC128 / BLK128x128 scale modes with NOT_SUPPORTED), so a
/// loader can pick another prefill path at load instead of failing mid-pass.
/// `weight_sf` / `act_sf` are any device pointers (the heuristic reads none).
pub fn fp8_blkscaled_pinned_prepare(weight_sf: u64, act_sf: u64, n: u32, k: u32) -> Result<()> {
    anyhow::ensure!(
        n > 0 && k > 0 && k.is_multiple_of(128),
        "cuBLASLt block FP8: {n}x{k}"
    );
    let ctx = ctx()?;
    let ops = super::fixed_algo::Operands::Blk128 { weight_sf, act_sf };
    super::mx_algo::pinned(ctx, n, k, ops).map(|_| ())
}

/// Block FP8 `out[m, n] = act[m, k] x weight[n, k]^T` (BF16) on ONE in-order
/// cuBLASLt kernel per `(n, k)`, chosen once without timing and never retuned
/// (`mx_algo::pinned`): the weight with its f32 `[n/128, k/128]` grid, the
/// activation with per-token, per-128 f32 scales (`[m, k/128]`, as
/// `per_token_group_quant_fp8` writes them). `m` must be a multiple of 16
/// (the caller pads). Each output row is one in-order accumulation of its
/// own inputs, so a row's bytes do not depend on the pass height. Errors
/// rather than switching kernels when the pinned one refuses a height.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_blkscaled_pinned(
    act_fp8: u64,
    act_scale: u64,
    weight_fp8: u64,
    weight_block_scale: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        m > 0 && m.is_multiple_of(16) && n > 0 && k > 0 && k.is_multiple_of(128),
        "cuBLASLt block FP8: {m}x{n}x{k} (m % 16, k % 128 or an empty dimension)"
    );
    let ctx = ctx()?;
    let ops = super::fixed_algo::Operands::Blk128 {
        weight_sf: weight_block_scale,
        act_sf: act_scale,
    };
    let p = super::fixed_algo::Problem::with(m, n, k, ops)?;
    let algo = super::mx_algo::pinned(ctx, n, k, ops)?;
    anyhow::ensure!(
        super::mx_algo::takes(ctx, &p, &algo),
        "cuBLASLt block FP8 {m}x{n}x{k}: the pinned kernel refuses {m} rows"
    );
    super::mx_algo::matmul(ctx, &p, &algo, [weight_fp8, act_fp8, out], stream)
}

/// [`fp8_gemm_act_weight_t_mx`] on the heuristic's `index`-th candidate for
/// exactly these `m` rows (tests and benches: the tuned pick is not
/// consulted or cached); `in_order_only` asks the heuristic for no-split-K
/// kernels only, as tuning does. `Ok(None)`: no such candidate, or it
/// refuses the problem; nothing launched. `Some((description, in_order))`
/// otherwise.
#[cfg(any(test, feature = "test-utils"))]
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_act_weight_t_mx_candidate(
    act_fp8: u64,
    act_sf: u64,
    weight_fp8: u64,
    weight_sf: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    index: usize,
    in_order_only: bool,
    stream: u64,
) -> Result<Option<(String, bool)>> {
    use super::mx_algo::{candidates, description, in_order, matmul, takes};
    let ctx = ctx()?;
    let p = super::fixed_algo::Problem::with(
        m,
        n,
        k,
        super::fixed_algo::Operands::Mxfp8 { weight_sf, act_sf },
    )?;
    let Some(algo) = candidates(ctx, &p, in_order_only)?.get(index).copied() else {
        return Ok(None);
    };
    if !takes(ctx, &p, &algo) {
        return Ok(None);
    }
    matmul(ctx, &p, &algo, [weight_fp8, act_fp8, out], stream)?;
    Ok(Some((description(&algo), in_order(&algo))))
}
