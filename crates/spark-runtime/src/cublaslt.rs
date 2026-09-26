// SPDX-License-Identifier: AGPL-3.0-only
//! Minimal cuBLASLt FFI for the high-efficiency GEMM path (`ATLAS_CUBLAS_GEMM`).
//!
//! The hand-written mma.sync projection/MoE GEMMs reach only ~30% of the cuBLAS
//! ceiling on GB10 (measured: 32 vs 85 TFLOPS bf16, 152 fp8, on the SSM-qkvz
//! shape 3537×12288×2048). This routes those GEMMs through cuBLASLt instead.
//! BF16 only for now — correctness-clean (no scale-format issues); native fp8
//! block-scaled is the follow-up once the end-to-end win is proven.

use anyhow::{Result, bail};
use std::ffi::c_void;
use std::sync::OnceLock;

// Native FP8 (E4M3) GEMM paths live in the `fp8` sibling (≤500 LoC split);
// re-exported so `spark_runtime::cublaslt::fp8_gemm_*` paths are unchanged.
mod fp8;
pub use fp8::{fp8_gemm_act_weight_t_blkscaled, fp8_gemm_act_weight_t_rowwise};

#[allow(non_camel_case_types)]
type cublasLtHandle_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulDesc_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatrixLayout_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulPreference_t = *mut c_void;

const CUDA_R_16BF: i32 = 14;
const CUDA_R_32F: i32 = 0;
const CUDA_R_8F_E4M3: i32 = 28;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const DESC_TRANSA: u32 = 3;
const DESC_TRANSB: u32 = 4;
const DESC_A_SCALE_POINTER: u32 = 17;
const DESC_B_SCALE_POINTER: u32 = 18;
const DESC_A_SCALE_MODE: u32 = 31;
const DESC_B_SCALE_MODE: u32 = 32;
const SCALE_MODE_OUTER_VEC_32F: i32 = 3;
const SCALE_MODE_VEC128_32F: i32 = 4;
const SCALE_MODE_BLK128X128_32F: i32 = 5;
const PREF_MAX_WORKSPACE_BYTES: u32 = 1;

unsafe extern "C" {
    fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> i32;
    fn cublasLtMatmulDescCreate(
        desc: *mut cublasLtMatmulDesc_t,
        compute_type: i32,
        scale_type: i32,
    ) -> i32;
    fn cublasLtMatmulDescSetAttribute(
        desc: cublasLtMatmulDesc_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> i32;
    fn cublasLtMatrixLayoutCreate(
        layout: *mut cublasLtMatrixLayout_t,
        dtype: i32,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> i32;
    fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> i32;
    fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> i32;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: cublasLtMatmulPreference_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoGetHeuristic(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        pref: cublasLtMatmulPreference_t,
        requested: i32,
        results: *mut c_void,
        returned: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmul(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        layout_a: cublasLtMatrixLayout_t,
        b: *const c_void,
        layout_b: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        layout_c: cublasLtMatrixLayout_t,
        d: *mut c_void,
        layout_d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cuLaunchKernel(
        f: u64,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem_bytes: u32,
        stream: u64,
        params: *mut *mut c_void,
        extra: *mut c_void,
    ) -> i32;
}

struct Ctx {
    handle: cublasLtHandle_t,
    workspace: u64,
    ws_size: usize,
}
// cuBLASLt handle + device workspace are process-global; matmul is invoked
// serially from the single-threaded scheduler forward.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

/// STATIC, DELIBERATELY — CUDA host. This is a workspace allocated in THE
/// process CUDA context (see `atlas_core::cuda_host`, which establishes one
/// per process) and sized by a fixed budget, not by any model's shapes: the
/// bounds below are generous upper limits chosen to fit any realistic serving
/// configuration, so a swap needs no reallocation and re-allocating per model
/// would churn hundreds of megabytes for no change in what is mapped.
///
/// It survives a model swap for the same reason the context does. Nothing in
/// it is derived from a model — no token ids, no weight pointers, no shapes —
/// only scratch the library plans within.
static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    let mut handle: cublasLtHandle_t = std::ptr::null_mut();
    let st = unsafe { cublasLtCreate(&mut handle) };
    if st != 0 {
        bail!("cublasLtCreate failed: {st}");
    }
    let ws_size = 64 * 1024 * 1024;
    let mut ws: u64 = 0;
    let st = unsafe { cuMemAlloc_v2(&mut ws, ws_size) };
    if st != 0 {
        bail!("cuMemAlloc cuBLASLt workspace failed: {st}");
    }
    let _ = CTX.set(Ctx {
        handle,
        workspace: ws,
        ws_size,
    });
    Ok(CTX.get().unwrap())
}

/// Force cuBLASLt's one-time costs at MODEL LOAD instead of on request 1.
///
/// The lazy `ctx()` means the first GEMM pays `cublasLtCreate`, the 64 MB
/// workspace alloc, and — the expensive part — the library's kernel-image
/// load and heuristic warm-up. Measured on the 35B flagship (2026-08-22,
/// dgx1): the first in-serve request read ~0.9 s slower than warm requests
/// once QKVZ routed through cuBLASLt, and cold TTFT is a headline metric.
/// One 64x64x64 BF16 GEMM here is trivial GPU work and moves that cost to
/// load time, where it overlaps the operator's mental model of "loading".
///
/// Never fails the serve: a pre-warm failure is logged and swallowed — the
/// lazy path remains and request 1 simply pays the old cost.
pub fn prewarm(stream: u64) {
    let r = (|| -> Result<()> {
        let bytes = 64usize * 64 * 2;
        let mut a = 0u64;
        let mut b = 0u64;
        let mut d = 0u64;
        unsafe {
            chk(cuMemAlloc_v2(&mut a, bytes), "prewarm alloc a")?;
            chk(cuMemAlloc_v2(&mut b, bytes), "prewarm alloc b")?;
            chk(cuMemAlloc_v2(&mut d, bytes), "prewarm alloc d")?;
        }
        let res = bf16_gemm_act_weight_t(a, b, d, 64, 64, 64, stream);
        unsafe {
            chk(cuStreamSynchronize(stream), "prewarm sync")?;
            let _ = cuMemFree_v2(a);
            let _ = cuMemFree_v2(b);
            let _ = cuMemFree_v2(d);
        }
        res
    })();
    match r {
        Ok(()) => tracing::info!("cuBLASLt pre-warmed (handle + workspace + kernel images)"),
        Err(e) => tracing::warn!("cuBLASLt pre-warm failed (request 1 pays lazy init): {e}"),
    }
}

/// Process-global BF16 GEMM for backends whose cuBLASLt is a stub (the HIP
/// target links `libcublaslt_stub.cpp`, which returns 1 for every call).
/// Same contract as [`bf16_gemm_act_weight_t`]: row-major
/// `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all BF16.
pub type Bf16GemmFallback = dyn Fn(u64, u64, u64, u32, u32, u32, u64) -> Result<()> + Send + Sync;

static BF16_FALLBACK: OnceLock<Box<Bf16GemmFallback>> = OnceLock::new();
static FALLBACK_LOGGED: std::sync::Once = std::sync::Once::new();

/// Single source of truth for the dense BF16 GEMM launch geometry — the
/// `dense_gemm_bf16` family shares one (A,B,C,M,N,K) contract across the
/// pipelined tensor-core kernel (128x128 CTA tile, 256 threads) and the
/// scalar fallback (16x16 tile, 16x16 threads). Used by `make_bf16_fallback`
/// and by `spark-model`'s `ops::gemm_dense` launchers.
pub const DENSE_GEMM_BF16_PIPELINED_TILE: u32 = 128;
pub const DENSE_GEMM_BF16_PIPELINED_THREADS: u32 = 256;
pub const DENSE_GEMM_BF16_SCALAR_TILE: u32 = 16;

/// Single source of truth for the `dense_gemv_bf16_batchm` launch geometry —
/// the M-row BF16 GEMV (`(A,B,C,M,N,K,out_stride)`, 4 outputs per 256-thread
/// block). The kernel CLAMPS `M` silently above `DENSE_GEMV_BATCHM_MAX_M`
/// (mirror of `MAX_M` in `kernels/gb10/common/dense_gemv_bf16_batchm.cu`), so
/// every caller must refuse wider batches rather than get stale rows. Used by
/// `make_bf16_fallback` and by spark-model's `ops::dense_gemv_batchm`.
pub const DENSE_GEMV_BATCHM_MAX_M: u32 = 8;
pub const DENSE_GEMV_BATCHM_OUTPUTS_PER_BLOCK: u32 = 4;
pub const DENSE_GEMV_BATCHM_THREADS: u32 = 256;

/// Install the GEMM [`bf16_gemm_act_weight_t`] routes to when `ctx()` fails.
/// Returns false if one was already installed — first install wins: the
/// fallback is a property of the process's backend, not of any model.
pub fn install_bf16_fallback(f: Box<Bf16GemmFallback>) -> bool {
    BF16_FALLBACK.set(f).is_ok()
}

/// Which kernel the fallback closure launches for one call. Kept as a pure
/// selector so the arm-choice logic is unit-testable without a GPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bf16FallbackArm {
    /// `dense_gemv_bf16_batchm` — one bandwidth-bound pass over the weight
    /// per row; the M<=8 arm (the decode projections, where a 128-row WMMA
    /// tile would do ~128x the useful MMA work). Per the kernel's own doc
    /// the M-row result is bit-identical to M x M=1 GEMVs.
    GemvBatchM,
    /// `dense_gemm_bf16_pipelined` — 128x128 WMMA tile.
    Pipelined,
    /// `dense_gemm_bf16` — scalar, no shape/alignment requirements.
    Scalar,
}

/// Arm selection: the GEMV only takes `1 <= m <= DENSE_GEMV_BATCHM_MAX_M`
/// under the vectorizable contract and only when its kernel resolved; wider
/// or unresolved falls to the pipelined tile, and anything unvectorizable
/// lands on the scalar kernel (whose unresolved handle is the launch-time
/// bail, as before).
fn bf16_fallback_arm(
    m: u32,
    vectorizable: bool,
    gemv_batchm: crate::gpu::KernelHandle,
    pipelined: crate::gpu::KernelHandle,
) -> Bf16FallbackArm {
    if vectorizable && (1..=DENSE_GEMV_BATCHM_MAX_M).contains(&m) && gemv_batchm.0 != 0 {
        Bf16FallbackArm::GemvBatchM
    } else if vectorizable && pipelined.0 != 0 {
        Bf16FallbackArm::Pipelined
    } else {
        Bf16FallbackArm::Scalar
    }
}

/// Build the Atlas-kernel BF16 GEMM fallback — three arms. For
/// `1 <= M <= 8` it launches `dense_gemv_bf16_batchm` (bandwidth-bound GEMV;
/// on Flash-Next the BF16 `in_proj_qkvz` decode projection is `16384x2560`
/// at M=1, where the 128-row tile spends ~128x the useful MMA work). Larger
/// vectorizable shapes launch `dense_gemm_bf16_pipelined` (tensor core,
/// ~40x scalar on large M) when the kernel resolved AND its
/// vectorized-load contract holds — 16-byte-aligned operands and
/// `K % 8 == 0` (uint4 row loads; both ports bounds-check M/N/K, so tile
/// multiples are not required) — else the scalar `dense_gemm_bf16`, which
/// has no alignment or shape requirements.
pub fn make_bf16_fallback(
    pipelined: crate::gpu::KernelHandle,
    scalar: crate::gpu::KernelHandle,
    gemv_batchm: crate::gpu::KernelHandle,
) -> Box<Bf16GemmFallback> {
    Box::new(move |act, weight, out, m, n, k, stream| {
        let vectorizable = k % 8 == 0 && act % 16 == 0 && weight % 16 == 0 && out % 16 == 0;
        let (mut a, mut w, mut o) = (act, weight, out);
        let (mut mm, mut nn, mut kk) = (m, n, k);
        // Only read by the GEMV arm's params; kept in this scope so the
        // pointer stays valid until the launch.
        let mut out_stride = n; // contiguous output rows
        let (func, grid, block, mut params): (_, _, _, Vec<*mut c_void>) =
            match bf16_fallback_arm(m, vectorizable, gemv_batchm, pipelined) {
                Bf16FallbackArm::GemvBatchM => (
                    gemv_batchm.0,
                    [n.div_ceil(DENSE_GEMV_BATCHM_OUTPUTS_PER_BLOCK), 1, 1],
                    [DENSE_GEMV_BATCHM_THREADS, 1, 1],
                    vec![
                        &mut a as *mut u64 as *mut c_void,
                        &mut w as *mut u64 as *mut c_void,
                        &mut o as *mut u64 as *mut c_void,
                        &mut mm as *mut u32 as *mut c_void,
                        &mut nn as *mut u32 as *mut c_void,
                        &mut kk as *mut u32 as *mut c_void,
                        &mut out_stride as *mut u32 as *mut c_void,
                    ],
                ),
                Bf16FallbackArm::Pipelined => (
                    pipelined.0,
                    [
                        n.div_ceil(DENSE_GEMM_BF16_PIPELINED_TILE),
                        m.div_ceil(DENSE_GEMM_BF16_PIPELINED_TILE),
                        1,
                    ],
                    [DENSE_GEMM_BF16_PIPELINED_THREADS, 1, 1],
                    vec![
                        &mut a as *mut u64 as *mut c_void,
                        &mut w as *mut u64 as *mut c_void,
                        &mut o as *mut u64 as *mut c_void,
                        &mut mm as *mut u32 as *mut c_void,
                        &mut nn as *mut u32 as *mut c_void,
                        &mut kk as *mut u32 as *mut c_void,
                    ],
                ),
                Bf16FallbackArm::Scalar => {
                    if scalar.0 == 0 {
                        anyhow::bail!(
                            "BF16 GEMM fallback: dense_gemv_bf16_batchm / \
                             dense_gemm_bf16_pipelined constraints unmet \
                             (M={m} N={n} K={k} or unresolved) and scalar \
                             dense_gemm_bf16 also unresolved"
                        );
                    }
                    (
                        scalar.0,
                        [
                            n.div_ceil(DENSE_GEMM_BF16_SCALAR_TILE),
                            m.div_ceil(DENSE_GEMM_BF16_SCALAR_TILE),
                            1,
                        ],
                        [DENSE_GEMM_BF16_SCALAR_TILE, DENSE_GEMM_BF16_SCALAR_TILE, 1],
                        vec![
                            &mut a as *mut u64 as *mut c_void,
                            &mut w as *mut u64 as *mut c_void,
                            &mut o as *mut u64 as *mut c_void,
                            &mut mm as *mut u32 as *mut c_void,
                            &mut nn as *mut u32 as *mut c_void,
                            &mut kk as *mut u32 as *mut c_void,
                        ],
                    )
                }
            };
        // SAFETY: `func` is a CUfunction resolved from this process's loaded
        // modules (lives as long as the backend, which outlives the serve);
        // params point at stack values that outlive the launch call, matching
        // each kernel's (ptr, ptr, ptr, u32, u32, u32[, u32]) signature; the
        // primary context is bound on the calling thread by
        // AtlasCudaBackend::new.
        let st = unsafe {
            cuLaunchKernel(
                func,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                0,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        chk(st, "Atlas-GEMM fallback launch")?;
        Ok(())
    })
}

fn chk(status: i32, what: &str) -> Result<()> {
    if status != 0 {
        bail!("cuBLASLt {what} failed: status {status}");
    }
    Ok(())
}

static CUBLASLT_LIVE_LOGGED: std::sync::Once = std::sync::Once::new();

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all BF16 — the standard
/// projection GEMM (activation × transposed weight). Maps to cuBLASLt's
/// column-major convention as `D[N,M] = opT(weightᶜ[K,N]) · opN(actᶜ[K,M])`.
/// Any failure of the cuBLASLt path (stub shim, missing hipBLASLt, or a
/// rejected shape) routes to the installed Atlas fallback; with none
/// installed the error propagates.
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    match bf16_gemm_act_weight_t_cublas(act, weight, out, m, n, k, stream) {
        Ok(()) => {
            CUBLASLT_LIVE_LOGGED.call_once(|| {
                tracing::info!(
                    "cuBLASLt BF16 GEMM live (first matmul status 0 — real library \
                     answered; on HIP targets this is the hipBLASLt shim)"
                );
            });
            Ok(())
        }
        Err(e) => {
            if let Some(fallback) = BF16_FALLBACK.get() {
                FALLBACK_LOGGED.call_once(|| {
                    tracing::info!(
                        "cuBLASLt unavailable ({e}); BF16 projections route through \
                         the installed Atlas GEMM fallback"
                    );
                });
                return fallback(act, weight, out, m, n, k, stream);
            }
            Err(e)
        }
    }
}

fn bf16_gemm_act_weight_t_cublas(
    act: u64,
    weight: u64,
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
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                desc,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        // A = weight stored row-major [N,K] == col-major [K,N], ld=K, opT → [N,K]
        // B = act    stored row-major [M,K] == col-major [K,M], ld=K, opN → [K,M]
        // D = out    row-major [M,N]        == col-major [N,M], ld=N
        let mut la: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut lb: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut ld_: cublasLtMatrixLayout_t = std::ptr::null_mut();
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_16BF, k as u64, n as u64, k as i64),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut lb, CUDA_R_16BF, k as u64, m as u64, k as i64),
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
        // cublasLtMatmulHeuristicResult_t = { algo[64B], workspaceSize, state,
        // wavesCount, reserved[4] } ≈ 96B; algo at offset 0. 128B for margin.
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
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = cublasLtMatmul(
            ctx.handle,
            desc,
            &alpha as *const f32 as *const c_void,
            weight as *const c_void,
            la,
            act as *const c_void,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::KernelHandle;
    use std::sync::Mutex;

    const OK: KernelHandle = KernelHandle(1);
    const MISSING: KernelHandle = KernelHandle(0);

    /// M=1 vectorizable projection — the Flash-Next GDN `in_proj_qkvz`
    /// decode shape — must take the GEMV arm when the kernel resolved.
    #[test]
    fn arm_selects_gemv_for_m1_vectorizable() {
        assert_eq!(
            bf16_fallback_arm(1, true, OK, OK),
            Bf16FallbackArm::GemvBatchM
        );
        assert_eq!(
            bf16_fallback_arm(8, true, OK, OK),
            Bf16FallbackArm::GemvBatchM
        );
    }

    /// Above the kernel's MAX_M the batched GEMV would clamp silently, so
    /// M=9 must route to the pipelined tile instead.
    #[test]
    fn arm_selects_pipelined_for_m9() {
        assert_eq!(
            bf16_fallback_arm(9, true, OK, OK),
            Bf16FallbackArm::Pipelined
        );
    }

    /// Unvectorizable operands (misaligned or K%8!=0) can't use the uint4
    /// loads of either fast kernel — the scalar GEMM serves.
    #[test]
    fn arm_selects_scalar_for_non_vectorizable_m1() {
        assert_eq!(bf16_fallback_arm(1, false, OK, OK), Bf16FallbackArm::Scalar);
    }

    /// A target whose registry lacks `dense_gemv_bf16_batchm` (handle 0)
    /// must skip the GEMV arm even at M=1, keeping the pipelined path.
    #[test]
    fn arm_skips_gemv_when_handle_unresolved() {
        assert_eq!(
            bf16_fallback_arm(1, true, MISSING, OK),
            Bf16FallbackArm::Pipelined
        );
    }

    /// Under the CI GPU stubs (`scripts/ci_gpu_stubs.sh`) `cublasLtCreate`
    /// returns 1, so `ctx()` fails deterministically and
    /// `bf16_gemm_act_weight_t` must route to the installed fallback with the
    /// exact (act, weight, out, m, n, k, stream) it was given. On a build
    /// where real cuBLASLt exists the fallback is never consulted — this
    /// test only asserts the wiring, which is also exercised on hosts where
    /// the FFI itself fails to resolve.
    #[test]
    fn install_bf16_fallback_first_wins_and_dispatches() {
        type Call = (u64, u64, u64, u32, u32, u32, u64);
        static CALLS: Mutex<Vec<Call>> = Mutex::new(Vec::new());
        let installed = install_bf16_fallback(Box::new(|act, weight, out, m, n, k, stream| {
            CALLS
                .lock()
                .unwrap()
                .push((act, weight, out, m, n, k, stream));
            Ok(())
        }));
        assert!(installed, "first install_bf16_fallback must succeed");
        assert!(
            !install_bf16_fallback(Box::new(|_, _, _, _, _, _, _| {
                anyhow::bail!("must never run: second install rejected")
            })),
            "second install_bf16_fallback must return false"
        );
        // Only meaningful where ctx() fails (CI stubs / HIP shim). On a host
        // with a real cuBLASLt this call would attempt a device GEMM instead,
        // so the assertion is conditional on the error path being reachable.
        if ctx().is_err() {
            bf16_gemm_act_weight_t(0xAA, 0xBB, 0xCC, 22, 16384, 2560, 0x11)
                .expect("fallback must serve the GEMM when ctx() fails");
            let calls = CALLS.lock().unwrap();
            assert_eq!(
                calls.as_slice(),
                &[(0xAA, 0xBB, 0xCC, 22, 16384, 2560, 0x11)]
            );
        }
    }
}
