// SPDX-License-Identifier: AGPL-3.0-only

//! BF16 Atlas-kernel GEMM fallback for `cublaslt` (install + arm select +
//! launch). Split out under the 500-LoC cap; the public items are re-exported
//! by the parent so `spark_runtime::cublaslt::*` paths are unchanged.

use std::ffi::c_void;
use std::sync::OnceLock;

use super::*;

/// Process-global BF16 GEMM for backends whose cuBLASLt is a stub (the HIP
/// target links `libcublaslt_stub.cpp`, which returns 1 for every call).
/// Same contract as [`bf16_gemm_act_weight_t`]: row-major
/// `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all BF16.
pub type Bf16GemmFallback = dyn Fn(u64, u64, u64, u32, u32, u32, u64) -> Result<()> + Send + Sync;

pub(crate) static BF16_FALLBACK: OnceLock<Box<Bf16GemmFallback>> = OnceLock::new();
pub(crate) static FALLBACK_LOGGED: std::sync::Once = std::sync::Once::new();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::KernelHandle;

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
}
