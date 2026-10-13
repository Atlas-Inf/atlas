// SPDX-License-Identifier: AGPL-3.0-only
//! One cuBLASLt kernel configuration per BF16 weight shape, whatever the row
//! count (`ATLAS_QWEN4EXP_PREFILL_ROWINV`): the heuristic's first pick for a
//! REFERENCE row count, cached per `(n, k, ref_m)` and replayed for every
//! `m` -- same algorithm, tile, split-K count and reduction scheme. Under one
//! configuration a row's output is a function of its own inputs (rows are
//! independent in every tile; the K slices are fixed by K and the split
//! count; the split-K reduction sums the slices in a fixed order), so the
//! GEMM is row-invariant where the heuristic, which re-picks per `m`, is not.
//! `examples/qwen4exp_rowinv_probe` byte-checks each shape it is used on.

use super::*;

/// `sizeof(cublasLtMatmulAlgo_t)`.
const ALGO_BYTES: usize = 64;

type Key = (u32, u32, u32);
static CACHE: std::sync::Mutex<Vec<(Key, [u8; ALGO_BYTES])>> = std::sync::Mutex::new(Vec::new());

/// The cached configuration for `(n, k, ref_m)`, described (`None` before
/// its first GEMM).
pub fn bf16_fixed_algo_description(n: u32, k: u32, ref_m: u32) -> Option<String> {
    let cache = CACHE.lock().unwrap();
    let algo = cache.iter().find(|e| e.0 == (n, k, ref_m))?.1;
    Some(super::kchain_pin::describe(algo.as_ptr() as *const c_void))
}

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoCheck(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        result: *mut c_void,
    ) -> i32;
}

/// Descriptor and layouts of `out[m, n] = act[m, k] @ weight[n, k]^T`;
/// destroyed on drop.
struct Problem {
    desc: cublasLtMatmulDesc_t,
    la: cublasLtMatrixLayout_t,
    lb: cublasLtMatrixLayout_t,
    ld: cublasLtMatrixLayout_t,
}

impl Problem {
    fn new(m: u32, n: u32, k: u32) -> Result<Self> {
        let mut p = Problem {
            desc: std::ptr::null_mut(),
            la: std::ptr::null_mut(),
            lb: std::ptr::null_mut(),
            ld: std::ptr::null_mut(),
        };
        // SAFETY: plain cuBLASLt object creation into the null-initialised
        // handles above; `Drop` destroys whatever was created.
        unsafe {
            chk(
                cublasLtMatmulDescCreate(&mut p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                "DescCreate",
            )?;
            let (ta, tb) = (CUBLAS_OP_T, CUBLAS_OP_N);
            chk(
                cublasLtMatmulDescSetAttribute(
                    p.desc,
                    DESC_TRANSA,
                    &ta as *const i32 as *const c_void,
                    4,
                ),
                "TRANSA",
            )?;
            chk(
                cublasLtMatmulDescSetAttribute(
                    p.desc,
                    DESC_TRANSB,
                    &tb as *const i32 as *const c_void,
                    4,
                ),
                "TRANSB",
            )?;
            chk(
                cublasLtMatrixLayoutCreate(&mut p.la, CUDA_R_16BF, k as u64, n as u64, k as i64),
                "LayoutA",
            )?;
            chk(
                cublasLtMatrixLayoutCreate(&mut p.lb, CUDA_R_16BF, k as u64, m as u64, k as i64),
                "LayoutB",
            )?;
            chk(
                cublasLtMatrixLayoutCreate(&mut p.ld, CUDA_R_16BF, n as u64, m as u64, n as i64),
                "LayoutD",
            )?;
        }
        Ok(p)
    }
}

impl Drop for Problem {
    fn drop(&mut self) {
        // SAFETY: each handle is null or was created in `new`.
        unsafe {
            if !self.la.is_null() {
                cublasLtMatrixLayoutDestroy(self.la);
            }
            if !self.lb.is_null() {
                cublasLtMatrixLayoutDestroy(self.lb);
            }
            if !self.ld.is_null() {
                cublasLtMatrixLayoutDestroy(self.ld);
            }
            if !self.desc.is_null() {
                cublasLtMatmulDescDestroy(self.desc);
            }
        }
    }
}

/// The heuristic's first algorithm for `ref_m` rows.
fn reference_algo(ctx: &Ctx, ref_m: u32, n: u32, k: u32) -> Result<[u8; ALGO_BYTES]> {
    let p = Problem::new(ref_m, n, k)?;
    let mut result = [0u8; 128];
    let mut returned = 0i32;
    // SAFETY: `result` holds one 96-byte heuristic result; the preference is
    // created and destroyed here.
    unsafe {
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let ws = ctx.ws_size;
        let st = cublasLtMatmulPreferenceSetAttribute(
            pref,
            PREF_MAX_WORKSPACE_BYTES,
            &ws as *const usize as *const c_void,
            std::mem::size_of::<usize>(),
        );
        let st = if st == 0 {
            cublasLtMatmulAlgoGetHeuristic(
                ctx.handle,
                p.desc,
                p.la,
                p.lb,
                p.ld,
                p.ld,
                pref,
                1,
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            )
        } else {
            st
        };
        cublasLtMatmulPreferenceDestroy(pref);
        chk(st, "AlgoGetHeuristic")?;
    }
    if returned < 1 {
        bail!("cuBLASLt: no algorithm for {ref_m}x{n}x{k}");
    }
    let mut algo = [0u8; ALGO_BYTES];
    algo.copy_from_slice(&result[..ALGO_BYTES]);
    tracing::info!(
        "cuBLASLt fixed algorithm for BF16 {n}x{k} (picked at {ref_m} rows): {}",
        super::kchain_pin::describe(algo.as_ptr() as *const c_void)
    );
    Ok(algo)
}

/// `out[m, n] = act[m, k] @ weight[n, k]^T` (BF16, FP32 accumulate) on the
/// configuration the heuristic picks for `ref_m` rows, at any `m`.
/// `Ok(false)`: unavailable, or the configuration does not take `m` rows
/// (nothing launched).
pub fn bf16_gemm_act_weight_t_fixed(
    act: u64,
    weight: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    ref_m: u32,
    stream: u64,
) -> Result<bool> {
    if !available() {
        return Ok(false);
    }
    let ctx = ctx()?;
    let key = (n, k, ref_m);
    let cached = CACHE
        .lock()
        .unwrap()
        .iter()
        .find(|e| e.0 == key)
        .map(|e| e.1);
    let algo = match cached {
        Some(a) => a,
        None => {
            let a = reference_algo(ctx, ref_m, n, k)?;
            CACHE.lock().unwrap().push((key, a));
            a
        }
    };
    let p = Problem::new(m, n, k)?;
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: the algorithm bytes are a heuristic result's algo field; the
    // problem's handles live until the end of this scope; pointers are the
    // caller's device buffers of the stated shapes.
    unsafe {
        let mut check = [0u8; 128];
        let st = cublasLtMatmulAlgoCheck(
            ctx.handle,
            p.desc,
            p.la,
            p.lb,
            p.ld,
            p.ld,
            algo.as_ptr() as *const c_void,
            check.as_mut_ptr() as *mut c_void,
        );
        if st != 0 {
            return Ok(false);
        }
        chk(
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
            ),
            "Matmul",
        )?;
    }
    Ok(true)
}
