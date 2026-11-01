// SPDX-License-Identifier: AGPL-3.0-only
//! The shared cuBLASLt BF16 GEMM body (split out of `cublaslt.rs` for the
//! 500-LoC cap).

use super::*;

/// Shared body. `op_a` selects the weight's stored layout: `CUBLAS_OP_T` for a
/// row-major `[N,K]` weight, `CUBLAS_OP_N` for a row-major `[K,N]` one. The A
/// LAYOUT MUST MATCH: `(k, n, ld=k)` under opT, `(n, k, ld=n)` under opN.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_bf16(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    op_a: i32,
    stream: u64,
) -> Result<()> {
    let pick = if kchain_pin::applies(m, n, op_a == CUBLAS_OP_T) {
        kchain_pin::Pick::Pin
    } else {
        kchain_pin::Pick::First
    };
    gemm_bf16_pick(act, weight, out, [m, n, k], op_a, pick, stream).map(|_| ())
}

/// [`gemm_bf16`] with the algorithm choice explicit (`kchain_pin::Pick`);
/// `Ok(false)`: `Pick::Force` found no k-chain kernel and launched nothing.
pub(super) fn gemm_bf16_pick(
    act: u64,
    weight: u64,
    out: u64,
    [m, n, k]: [u32; 3],
    op_a: i32,
    pick: kchain_pin::Pick,
    stream: u64,
) -> Result<bool> {
    let ctx = ctx()?;
    unsafe {
        let mut desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        chk(
            cublasLtMatmulDescCreate(&mut desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let ta = op_a;
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
        let (a_rows, a_cols, a_ld) = if op_a == CUBLAS_OP_T {
            (k as u64, n as u64, k as i64) // row-major [N,K] -> col-major (K,N)
        } else {
            (n as u64, k as u64, n as i64) // row-major [K,N] -> col-major (N,K)
        };
        chk(
            cublasLtMatrixLayoutCreate(&mut la, CUDA_R_16BF, a_rows, a_cols, a_ld),
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
        // Heuristic results: 96 B each, algo at offset 0 (+ margin, `kchain_pin`).
        let mut result = [0u8; 128 * kchain_pin::CANDIDATES];
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
                pick.requested(),
                result.as_mut_ptr() as *mut c_void,
                &mut returned,
            ),
            "AlgoGetHeuristic",
        )?;
        if returned < 1 {
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }
        let Some(algo_off) = pick.offset(&result, returned) else {
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(la);
            cublasLtMatrixLayoutDestroy(lb);
            cublasLtMatrixLayoutDestroy(ld_);
            cublasLtMatmulDescDestroy(desc);
            return Ok(false);
        };
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
            result[algo_off..].as_ptr() as *const c_void,
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
    Ok(true)
}
