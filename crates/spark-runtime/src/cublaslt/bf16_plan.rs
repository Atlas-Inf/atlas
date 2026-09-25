// SPDX-License-Identifier: AGPL-3.0-only
//! Per-shape cuBLASLt plans for the BF16 projection GEMM.
//!
//! `gemm_bf16` used to create a matmul descriptor, three matrix layouts and a
//! preference, run `cublasLtMatmulAlgoGetHeuristic`, launch, and destroy all
//! of it — on EVERY call. None of that depends on a data pointer: it is a pure
//! function of `(m, n, k, op_a)` and the fixed workspace size. The heuristic is
//! deterministic for identical inputs, so caching its answer launches the SAME
//! algorithm the per-call path would have picked: bit-identical output, minus
//! the host-side setup that made `ATLAS_CUBLAS_GEMM=1` measure slower than the
//! tile kernels it was meant to beat.
//!
//! Bounded: prefill `m` follows the chunk length, so a long-lived server sees
//! many shapes. Past `MAX_PLANS` the cache is cleared and refilled; a rebuild
//! costs exactly what every call cost before.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail};

use super::{
    CUBLAS_COMPUTE_32F, CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16BF, CUDA_R_32F, DESC_TRANSA,
    DESC_TRANSB, PREF_MAX_WORKSPACE_BYTES, chk, cublasLtHandle_t, cublasLtMatmulAlgoGetHeuristic,
    cublasLtMatmulDesc_t, cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy,
    cublasLtMatmulDescSetAttribute, cublasLtMatmulPreference_t, cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceSetAttribute, cublasLtMatrixLayout_t,
    cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy,
};

const MAX_PLANS: usize = 4096;

/// Everything `cublasLtMatmul` needs besides the data pointers.
pub(super) struct Bf16Plan {
    pub desc: cublasLtMatmulDesc_t,
    pub la: cublasLtMatrixLayout_t,
    pub lb: cublasLtMatrixLayout_t,
    pub ld: cublasLtMatrixLayout_t,
    /// `cublasLtMatmulHeuristicResult_t` (~96 B, algo at offset 0); 128 B margin.
    pub algo: [u8; 128],
}
// cuBLASLt descriptors are plain host objects; the GEMM path is invoked
// serially from the scheduler, and the map is behind a Mutex regardless.
unsafe impl Send for Bf16Plan {}

impl Drop for Bf16Plan {
    fn drop(&mut self) {
        unsafe {
            cublasLtMatrixLayoutDestroy(self.la);
            cublasLtMatrixLayoutDestroy(self.lb);
            cublasLtMatrixLayoutDestroy(self.ld);
            cublasLtMatmulDescDestroy(self.desc);
        }
    }
}

type Key = (u32, u32, u32, i32);
static PLANS: OnceLock<Mutex<HashMap<Key, Bf16Plan>>> = OnceLock::new();

/// Run `f` with the cached plan for `(m, n, k, op_a)`, building it on a miss.
pub(super) fn with_plan<R>(
    handle: cublasLtHandle_t,
    ws_size: usize,
    m: u32,
    n: u32,
    k: u32,
    op_a: i32,
    f: impl FnOnce(&Bf16Plan) -> Result<R>,
) -> Result<R> {
    let map = PLANS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().unwrap_or_else(|e| e.into_inner());
    let key = (m, n, k, op_a);
    if !map.contains_key(&key) {
        if map.len() >= MAX_PLANS {
            map.clear();
        }
        let plan = build(handle, ws_size, m, n, k, op_a)?;
        map.insert(key, plan);
    }
    f(map.get(&key).expect("plan inserted above"))
}

fn build(
    handle: cublasLtHandle_t,
    ws_size: usize,
    m: u32,
    n: u32,
    k: u32,
    op_a: i32,
) -> Result<Bf16Plan> {
    unsafe {
        let mut plan = Bf16Plan {
            desc: std::ptr::null_mut(),
            la: std::ptr::null_mut(),
            lb: std::ptr::null_mut(),
            ld: std::ptr::null_mut(),
            algo: [0u8; 128],
        };
        chk(
            cublasLtMatmulDescCreate(&mut plan.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "DescCreate",
        )?;
        let (ta, tb) = (op_a, CUBLAS_OP_N);
        chk(
            cublasLtMatmulDescSetAttribute(
                plan.desc,
                DESC_TRANSA,
                &ta as *const i32 as *const c_void,
                4,
            ),
            "TRANSA",
        )?;
        chk(
            cublasLtMatmulDescSetAttribute(
                plan.desc,
                DESC_TRANSB,
                &tb as *const i32 as *const c_void,
                4,
            ),
            "TRANSB",
        )?;
        // A = weight: row-major [N,K] (opT) or [K,N] (opN), as col-major.
        // B = act row-major [M,K] == col-major [K,M]; D = out [M,N] == col-major [N,M].
        let (a_rows, a_cols, a_ld) = if op_a == CUBLAS_OP_T {
            (k as u64, n as u64, k as i64)
        } else {
            (n as u64, k as u64, n as i64)
        };
        chk(
            cublasLtMatrixLayoutCreate(&mut plan.la, CUDA_R_16BF, a_rows, a_cols, a_ld),
            "LayoutA",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut plan.lb, CUDA_R_16BF, k as u64, m as u64, k as i64),
            "LayoutB",
        )?;
        chk(
            cublasLtMatrixLayoutCreate(&mut plan.ld, CUDA_R_16BF, n as u64, m as u64, n as i64),
            "LayoutD",
        )?;
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(cublasLtMatmulPreferenceCreate(&mut pref), "PrefCreate")?;
        let set = cublasLtMatmulPreferenceSetAttribute(
            pref,
            PREF_MAX_WORKSPACE_BYTES,
            &ws_size as *const usize as *const c_void,
            std::mem::size_of::<usize>(),
        );
        let mut returned: i32 = 0;
        let heur = if set == 0 {
            cublasLtMatmulAlgoGetHeuristic(
                handle,
                plan.desc,
                plan.la,
                plan.lb,
                plan.ld,
                plan.ld,
                pref,
                1,
                plan.algo.as_mut_ptr() as *mut c_void,
                &mut returned,
            )
        } else {
            set
        };
        cublasLtMatmulPreferenceDestroy(pref);
        chk(heur, "PrefWorkspace/AlgoGetHeuristic")?;
        if returned < 1 {
            bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
        }
        Ok(plan)
    }
}
