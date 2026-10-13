// SPDX-License-Identifier: AGPL-3.0-only
//! BF16 token-interleaved head GEMMs. No packing or precision conversion.
use super::*;
use anyhow::ensure;

// CUDA 13 cublasLt.h: int32 batch count; int64 batch stride in ELEMENTS.
const LAYOUT_BATCH_COUNT: u32 = 5;
const LAYOUT_STRIDED_BATCH_OFFSET: u32 = 6;

#[derive(Default)]
struct Descriptors {
    desc: cublasLtMatmulDesc_t,
    layouts: [cublasLtMatrixLayout_t; 3],
    pref: cublasLtMatmulPreference_t,
}
impl Drop for Descriptors {
    fn drop(&mut self) {
        unsafe {
            if !self.pref.is_null() {
                cublasLtMatmulPreferenceDestroy(self.pref);
            }
            for layout in self.layouts {
                if !layout.is_null() {
                    cublasLtMatrixLayoutDestroy(layout);
                }
            }
            if !self.desc.is_null() {
                cublasLtMatmulDescDestroy(self.desc);
            }
        }
    }
}

/// `out[t*C_stride+h*N+n] = sum_k act[t*A_stride+h*K+k]*weight[h*N*K+n*K+k]`.
/// Each head is a strided batch in a column-major `Wᵀ[N,K] * A[K,M]`.
/// Accumulation is FP32 with BF16 output. As with the other cuBLAS helpers,
/// callers serialize use of the shared workspace on the model's stream.
#[allow(clippy::too_many_arguments)]
pub fn bf16_grouped_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    g: u32,
    n: u32,
    k: u32,
    a_stride: u32,
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(m > 0 && g > 0 && n > 0 && k > 0, "empty grouped GEMM");
    let batches = i32::try_from(g)?;
    let a_width = u64::from(g) * u64::from(k);
    let c_width = u64::from(g) * u64::from(n);
    ensure!(
        u64::from(a_stride) >= a_width,
        "grouped GEMM input rows overlap"
    );
    ensure!(
        u64::from(c_stride) >= c_width,
        "grouped GEMM output rows overlap"
    );
    let a_end = end_address(act, u64::from(m - 1) * u64::from(a_stride) + a_width)?;
    let c_end = end_address(out, u64::from(m - 1) * u64::from(c_stride) + c_width)?;
    let w_count = u64::from(g)
        .checked_mul(u64::from(n) * u64::from(k))
        .ok_or_else(|| anyhow::anyhow!("grouped GEMM weight size overflow"))?;
    let w_end = end_address(weight, w_count)?;
    ensure!(
        out >= a_end || act >= c_end,
        "grouped GEMM aliases input/output"
    );
    ensure!(
        out >= w_end || weight >= c_end,
        "grouped GEMM aliases weight/output"
    );

    let weight_stride = i64::try_from(u64::from(n) * u64::from(k))?;
    let context = ctx()?;
    let mut handles = Descriptors::default();
    unsafe {
        chk(
            cublasLtMatmulDescCreate(&mut handles.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
            "grouped DescCreate",
        )?;
        for (attr, value) in [(DESC_TRANSA, CUBLAS_OP_T), (DESC_TRANSB, CUBLAS_OP_N)] {
            chk(
                cublasLtMatmulDescSetAttribute(
                    handles.desc,
                    attr,
                    (&value as *const i32).cast(),
                    size_of::<i32>(),
                ),
                "grouped transpose",
            )?;
        }
        // The token stride is the column leading dimension; head stride is
        // independent, so padded token rows need no gather/scatter operation.
        for (i, (rows, cols, ld, stride)) in [
            (k, n, i64::from(k), weight_stride),
            (k, m, i64::from(a_stride), i64::from(k)),
            (n, m, i64::from(c_stride), i64::from(n)),
        ]
        .into_iter()
        .enumerate()
        {
            chk(
                cublasLtMatrixLayoutCreate(
                    &mut handles.layouts[i],
                    CUDA_R_16BF,
                    u64::from(rows),
                    u64::from(cols),
                    ld,
                ),
                "grouped LayoutCreate",
            )?;
            chk(
                cublasLtMatrixLayoutSetAttribute(
                    handles.layouts[i],
                    LAYOUT_BATCH_COUNT,
                    (&batches as *const i32).cast(),
                    size_of::<i32>(),
                ),
                "grouped batch count",
            )?;
            chk(
                cublasLtMatrixLayoutSetAttribute(
                    handles.layouts[i],
                    LAYOUT_STRIDED_BATCH_OFFSET,
                    (&stride as *const i64).cast(),
                    size_of::<i64>(),
                ),
                "grouped batch stride",
            )?;
        }
        chk(
            cublasLtMatmulPreferenceCreate(&mut handles.pref),
            "grouped PreferenceCreate",
        )?;
        chk(
            cublasLtMatmulPreferenceSetAttribute(
                handles.pref,
                PREF_MAX_WORKSPACE_BYTES,
                (&context.ws_size as *const usize).cast(),
                size_of::<usize>(),
            ),
            "grouped workspace",
        )?;
        // Heuristic defaults assume 256-byte alignment. Respect every head
        // base and token/weight-column stride, including padded small shapes.
        for (attr, address_bits) in [
            (5, weight | (weight_stride as u64 * 2) | (u64::from(k) * 2)),
            (6, act | (u64::from(k) * 2) | (u64::from(a_stride) * 2)),
            (7, out | (u64::from(n) * 2) | (u64::from(c_stride) * 2)),
            (8, out | (u64::from(n) * 2) | (u64::from(c_stride) * 2)),
        ] {
            let alignment = 1u32 << address_bits.trailing_zeros().min(8);
            chk(
                cublasLtMatmulPreferenceSetAttribute(
                    handles.pref,
                    attr,
                    (&alignment as *const u32).cast(),
                    size_of::<u32>(),
                ),
                "grouped minimum alignment",
            )?;
        }
        // HeuristicResult contains a 64-byte algorithm followed by metadata;
        // 128 bytes cover the CUDA ABI with its required 8-byte alignment.
        let mut result = [0u64; 16];
        let mut returned = 0;
        let [la, lb, lc] = handles.layouts;
        chk(
            cublasLtMatmulAlgoGetHeuristic(
                context.handle,
                handles.desc,
                la,
                lb,
                lc,
                lc,
                handles.pref,
                1,
                result.as_mut_ptr().cast(),
                &mut returned,
            ),
            "grouped AlgoGetHeuristic",
        )?;
        ensure!(
            returned > 0,
            "cuBLASLt: no grouped algorithm M={m} G={g} N={n} K={k} A_stride={a_stride} C_stride={c_stride}"
        );
        let alpha = 1.0f32;
        let beta = 0.0f32;
        chk(
            cublasLtMatmul(
                context.handle,
                handles.desc,
                (&alpha as *const f32).cast(),
                weight as *const c_void,
                la,
                act as *const c_void,
                lb,
                (&beta as *const f32).cast(),
                out as *const c_void,
                lc,
                out as *mut c_void,
                lc,
                result.as_ptr().cast(),
                context.workspace as *mut c_void,
                context.ws_size,
                stream as *mut c_void,
            ),
            "grouped Matmul",
        )?;
    }
    Ok(())
}

fn end_address(ptr: u64, elements: u64) -> Result<u64> {
    ensure!(
        ptr > 0 && ptr.is_multiple_of(2),
        "grouped GEMM requires BF16 aligned pointers"
    );
    elements
        .checked_mul(2)
        .and_then(|bytes| ptr.checked_add(bytes))
        .ok_or_else(|| anyhow::anyhow!("grouped GEMM address overflow"))
}
