// SPDX-License-Identifier: AGPL-3.0-only
//! Dense CUTLASS GEMM host wrappers (BF16 + native NVFP4 projection).

use anyhow::{Result, bail};

#[cfg(atlas_cutlass)]
use std::ffi::c_void;

#[cfg(atlas_cutlass)]
use super::*;

/// Row-major `out[M,N] = act[M,K] @ weight[N,K]^T`, all BF16.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let ctx = ctx()?;
        let status = drained("bf16_gemm_act_weight_t", unsafe {
            atlas_cutlass_bf16_gemm_act_weight_t(
                act as *const c_void,
                weight as *const c_void,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        });
        if status != 0 {
            bail!("CUTLASS bf16 GEMM failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (act, weight, out, m, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Row-major `out[m, ldc] = act[m, lda] @ weight[n, k]^T` (BF16, FP32
/// accumulate) through a CUTLASS tile/stage/CTA-swizzle `config`:
/// 0 = 128x128x32/4-stage, 4 = 128x256x32/3-stage swizzle 4,
/// 5 = 128x256x32/3-stage swizzle 8, 9 = 128x128x64/3-stage swizzle 8.
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_tuned(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    config: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let status = drained("bf16_gemm_tuned", unsafe {
            atlas_cutlass_bf16_gemm_tuned(
                act as *const c_void,
                weight as *const c_void,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                lda as i32,
                ldc as i32,
                config as i32,
                stream as *mut c_void,
            )
        });
        if status != 0 {
            return Err(super::status_error(
                status,
                format!("CUTLASS tuned bf16 GEMM {m}x{n}x{k} config {config}"),
            ));
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (act, weight, out, m, n, k, lda, ldc, config, stream);
        Err(anyhow::Error::new(super::RejectedBeforeLaunch)
            .context("CUTLASS support was not built; set CUTLASS_HOME when building"))
    }
}

/// Head-batched BF16 GEMM with the `cublaslt::bf16_grouped_gemm_act_weight_t`
/// contract: `out[t*c_stride+h*n+j] = sum_i act[t*a_stride+h*k+i]*w[h,j,i]`.
/// One strided batch per head, 128x128x32 tiles, 3 stages.
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
    #[cfg(atlas_cutlass)]
    {
        let status = drained("bf16_grouped_gemm_act_weight_t", unsafe {
            atlas_cutlass_bf16_grouped_gemm_act_weight_t(
                act as *const c_void,
                weight as *const c_void,
                out as *mut c_void,
                m as i32,
                g as i32,
                n as i32,
                k as i32,
                a_stride as i32,
                c_stride as i32,
                stream as *mut c_void,
            )
        });
        if status != 0 {
            bail!("CUTLASS grouped bf16 GEMM failed: status {status} for {m}x{g}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (act, weight, out, m, g, n, k, a_stride, c_stride, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// Native CUTLASS NVFP4 dense projection:
/// `out[M,N] = quant_nvfp4(act[M,K]) @ weight_t[N,K]^T -> BF16`.
///
/// `weight_packed_t` and `weight_scale_t` are Atlas's transposed NVFP4
/// prefill layout: packed data `[K/2,N]`, scales `[K/16,N]`. The wrapper
/// repacks activation and scale tensors into CUTLASS's SM120 blockscaled
/// layouts in the shared CUTLASS workspace before dispatch.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_bf16_act_weight_t(
    act: u64,
    weight_packed_t: u64,
    weight_scale_t: u64,
    weight_scale_2: f32,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let ctx = ctx()?;
        let status = drained("nvfp4_gemm_bf16_act_weight_t", unsafe {
            atlas_cutlass_nvfp4_gemm_bf16_act_weight_t(
                act as *const c_void,
                weight_packed_t as *const c_void,
                weight_scale_t as *const c_void,
                weight_scale_2,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                ctx.workspace as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        });
        if status != 0 {
            bail!("CUTLASS nvfp4 GEMM failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (
            act,
            weight_packed_t,
            weight_scale_t,
            weight_scale_2,
            out,
            m,
            n,
            k,
            stream,
        );
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}
