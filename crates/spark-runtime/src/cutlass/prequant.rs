// SPDX-License-Identifier: AGPL-3.0-only
//! Dense SM120 NVFP4 GEMM on pre-quantized operands (`cuda/cutlass_nvfp4_prequant.cu`):
//! activations quantized at a checkpoint's static `input_scale`, weights in the
//! checkpoint's native `[N, K/2]` layout with an SFB packed once at load
//! ([`super::pack_weight_sfb`], N-major source), alpha = `input_scale * weight_scale_2`.

use anyhow::{Result, bail};

#[cfg(atlas_cutlass)]
use std::ffi::c_void;

#[cfg(atlas_cutlass)]
use super::*;

#[cfg(atlas_cutlass)]
unsafe extern "C" {
    fn atlas_cutlass_nvfp4_quant_act_scaled(
        act: *const c_void,
        packed: *mut c_void,
        sf: *mut c_void,
        m: i32,
        k: i32,
        inv_global_scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn atlas_cutlass_nvfp4_quant_swiglu_clamp(
        gate: *const c_void,
        up: *const c_void,
        packed: *mut c_void,
        sf: *mut c_void,
        m: i32,
        k: i32,
        inv_global_scale: f32,
        limit: f32,
        stream: *mut c_void,
    ) -> i32;
    fn atlas_cutlass_nvfp4_gemm_prequant(
        a_packed: *const c_void,
        a_sf: *const c_void,
        w_packed: *const c_void,
        w_sf: *const c_void,
        alpha: f32,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        swizzle: i32,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
}

/// Persistent-scheduler tile swizzle for a `k`-deep GEMM. GB10, M = 6618
/// (`prequant_bench`): K 5120 x N 40960 234 -> 309 TFLOPS at 8; K 40960 x
/// N 5120 92 -> 215 at 4 (8: 211). Unswizzled, a wave's tiles stream fresh
/// long-K operand panels from DRAM.
// Only the CUTLASS (SM120) build calls this; strix-hip compiles the stubs.
#[cfg_attr(not(atlas_cutlass), allow(dead_code))]
fn swizzle(k: u32) -> i32 {
    if k > 8192 { 4 } else { 8 }
}

/// Bytes of a swizzled SM120 block-scale atom (SFA or SFB) for `rows` x `k`:
/// rows padded to 128, K/16 scale columns padded to 4.
pub fn nvfp4_sf_bytes(rows: usize, k: usize) -> usize {
    rows.div_ceil(128) * 128 * (k / 16).div_ceil(4) * 4
}

/// One activation operand: packed E2M1 `[m, k/2]` plus its SFA atom
/// ([`nvfp4_sf_bytes`]`(m, k)` bytes).
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4Act {
    pub packed: u64,
    pub sf: u64,
}

/// One weight operand: native packed E2M1 `[n, k/2]` and its load-time SFB.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4WeightSfb {
    pub packed: u64,
    pub sfb: u64,
}

// Only the CUTLASS (SM120) build calls this; strix-hip compiles the stubs.
#[cfg_attr(not(atlas_cutlass), allow(dead_code))]
fn check(what: &str, status: i32, m: u32, k: u32) -> Result<()> {
    if status != 0 {
        bail!("CUTLASS {what} failed: status {status} for {m}x{k}");
    }
    Ok(())
}

/// Quantize BF16 `[m, k]` at global scale `g` (`input_scale`) into `out`.
pub fn nvfp4_quant_act_scaled(
    act: u64,
    out: Nvfp4Act,
    m: u32,
    k: u32,
    g: f32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let status = drained("nvfp4_quant_act_scaled", unsafe {
            atlas_cutlass_nvfp4_quant_act_scaled(
                act as *const c_void,
                out.packed as *mut c_void,
                out.sf as *mut c_void,
                m as i32,
                k as i32,
                1.0 / g,
                stream as *mut c_void,
            )
        });
        check("nvfp4_quant_act_scaled", status, m, k)
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (act, out, m, k, g, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// `silu(min(gate, limit)) * clamp(up, ±limit)` over BF16 `[m, k]`, rounded to
/// BF16, quantized at global scale `g` into `out` (the down projection's input).
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_quant_swiglu_clamp(
    gate: u64,
    up: u64,
    out: Nvfp4Act,
    m: u32,
    k: u32,
    g: f32,
    limit: f32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let status = drained("nvfp4_quant_swiglu_clamp", unsafe {
            atlas_cutlass_nvfp4_quant_swiglu_clamp(
                gate as *const c_void,
                up as *const c_void,
                out.packed as *mut c_void,
                out.sf as *mut c_void,
                m as i32,
                k as i32,
                1.0 / g,
                limit,
                stream as *mut c_void,
            )
        });
        check("nvfp4_quant_swiglu_clamp", status, m, k)
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (gate, up, out, m, k, g, limit, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

/// `out[m, n]` (BF16) = `alpha * a[m, k] x w[n, k]^T`.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemm_prequant(
    a: Nvfp4Act,
    w: Nvfp4WeightSfb,
    alpha: f32,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    #[cfg(atlas_cutlass)]
    {
        let ctx = ctx()?;
        let status = drained("nvfp4_gemm_prequant", unsafe {
            atlas_cutlass_nvfp4_gemm_prequant(
                a.packed as *const c_void,
                a.sf as *const c_void,
                w.packed as *const c_void,
                w.sfb as *const c_void,
                alpha,
                out as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                swizzle(k),
                ctx.workspace()? as *mut c_void,
                ctx.ws_size,
                stream as *mut c_void,
            )
        });
        if status != 0 {
            bail!("CUTLASS nvfp4 prequant GEMM failed: status {status} for {m}x{n}x{k}");
        }
        Ok(())
    }
    #[cfg(not(atlas_cutlass))]
    {
        let _ = (a, w, alpha, out, m, n, k, stream);
        bail!("CUTLASS support was not built; set CUTLASS_HOME when building")
    }
}

#[cfg(test)]
mod tests {
    use super::nvfp4_sf_bytes;

    #[test]
    fn sf_atom_pads_rows_to_128_and_scale_columns_to_4() {
        assert_eq!(nvfp4_sf_bytes(128, 64), 128 * 4);
        assert_eq!(nvfp4_sf_bytes(129, 5120), 256 * 320);
        assert_eq!(nvfp4_sf_bytes(1, 16), 128 * 4);
    }
}
