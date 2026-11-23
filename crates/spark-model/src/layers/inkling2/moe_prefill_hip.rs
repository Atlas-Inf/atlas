// SPDX-License-Identifier: AGPL-3.0-only

//! AMD (strix-hip) grouped-expert prefill over BF16 rows, split out of `moe_ops.rs`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::moe_ops::GEMM_N_TILE;
use super::moe_ops::MoePrefill;
use super::ops::{Inkling2Kernels, u32_of};

/// strix-hip BF16-row grouped GEMM tile rows: MXFP8 64, NVFP4 128.
const GEMM_M_TILE_MX: usize = 64;
const GEMM_M_TILE_FP4: usize = 128;

/// AMD (strix-hip): the grouped experts of [`moe_prefill`] over BF16 rows. The W4A4 / W8A8
/// block-scaled GEMMs and their activation quantizers are gb10 tensor-core PTX; here the
/// checkpoint's stacked experts are dequantised to BF16 per K step on WMMA
/// (kernels/strix-hip/inkling-2-mini/nvfp4/inkling2_moe_gemm.cu), gate_up, the softcapped
/// SwiGLU, down, then the weighted combine (gate and up rows stay interleaved, as stored).
#[allow(clippy::too_many_arguments)]
pub(super) fn moe_prefill_bf16(
    gpu: &dyn GpuBackend,
    k: &Inkling2Kernels,
    a: &MoePrefill,
    sorted_rows: DevicePtr,
    offsets: DevicePtr,
    to_perm: DevicePtr,
    stream: u64,
) -> Result<()> {
    let e = a.experts;
    e.ensure_aligned(4)?;
    let te = a.rows * a.top_k;
    let (ne, h, i) = (
        u32_of(a.num_experts, "experts")?,
        u32_of(a.hidden, "hidden")?,
        u32_of(a.inter, "inter")?,
    );
    // an expert takes each row at most once
    let m_tiles = u32_of(
        a.rows
            .div_ceil(if e.mxfp8 {
                GEMM_M_TILE_MX
            } else {
                GEMM_M_TILE_FP4
            })
            .max(1),
        "m tiles",
    )?;
    let gemm = |x: DevicePtr,
                w: DevicePtr,
                sc: DevicePtr,
                g: DevicePtr,
                y: DevicePtr,
                gather: DevicePtr,
                n: u32,
                kk: u32| {
        let l = KernelLaunch::new(
            gpu,
            if e.mxfp8 {
                k.moe_gemm_mxfp8
            } else {
                k.moe_gemm_nvfp4
            },
        )
        .grid([n.div_ceil(GEMM_N_TILE), m_tiles, ne])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(sc);
        let l = if e.mxfp8 { l } else { l.arg_ptr(g) };
        l.arg_ptr(y)
            .arg_ptr(offsets)
            .arg_ptr(gather)
            .arg_u32(n)
            .arg_u32(kk)
            .launch(stream)
    };
    gemm(
        a.x,
        e.w13,
        e.w13_scale,
        e.w13_global,
        a.gu,
        sorted_rows,
        2 * i,
        h,
    )?;
    KernelLaunch::new(gpu, k.softcap_swiglu)
        .grid([u32_of(te, "slots")?, i.div_ceil(256), 1])
        .block([256, 1, 1])
        .arg_ptr(a.gu)
        .arg_ptr(a.act)
        .arg_u32(i)
        .arg_f32(a.gate_softcap)
        .arg_f32(a.up_softcap)
        .launch(stream)?;
    gemm(
        a.act,
        e.w2,
        e.w2_scale,
        e.w2_global,
        a.y,
        DevicePtr::NULL,
        h,
        i,
    )?;
    super::super::ops::moe_unpermute_reduce_indexed(
        gpu,
        k.moe_unpermute,
        a.y,
        a.out,
        to_perm,
        a.weights,
        h,
        u32_of(a.rows, "rows")?,
        u32_of(a.top_k, "top_k")?,
        stream,
    )
}
