// SPDX-License-Identifier: AGPL-3.0-only

//! The block-FP8 (`Family::W8a16`) launch of `skinny_mma`.

use super::*;

/// `m` rows through one or two block-scaled FP8 weights: part
/// `((weight, block_scale), out, n)`, FP32 block scales
/// `[ceil(n/128), k/128]`, writes `out[t * c_stride + j]`.
pub fn launch_w8a16(
    gpu: &dyn GpuBackend,
    kernels: &SkinnyKernels,
    input: DevicePtr,
    parts: &[((DevicePtr, DevicePtr), DevicePtr, u32)],
    (m, k): (u32, u32),
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    kernels.check(Family::W8a16, m, k)?;
    ensure!(
        (1..=2).contains(&parts.len()),
        "skinny_w8a16: {} parts",
        parts.len()
    );
    let tiles: u32 = parts.iter().map(|p| div_ceil(p.2, CTA_OUTPUTS)).sum();
    for (kernel, first, rows, per_cta) in kernels.launches(Family::W8a16, m) {
        let mut l = launch(gpu, kernel, div_ceil(tiles, per_cta))
            .arg_ptr(input.offset(row_off(first, k)))
            .arg_u32(rows)
            .arg_u32(k)
            .arg_u32(c_stride);
        for p in 0..2 {
            let ((weight, scale), out, n) =
                parts
                    .get(p)
                    .copied()
                    .unwrap_or(((DevicePtr(0), DevicePtr(0)), DevicePtr(0), 0));
            let out = if out.0 == 0 {
                out
            } else {
                out.offset(row_off(first, c_stride))
            };
            l = l.arg_ptr(weight).arg_ptr(scale).arg_ptr(out).arg_u32(n);
        }
        l.launch(stream)?;
    }
    Ok(())
}
