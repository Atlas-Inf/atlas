// SPDX-License-Identifier: AGPL-3.0-only

//! Patch packing shared by the model-specific video samplers.
use super::*;

/// One temporal group's `tp` frames, already at the `grid × patch` canvas,
/// as `[grid_h * grid_w, C * tp * patch * patch]` normalized patches in the
/// tower's patch order.
pub(super) fn patchify_group(
    vcfg: &VisionConfig,
    frames: &[RgbImage],
    grid_h: usize,
    grid_w: usize,
) -> Vec<f32> {
    let (ps, tp) = (vcfg.patch_size, vcfg.temporal_patch_size);
    let patch_dim = 3 * tp * ps * ps;
    let mut pixels = vec![0.0f32; grid_h * grid_w * patch_dim];
    for (patch_idx, (ph, pw)) in patch_coordinates(vcfg, grid_h, grid_w)
        .into_iter()
        .enumerate()
    {
        for c in 0..3usize {
            for (t, frame) in frames.iter().enumerate() {
                for py in 0..ps {
                    for px in 0..ps {
                        let raw = frame.get_pixel((pw * ps + px) as u32, (ph * ps + py) as u32)[c];
                        let off = c * (tp * ps * ps) + t * (ps * ps) + py * ps + px;
                        pixels[patch_idx * patch_dim + off] = normalize_channel(vcfg, c, raw);
                    }
                }
            }
        }
    }
    pixels
}
