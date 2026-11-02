// SPDX-License-Identifier: AGPL-3.0-only

//! Incremental PNG framing: bounded compressed bytes and decoded pixel volume.
use super::*;

pub(super) fn decoded_frame_budget(policy: &FfmpegPolicy, frames: usize) -> Result<usize> {
    let area = policy
        .max_total_decoded_pixels
        .map_or(policy.max_decoded_pixels, |total| {
            policy.max_decoded_pixels.min(total / frames.max(1))
        });
    ensure!(
        area > 0,
        "video decode pixel budget cannot hold {frames} frames"
    );
    Ok(area)
}

pub(super) fn read_frames(
    input: &mut impl Read,
    expected: usize,
    policy: &FfmpegPolicy,
) -> Result<Vec<RgbImage>> {
    let area = decoded_frame_budget(policy, expected)?;
    let mut frames = Vec::with_capacity(expected);
    let mut total = 0usize;
    loop {
        let mut signature = [0u8; 8];
        let first = input.read(&mut signature[..1])?;
        if first == 0 {
            break;
        }
        input.read_exact(&mut signature[1..])?;
        ensure!(signature == PNG_MAGIC, "invalid PNG frame signature");
        ensure!(
            frames.len() < expected,
            "decoder returned more than {expected} frames"
        );
        let mut png = signature.to_vec();
        total += 8;
        loop {
            let mut header = [0u8; 8];
            input.read_exact(&mut header)?;
            let len = u32::from_be_bytes(header[..4].try_into()?) as usize;
            let extra = len.checked_add(12).context("PNG chunk size overflow")?;
            total = total
                .checked_add(extra)
                .context("PNG output size overflow")?;
            ensure!(
                total <= policy.max_output_bytes,
                "decoded output exceeded the {}-byte cap",
                policy.max_output_bytes
            );
            png.extend_from_slice(&header);
            let offset = png.len();
            png.resize(offset + len + 4, 0);
            input.read_exact(&mut png[offset..])?;
            if &header[4..] == b"IEND" {
                break;
            }
        }
        let img = image::load_from_memory_with_format(&png, image::ImageFormat::Png)?;
        ensure!(
            (img.width() as usize).saturating_mul(img.height() as usize) <= area,
            "decoded frame exceeded its {area}-pixel budget"
        );
        frames.push(img.to_rgb8());
    }
    Ok(frames)
}
