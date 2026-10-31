// SPDX-License-Identifier: AGPL-3.0-only

/// nemotron_h_omni: each bare `<image>`(img) -> `<img>` + N×img + `</img>`.
/// `pad_counts[i]` is the i-th image's token count (target_patches /
/// downsample²), matching the checkpoint processor's `num_tokens`.
/// Refuse-not-truncate is enforced downstream: a pad run that would
/// not fit the sequence budget makes the request fail rather than
/// being cut mid-image.
pub(crate) fn expand_omni_pads(
    tokens: Vec<u32>,
    pad_counts: &[usize],
    img: u32,
    start: u32,
    end: u32,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(tokens.len() + pad_counts.iter().sum::<usize>());
    let mut idx = 0usize;
    for t in tokens {
        if t == img && idx < pad_counts.len() {
            out.push(start);
            for _ in 0..pad_counts[idx].max(1) {
                out.push(img);
            }
            out.push(end);
            idx += 1;
        } else {
            out.push(t);
        }
    }
    out
}
