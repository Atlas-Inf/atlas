// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

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
