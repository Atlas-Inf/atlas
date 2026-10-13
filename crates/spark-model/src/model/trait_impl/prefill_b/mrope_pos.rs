// SPDX-License-Identifier: AGPL-3.0-only

//! MRoPE (T, H, W) position streams for a prefill chunk.
//!
//! Pure arithmetic, extracted from `upload_meta` so it can be tested: the
//! caller needs a GPU, a KV cache and a pinned staging allocation, none of
//! which the position rule depends on. This decides where every vision token
//! sits in all three rotary streams, and a mistake here is invisible — the
//! model produces fluent, confidently wrong output rather than failing.

/// Append the (T, H, W) streams for `chunk_tokens` to the three output
/// vectors, starting the running position at `start_pos`.
///
/// Matches HF Qwen3-VL's `get_rope_index` / `get_vision_position_ids`:
///
/// - a TEXT token takes `T = H = W = pos` and advances `pos` by one;
/// - a VISION item of `t_len` temporal groups over a post-merge `gh × gw`
///   grid occupies `t_len * gh * gw` consecutive pad tokens, where token `k`
///   of group `g` takes `T = base + g`, `H = base + row`, `W = base + col`,
///   and afterwards `pos` advances by `max(t_len, gh, gw)`.
///
/// An IMAGE is the `t_len = 1` case and reduces exactly to the image-only
/// rule that preceded this: `base + g` collapses to `base`, and the advance
/// to `max(gh, gw)`.
///
/// Both pad tokens are recognized. They are consumed identically — the item's
/// own `t_len` already says whether it is a still or a clip — but a video run
/// whose token went unrecognized would be walked one text token at a time,
/// handing each of its thousands of pad positions a distinct index and
/// shifting every token after it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build(
    chunk_tokens: &[u32],
    grids: &[(usize, usize, usize)],
    grid_base: usize,
    grid_hi: usize,
    start_pos: u32,
    image_pad: u32,
    video_pad: u32,
    t_out: &mut Vec<u32>,
    h_out: &mut Vec<u32>,
    w_out: &mut Vec<u32>,
) {
    let is_pad = |tok: u32| tok == image_pad || tok == video_pad;
    let mut pos = start_pos;
    let mut item = grid_base;
    let mut i = 0usize;
    while i < chunk_tokens.len() {
        if is_pad(chunk_tokens[i]) && item < grid_hi {
            let (t_len, gh, gw) = grids[item];
            let t_len = t_len.max(1);
            let plane = (gh * gw).max(1);
            let run_len = t_len * plane;
            let base = pos;
            for k in 0..run_len {
                // [group, row, col] order — the order the encoder emitted the
                // groups, and therefore the order the merged rows are spliced.
                let g = (k / plane) as u32;
                let within = k % plane;
                let row = (within / gw.max(1)) as u32;
                let col = (within % gw.max(1)) as u32;
                t_out.push(base + g);
                h_out.push(base + row);
                w_out.push(base + col);
            }
            // The item's extent on EVERY axis, so the next text token starts
            // clear of all three streams. A long clip can exceed its own
            // spatial extent, which is why t_len joins the max rather than
            // the spatial pair being assumed to dominate.
            pos += t_len.max(gh).max(gw) as u32;
            i += run_len;
            item += 1;
        } else {
            t_out.push(pos);
            h_out.push(pos);
            w_out.push(pos);
            pos += 1;
            i += 1;
        }
    }
}

/// GLM-5 position walk for the checkpoint's expanded prompt form. Images use
/// one image marker triple. Videos use
/// `video_start, (image_start, image_pad*, image_end, timestamp*)*, video_end`;
/// each frame's patch plane is a separate multimodal range, so the structural
/// boundary and timestamp tokens remain ordinary 1-D text positions. This is
/// the same range layout produced by the pinned vLLM prompt updater.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_glm5(
    chunk_tokens: &[u32],
    grids: &[(usize, usize, usize)],
    grid_base: usize,
    grid_hi: usize,
    start_pos: u32,
    image_start: u32,
    _image_end: u32,
    video_start: u32,
    video_end: u32,
    image_pad: u32,
    video_pad: u32,
    t_out: &mut Vec<u32>,
    h_out: &mut Vec<u32>,
    w_out: &mut Vec<u32>,
) {
    let is_pad = |tok: u32| tok == image_pad || tok == video_pad;
    let mut pos = start_pos;
    let mut item = grid_base;
    let mut video_item: Option<(usize, usize)> = None;
    let mut i = 0usize;

    let push_text =
        |pos: &mut u32, t_out: &mut Vec<u32>, h_out: &mut Vec<u32>, w_out: &mut Vec<u32>| {
            t_out.push(*pos);
            h_out.push(*pos);
            w_out.push(*pos);
            *pos += 1;
        };
    let push_plane = |pos: &mut u32,
                      t_out: &mut Vec<u32>,
                      h_out: &mut Vec<u32>,
                      w_out: &mut Vec<u32>,
                      gh: usize,
                      gw: usize,
                      temporal: u32| {
        let base = *pos;
        for k in 0..gh.max(1) * gw.max(1) {
            t_out.push(base + temporal);
            h_out.push(base + (k / gw.max(1)) as u32);
            w_out.push(base + (k % gw.max(1)) as u32);
        }
        *pos += gh.max(gw).max(1) as u32;
    };

    while i < chunk_tokens.len() {
        if chunk_tokens[i] == video_start && video_item.is_none() {
            if item < grid_hi {
                video_item = Some((item, 0));
            }
            push_text(&mut pos, t_out, h_out, w_out);
            i += 1;
            continue;
        }

        if let Some((video_idx, frame)) = video_item {
            if chunk_tokens[i] == video_end {
                video_item = None;
                item = item.saturating_add(1);
                push_text(&mut pos, t_out, h_out, w_out);
                i += 1;
                continue;
            }
            if is_pad(chunk_tokens[i]) && video_idx < grid_hi {
                let (t_len, gh, gw) = grids[video_idx];
                let plane = gh.max(1) * gw.max(1);
                let contiguous = i + plane <= chunk_tokens.len()
                    && chunk_tokens[i..i + plane].iter().copied().all(is_pad);
                if frame < t_len.max(1) && contiguous {
                    // vLLM's per-frame ranges each have grid_t=1, so T is
                    // the range base. Timestamp and boundary tokens advance
                    // the next frame's base as ordinary text positions.
                    push_plane(&mut pos, t_out, h_out, w_out, gh, gw, 0);
                    video_item = Some((video_idx, frame + 1));
                    i += plane;
                    continue;
                }
            }
            push_text(&mut pos, t_out, h_out, w_out);
            i += 1;
            continue;
        }

        if is_pad(chunk_tokens[i]) && item < grid_hi {
            let (t_len, gh, gw) = grids[item];
            let plane = gh.max(1) * gw.max(1);
            // A still image is a single structured plane. A compact legacy
            // video run is kept as a defensive fallback for a caller that
            // bypasses the GLM prompt updater; normal GLM videos enter the
            // branch above and carry their boundaries/timestamps.
            let run = if t_len <= 1 {
                plane
            } else {
                plane * t_len.max(1)
            };
            let contiguous = i + run <= chunk_tokens.len()
                && chunk_tokens[i..i + run].iter().copied().all(is_pad);
            if contiguous {
                if t_len <= 1 {
                    push_plane(&mut pos, t_out, h_out, w_out, gh, gw, 0);
                } else {
                    let base = pos;
                    for k in 0..run {
                        let g = (k / plane) as u32;
                        let within = k % plane;
                        t_out.push(base + g);
                        h_out.push(base + (within / gw.max(1)) as u32);
                        w_out.push(base + (within % gw.max(1)) as u32);
                    }
                    pos += t_len.max(gh).max(gw) as u32;
                }
                item += 1;
                i += run;
                continue;
            }
        }

        // `image_start` is intentionally accepted as an ordinary text token;
        // the following image pad plane is what carries the grid positions.
        let _ = image_start;
        push_text(&mut pos, t_out, h_out, w_out);
        i += 1;
    }
}

#[cfg(test)]
#[path = "mrope_pos_tests.rs"]
mod tests;
