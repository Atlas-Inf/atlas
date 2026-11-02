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
/// - a VISION image occupies one `gh × gw` pad plane;
/// - a Qwen video item of `t_len` temporal groups is rendered as one
///   timestamped `gh × gw` pad plane per group, with its boundary/timestamp
///   tokens walked as ordinary text between planes. Each frame plane therefore
///   starts a fresh base, matching Qwen3.8's reference `get_rope_index` after
///   it repeats `video_grid_thw` and sets each temporal extent to one.
///
/// An IMAGE is the `t_len = 1` case and reduces exactly to the image-only
/// rule that preceded this: `base + g` collapses to `base`, and the advance
/// to `max(gh, gw)`.
///
/// Both pad tokens are recognized. Video pad planes are consumed one at a time
/// when the owning item has `t_len > 1`; this handles both the timestamped
/// Qwen3.8 form and a compact legacy run without changing image arithmetic.
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
    let mut video_frame = 0usize;
    let mut i = 0usize;
    while i < chunk_tokens.len() {
        if is_pad(chunk_tokens[i]) && item < grid_hi {
            let (t_len, gh, gw) = grids[item];
            let t_len = t_len.max(1);
            let plane = (gh * gw).max(1);
            let run_len = t_len * plane;
            let base = pos;
            if chunk_tokens[i] == video_pad && t_len > 1 {
                // Qwen3.8's processor repeats one video_grid_thw row per
                // temporal group and emits a timestamped vision range for
                // every row. The timestamp and vision boundaries are ordinary
                // text positions, so each pad plane starts at the current
                // base and advances the spatial extent only.
                for k in 0..plane {
                    let row = (k / gw.max(1)) as u32;
                    let col = (k % gw.max(1)) as u32;
                    t_out.push(base);
                    h_out.push(base + row);
                    w_out.push(base + col);
                }
                pos += gh.max(gw).max(1) as u32;
                i += plane;
                video_frame += 1;
                if video_frame >= t_len {
                    video_frame = 0;
                    item += 1;
                }
            } else {
                for k in 0..run_len {
                    // [group, row, col] order — the order the encoder emitted
                    // the groups, and therefore the order the merged rows are
                    // spliced.
                    let g = (k / plane) as u32;
                    let within = k % plane;
                    let row = (within / gw.max(1)) as u32;
                    let col = (within % gw.max(1)) as u32;
                    t_out.push(base + g);
                    h_out.push(base + row);
                    w_out.push(base + col);
                }
                // The item's extent on EVERY axis, so the next text token
                // starts clear of all three streams.
                pos += t_len.max(gh).max(gw) as u32;
                i += run_len;
                item += 1;
                video_frame = 0;
            }
        } else {
            t_out.push(pos);
            h_out.push(pos);
            w_out.push(pos);
            pos += 1;
            i += 1;
        }
    }
}

/// Qwen3-VL position walk for the checkpoint's expanded prompt form. HF's
/// `get_rope_index` splits `video_grid_thw` into one `t=1` plane per
/// temporal group (`repeat_interleave` + `[:,0]=1`), so every contiguous
/// pad run is a single `gh × gw` plane whose T is the plane base, and the
/// timestamp/start/end tokens between planes are ordinary 1-D text
/// positions. Images are the `t_len = 1` case.
///
/// Consumes `grids` item-by-item in media order: an image takes ONE plane
/// (its `t_len` is 1), a video takes `t_len` planes — one per group — then
/// advances to the next item. Boundary markers and timestamp text are plain
/// text positions, matching the processor's nested expansion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_qwen3vl(
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
    let mut group = 0usize;
    let mut i = 0usize;
    while i < chunk_tokens.len() {
        if is_pad(chunk_tokens[i]) && item < grid_hi {
            let (t_len, gh, gw) = grids[item];
            let plane = (gh * gw).max(1);
            let contiguous = i + plane <= chunk_tokens.len()
                && chunk_tokens[i..i + plane].iter().copied().all(is_pad);
            if group < t_len.max(1) && contiguous {
                // One t=1 plane: T stays at the base for the whole group,
                // H/W carry the grid. Pos advances by the spatial extent so
                // the next group's timestamp starts clear of it — HF's
                // `pos += max(gh, gw) // spatial_merge_size`, with gh/gw
                // already post-merge here.
                let base = pos;
                for k in 0..plane {
                    t_out.push(base);
                    h_out.push(base + (k / gw.max(1)) as u32);
                    w_out.push(base + (k % gw.max(1)) as u32);
                }
                pos += gh.max(gw).max(1) as u32;
                group += 1;
                if group >= t_len.max(1) {
                    item += 1;
                    group = 0;
                }
                i += plane;
                continue;
            }
        }
        t_out.push(pos);
        h_out.push(pos);
        w_out.push(pos);
        pos += 1;
        i += 1;
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

/// Which walker positions a vision prompt (the checkpoint's prompt form).
#[derive(Clone, Copy)]
pub(crate) enum Walk {
    /// [`build`]: one contiguous pad run per item.
    Generic,
    /// [`build_qwen3vl`]: one timestamp-separated `t=1` plane per group.
    Qwen3vl,
    /// [`build_glm5`]: per-frame image markers; the four marker ids.
    Glm5 {
        image_start: u32,
        image_end: u32,
        video_start: u32,
        video_end: u32,
    },
}

/// The (T, H, W) positions of prompt rows `rows`, appended to `out`: the
/// walker runs over the prompt from position 0 and the range is sliced out,
/// so text after an image keeps the image's compressed position base in
/// every later chunk, an item straddling a chunk boundary keeps its plane,
/// and a later chunk's item takes its own grid. A prefill that starts after
/// a cached prefix therefore positions its rows exactly as a full prefill
/// does. A walk shorter than the range means the placeholders and grids
/// disagree; every rank decides that from the same tokens and grids
/// (`request_local`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn walk_rows(
    walk: Walk,
    tokens: &[u32],
    grids: &[(usize, usize, usize)],
    grid_base: usize,
    grid_hi: usize,
    (image_pad, video_pad): (u32, u32),
    rows: std::ops::Range<usize>,
    out: [&mut Vec<u32>; 3],
) -> anyhow::Result<()> {
    if rows.end > tokens.len() || rows.start > rows.end {
        return Err(crate::request_local::request_local(format!(
            "MRoPE position range {rows:?} exceeds the {}-token prompt",
            tokens.len()
        )));
    }
    let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
    match walk {
        Walk::Generic => build(
            tokens, grids, grid_base, grid_hi, 0, image_pad, video_pad, &mut t, &mut h, &mut w,
        ),
        Walk::Qwen3vl => build_qwen3vl(
            tokens, grids, grid_base, grid_hi, 0, image_pad, video_pad, &mut t, &mut h, &mut w,
        ),
        Walk::Glm5 {
            image_start,
            image_end,
            video_start,
            video_end,
        } => build_glm5(
            tokens,
            grids,
            grid_base,
            grid_hi,
            0,
            image_start,
            image_end,
            video_start,
            video_end,
            image_pad,
            video_pad,
            &mut t,
            &mut h,
            &mut w,
        ),
    }
    if rows.end > t.len().min(h.len()).min(w.len()) {
        return Err(crate::request_local::request_local(format!(
            "MRoPE position range {rows:?} exceeds the {} walked positions",
            t.len()
        )));
    }
    let [ot, oh, ow] = out;
    ot.extend_from_slice(&t[rows.clone()]);
    oh.extend_from_slice(&h[rows.clone()]);
    ow.extend_from_slice(&w[rows]);
    Ok(())
}

#[cfg(test)]
#[path = "mrope_pos_tests.rs"]
mod tests;
