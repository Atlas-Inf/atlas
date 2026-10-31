// SPDX-License-Identifier: AGPL-3.0-only

//! Deterministic sample planning for a decoded clip.
//!
//! Pure arithmetic, no I/O and no decoder: given a clip's duration and the
//! operator's sampling policy, decide WHERE the frames come from and at what
//! time each one sits. The decoder (`video_decode_ffmpeg`) and the in-process
//! GIF path share it so a GIF and an MP4 of the same clip sample the same way.
//!
//! # Why an explicit timestamp plan instead of `fps=`
//!
//! ffmpeg's `-vf fps=N` walks the timeline at a fixed interval and stops when
//! the requested frame count is reached. On a clip longer than
//! `max_frames / fps` seconds that silently keeps only the FIRST
//! `max_frames / fps` seconds: a 4 s clip decoded at the 32-frame cap and
//! 2 fps samples the first 16 s, and a 20-minute clip at the 768-frame cap
//! samples the first 6.4 minutes. Everything after that is invisible to the
//! model with no error and no log line.
//!
//! Emitting one timestamp per wanted frame and selecting each with
//! `select=eq(n\,K)` removes the ambiguity: the positions span the whole
//! duration by construction, they include both endpoints, and the plan says
//! exactly which frame times the returned frames represent — which is what
//! the prompt-side timestamps are derived from.

use anyhow::{Result, ensure};

/// Refuse to build a `select` filter with more clauses than this.
///
/// The bound exists because the clauses are handed to ffmpeg as ONE argv
/// element: a container that declares an absurd duration must fail with a
/// message about the container rather than build a megabyte-long argument.
/// Well above the 768 the checkpoint's `video_processor` declares.
pub const MAX_SAMPLE_PLAN: usize = 4_096;

/// What to decode, and when each returned frame sits in the source clip.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplePlan {
    /// Source timestamps of the frames to decode, in seconds, ascending, with
    /// the first at 0.0 and the last at the final decodable frame's time.
    pub times: Vec<f32>,
}

impl SamplePlan {
    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// The ffmpeg filter that selects exactly these frames.
    ///
    /// `eq(n\,K)` matches one input frame by its index. Frame indices are the
    /// sampling grid's own output positions, so this selects frame K of the
    /// grid rather than a decoder-internal frame — which keeps the plan
    /// independent of the container's native rate.
    pub fn select_filter(&self) -> String {
        let clauses: Vec<String> = (0..self.times.len())
            .map(|k| format!("eq(n\\,{k})"))
            .collect();
        format!("select='{}'", clauses.join("+"))
    }

    /// The ffmpeg `select` filter that advances through this plan by source
    /// timestamp rather than by a synthetic frame index. `selected_n` is the
    /// count of frames already accepted; for each plan position the filter
    /// takes the first source frame whose `t` reaches that position — CFR and
    /// VFR inputs keep their own timeline and `showinfo` can report the
    /// selected frame's real PTS. Enntity/atlas d0cead1.
    pub fn select_time_filter(&self) -> String {
        // Plans are uniformly spaced. Express the next target from selected_n
        // in constant space: thousands of clauses exceed Linux's argv bound
        // and require quadratic work across an hour of source frames.
        if self.times.len() <= 1 {
            return "select='eq(selected_n,0)'".into();
        }
        let start = f64::from(self.times[0]);
        let end = f64::from(self.times[self.times.len() - 1]);
        // Source times are f32: at an hour their rounding can put the final
        // target beyond the last actual PTS. Compute the repeated step in f64
        // and inset by two f32 rounding units (under 1 ms at an hour).
        let rounding = f64::from(f32::EPSILON) * end.abs().max(1.0) * 2.0;
        let step = (end - start - rounding).max(0.0) / (self.times.len() - 1) as f64;
        format!(
            "select='lt(selected_n\\,{})*gte(t\\,{start:.12}+selected_n*{step:.12})'",
            self.times.len()
        )
    }
}

/// Spacing of the sampling grid, in the decoder's own frame-index space.
///
/// `ceil` rather than `floor`: rounding the interval DOWN asks ffmpeg to
/// produce MORE frames than the source has. The fps filter repeats the last
/// frame to honour that (measured on 7.1.1: `duration=1s, fps=2` yields three
/// samples, the third a duplicate), and a duplicated frame is a frame the
/// model is charged for and cannot learn from. Rounding up can only skip, and
/// the endpoints are pinned separately.
pub fn grid_step(native_fps: f32, target_fps: f32, fallback: f32) -> u32 {
    let native = finite_positive(native_fps).unwrap_or(fallback);
    let target = finite_positive(target_fps).unwrap_or(fallback);
    ((native / target).ceil() as u32).max(1)
}

/// How many frames to sample from a clip of `duration_secs`.
///
/// At the source rate when the clip is shorter than the cap, at the target
/// rate otherwise — never more than `max_frames`.
pub fn wanted_frames(
    duration_secs: f32,
    native_fps: f32,
    target_fps: f32,
    max_frames: usize,
    fallback_fps: f32,
) -> usize {
    let native = finite_positive(native_fps).unwrap_or(fallback_fps);
    let target = finite_positive(target_fps).unwrap_or(fallback_fps);
    let cap = max_frames.max(1);
    let bounded = |x: f32| x.floor().max(1.0).min(cap as f32) as usize;
    // An unprobed duration is planned for the CAP rather than for a guess:
    // leaving samples on the table would shorten the clip silently, which is
    // the failure this module exists to remove. The decoder still refuses a
    // clip it cannot probe — it needs the duration for the timestamps.
    let duration = finite_positive(duration_secs).unwrap_or(cap as f32);
    let natural = duration * native;
    // `<=`: a clip whose source rate lands exactly on the cap still keeps every
    // source frame. Only a clip that would EXCEED the cap falls back to the
    // requested rate, because then something has to be dropped.
    if natural <= cap as f32 {
        bounded(natural)
    } else {
        bounded(duration * target)
    }
}

/// Plan `wanted` evenly spaced frames across `duration_secs`.
///
/// Both endpoints are always present: the first frame of the clip and the
/// last one that can be decoded without reaching past the end. The spacing is
/// uniform, so the sample rate never exceeds `wanted / duration` — which is
/// where the operator's `fps` bound comes back in.
///
/// The caller is responsible for the two clip properties this arithmetic
/// cannot see: that the duration was actually probed, and that it is long
/// enough for a temporal group. Refusing here would put a checkpoint-shaped
/// (`temporal_patch_size`) rule inside a helper that deliberately has none.
pub fn sample_plan(wanted: usize, duration_secs: f32) -> Result<SamplePlan> {
    ensure!(
        wanted > 0,
        "video sampling asked for zero frames; a clip must carry at least one"
    );
    ensure!(
        wanted <= MAX_SAMPLE_PLAN,
        "video sampling asked for {wanted} frames, above the {MAX_SAMPLE_PLAN}-frame planning limit"
    );
    let duration = finite_positive(duration_secs)
        .ok_or_else(|| anyhow::anyhow!("video duration is not a finite positive number"))?;

    // A wanted count of one cannot cover a span, and the arithmetic below
    // degenerates for it by construction: the single sample sits at 0.0 with
    // no claim about the end of the clip.
    if wanted == 1 {
        return Ok(SamplePlan { times: vec![0.0] });
    }
    let mut times = vec![0.0f32; wanted];
    // The frame at `duration` does not exist — a 4.000 s clip holds its final
    // frame at 3.875 s at 8 fps. Aim one grid step short of the end so the
    // selection names a frame the decoder actually has, rather than a time
    // past the last one that silently returns nothing.
    let end = (duration - 1.0 / PLAN_FPS_DENOM).max(0.0);
    let step = end / (wanted - 1) as f32;
    for (k, t) in times.iter_mut().enumerate() {
        *t = (k as f32 * step).min(end);
    }
    Ok(SamplePlan { times })
}

/// Denominator of the grid the endpoint inset approximates: a plan works in
/// fractional seconds, and one grid step at any plausible container rate is
/// around a hundredth of a second. The inset only has to keep the end time
/// off the exact boundary, so an approximate value is not a correctness
/// claim about any particular rate.
const PLAN_FPS_DENOM: f32 = 100.0;

/// GLM reference sampling defaults (`max_frame_count_dynamic`, `fps` in
/// vLLM's `glm5next` video processor).
pub const GLM_DEFAULT_MAX_FRAMES: usize = 2_048;
pub const GLM_DEFAULT_FPS: f64 = 2.0;

/// The ffmpeg `select` filter that keeps exactly the input frames numbered
/// `indices` (`eq(n\,K)` matches one frame by its index).
/// (Enntity/atlas d0cead1.)
pub fn select_frames(indices: impl IntoIterator<Item = usize>) -> String {
    let clauses: Vec<String> = indices
        .into_iter()
        .map(|k| format!("eq(n\\,{k})"))
        .collect();
    format!("select='{}'", clauses.join("+"))
}

/// vLLM `glm5next` frame-index sampling: `linspace` over the clip when it is
/// shorter than the extraction count, else a `1/(tp*fps)` walk over declared
/// frame times, then a linspace re-count. The last index is duplicated when
/// the count is odd — the processor pads the final temporal pair.
/// (Enntity/atlas d0cead1.)
pub fn glm_sample_frame_indices(
    total_frames: usize,
    fps: f64,
    duration: f64,
    target_fps: Option<f64>,
    max_frame_count: Option<usize>,
    temporal_patch_size: usize,
) -> Vec<usize> {
    let max_frame_idx = total_frames as f64 - 1.0;
    let duration = if duration != 0.0 {
        duration
    } else if fps != 0.0 {
        (max_frame_idx / fps).round_ties_even() + 1.0
    } else {
        0.0
    };
    let target_fps = target_fps.unwrap_or(GLM_DEFAULT_FPS);
    let extract_t =
        ((duration * target_fps) as usize).min(max_frame_count.unwrap_or(GLM_DEFAULT_MAX_FRAMES));
    let mut indices: Vec<usize> = if total_frames < extract_t {
        (0..extract_t)
            .map(|i| i * total_frames / extract_t)
            .collect()
    } else {
        let per_frame = 1.0 / fps;
        let step = 1.0 / (temporal_patch_size as f64 * target_fps);
        let max_second = duration.trunc();
        let mut walk = Vec::new();
        let mut current = 0.0f64;
        for index in 0..total_frames {
            if index as f64 * per_frame >= current {
                current += step;
                walk.push(index);
                if current >= max_second {
                    break;
                }
            }
        }
        walk
    };
    if indices.len() < extract_t {
        let (start, end) = match (indices.first(), indices.last()) {
            (Some(&first), Some(&last)) => (first, last),
            _ => (0, total_frames.saturating_sub(1)),
        };
        indices = linspace_floor(start, end, extract_t);
    } else if indices.len() > extract_t {
        indices = linspace_floor(0, total_frames.saturating_sub(1), extract_t);
    }
    let mut seen = std::collections::HashSet::new();
    indices.retain(|&index| seen.insert(index));
    if indices.len() % 2 == 1 {
        indices.push(indices[indices.len() - 1]);
    }
    indices
}

/// `np.linspace(start, end, num, dtype=int)`: `k × step + start` in f64,
/// floored, with the endpoint exact. (Enntity/atlas d0cead1.)
fn linspace_floor(start: usize, end: usize, num: usize) -> Vec<usize> {
    if num <= 1 {
        return vec![start; num];
    }
    let step = (end as f64 - start as f64) / (num - 1) as f64;
    (0..num)
        .map(|k| {
            if k == num - 1 {
                end
            } else {
                (k as f64 * step + start as f64).floor() as usize
            }
        })
        .collect()
}

/// `Some` only for a rate that can be divided by.
pub(crate) fn finite_positive(x: f32) -> Option<f32> {
    (x.is_finite() && x > 0.0).then_some(x)
}

#[cfg(test)]
#[path = "video_sample_tests.rs"]
mod tests;
