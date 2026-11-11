// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the sample planner — pure arithmetic, no decoder.

use super::*;

/// The regression this module exists for: a 4 s clip at the 32-frame cap and
/// 2 fps. ffmpeg's `fps=2` + `-frames:v 32` stops after 32 samples, i.e. after
/// 16 s of a 4 s clip — the whole clip here, but the same rule keeps only the
/// first 16 s of a 5-minute one. The plan must span the clip instead.
#[test]
fn the_plan_spans_the_whole_clip_and_keeps_both_endpoints() {
    let p = sample_plan(32, 4.0).expect("plan");
    assert_eq!(p.times.len(), 32);
    assert_eq!(p.times[0], 0.0, "the first frame must be the clip's first");
    let last = *p.times.last().unwrap();
    assert!(
        (last - 4.0).abs() <= 0.05,
        "the last sample must sit at the end of the clip, got {last}"
    );
    // No claim may exceed the clip: ffmpeg's terminal sample at exactly
    // `duration` duplicates the previous frame (verified on 7.1.1).
    for t in &p.times {
        assert!(*t >= 0.0 && *t < 4.0, "sample at {t} is outside 0..4");
    }
    for w in p.times.windows(2) {
        assert!(w[1] > w[0], "times must ascend: {:?}", w);
    }
}

/// The truncation case, at a duration long enough that the old behaviour was
/// plainly wrong: 300 s at 2 fps is 600 samples, above the 32-frame cap, and
/// `fps=2 -frames:v 32` would have covered only the first 16 s.
#[test]
fn a_long_clip_is_sampled_past_the_point_the_old_cap_stopped() {
    let p = sample_plan(32, 300.0).expect("plan");
    let last = *p.times.last().unwrap();
    assert!(
        last > 290.0,
        "a 300 s clip must be sampled to its end, not to 16 s; got {last}"
    );
    // The old behaviour's horizon is not merely missed, it is covered evenly.
    assert!(
        p.times.iter().any(|&t| t > 150.0),
        "nothing sampled in the second half: {:?}",
        p.times
    );
}

/// The effective rate must never exceed the requested one — that is the half
/// of "preserve at most the configured fps" that a coverage-only fix could
/// break by packing 32 samples into a 4 s clip.
#[test]
fn the_effective_rate_stays_within_the_requested_fps() {
    for (max_frames, duration, fps) in [
        (32usize, 4.0f32, 2.0f32),
        (32, 300.0, 2.0),
        (768, 600.0, 2.0),
        (1, 300.0, 2.0),
        (2, 300.0, 2.0),
    ] {
        let want = wanted_frames(duration, fps, fps, max_frames, 2.0);
        let plan = sample_plan(want, duration).expect("plan");
        let span = plan.times.last().copied().unwrap_or(0.0);
        let observed = plan.len() as f32 / (span.max(1e-6));
        if plan.len() > 1 {
            assert!(
                observed <= fps * 1.01,
                "cap={max_frames} dur={duration}: {observed} fps vs requested {fps}"
            );
        }
    }
}

/// A frame cap of one cannot cover the span, so it must not pretend to: the
/// plan is one frame, at the start, and says so.
#[test]
fn a_frame_cap_of_one_plans_a_single_first_frame() {
    let p = sample_plan(1, 300.0).expect("plan");
    assert_eq!(p.times, vec![0.0]);
}

#[test]
fn a_frame_cap_of_two_keeps_the_first_and_last_available_frames() {
    let p = sample_plan(2, 120.0).expect("plan");
    assert_eq!(p.times.len(), 2);
    assert_eq!(p.times[0], 0.0);
    assert!(p.times[1] > 119.0, "got {:?}", p.times);
}

/// An unprobed or broken duration must fail loudly. The old path's failure
/// mode was to sample the first frames and say nothing, which is the bug.
#[test]
fn an_unknown_or_invalid_duration_is_refused() {
    for bad in [f32::NAN, f32::INFINITY, 0.0, -3.0] {
        let err = sample_plan(32, bad).unwrap_err().to_string();
        assert!(err.contains("duration"), "duration={bad}: {err}");
    }
    assert!(sample_plan(0, 4.0).is_err(), "zero frames is not a plan");
}

/// The select filter must name exactly the frames the plan holds, and it must
/// be a single argv-shaped string (ffmpeg receives it as one argument).
#[test]
fn the_select_filter_names_every_planned_frame() {
    let p = sample_plan(4, 10.0).expect("plan");
    assert_eq!(
        p.select_filter(),
        "select='eq(n\\,0)+eq(n\\,1)+eq(n\\,2)+eq(n\\,3)'"
    );
    assert_eq!(
        sample_plan(1, 10.0).unwrap().select_filter(),
        "select='eq(n\\,0)'"
    );
}

#[test]
fn the_timestamp_filter_advances_once_per_planned_source_time() {
    let p = sample_plan(3, 10.0).expect("plan");
    let filter = p.select_time_filter();
    assert!(filter.contains("lt(selected_n\\,3)"));
    assert!(filter.len() < 100);
    assert!(
        sample_plan(7200, 3600.0)
            .unwrap()
            .select_time_filter()
            .len()
            < 100
    );
}

/// A container declaring an absurd rate must be refused by the planner rather
/// than build a filter with millions of clauses.
#[test]
fn an_absurd_frame_count_is_refused_rather_than_built() {
    assert!(sample_plan(MAX_SAMPLE_PLAN, 1.0).is_ok());
    let err = sample_plan(MAX_SAMPLE_PLAN + 1, 1.0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("planning limit"), "{err}");
}

/// Rounding the grid interval DOWN asks ffmpeg for frames the clip does not
/// have, and the fps filter answers with duplicates of the last one.
#[test]
fn the_grid_interval_rounds_up_so_the_decoder_is_never_asked_to_repeat() {
    assert_eq!(grid_step(8.0, 2.0, 2.0), 4);
    assert_eq!(grid_step(10.0, 2.0, 2.0), 5);
    assert_eq!(grid_step(3.0, 2.0, 2.0), 2, "3/2 must not round down to 1");
    for step in [grid_step(8.0, 2.0, 2.0), grid_step(7.0, 2.0, 2.0)] {
        assert!(step as f32 >= 8.0 / 2.0 - 0.5);
    }
}

/// Short clips sample at their own rate so no information is dropped; long
/// clips at the requested rate, capped.
#[test]
fn short_clips_sample_at_source_rate_and_long_ones_at_the_cap() {
    // 1 s at 8 fps = 8 frames, under the 32 cap → all 8.
    assert_eq!(wanted_frames(1.0, 8.0, 2.0, 32, 2.0), 8);
    // 4 s at 8 fps = 32 frames, at the cap → 32 (and a 2 fps request is
    // allowed to be beaten, because dropping 24 source frames while the cap
    // can hold them would throw away motion for nothing).
    assert_eq!(wanted_frames(4.0, 8.0, 2.0, 32, 2.0), 32);
    // 300 s at 2 fps = 600, capped to 32.
    assert_eq!(wanted_frames(300.0, 8.0, 2.0, 32, 2.0), 32);
    // Degenerate rates fall back rather than divide by zero.
    for native in [0.0f32, f32::NAN, -1.0, f32::INFINITY] {
        assert!(wanted_frames(2.0, native, 0.0, 32, 2.0) >= 1);
    }
}

/// `(total_frames, fps, duration, max_frames) -> (count, first, last, sum)`
/// from the serving processor's `glm_sample_frame_indices` (fps_interval 2,
/// temporal pairs), run on the reference Python: dense clips re-spread by
/// linspace, short clips by floor positions, clips under half a second empty,
/// long clips at the 2048-frame cap, and the deployed 32-frame cap.
#[allow(clippy::type_complexity)]
const GLM_SAMPLER_PARITY: [(
    (usize, f64, f64, Option<usize>),
    (usize, &[usize], &[usize], usize),
); 16] = [
    (
        (1800, 30.0, 60.0, None),
        (120, &[0, 15, 30, 45, 60, 75], &[1768, 1783, 1799], 107884),
    ),
    (
        (300, 30.0, 10.0, None),
        (20, &[0, 15, 31, 47, 62, 78], &[267, 283, 299], 2981),
    ),
    ((15, 30.0, 0.5, None), (2, &[0, 0], &[0, 0], 0)),
    ((12, 30.0, 0.4, None), (0, &[], &[], 0)),
    (
        (40, 10.0, 4.0, None),
        (8, &[0, 5, 11, 16, 22, 27], &[27, 33, 39], 153),
    ),
    (
        (5, 1.0, 5.0, None),
        (6, &[0, 1, 2, 3, 4, 4], &[3, 4, 4], 14),
    ),
    (
        (7, 2.0, 3.5, None),
        (8, &[0, 1, 2, 3, 4, 5], &[5, 6, 6], 27),
    ),
    (
        (36000, 30.0, 1200.0, None),
        (
            2048,
            &[0, 17, 35, 52, 70, 87],
            &[35963, 35981, 35999],
            36861953,
        ),
    ),
    (
        (129600, 30.0, 4320.0, None),
        (
            2048,
            &[0, 63, 126, 189, 253, 316],
            &[129472, 129535, 129599],
            132708353,
        ),
    ),
    (
        (1798, 29.97002997002997, 59.99326666666667, None),
        (120, &[0, 15, 30, 45, 60, 76], &[1781, 1797, 1797], 108660),
    ),
    (
        (50, 25.0, 0.0, None),
        (6, &[0, 9, 19, 29, 39, 49], &[29, 39, 49], 145),
    ),
    (
        (240, 24.0, 10.0, None),
        (20, &[0, 12, 25, 37, 50, 62], &[213, 226, 239], 2381),
    ),
    (
        (9, 3.0, 3.0, None),
        (6, &[0, 1, 3, 4, 6, 8], &[4, 6, 8], 22),
    ),
    (
        (1800, 30.0, 60.0, Some(32)),
        (32, &[0, 58, 116, 174, 232, 290], &[1682, 1740, 1799], 28769),
    ),
    (
        (600, 60.0, 10.0, None),
        (20, &[0, 31, 63, 94, 126, 157], &[535, 567, 599], 5981),
    ),
    (
        (100, 7.5, 13.333333333333334, None),
        (26, &[0, 3, 7, 11, 15, 19], &[91, 95, 99], 1275),
    ),
];

#[test]
fn glm_frame_sampling_matches_the_reference_processor() {
    for ((total, fps, duration, cap), (count, first, last, sum)) in GLM_SAMPLER_PARITY {
        let got = glm_sample_frame_indices(total, fps, duration, None, cap, 2);
        let case = format!("{total} frames at {fps} fps over {duration} s, cap {cap:?}");
        assert_eq!(got.len(), count, "{case}");
        assert_eq!(&got[..first.len()], first, "{case}");
        assert_eq!(&got[got.len() - last.len()..], last, "{case}");
        assert_eq!(got.iter().sum::<usize>(), sum, "{case}");
    }
}
