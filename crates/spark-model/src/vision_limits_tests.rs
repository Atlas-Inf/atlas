// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// qwen4_exp at the checkpoint bound: 2048x2048 = 16384 patches, merge 2.
const P_MAX: usize = 16_384;
const FULL: usize = 4096; // one 2048x2048 image, merged rows
/// `--max-prefill-tokens 16384 --max-num-seqs 4` → max_batch_tokens.
const CHUNK: usize = 16_388;

fn limits(request_rows: usize) -> VisionRowLimits {
    VisionRowLimits {
        window_rows: window_rows(P_MAX, 2, CHUNK),
        chunk_tokens: CHUNK,
        request_rows,
    }
}

#[test]
fn the_window_covers_a_chunk_plus_two_cut_images() {
    assert_eq!(window_rows(P_MAX, 2, CHUNK), CHUNK + 2 * FULL);
    // Never below the historical one-batch buffer, which every request that
    // fit before relied on.
    assert_eq!(
        window_rows(P_MAX, 2, 0),
        P_MAX,
        "unknown chunk keeps the size"
    );
    assert_eq!(window_rows(P_MAX, 2, 1024), P_MAX);
    assert_eq!(window_rows(6400, 2, 0), 6400);
}

#[test]
fn worst_chunk_rows_is_the_two_largest_plus_what_fits_between() {
    assert_eq!(worst_chunk_rows(&[], CHUNK), 0);
    assert_eq!(worst_chunk_rows(&[FULL], CHUNK), FULL, "one item: its rows");
    // Few items: the whole request can sit in one chunk.
    assert_eq!(worst_chunk_rows(&[FULL; 3], CHUNK), 3 * FULL);
    // The reporter's agent history, 16680 rows: e.g. four full-size
    // screenshots and a 256-row thumbnail.
    let report = [FULL, FULL, FULL, FULL, 296];
    assert_eq!(report.iter().sum::<usize>(), 16_680);
    assert_eq!(worst_chunk_rows(&report, CHUNK), 16_680);
    // Many: two cut at the ends plus a chunk's worth between.
    assert_eq!(worst_chunk_rows(&[FULL; 20], CHUNK), 2 * FULL + CHUNK - 2);
    // A tiny chunk still needs the two largest items whole.
    assert_eq!(worst_chunk_rows(&[10, FULL, 7, 3000], 2), FULL + 3000);
}

#[test]
fn every_still_image_request_within_the_request_limit_is_servable() {
    // The window is sized so the encoder never decides: any mix of still
    // images under the request limit passes the window check.
    let l = limits(usize::MAX);
    for n in [1, 2, 4, 5, 7, 16, 64, 1024] {
        assert_eq!(
            l.check(&vec![FULL; n], false),
            Ok(()),
            "{n} full-size images"
        );
        assert_eq!(
            l.check(&vec![1024; n], false),
            Ok(()),
            "{n} 1024x1024 images"
        );
    }
}

#[test]
fn a_request_over_the_old_one_batch_cap_is_accepted() {
    // Issue #180: 16680 rows used to fail at prefill start against 16384.
    let l = limits(32_768);
    assert_eq!(l.check(&[FULL, FULL, FULL, FULL, 296], false), Ok(()));
    assert_eq!(
        l.check(&[FULL; 8], false),
        Ok(()),
        "exactly the request limit"
    );
}

#[test]
fn the_request_limit_rejects_with_counts_and_the_limit() {
    let err = limits(16_384)
        .check(&[FULL, FULL, FULL, FULL, 296], false)
        .unwrap_err();
    assert!(err.contains("5 images"), "{err}");
    assert!(err.contains("16680"), "{err}");
    assert!(err.contains("16384"), "{err}");
}

#[test]
fn items_past_the_protocol_limit_are_rejected() {
    let l = limits(usize::MAX);
    assert_eq!(l.check(&vec![1; MAX_VISION_ITEMS], false), Ok(()));
    let err = l.check(&vec![1; MAX_VISION_ITEMS + 1], false).unwrap_err();
    assert!(err.contains("1025 images"), "{err}");
}

#[test]
fn long_videos_that_cannot_share_a_chunk_are_rejected() {
    // Videos are items of many temporal groups; two that large cut by one
    // chunk would overflow the window, so the request is refused up front.
    let l = limits(usize::MAX);
    let video = 12_000;
    assert_eq!(
        l.check(&[video], false),
        Ok(()),
        "one video fits on its own"
    );
    assert_eq!(
        l.check(&[video, 100], false),
        Ok(()),
        "the whole request fits one window"
    );
    let err = l.check(&[video, video, FULL], false).unwrap_err();
    assert!(err.contains("window"), "{err}");
    assert!(err.contains(&l.window_rows.to_string()), "{err}");
    // A single item larger than the window can never be encoded.
    assert!(l.check(&[l.window_rows + 1], false).is_err());
}

#[test]
fn a_request_that_fit_the_old_cap_still_fits() {
    // Any request of <= 16384 rows (the historical per-request bound) passes,
    // whatever its items: its worst chunk is at most its total.
    let l = limits(16_384);
    assert_eq!(l.check(&[16_384], false), Ok(()));
    assert_eq!(l.check(&[8000, 8000, 384], false), Ok(()));
}

#[test]
fn messages_carry_no_request_content() {
    // Counts and limits only: digits, ASCII words, punctuation.
    let err = limits(10).check(&[FULL], false).unwrap_err();
    assert!(err.is_ascii());
    assert!(err.starts_with("vision input too large"));
}

#[test]
fn the_request_limit_is_the_knob_capped_by_the_context() {
    let window = window_rows(P_MAX, 2, CHUNK);
    let l = VisionRowLimits::resolve(window, CHUNK, 32_768, 262_144, 0);
    assert_eq!(l, limits(32_768));
    assert_eq!(
        VisionRowLimits::resolve(window, CHUNK, 0, 262_144, 0).request_rows,
        262_144,
        "0 = the context alone"
    );
    assert_eq!(
        VisionRowLimits::resolve(window, CHUNK, 1 << 20, 8192, 0).request_rows,
        8192,
        "never past the context"
    );
}

#[test]
fn the_request_limit_never_exceeds_the_inflight_budget_or_the_engine_bound() {
    // A request larger than the in-flight budget could never acquire it: it
    // would be told to retry forever, so it is refused as too large instead.
    let window = window_rows(P_MAX, 2, CHUNK);
    assert_eq!(
        VisionRowLimits::resolve(window, CHUNK, 32_768, 262_144, 16_384).request_rows,
        16_384
    );
    // An engine that bounds a deferred request by its window (GLM) caps it.
    assert_eq!(
        VisionRowLimits::resolve(16_384, 8192, 32_768, 16_384, 65_536).request_rows,
        16_384
    );
}

#[test]
fn the_running_check_refuses_at_the_first_item_past_a_bound() {
    let l = limits(16_384);
    assert_eq!(l.check_running(0, 5, FULL, FULL, false), Ok(()));
    assert_eq!(l.check_running(3, 5, FULL, 4 * FULL, false), Ok(()));
    let err = l.check_running(4, 5, 296, 16_680, false).unwrap_err();
    assert!(err.contains("first 5 of 5 images"), "{err}");
    assert!(err.contains("16680") && err.contains("16384"), "{err}");
    let err = l
        .check_running(1, 3, l.window_rows + 1, l.window_rows + 2, true)
        .unwrap_err();
    assert!(err.contains("item 2 of 3 images or videos"), "{err}");
    assert!(err.contains("window"), "{err}");
}

#[test]
fn messages_say_videos_when_videos_are_present() {
    let err = limits(10).check(&[FULL, 8], true).unwrap_err();
    assert!(err.contains("2 images or videos"), "{err}");
    let err = limits(10).check(&[FULL], true).unwrap_err();
    assert!(err.contains("1 video"), "{err}");
}

#[test]
fn an_unset_request_limit_is_half_the_inflight_budget() {
    let window = window_rows(P_MAX, 2, CHUNK);
    // The defaults: 32768 per request, 65536 in flight (already half).
    assert_eq!(
        VisionRowLimits::resolve(window, CHUNK, 32_768, 262_144, 65_536).request_rows,
        32_768
    );
    // `--vision-max-request-rows 0`: half the budget, not all of it.
    assert_eq!(
        VisionRowLimits::resolve(window, CHUNK, 0, 262_144, 65_536).request_rows,
        32_768
    );
    assert_eq!(
        VisionRowLimits::resolve(window, CHUNK, 0, 262_144, 0).request_rows,
        262_144
    );
}
