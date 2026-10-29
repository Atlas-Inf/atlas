// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::ir::{
    MediaKind,
    message::{ContentPart, ImageData, ImageSource, Message, Role, VideoSource},
};

fn limits(request_rows: usize) -> VisionRowLimits {
    VisionRowLimits::resolve(
        spark_model::vision_limits::window_rows(16_384, 2, 16_388),
        16_388,
        request_rows,
        262_144,
        0,
    )
}

async fn body(resp: Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn admit<'a>(l: &'a VisionRowLimits, b: Option<&'a Arc<VisionRowBudget>>) -> AdmitCtx<'a> {
    AdmitCtx {
        limits: Some(l),
        budget: b,
    }
}

fn video(uri: String) -> ContentPart {
    ContentPart::Video(VideoSource {
        data: ImageData::Base64(uri),
    })
}

#[tokio::test]
async fn the_running_total_is_refused_with_counts_and_the_limit() {
    // The reporter's 16680-row history against a 16384-row request limit,
    // refused at the fifth item, before it is decoded.
    let l = limits(16_384);
    let mut a = Admission::begin(admit(&l, None), 5, false).unwrap();
    for _ in 0..4 {
        a.note(4096).unwrap();
    }
    let resp = a.note(296).err().unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body(resp).await;
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["code"], VISION_TOO_LARGE_CODE);
    assert_eq!(v["error"]["param"], "messages");
    let msg = v["error"]["message"].as_str().unwrap();
    for part in ["5 images", "16680", "16384"] {
        assert!(msg.contains(part), "{part} missing from {msg}");
    }
}

#[test]
fn too_many_items_are_refused_before_any_is_read() {
    let l = limits(usize::MAX);
    assert!(Admission::begin(admit(&l, None), MAX_VISION_ITEMS, false).is_ok());
    let resp = Admission::begin(admit(&l, None), MAX_VISION_ITEMS + 1, true)
        .err()
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn the_worst_chunk_is_checked_once_every_item_is_sized() {
    let l = limits(usize::MAX);
    let mut a = Admission::begin(admit(&l, None), 3, true).unwrap();
    for rows in [12_000, 12_000, 4096] {
        a.note(rows).unwrap();
    }
    assert_eq!(a.admit().err().unwrap().status(), StatusCode::BAD_REQUEST);
}

#[test]
fn a_video_reservation_is_reconciled_into_the_shared_request_permit() {
    let l = limits(usize::MAX);
    let budget = VisionRowBudget::new(1_000);
    let mut a = Admission::begin(admit(&l, Some(&budget)), 1, true).unwrap();

    // The decoder needs a conservative 800-row reservation, but the actual
    // clip produces only 250 rows. The same permit is shrunk before it is
    // attached to the item; no drop/reacquire window exists.
    a.reserve_video(800).unwrap();
    assert_eq!(budget.in_use(), 800);
    a.note(250).unwrap();
    let admitted = a.admit().unwrap();
    assert_eq!(budget.in_use(), 250);
    let item = admitted.attach(VisionItem::image(vec![0.0; 4], 2, 2));
    drop(admitted);
    assert_eq!(budget.in_use(), 250);
    drop(item);
    assert_eq!(budget.in_use(), 0);
}

#[test]
fn sequential_short_videos_release_their_unused_ceiling_before_the_next_decode() {
    let l = limits(usize::MAX);
    let budget = VisionRowBudget::new(1_000);
    let mut a = Admission::begin(admit(&l, Some(&budget)), 2, true).unwrap();

    a.reserve_video(800).unwrap();
    a.reconcile_video(50).unwrap();
    assert_eq!(budget.in_use(), 50);
    a.note(50).unwrap();

    // A second short clip can reserve the same ceiling because the first
    // clip's unused 750 rows were returned before this decode began.
    a.reserve_video(800).unwrap();
    assert_eq!(budget.in_use(), 850);
    a.reconcile_video(100).unwrap();
    assert_eq!(budget.in_use(), 100);
    a.note(50).unwrap();
    let admitted = a.admit().unwrap();
    assert_eq!(budget.in_use(), 100);
    drop(admitted);
    assert_eq!(budget.in_use(), 0);
}

#[test]
fn a_video_reservation_refuses_before_decode_when_budget_is_spent() {
    let l = limits(usize::MAX);
    let budget = VisionRowBudget::new(600);
    let held = budget.try_acquire(300).unwrap();
    let mut a = Admission::begin(admit(&l, Some(&budget)), 1, true).unwrap();
    assert_eq!(
        a.reserve_video(400).unwrap_err().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        budget.in_use(),
        300,
        "the failed reservation allocated nothing"
    );
    drop(held);
}

#[test]
fn a_video_reservation_above_the_request_bound_is_a_bad_request_before_decode() {
    let l = limits(600);
    let budget = VisionRowBudget::new(1_000);
    let mut a = Admission::begin(admit(&l, Some(&budget)), 1, true).unwrap();
    assert_eq!(
        a.reserve_video(601).unwrap_err().status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(budget.in_use(), 0);
}

#[tokio::test]
async fn the_budget_is_taken_for_the_whole_request_or_not_at_all() {
    // Atomic: a request that does not fit holds nothing, so two large
    // requests cannot each hold part and both fail, and the 503 says
    // nothing about what other requests hold.
    let l = limits(usize::MAX);
    let budget = VisionRowBudget::new(8192);
    let mut a = Admission::begin(admit(&l, Some(&budget)), 3, false).unwrap();
    for rows in [4096, 4096, 1] {
        a.note(rows).unwrap();
    }
    let resp = a.admit().err().unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(resp.headers()[RETRY_AFTER], RETRY_AFTER_SECS);
    assert_eq!(budget.in_use(), 0, "a deferred request holds nothing");
    let v = body(resp).await;
    assert_eq!(v["error"]["code"], VISION_BUSY_CODE);
    assert_eq!(v["error"]["type"], "server_error");
    assert_eq!(
        v["error"]["message"],
        "vision capacity is busy; retry shortly"
    );

    // One permit, shared by the request's items, back on the last drop.
    let mut a = Admission::begin(admit(&l, Some(&budget)), 2, false).unwrap();
    a.note(4096).unwrap();
    a.note(4096).unwrap();
    let admitted = a.admit().unwrap();
    let x = admitted.attach(VisionItem::image(vec![0.0; 4], 2, 2));
    let y = admitted.attach(VisionItem::image(vec![0.0; 4], 2, 2));
    drop(admitted);
    assert_eq!(budget.in_use(), 8192);
    drop(x);
    assert_eq!(budget.in_use(), 8192, "y still holds the request's rows");
    drop(y);
    assert_eq!(budget.in_use(), 0);
}

#[test]
fn a_request_that_can_never_fit_is_a_400_even_when_the_budget_is_spent() {
    let l = limits(16_384);
    let budget = VisionRowBudget::new(65_536);
    let _held = budget.try_acquire(65_536).unwrap();
    let mut a = Admission::begin(admit(&l, Some(&budget)), 5, false).unwrap();
    for _ in 0..4 {
        a.note(4096).unwrap();
    }
    assert_eq!(a.note(296).err().unwrap().status(), StatusCode::BAD_REQUEST);
}
