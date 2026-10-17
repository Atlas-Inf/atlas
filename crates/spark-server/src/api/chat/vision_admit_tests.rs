// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Role};
use spark_model::vision_preprocess::fixtures::{header_only_png, png, qwen_config};

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

#[tokio::test]
async fn the_item_count_is_checked_before_any_url_is_fetched() {
    // 1025 image URLs: refused for their count, before any is resolved
    // (here fetching is off, which would have answered "not fetched").
    let l = limits(usize::MAX);
    let parts: Vec<ContentPart> = (0..=MAX_VISION_ITEMS)
        .map(|i| {
            ContentPart::Image(ImageSource {
                data: ImageData::Url(format!("http://127.0.0.1/{i}.png")),
            })
        })
        .collect();
    let resp = build_parts(parts, admit(&l, None)).err().unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert_eq!(body(resp).await["error"]["code"], VISION_TOO_LARGE_CODE);
}

#[allow(clippy::result_large_err)]
fn build(
    uris: &[String],
    ctx: AdmitCtx<'_>,
) -> Result<super::super::msg_entry::BuildOut, Response> {
    let parts = uris
        .iter()
        .map(|u| {
            ContentPart::Image(ImageSource {
                data: ImageData::Base64(u.clone()),
            })
        })
        .collect();
    build_parts(parts, ctx)
}

#[allow(clippy::result_large_err)]
fn build_parts(
    mut parts: Vec<ContentPart>,
    ctx: AdmitCtx<'_>,
) -> Result<super::super::msg_entry::BuildOut, Response> {
    parts.push(ContentPart::Text("describe".into()));
    let msg = Message {
        role: Role::User,
        content: parts,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    };
    let mut vcfg = qwen_config();
    vcfg.max_pixels = Some(16_777_216);
    super::super::msg_entry::build_msg_entries_admitted(
        Some(&vcfg),
        Some(16_777_216),
        &crate::api::chat::remote_image::RemoteImagePolicy::default(),
        &crate::api::chat::msg_entry::VideoDecode {
            ffmpeg: &spark_model::video_decode_ffmpeg::FfmpegPolicy {
                enabled: false,
                ..Default::default()
            },
            fps: 2.0,
        },
        &[msg],
        false,
        &crate::api::chat::levers::ChatLevers::OFF,
        false,
        ctx,
    )
}

#[tokio::test]
async fn a_huge_declared_image_is_refused_from_its_header_without_decoding() {
    // M1: a few-byte PNG declaring 8000x8000 would decode to hundreds of
    // MiB of f32. Sized from its header (4096 vision tokens at the encoder's
    // 2048^2 capacity), it is refused as too large; decoded, its truncated
    // data would have failed as a decode error instead.
    let l = limits(1000);
    let budget = VisionRowBudget::new(65_536);
    let t = std::time::Instant::now();
    let resp = build(&[header_only_png(8000, 8000)], admit(&l, Some(&budget)))
        .err()
        .unwrap();
    assert!(t.elapsed().as_millis() < 500, "{:?}", t.elapsed());
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body(resp).await;
    assert_eq!(v["error"]["code"], VISION_TOO_LARGE_CODE, "{v}");
    assert_eq!(budget.in_use(), 0, "nothing is held after a refusal");
}

#[tokio::test]
async fn a_spent_budget_defers_the_request_before_any_decode() {
    // The header-only image would fail to decode: a 503 proves it was not.
    let l = limits(usize::MAX);
    let budget = VisionRowBudget::new(6000);
    let held = budget.try_acquire(3000).unwrap();
    let resp = build(&[header_only_png(8000, 8000)], admit(&l, Some(&budget)))
        .err()
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    drop(held);
    // Admitted, its decode fails: the request's rows come back with it.
    let resp = build(&[header_only_png(8000, 8000)], admit(&l, Some(&budget)))
        .err()
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "decode error");
    assert_eq!(budget.in_use(), 0);
}

#[test]
fn an_admitted_request_holds_its_rows_until_its_items_drop() {
    // The items travel to the scheduler and are dropped when the prefill
    // ends (`scheduler::vision_release`); here, when the build output drops.
    let l = limits(usize::MAX);
    let budget = VisionRowBudget::new(65_536);
    let out = build(&[png(640, 480), png(320, 320)], admit(&l, Some(&budget))).unwrap();
    let rows: usize = out.image_pad_counts.iter().sum();
    assert_eq!(rows, 300 + 100);
    assert_eq!(budget.in_use(), rows);
    let items = out.image_pixels;
    assert_eq!(budget.in_use(), rows);
    drop(items);
    assert_eq!(budget.in_use(), 0);
}
