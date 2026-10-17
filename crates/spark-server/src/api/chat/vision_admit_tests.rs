// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
<<<<<<< HEAD
use crate::ir::{
    MediaKind,
    message::{ContentPart, ImageData, ImageSource, Message, Role, VideoSource},
};
=======
use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Role};
>>>>>>> 7d33dd4cc (shared: admit vision items from their headers under the request limit and the in-flight budget)

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

<<<<<<< HEAD
fn video(uri: String) -> ContentPart {
    ContentPart::Video(VideoSource {
        data: ImageData::Base64(uri),
    })
}

=======
>>>>>>> 7d33dd4cc (shared: admit vision items from their headers under the request limit and the in-flight budget)
#[tokio::test]
async fn the_running_total_is_refused_with_counts_and_the_limit() {
    // The reporter's 16680-row history against a 16384-row request limit,
    // refused at the fifth item, before it is decoded.
    let l = limits(16_384);
    let mut a = Admission::begin(admit(&l, None), 5, false).unwrap();
    for _ in 0..4 {
<<<<<<< HEAD
        a.note(4096).unwrap();
    }
    let resp = a.note(296).err().unwrap();
=======
        a.charge(4096).unwrap();
    }
    let resp = a.charge(296).err().unwrap();
>>>>>>> 7d33dd4cc (shared: admit vision items from their headers under the request limit and the in-flight budget)
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
<<<<<<< HEAD
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
=======
fn the_worst_chunk_is_checked_once_every_item_is_charged() {
    let l = limits(usize::MAX);
    let mut a = Admission::begin(admit(&l, None), 3, true).unwrap();
    for rows in [12_000, 12_000, 4096] {
        a.charge(rows).unwrap();
    }
    assert_eq!(a.finish().err().unwrap().status(), StatusCode::BAD_REQUEST);
    let mut a = Admission::begin(admit(&l, None), 4, false).unwrap();
    for _ in 0..4 {
        a.charge(4096).unwrap();
    }
    assert!(a.finish().is_ok());
}

#[tokio::test]
async fn a_spent_budget_is_a_503_with_retry_after_and_permits_come_back() {
    let l = limits(usize::MAX);
    let budget = VisionRowBudget::new(8192);
    let mut a = Admission::begin(admit(&l, Some(&budget)), 3, false).unwrap();
    let first = a.charge(4096).unwrap();
    let second = a.charge(4096).unwrap();
    let resp = a.charge(1).err().unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(resp.headers()[RETRY_AFTER], RETRY_AFTER_SECS);
    let v = body(resp).await;
    assert_eq!(v["error"]["code"], VISION_BUSY_CODE);
    assert_eq!(v["error"]["type"], "server_error");
    // A charged item holds its rows until its last copy drops.
    let item = first.attach(VisionItem::image(vec![0.0; 4], 2, 2));
    let copy = item.clone();
    drop(second);
    drop(item);
    assert_eq!(budget.in_use(), 4096);
    drop(copy);
    assert_eq!(budget.in_use(), 0);
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// A ~60-byte PNG declaring `w x h` RGB with a truncated IDAT: the header
/// reads, any decode fails.
fn header_only_png(w: u32, h: u32) -> String {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        png.extend((data.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend(kind);
        png.extend(data);
        let crc = crc32(&png[start..]);
        png.extend(crc.to_be_bytes());
    };
    let mut ihdr = [w.to_be_bytes(), h.to_be_bytes()].concat();
    ihdr.extend([8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &[0x78, 0x9c]);
    chunk(b"IEND", &[]);
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png);
    format!("data:image/png;base64,{b64}")
}

fn qwen() -> atlas_core::config::VisionConfig {
    atlas_core::config::VisionConfig {
        is_glm5_next: false,
        in_channels: 3,
        depth: 27,
        hidden_size: 1152,
        num_heads: 16,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        intermediate_size: 4304,
        out_hidden_size: 2560,
        projection_intermediate_size: 0,
        rms_norm_eps: 1e-6,
        swiglu_limit: 0.0,
        deepstack_visual_indexes: vec![],
        image_pad_token_id: 248_056,
        video_pad_token_id: 248_057,
        image_start_token_id: 0,
        image_end_token_id: 0,
        video_start_token_id: 0,
        video_end_token_id: 0,
        max_pixels: Some(16_777_216),
    }
}

#[allow(clippy::result_large_err)]
fn build(uris: &[String], ctx: AdmitCtx<'_>) -> Result<(), Response> {
    let msg = Message {
        role: Role::User,
        content: uris
            .iter()
            .map(|u| {
                ContentPart::Image(ImageSource {
                    data: ImageData::Base64(u.clone()),
                })
            })
            .chain([ContentPart::Text("describe".into())])
            .collect(),
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    };
    super::super::msg_entry::build_msg_entries_admitted(
        Some(&qwen()),
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
    .map(drop)
}

#[tokio::test]
async fn a_huge_declared_image_is_refused_from_its_header_without_decoding() {
    // M1: a few-byte PNG declaring 16000x12000 would decode to hundreds of
    // MiB of f32. Sized from its header (~4070 vision tokens at the encoder's
    // 2048^2 cap), it is refused as too large; had it been decoded, the
    // truncated data would have failed as a decode error instead.
    let l = limits(1000);
    let budget = VisionRowBudget::new(65_536);
    let t = std::time::Instant::now();
    let resp = build(&[header_only_png(16_000, 12_000)], admit(&l, Some(&budget)))
        .err()
        .unwrap();
    assert!(t.elapsed().as_millis() < 500, "{:?}", t.elapsed());
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v = body(resp).await;
    assert_eq!(v["error"]["code"], VISION_TOO_LARGE_CODE, "{v}");
    assert_eq!(budget.in_use(), 0, "nothing is held after a refusal");
}

#[tokio::test]
async fn a_request_the_budget_cannot_hold_is_deferred_and_holds_nothing() {
    // A charged item that then fails to decode returns its rows; an item the
    // budget cannot hold is deferred (503) before it is decoded.
    let l = limits(usize::MAX);
    let budget = VisionRowBudget::new(6000);
    let resp = build(&[header_only_png(16_000, 12_000)], admit(&l, Some(&budget)))
        .err()
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "decode error");
    assert_eq!(budget.in_use(), 0, "the charged rows came back");
    let held = budget.try_acquire(3000).unwrap();
    let resp = build(&[header_only_png(16_000, 12_000)], admit(&l, Some(&budget)))
        .err()
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    drop(held);
    assert_eq!(budget.in_use(), 0);
>>>>>>> 7d33dd4cc (shared: admit vision items from their headers under the request limit and the in-flight budget)
}
