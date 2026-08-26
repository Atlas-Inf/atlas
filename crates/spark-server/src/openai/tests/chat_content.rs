// SPDX-License-Identifier: AGPL-3.0-only

//! Chat-completions message-content parsing: audio/video content parts
//! (base64 data URIs and URLs) alongside text and images. Wave 1 of the
//! multimodal bring-up — IR + wire-format layer only.

use crate::ir::message::{ContentPart, ImageData, ImageSource};
use crate::ir::{MediaData, MediaSource, VideoSource};
use crate::openai::*;

fn parse_message(content: serde_json::Value) -> IncomingMessage {
    serde_json::from_value(serde_json::json!({
        "role": "user",
        "content": content,
    }))
    .expect("message parses")
}

#[test]
fn audio_and_video_parts_parse_into_parsed_content() {
    let m = parse_message(serde_json::json!([
        {"type": "text", "text": "see"},
        {"type": "audio", "audio": {"data": "data:audio/wav;base64,QUJD"}},
        {"type": "video", "video": {"data": "data:video/mp4;base64,REVG"}},
    ]));
    assert_eq!(m.content.text, "see");
    assert!(m.content.images.is_empty());
    assert_eq!(
        m.content.audios,
        vec!["data:audio/wav;base64,QUJD".to_string()]
    );
    assert_eq!(
        m.content.videos,
        vec!["data:video/mp4;base64,REVG".to_string()]
    );
}

#[test]
fn audio_url_and_input_video_forms_parse() {
    let m = parse_message(serde_json::json!([
        {"type": "audio", "audio": {"url": "https://example.com/a.wav"}},
        {"type": "input_video", "video": {"data": "data:video/webm;base64,R0lG"}},
    ]));
    assert_eq!(
        m.content.audios,
        vec!["https://example.com/a.wav".to_string()]
    );
    assert_eq!(
        m.content.videos,
        vec!["data:video/webm;base64,R0lG".to_string()]
    );
}

#[test]
fn raw_base64_with_format_becomes_data_uri() {
    let m = parse_message(serde_json::json!([
        {"type": "input_audio", "audio": {"data": "QUJD", "format": "wav"}},
        {"type": "video", "video": {"data": "REVG", "format": "video/mp4"}},
    ]));
    assert_eq!(
        m.content.audios,
        vec!["data:audio/wav;base64,QUJD".to_string()]
    );
    assert_eq!(
        m.content.videos,
        vec!["data:video/mp4;base64,REVG".to_string()]
    );
}

#[test]
fn media_lowers_to_ir_parts_in_canonical_order() {
    let m = parse_message(serde_json::json!([
        {"type": "text", "text": "see"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUFB"}},
        {"type": "audio", "audio": {"data": "data:audio/wav;base64,QUJD"}},
        {"type": "video", "video": {"data": "data:video/mp4;base64,REVG"}},
    ]));
    let ir: crate::ir::Message = (&m).into();
    // Canonical part order: [image*, audio*, video*, text] — the
    // [image*, text] convention extended with the new modalities.
    assert_eq!(
        ir.content,
        vec![
            ContentPart::Image(ImageSource {
                data: ImageData::Base64("data:image/png;base64,QUFB".into()),
            }),
            ContentPart::Audio(MediaSource {
                data: MediaData::Base64("data:audio/wav;base64,QUJD".into()),
            }),
            ContentPart::Video(VideoSource {
                frames: vec![MediaData::Base64("data:video/mp4;base64,REVG".into())],
            }),
            ContentPart::Text("see".into()),
        ]
    );
}

#[test]
fn remote_url_media_classified_as_url_variant() {
    let m = parse_message(serde_json::json!([
        {"type": "audio", "audio": {"url": "https://example.com/a.wav"}},
    ]));
    let ir: crate::ir::Message = (&m).into();
    assert_eq!(
        ir.content,
        vec![ContentPart::Audio(MediaSource {
            data: MediaData::Url("https://example.com/a.wav".into()),
        })]
    );
}

#[test]
fn video_frames_array_parses_into_flat_videos_with_clip_counts() {
    // Wave 5A: a video clip is an ARRAY of base64 image frames (no ffmpeg
    // — the caller supplies the frames). All frame URIs land in `videos`
    // (flat, clip order); `video_clip_frame_counts` tracks frames-per-clip.
    let m = parse_message(serde_json::json!([
        {"type": "video", "video": {"frames": [
            "data:image/png;base64,QUFB",
            "data:image/png;base64,QUJD"
        ]}},
    ]));
    assert_eq!(
        m.content.videos,
        vec![
            "data:image/png;base64,QUFB".to_string(),
            "data:image/png;base64,QUJD".to_string()
        ]
    );
    assert_eq!(m.content.video_clip_frame_counts, vec![2]);
}

#[test]
fn multiple_video_clips_track_separate_frame_counts() {
    let m = parse_message(serde_json::json!([
        {"type": "video", "video": {"frames": ["data:image/png;base64,QUFB"]}},
        {"type": "video", "video": {"frames": [
            "data:image/png;base64,QUJD",
            "data:image/png;base64,QUFF"
        ]}},
    ]));
    assert_eq!(m.content.videos.len(), 3);
    assert_eq!(m.content.video_clip_frame_counts, vec![1, 2]);
}

#[test]
fn single_uri_video_is_a_one_frame_clip() {
    // Wave-1 legacy shape (`data`/`url`/`format`) still parses — as a
    // 1-frame clip, so the clip-boundary bookkeeping is uniform.
    let m = parse_message(serde_json::json!([
        {"type": "video", "video": {"data": "data:video/mp4;base64,REVG"}},
    ]));
    assert_eq!(
        m.content.videos,
        vec!["data:video/mp4;base64,REVG".to_string()]
    );
    assert_eq!(m.content.video_clip_frame_counts, vec![1]);
}

#[test]
fn video_frames_lower_to_one_ir_part_per_clip() {
    // One ContentPart::Video per CLIP (not per frame): the template emits
    // one `{"type":"video"}` marker per clip and the expander generates
    // `video_frames` (32) frames internally — per-frame markers would
    // multiply the expansion 32x.
    let m = parse_message(serde_json::json!([
        {"type": "video", "video": {"frames": [
            "data:image/png;base64,QUFB",
            "data:image/png;base64,QUJD"
        ]}},
    ]));
    let ir: crate::ir::Message = (&m).into();
    assert_eq!(ir.video_count(), 1, "one part per clip, not per frame");
    match &ir.content[0] {
        ContentPart::Video(src) => assert_eq!(src.frames.len(), 2),
        other => panic!("expected one Video part, got {other:?}"),
    }
}

#[test]
fn text_and_image_parts_unchanged() {
    // Regression: the image/text wire shape must parse exactly as before.
    let m = parse_message(serde_json::json!([
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]));
    assert_eq!(m.content.text, "what is this?");
    assert_eq!(
        m.content.images,
        vec!["data:image/png;base64,AAA".to_string()]
    );
    assert!(m.content.audios.is_empty());
    assert!(m.content.videos.is_empty());
    let ir: crate::ir::Message = (&m).into();
    assert_eq!(ir.image_count(), 1);
    assert_eq!(ir.text(), "what is this?");
}
