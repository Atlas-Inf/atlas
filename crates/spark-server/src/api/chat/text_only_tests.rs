// SPDX-License-Identifier: AGPL-3.0-only

//! Text-only media rejection (`reject_media_when_text_only`): image/video
//! parts on a model without a vision config are a 400 naming the model.

use super::prepare::reject_media_when_text_only;
use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Role, VideoSource};

fn msg(parts: Vec<ContentPart>) -> Message {
    Message {
        role: Role::User,
        content: parts,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

#[test]
fn text_only_rejects_image() {
    let m = msg(vec![
        ContentPart::Text("what is this".into()),
        ContentPart::Image(ImageSource {
            data: ImageData::Base64("AAA".into()),
        }),
    ]);
    let err = reject_media_when_text_only(true, "nemotron-3.5-super-vl", &[m]).unwrap_err();
    assert_eq!(err.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[test]
fn text_only_rejects_video() {
    let m = msg(vec![ContentPart::Video(VideoSource {
        data: ImageData::Url("https://x/v.mp4".into()),
    })]);
    assert!(reject_media_when_text_only(true, "m", &[m]).is_err());
}

#[test]
fn text_only_allows_text() {
    let m = msg(vec![ContentPart::Text("hi".into())]);
    assert!(reject_media_when_text_only(true, "m", &[m]).is_ok());
}

#[test]
fn vision_capable_allows_media() {
    let m = msg(vec![ContentPart::Image(ImageSource {
        data: ImageData::Base64("AAA".into()),
    })]);
    assert!(reject_media_when_text_only(false, "m", &[m]).is_ok());
}
