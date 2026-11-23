// SPDX-License-Identifier: AGPL-3.0-only

use super::build_json_messages;
use super::{MsgContentPart, MsgEntry};

fn entry(role: &str, content: &str, image_count: usize) -> MsgEntry {
    let mut parts = Vec::new();
    for _ in 0..image_count {
        parts.push(MsgContentPart::Media(crate::ir::MediaKind::Image));
    }
    if !content.is_empty() {
        parts.push(MsgContentPart::Text(content.to_string()));
    }
    MsgEntry {
        role: role.to_string(),
        parts,
        tool_calls: None,
        tool_call_id: None,
        reasoning_content: None,
    }
}

/// PROMPT-STABILITY GATE. Characterization golden for the full
/// `Vec<ir::Message>` → `build_msg_entries` → `build_json_messages`
/// path — the exact JSON the Jinja chat template consumes. Any
/// change here changes rendered prompts, which breaks kv-cache
/// prefix reuse across requests of the same conversation. Update
/// ONLY for an intentional, documented behavior change.
#[test]
fn prompt_json_stability_gate() {
    use crate::ir::message::{Reasoning, ToolCall};
    use crate::ir::{ContentPart, Message, Role};

    fn text_msg(role: Role, t: &str) -> Message {
        Message {
            role,
            content: vec![ContentPart::Text(t.into())],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        }
    }

    let mut assistant = text_msg(Role::Assistant, "Sure.");
    assistant.reasoning = Some(Reasoning {
        text: "plan the verification".into(),
    });
    assistant.tool_calls = vec![
        ToolCall {
            id: "c1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "cargo test"}),
        },
        ToolCall {
            id: "c2".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path": "a.rs"}),
        },
    ];
    let mut tool_result = text_msg(Role::Tool, "total 0");
    tool_result.tool_call_id = Some("c1".into());
    tool_result.name = Some("bash".into());

    let msgs = vec![
        text_msg(
            Role::System,
            "You are helpful.\nworking directory: /tmp/proj",
        ),
        text_msg(Role::Other("developer".into()), "be terse"),
        text_msg(Role::User, "run the tests"),
        assistant,
        tool_result,
        text_msg(Role::User, "thanks"),
    ];

    let out = super::super::msg_entry::build_msg_entries_admitted(
        None,
        None,
        &crate::api::chat::remote_image::RemoteImagePolicy::default(),
        &crate::api::chat::msg_entry::VideoDecode {
            ffmpeg: &spark_model::video_decode_ffmpeg::FfmpegPolicy {
                enabled: false,
                ..Default::default()
            },
            fps: 2.0,
        },
        &msgs,
        true,
        &crate::api::chat::levers::ChatLevers::OFF,
        false,
        crate::api::chat::vision_admit::AdmitCtx::default(),
    )
    .expect("fixture builds");
    assert_eq!(out.cwd_hint.as_deref(), Some("/tmp/proj"));
    let json = build_json_messages(&out.messages);

    let expected = serde_json::json!([
        {
            "role": "system",
            "content": "You are helpful.\nworking directory: /tmp/proj\n<environment>\nworking_directory: /tmp/proj\n</environment>"
        },
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "run the tests"},
        {
            "role": "assistant",
            "content": "Sure.",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": {"command": "cargo test"}}},
                {"id": "c2", "type": "function", "function": {"name": "read", "arguments": {"path": "a.rs"}}}
            ],
            "reasoning_content": "plan the verification"
        },
        {"role": "tool", "content": "total 0"},
        {"role": "user", "content": "thanks"}
    ]);
    assert_eq!(
        serde_json::Value::Array(json.clone()),
        expected,
        "prompt JSON drifted — this breaks kv-cache prefix stability:\n{}",
        serde_json::to_string_pretty(&json).unwrap()
    );
}

#[test]
fn plain_text_message_serializes_to_string_content() {
    let out = build_json_messages(&[entry("user", "hi", 0)]);
    assert_eq!(
        out,
        vec![serde_json::json!({"role": "user", "content": "hi"})]
    );
}

#[test]
fn images_expand_to_structured_content_array_with_text_last() {
    let out = build_json_messages(&[entry("user", "look", 2)]);
    assert_eq!(
        out,
        vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "look"}
            ]
        })]
    );
}

#[test]
fn empty_text_with_images_omits_text_part() {
    let out = build_json_messages(&[entry("user", "", 1)]);
    assert_eq!(
        out,
        vec![serde_json::json!({"role": "user", "content": [{"type": "image"}]})]
    );
}

/// Anthropic can send `[text, image, text]`; the renderer must keep that
/// interleaving instead of grouping every media marker before the prose.
#[test]
fn text_and_media_parts_render_in_content_order() {
    let mut e = entry("user", "", 0);
    e.parts = vec![
        MsgContentPart::Text("before".to_string()),
        MsgContentPart::Media(crate::ir::MediaKind::Image),
        MsgContentPart::Text("after".to_string()),
    ];
    let out = build_json_messages(&[e]);
    assert_eq!(
        out,
        vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "after"}
            ]
        })]
    );
}

/// ★ The reordering regression, at the rendering end. The builder used
/// to emit `image_count` image markers and then `video_count` video
/// markers, so a client that sent a video first had its still
/// described instead — silently, since the marker count, the pad runs
/// and the encoder rows all still matched.
///
/// The bundled template walks this array left to right and numbers its
/// `Picture N:` / `Video N:` labels from it, so the array order IS the
/// order the model reads.
#[test]
fn media_markers_render_in_the_clients_order() {
    use crate::ir::MediaKind;

    let mut e = entry("user", "which came first?", 0);
    e.parts = vec![
        MsgContentPart::Media(MediaKind::Video),
        MsgContentPart::Media(MediaKind::Image),
        MsgContentPart::Media(MediaKind::Video),
        MsgContentPart::Text("which came first?".to_string()),
    ];
    let out = build_json_messages(&[e]);
    assert_eq!(
        out,
        vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "image"},
                {"type": "video"},
                {"type": "text", "text": "which came first?"}
            ]
        })]
    );
}

#[test]
fn tool_calls_and_reasoning_are_attached() {
    let mut e = entry("assistant", "", 0);
    e.tool_calls = Some(vec![serde_json::json!({"id": "c1"})]);
    e.reasoning_content = Some("because".to_string());
    let out = build_json_messages(&[e]);
    assert_eq!(
        out,
        vec![serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1"}],
            "reasoning_content": "because"
        })]
    );
}
