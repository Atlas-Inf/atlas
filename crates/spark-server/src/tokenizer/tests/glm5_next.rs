// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 media markers through the checkpoint's own template. Prompt parity
//! with the reference engine for every request form is pinned end to end in
//! `api/chat/glm_prompt_parity_tests.rs`.

use serde_json::json;

fn render(messages: &[serde_json::Value]) -> String {
    super::qwen_dense::render_fixture(
        "glm-5.3-flash-fp4-local",
        messages,
        None,
        crate::tokenizer::chat_render::RenderFlags {
            enable_thinking: true,
            ..Default::default()
        },
    )
    .expect("GLM-5.3 checkpoint template renders")
}

#[test]
fn glm5_rendered_prompt_keeps_two_image_markers_in_order() {
    let messages = [json!({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "compare"},
            {"type": "image"}
        ]
    })];
    let rendered = render(&messages);
    let marker = "<|begin_of_image|><|image|><|end_of_image|>";
    assert_eq!(rendered.matches(marker).count(), 2, "{rendered}");
    assert!(!rendered.contains("<|vision_start|>"));
    assert!(!rendered.contains("<|image_pad|>"));
}

#[test]
fn glm5_multiturn_media_markers_are_not_grouped_or_dropped() {
    let messages = [
        json!({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "first"}]}),
        json!({"role": "assistant", "content": "noted"}),
        json!({"role": "user", "content": [{"type": "text", "text": "second"}, {"type": "image"}]}),
    ];
    let rendered = render(&messages);
    let marker = "<|begin_of_image|><|image|><|end_of_image|>";
    assert_eq!(rendered.matches(marker).count(), 2, "{rendered}");
    let first = rendered.find(marker).expect("first image marker");
    let second = rendered[first + marker.len()..]
        .find(marker)
        .map(|offset| first + marker.len() + offset)
        .expect("second image marker");
    assert!(first < second);
}

#[test]
fn glm5_video_template_emits_one_canonical_compact_marker() {
    // Temporal grouping and odd-frame repair happen in the processor; the
    // shipped chat template must contribute exactly one video triple for both
    // a two-frame clip and a clip whose sampled frames are repaired to a pair.
    for content in [
        json!([{"type": "video"}, {"type": "text", "text": "two frames"}]),
        json!([{"type": "video"}, {"type": "text", "text": "odd source"}]),
    ] {
        let rendered = render(&[json!({"role": "user", "content": content})]);
        assert_eq!(
            rendered
                .matches("<|begin_of_video|><|video|><|end_of_video|>")
                .count(),
            1,
            "{rendered}"
        );
        assert!(!rendered.contains("<|vision_start|>"));
    }
}
