// SPDX-License-Identifier: AGPL-3.0-only

//! Regression coverage for the GLM-5.3 OpenAI template override.

use serde_json::json;

fn render(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> String {
    render_flags(
        messages,
        tools,
        crate::tokenizer::chat_render::RenderFlags {
            enable_thinking,
            ..Default::default()
        },
    )
}

fn render_flags(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    flags: crate::tokenizer::chat_render::RenderFlags<'_>,
) -> String {
    let raw = include_str!("../../../../../jinja-templates/openai/glm5_next.jinja");
    let converted = crate::tokenizer::jinja_helpers::convert_python_jinja_to_minijinja(raw);
    let env = crate::tokenizer::jinja_helpers::build_jinja_env(&converted)
        .expect("GLM-5.3 OpenAI template compiles");
    crate::tokenizer::chat_render::render_chat(&env, messages, tools, flags)
        .expect("GLM-5.3 OpenAI template renders")
}

#[test]
fn glm5_tools_respect_resolved_thinking_before_generation() {
    let tools = [json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    })];
    let messages = [json!({"role": "user", "content": "What is the weather in Paris?"})];
    // The API resolver supplies false for silent clients under the GLM tool
    // default, but explicit enable wins that default (thinking.rs tests).
    for enabled in [false, true] {
        let rendered = render(&messages, Some(&tools), enabled);
        let suffix = if enabled {
            "<|assistant|><think>"
        } else {
            "<|assistant|><think></think>"
        };
        assert!(rendered.ends_with(suffix), "resolved thinking={enabled}");
        // An empty think block carries no effort header, as in the upstream
        // template's own thinking switch.
        assert_eq!(rendered.contains("Reasoning Effort: Max"), enabled);
        assert!(rendered.contains("<tools>"));
        assert!(rendered.contains("get_weather"));
    }
}

#[test]
fn glm5_without_tools_opens_think_even_when_thinking_is_off() {
    // GLM-5.3 always reasons; an empty `<think></think>` is out of
    // distribution (lower draft acceptance, degraded long outputs), so
    // thinking-off renders the model's own low-effort mode instead.
    let messages = [json!({"role": "user", "content": "What is the weather in Paris?"})];
    let off = render(&messages, None, false);
    assert!(off.ends_with("<|assistant|><think>"));
    assert!(off.contains("Reasoning Effort: Low"));
    let on = render(&messages, None, true);
    assert!(on.ends_with("<|assistant|><think>"));
    assert!(on.contains("Reasoning Effort: Max"));
}

#[test]
fn glm5_response_format_thinking_off_renders_the_empty_think_block() {
    // response_format requests set `empty_think_off`: thinking off then leaves
    // no reasoning and the grammar owns the first token. Thinking on is as-is.
    let messages = [json!({"role": "user", "content": "Name one bridge."})];
    let flags = |enable_thinking| crate::tokenizer::chat_render::RenderFlags {
        enable_thinking,
        empty_think_off: true,
        ..Default::default()
    };
    let off = render_flags(&messages, None, flags(false));
    assert!(off.ends_with("<|assistant|><think></think>"), "{off}");
    assert!(!off.contains("Reasoning Effort: Low"));
    assert_eq!(
        render_flags(&messages, None, flags(true)),
        render(&messages, None, true)
    );
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
    let rendered = render(&messages, None, false);
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
    let rendered = render(&messages, None, false);
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
        let rendered = render(&[json!({"role": "user", "content": content})], None, false);
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
