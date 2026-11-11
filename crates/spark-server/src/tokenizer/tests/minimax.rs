// SPDX-License-Identifier: AGPL-3.0-only

//! MiniMax Jinja-template render tests. Split from `tests.rs` (500-LoC cap).

use super::*;

fn render_minimax_openai_template(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> String {
    let template_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/openai/minimax_m2.jinja"
    );
    let raw = std::fs::read_to_string(template_path)
        .expect("bundled MiniMax OpenAI template must be present in the repo");
    let converted = super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages_for_render = normalize_tool_call_arguments(messages);
    let messages_val = minijinja::Value::from_serialize(&messages_for_render);
    let tools_val = tools.map(minijinja::Value::from_serialize);
    let reasoning_effort: minijinja::Value = if enable_thinking {
        "high".into()
    } else {
        "none".into()
    };
    let ctx = minijinja::context! {
        messages => messages_val,
        tools => tools_val.unwrap_or(minijinja::Value::UNDEFINED),
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        reasoning_effort => reasoning_effort,
        disable_tool_steering => false,
        add_vision_id => false,
    };
    tmpl.render(ctx).expect("template renders")
}

/// F76 integration: render the actual MiniMax M2.7 chat template
/// with a second-turn shape (assistant has tool_calls with string
/// args). Without F76 this errors with `unknown method: map has
/// no method named items` on line 112.
#[test]
fn render_minimax_template_with_string_tool_call_args() {
    let template_path = "/workspace/.cache/huggingface/hub/models--lukealonso--MiniMax-M2.7-NVFP4/snapshots/ba6a625013cdacdc560f6203d177c0f27d41775e/chat_template.jinja";
    let Ok(template) = std::fs::read_to_string(template_path) else {
        eprintln!("MiniMax template not on disk; skipping");
        return;
    };
    let env = super::jinja_helpers::build_jinja_env(&template).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    // The exact wire shape opencode sends back on turn 2.
    let messages = vec![
        json!({"role": "user", "content": "List /tmp"}),
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": "{\"command\":\"ls -la /tmp\"}"
                }
            }]
        }),
        json!({"role": "tool", "tool_call_id": "call_0", "content": "total 0"}),
        json!({"role": "user", "content": "Now uname -r"}),
    ];
    let normalized = normalize_tool_call_arguments(&messages);
    let messages_val = minijinja::Value::from_serialize(&normalized);
    let ctx = minijinja::context! {
        messages => messages_val,
        tools => minijinja::Value::UNDEFINED,
        add_generation_prompt => true,
        enable_thinking => true,
        reasoning_effort => "high",
        disable_tool_steering => false,
        add_vision_id => false,
    };
    let rendered = tmpl
        .render(ctx)
        .expect("F76 must keep MiniMax template from raising on second-turn");
    // Sanity check: rendered output should contain the bash invoke
    // with command parameter — the items() iteration produced output.
    assert!(
        rendered.contains("<invoke name=\"bash\">"),
        "expected `<invoke name=\"bash\">` in render: {rendered}"
    );
    assert!(
        rendered.contains("<parameter name=\"command\">"),
        "expected `<parameter name=\"command\">` from .items() iteration: {rendered}"
    );
    assert!(
        rendered.contains("ls -la /tmp"),
        "expected the parsed command value in render: {rendered}"
    );
}

#[test]
fn render_minimax_openai_template_closes_think_prompt_when_disabled() {
    let messages = vec![json!({"role": "user", "content": "Reply with exactly: OK"})];
    let rendered = render_minimax_openai_template(&messages, None, false);
    assert!(
        rendered.ends_with("]~b]ai\n<think>\n\n</think>\n\n"),
        "expected closed-thinking assistant generation prompt: {rendered}"
    );
    let generation_tail = rendered
        .rsplit_once("]~b]ai\n")
        .map(|(_, tail)| tail)
        .expect("assistant generation prompt is present");
    assert_eq!(
        generation_tail, "<think>\n\n</think>\n\n",
        "disabled thinking must not leave the model inside <think>: {rendered}"
    );
}

#[test]
fn render_minimax_openai_template_opens_think_prompt_when_enabled() {
    let messages = vec![json!({"role": "user", "content": "Think before answering"})];
    let rendered = render_minimax_openai_template(&messages, None, true);
    assert!(
        rendered.ends_with("]~b]ai\n<think>\n"),
        "expected thinking assistant generation prompt: {rendered}"
    );
}

#[test]
fn render_minimax_openai_template_omits_think_prompt_with_tools_when_disabled() {
    let messages = vec![json!({"role": "user", "content": "List the current directory"})];
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    })];
    let rendered = render_minimax_openai_template(&messages, Some(&tools), false);
    assert!(
        rendered.contains("<tools>"),
        "expected tool schema block in render: {rendered}"
    );
    assert!(
        rendered.contains("<minimax:tool_call>"),
        "expected MiniMax tool-call instructions in render: {rendered}"
    );
    assert!(
        rendered.ends_with("]~b]ai\n<think>\n\n</think>\n\n"),
        "tool-active disabled-thinking requests must use a closed-thinking assistant prompt: {rendered}"
    );
}
