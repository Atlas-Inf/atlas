// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron 3.5 Super VL Jinja-template render tests. The template ships
//! verbatim from the checkpoint (`nemotron_3_5_super_vl.jinja`); these
//! tests pin the two contracts serving depends on: the empty
//! `<think></think>` prefill when thinking is off, and the qwen3_coder XML
//! `<tool_call>\n<function=...>` format instructions.

use serde_json::json;

use crate::tokenizer::normalize_tool_call_arguments;

fn render_nemotron_super_vl_template(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    enable_thinking: bool,
) -> String {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/nemotron_3_5_super_vl.jinja"
    ))
    .expect("bundled Nemotron 3.5 Super VL template must be present");
    let env = crate::tokenizer::jinja_helpers::build_jinja_env(&raw).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages = normalize_tool_call_arguments(messages);
    tmpl.render(minijinja::context! {
        messages => minijinja::Value::from_serialize(&messages),
        tools => tools.map(minijinja::Value::from_serialize).unwrap_or(minijinja::Value::UNDEFINED),
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
    })
    .expect("template renders")
}

#[test]
fn nemotron_super_vl_thinking_off_renders_empty_think_prefill() {
    let messages = vec![json!({"role": "user", "content": "2+3?"})];
    let rendered = render_nemotron_super_vl_template(&messages, None, false);
    assert!(
        rendered.ends_with("<think></think>"),
        "thinking-off must end on the empty <think></think> prefill: {rendered:?}"
    );

    let think = render_nemotron_super_vl_template(&messages, None, true);
    assert!(
        think.trim_end().ends_with("<think>"),
        "thinking-on must end inside an open think block: {think:?}"
    );
}

#[test]
fn nemotron_super_vl_tools_render_qwen3_coder_instructions() {
    let messages = vec![json!({"role": "user", "content": "weather in Lyon?"})];
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    })];
    let rendered = render_nemotron_super_vl_template(&messages, Some(&tools), false);
    assert!(
        rendered.contains("<tool_call>") && rendered.contains("<function=example_function_name>"),
        "tools must render the qwen3_coder XML format instructions: {rendered:?}"
    );
    assert!(rendered.contains("get_weather"));
}

/// A real BFCL `bfcl-subset` request shape (2022 run): user-only message,
/// OpenAI tool nesting, `tool_choice = "auto"`, `parameters.type =
/// "object"`. The gate fell back to the checkpoint's own
/// `chat_template.jinja` when its cwd missed `jinja-templates/` — this
/// renders BOTH templates to prove they agree on the path 775 requests
/// died on.
fn render_file_template(
    path: &std::path::Path,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
) -> Result<String, minijinja::Error> {
    let raw = std::fs::read_to_string(path).expect("template file reads");
    let converted = crate::tokenizer::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env =
        crate::tokenizer::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let messages = normalize_tool_call_arguments(messages);
    tmpl.render(minijinja::context! {
        messages => minijinja::Value::from_serialize(&messages),
        tools => tools.map(minijinja::Value::from_serialize).unwrap_or(minijinja::Value::UNDEFINED),
        add_generation_prompt => true,
        enable_thinking => false,
    })
}

fn bfcl_body() -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let messages = vec![json!({
        "role": "user",
        "content": "Find the area of a triangle with a base of 10 units and height of 5 units."
    })];
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "calculate_triangle_area",
            "description": "Calculate the area of a triangle given its base and height.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "integer", "description": "The base of the triangle."},
                    "height": {"type": "integer", "description": "The height of the triangle."},
                    "unit": {"type": "string", "description": "Unit (defaults to units)"}
                },
                "required": ["base", "height"]
            }
        }
    })];
    (messages, tools)
}

#[test]
fn bfcl_user_only_tools_render_bundled_template() {
    let (messages, tools) = bfcl_body();
    let rendered = render_nemotron_super_vl_template(&messages, Some(&tools), false);
    assert!(
        rendered.contains("<tool_call>") && rendered.contains("calculate_triangle_area"),
        "BFCL-shaped body must render the tool contract: {rendered:?}"
    );
}

#[test]
fn bfcl_user_only_tools_render_checkpoint_template() {
    // The gate's fallback path: the checkpoint's own chat_template.jinja.
    // Present where the checkpoint is downloaded to ~/models; skipped elsewhere.
    let p = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join("models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/chat_template.jinja");
    if !p.exists() {
        eprintln!("checkpoint dir absent — skipping");
        return;
    }
    let (messages, tools) = bfcl_body();
    let rendered = render_file_template(&p, &messages, Some(&tools))
        .expect("checkpoint template must render a tools+user-only body");
    assert!(rendered.contains("calculate_triangle_area"));
}

/// BFCL python-type schemas: `dict`/`float`/`any`/`tuple` parameter types
/// and a flat (non-nested) tool record.
#[test]
fn bfcl_python_types_and_flat_tool_record_render() {
    let messages = vec![json!({"role": "user", "content": "do it"})];
    let tools = vec![
        json!({
            "name": "f", "description": "d",
            "parameters": {"type": "dict", "properties": {
                "a": {"type": "float"}, "b": {"type": "any"},
                "c": {"type": "tuple"}, "d": {"type": "dict"}
            }}
        }),
        json!({
            "type": "function",
            "function": {"name": "g", "description": "java style",
                "parameters": {"type": "object", "properties": {
                    "x": {"type": "HashMap"}, "y": {"type": "ArrayList"}}}}
        }),
    ];
    let rendered = render_nemotron_super_vl_template(&messages, Some(&tools), false);
    assert!(rendered.contains("<name>f</name>") && rendered.contains("<name>g</name>"));
}

/// No system + several tools (BFCL parallel subset) and a with-system case.
#[test]
fn no_system_parallel_tools_render() {
    let messages = vec![json!({"role": "user", "content": "both please"})];
    let tools: Vec<serde_json::Value> = (0..4)
        .map(|i| {
            json!({"type":"function","function":{"name":format!("f{i}"),
            "description":"d","parameters":{"type":"object","properties":{}}}})
        })
        .collect();
    let rendered = render_nemotron_super_vl_template(&messages, Some(&tools), false);
    assert!(rendered.contains("<name>f3</name>"));

    let with_sys = vec![
        json!({"role": "system", "content": "be terse"}),
        json!({"role": "user", "content": "hi"}),
    ];
    let rendered = render_nemotron_super_vl_template(&with_sys, Some(&tools), false);
    assert!(rendered.contains("be terse"));
}

#[test]
fn omni_expand_wraps_image_pads() {
    use crate::tokenizer::chat_impl::expand_omni_pads;
    // 5 + one <image> + 7: expands to 19 + N*18 + 20 in place.
    let toks = vec![10, 11, 18, 12, 13];
    let out = expand_omni_pads(toks, &[3], 18, 19, 20);
    assert_eq!(out, vec![10, 11, 19, 18, 18, 18, 20, 12, 13]);
    // Two images, counts in order.
    let toks = vec![18, 99, 18];
    let out = expand_omni_pads(toks, &[1, 2], 18, 19, 20);
    assert_eq!(out, vec![19, 18, 20, 99, 19, 18, 18, 20]);
}
