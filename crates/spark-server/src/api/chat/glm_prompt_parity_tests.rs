// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash prompts are byte-identical to the reference engine's.
//!
//! `test_data/chat_templates/glm-5.3-flash-fp4-local.cases.json` holds, per
//! request form (no thinking flag, `enable_thinking` true/false, every
//! `reasoning_effort`, `chat_template_kwargs`, tools, `response_format`,
//! single and multi-turn with past reasoning and out-of-order parallel tool
//! results), the prompt vLLM renders from the checkpoint's own
//! `chat_template.jinja` (a byte copy sits beside it; regenerate the cases
//! with the `.cases.py` there). Each request runs the production path here —
//! wire request → IR → [`prepare_chat_prompt`] → tokens — through a
//! byte-level tokenizer, so decoding the prompt tokens is exact.
//!
//! Ported from Enntity/atlas f4d95a10 and b1b53814 (AGPL-3.0-only).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::prepare_chat_prompt;
use crate::AppState;

/// Every special token the template emits, so the prompt-tail reasoning
/// detection sees `<think>` / `</think>` as single tokens, as in the real
/// GLM tokenizer.
const SPECIAL: &[&str] = &[
    "[gMASK]",
    "<sop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
];

fn manifest_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// SparkGLM 0.2.0's GLM `[behavior]`, for the keys the prompt path reads
/// (Enntity/atlas c87f2790 `kernels/gb10/glm-5.3-flash/MODEL.toml`).
fn glm_behavior() -> atlas_kernels::ModelBehavior {
    atlas_kernels::ModelBehavior {
        thinking_default: true,
        thinking_in_tools: true,
        max_thinking_budget: 0,
        cap_thinking_at_max_tokens: false,
        verbatim_reasoning_effort: true,
        disable_cwd_hint_injection: true,
        ..Default::default()
    }
}

/// A byte-level BPE with no merges plus the GLM special tokens: every
/// string round-trips through encode/decode unchanged.
/// `hf_json`: tool JSON spaced as SparkGLM serves it
/// (`ATLAS_USE_HF_REF_JSON_DUMPS=1`), through the verbatim-prompt tokenizer.
fn byte_level_tokenizer(dir: &Path, hf_json: bool) -> crate::tokenizer::ChatTokenizer {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    let vocab: tokenizers::models::bpe::Vocab = ByteLevel::alphabet()
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect();
    let model = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .build()
        .unwrap();
    let mut inner = tokenizers::Tokenizer::new(model);
    inner.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));
    inner.with_decoder(Some(ByteLevel::default()));
    let special: Vec<_> = SPECIAL
        .iter()
        .map(|t| tokenizers::AddedToken::from(*t, true))
        .collect();
    inner.add_special_tokens(special).unwrap();
    inner.save(dir.join("tokenizer.json"), false).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    std::fs::copy(
        manifest_path("test_data/chat_templates/glm-5.3-flash-fp4-local.jinja"),
        dir.join("chat_template.jinja"),
    )
    .unwrap();
    // The repo root is searched for a template override, exactly as serve
    // does: GLM must find none and render the checkpoint's own template.
    crate::tokenizer::ChatTokenizer::from_model_dir(
        dir,
        0,
        true,
        "glm5_next",
        "",
        Some(&manifest_path("../..")),
        false,
        hf_json,
    )
    .unwrap()
}

fn glm_state(dir: &Path) -> Arc<AppState> {
    glm_state_with(dir, true)
}

fn glm_state_with(dir: &Path, hf_json: bool) -> Arc<AppState> {
    let tokenizer = byte_level_tokenizer(dir, hf_json);
    let id = |t: &str| {
        let ids = tokenizer.encode(t).unwrap();
        assert_eq!(ids.len(), 1, "{t} is one token");
        ids[0]
    };
    Arc::new(AppState {
        think_start_token_id: Some(id("<think>")),
        think_end_token_id: Some(id("</think>")),
        tool_call_start_token_id: Some(id("<tool_call>")),
        tool_call_parser: Some(Arc::from(
            crate::tool_parser::ToolCallFormat::PoolsideV1.into_parser(),
        )),
        behavior: glm_behavior(),
        ..crate::api::test_app_state::app_state(tokenizer)
    })
}

fn cases() -> Vec<serde_json::Value> {
    serde_json::from_str(
        &std::fs::read_to_string(manifest_path(
            "test_data/chat_templates/glm-5.3-flash-fp4-local.cases.json",
        ))
        .unwrap(),
    )
    .unwrap()
}

/// The prompt `state` renders for the wire request `request`.
fn render(
    state: &Arc<AppState>,
    name: &str,
    request: &serde_json::Value,
) -> (String, bool, Option<u32>) {
    let wire: crate::openai::ChatCompletionRequest =
        serde_json::from_value(request.clone()).unwrap();
    wire.validate_reasoning_effort().unwrap();
    let mut req: crate::ir::ChatRequest = wire.into();
    let Ok(prepared) = prepare_chat_prompt(state, &mut req) else {
        panic!("{name}: prepare failed");
    };
    let prompt = state
        .tokenizer
        .decode_with_special(&prepared.prompt_tokens)
        .unwrap();
    (prompt, prepared.enable_thinking, prepared.thinking_budget)
}

#[test]
fn every_request_form_renders_the_reference_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let state = glm_state(dir.path());
    let cases = cases();
    assert!(cases.len() >= 40, "case table present");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let (prompt, enable_thinking, thinking_budget) = render(&state, name, &case["request"]);
        assert_eq!(prompt, case["prompt"].as_str().unwrap(), "{name}");
        // The template opens reasoning on every request, and no server
        // budget closes it: the model ends it, or max_tokens does.
        assert!(enable_thinking, "{name}: reasoning is parsed");
        assert_eq!(thinking_budget, None, "{name}: no server budget");
    }
}

/// Without the verbatim-prompt tokenizer an effort-silent render passes the
/// cross-template `medium` default, which GLM's template renders as its own
/// default (Max): tool-free prompts are the same bytes.
#[test]
fn the_effort_silent_default_renders_the_template_default() {
    let dir = tempfile::tempdir().unwrap();
    let state = glm_state_with(dir.path(), false);
    for case in cases() {
        if case["request"].get("tools").is_some() {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let (prompt, ..) = render(&state, name, &case["request"]);
        assert_eq!(prompt, case["prompt"].as_str().unwrap(), "{name}");
    }
}

#[test]
fn an_explicit_client_budget_is_the_only_thinking_cap() {
    let dir = tempfile::tempdir().unwrap();
    let state = glm_state(dir.path());
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "glm", "max_tokens": 4096, "thinking_token_budget": 300,
        "messages": [{"role": "user", "content": "Hi"}]
    }))
    .unwrap();
    let mut req: crate::ir::ChatRequest = wire.into();
    let Ok(prepared) = prepare_chat_prompt(&state, &mut req) else {
        panic!("prepare failed");
    };
    assert!(prepared.enable_thinking);
    assert_eq!(prepared.thinking_budget, Some(300));
}

#[test]
fn literal_glm_tool_markup_stays_text_while_structured_call_stays_native() {
    let dir = tempfile::tempdir().unwrap();
    let state = glm_state(dir.path());
    let tool_open = state.tokenizer.encode("<tool_call>").unwrap();
    assert_eq!(tool_open.len(), 1, "GLM opener must be one native token");
    let native_tool_open = tool_open[0];
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "glm",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": "Quote <tool_call> and </tool_call> literally."
            },
            {
                "role": "assistant",
                "content": "I will check.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Oslo <tool_call>\"}"
                    }
                }]
            }
        ]
    }))
    .unwrap();
    let mut req: crate::ir::ChatRequest = wire.into();
    let prepared = prepare_chat_prompt(&state, &mut req).expect("GLM prompt prepares");
    let prompt = state
        .tokenizer
        .decode_with_special(&prepared.prompt_tokens)
        .unwrap();
    assert!(
        prompt.contains("Quote <tool_call> and </tool_call> literally."),
        "literal tool markup must survive prompt decoding: {prompt:?}"
    );
    assert!(
        prompt.contains("Oslo <tool_call>"),
        "historical literal value survives: {prompt:?}"
    );
    assert_eq!(
        prepared
            .prompt_tokens
            .iter()
            .filter(|&&id| id == native_tool_open)
            .count(),
        1,
        "only the structured historical call may use the native opener"
    );
}
