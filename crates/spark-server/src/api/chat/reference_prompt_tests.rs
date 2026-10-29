// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.8-Flash-Next prompts are the reference server's prompts, byte for
//! byte.
//!
//! The reference for nvidia/Qwen3.8-Flash-Next-FP4-Local is vLLM rendering
//! the checkpoint's own `chat_template.jinja`
//! (`test_data/chat_templates/qwen3.8-flash-next-fp4-local.jinja`, a byte
//! copy of the checkpoint's, md5 519239a4908bb1f805bbce5fa8c8a242, identical
//! to the `chat_template` in its `tokenizer_config.json`; Qwen template,
//! vendored like the other checkpoint templates beside it). `scripts/qwen38_flash_next_template_golden.py`
//! renders it with transformers and the kwargs vLLM derives from each request
//! and stores request + expected prompt in
//! `qwen3.8-flash-next-fp4-local.cases.json`. Each case here goes through the
//! production request path — OpenAI wire parse, the edge's effort check, IR
//! lowering, `prepare_chat_prompt` under this target's own MODEL.toml
//! `[behavior]` — and the prompt tokens must decode to exactly that text.
//! The same requests must resolve no server thinking budget and no
//! `<tool_call>` bias, as on the reference.
//!
//! The tokenizer is a byte-level BPE with no merges plus the template's
//! control tokens: every string round-trips exactly, so a mismatch is a
//! rendering difference, not a tokenization artifact.

use std::sync::Arc;

use serde::Deserialize;

use crate::AppState;
use crate::api::chat::prepare::prepare_chat_prompt;
use crate::api::test_app_state::build_behavior;

const CASES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_data/chat_templates/qwen3.8-flash-next-fp4-local.cases.json"
));
const TEMPLATE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_data/chat_templates/qwen3.8-flash-next-fp4-local.jinja"
));
const TARGET: &str = "../../kernels/gb10/qwen3.8-flash-next";

/// The checkpoint's added tokens that the template emits.
const CONTROL_TOKENS: [&str; 13] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
];

#[derive(Deserialize)]
struct Case {
    name: String,
    request: serde_json::Value,
    expected: String,
}

/// A checkpoint directory holding the template and a lossless tokenizer.
fn model_dir(dir: &std::path::Path, verbatim_prompt: bool) -> crate::tokenizer::ChatTokenizer {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let vocab: tokenizers::models::bpe::Vocab = alphabet
        .iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32))
        .collect();
    let bpe = tokenizers::models::bpe::BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    tokenizer
        .with_pre_tokenizer(Some(ByteLevel::new(false, true, false)))
        .with_decoder(Some(ByteLevel::default()));
    let control: Vec<_> = CONTROL_TOKENS
        .iter()
        .map(|t| tokenizers::AddedToken::from(*t, true))
        .collect();
    tokenizer.add_special_tokens(control).unwrap();
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::json!({ "chat_template": TEMPLATE }).to_string(),
    )
    .unwrap();
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    crate::tokenizer::ChatTokenizer::from_model_dir(
        dir,
        eos,
        true,
        "qwen4_exp",
        "",
        None,
        false,
        verbatim_prompt,
    )
    .unwrap()
}

/// This target's `[behavior]` as the server is built with it, for the fields
/// the prompt and sampling setup read.
fn behavior() -> atlas_kernels::ModelBehavior {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(TARGET);
    let b = build_behavior::parse_behavior(&dir);
    assert_eq!(
        b.tool_call_parser, "qwen3_coder",
        "the target's MODEL.toml was read"
    );
    atlas_kernels::ModelBehavior {
        thinking_in_tools: b.thinking_in_tools,
        max_thinking_budget: b.max_thinking_budget,
        effort_capped_at_ceiling: b.effort_capped_at_ceiling,
        thinking_default: b.thinking_default,
        disable_tool_steering: b.disable_tool_steering,
        template_owns_tool_definitions: b.template_owns_tool_definitions,
        disable_cwd_hint_injection: b.disable_cwd_hint_injection,
        cap_thinking_at_max_tokens: b.cap_thinking_at_max_tokens,
        tscg: b.tscg,
        preserve_thinking: b.preserve_thinking,
        verbatim_prompt: b.verbatim_prompt,
        tool_call_steering: b.tool_call_steering,
        ..Default::default()
    }
}

fn state(dir: &std::path::Path) -> Arc<AppState> {
    let behavior = behavior();
    let tokenizer = model_dir(dir, behavior.verbatim_prompt);
    let mut state = crate::api::test_app_state::app_state(tokenizer);
    state.tool_call_parser = Some(Arc::from(
        "qwen3_coder"
            .parse::<crate::tool_parser::ToolCallFormat>()
            .unwrap()
            .into_parser(),
    ));
    state.think_start_token_id = state.tokenizer.inner().token_to_id("<think>");
    state.think_end_token_id = state.tokenizer.inner().token_to_id("</think>");
    state.tool_call_start_token_id = state.tokenizer.inner().token_to_id("<tool_call>");
    state.chat = super::levers::ChatLevers::resolve(&behavior);
    state.max_seq_len = 262_144;
    state.behavior = behavior;
    Arc::new(state)
}

/// One wire request through the edge and `prepare_chat_prompt`.
fn prepare(
    state: &Arc<AppState>,
    request: &serde_json::Value,
) -> Result<(crate::ir::ChatRequest, super::prepare::PreparedChat), String> {
    let wire: crate::openai::ChatCompletionRequest =
        serde_json::from_value(request.clone()).map_err(|e| e.to_string())?;
    wire.validate_reasoning_effort()?;
    let mut req: crate::ir::ChatRequest = wire.into();
    crate::api::chat_phases::validate_input(&req).map_err(|r| format!("{:?}", r.status()))?;
    let prepared = prepare_chat_prompt(state, &mut req).map_err(|r| format!("{:?}", r.status()))?;
    Ok((req, prepared))
}

/// The prompt text Atlas prefills for one wire request.
fn atlas_prompt(state: &Arc<AppState>, request: &serde_json::Value) -> Result<String, String> {
    let (_, prepared) = prepare(state, request)?;
    state
        .tokenizer
        .decode_with_special(&prepared.prompt_tokens)
        .map_err(|e| e.to_string())
}

#[test]
fn every_request_form_renders_the_reference_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let state = state(dir.path());
    let cases: Vec<Case> = serde_json::from_str(CASES).unwrap();
    let mut failed = Vec::new();
    for case in &cases {
        match atlas_prompt(&state, &case.request) {
            Ok(got) if got == case.expected => {}
            Ok(got) => failed.push(format!(
                "{}:\n--- reference\n{}\n--- atlas\n{got}",
                case.name, case.expected
            )),
            Err(e) => failed.push(format!("{}: request refused: {e}", case.name)),
        }
    }
    assert!(
        failed.is_empty(),
        "{} of {} cases differ from the reference render:\n\n{}",
        failed.len(),
        cases.len(),
        failed.join("\n\n")
    );
}

/// The tokenizer round-trips arbitrary text, so the comparison above sees
/// rendering differences only.
#[test]
fn the_fixture_tokenizer_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = model_dir(dir.path(), false);
    let text = "<|im_start|>system\n{\"a\": \"<b> & é\"}\n\n  x<|im_end|>\n<think>\n";
    let ids = tokenizer.encode(text).unwrap();
    assert_eq!(tokenizer.decode_with_special(&ids).unwrap(), text);
}

/// The decode side of the same requests: reasoning carries no server budget
/// (only a client's explicit token budget binds), and a tools request leaves
/// `<tool_call>` unbiased.
#[test]
fn no_request_form_gets_a_thinking_budget_or_a_tool_call_bias() {
    let dir = tempfile::tempdir().unwrap();
    let state = state(dir.path());
    let cases: Vec<Case> = serde_json::from_str(CASES).unwrap();
    for case in &cases {
        let (req, prepared) = prepare(&state, &case.request).unwrap();
        let client_budget = case.request["thinking_token_budget"].as_u64();
        assert_eq!(
            prepared.thinking_budget,
            client_budget.map(|b| b as u32),
            "{}: thinking budget",
            case.name
        );
        // Thinking is on exactly when the template opened `<think>` for it.
        assert_eq!(
            prepared.enable_thinking,
            case.expected.ends_with("<think>\n"),
            "{}: thinking",
            case.name
        );
        let sampling = super::sampling_setup::build_sampling(
            &state,
            &req,
            prepared.enable_thinking,
            prepared.tools_active,
            false,
            0,
        )
        .unwrap_or_else(|_| panic!("{}: sampling setup", case.name));
        assert!(sampling.logit_bias.is_empty(), "{}: logit bias", case.name);
    }
}
