// SPDX-License-Identifier: AGPL-3.0-only

//! Decode native atomic markers into Atlas's existing reasoning/Poolside
//! parser vocabulary. Prompt rendering always keeps the HF namespace.
use tokenizers::Tokenizer;

pub(super) fn validate(tokenizer: &Tokenizer) -> anyhow::Result<()> {
    for marker in [
        "<｜hy_start:opensource｜>",
        "<think:opensource>",
        "</think:opensource>",
        "<tool_calls:opensource>",
        "</tool_calls:opensource>",
        "<tool_call:opensource>",
        "</tool_call:opensource>",
        "<arg_key:opensource>",
        "</arg_key:opensource>",
        "<arg_value:opensource>",
        "</arg_value:opensource>",
    ] {
        anyhow::ensure!(
            tokenizer.token_to_id(marker).is_some(),
            "Hy4 tokenizer requires atomic native marker {marker}"
        );
    }
    Ok(())
}
pub(super) fn is_native(tokenizer: &Tokenizer) -> bool {
    tokenizer.token_to_id("<｜hy_start:opensource｜>").is_some()
}
pub(super) fn think_alias(tokenizer: &Tokenizer, text: &str) -> Option<u32> {
    let native = match text {
        "<think>" => "<think:opensource>",
        "</think>" => "</think:opensource>",
        _ => return None,
    };
    tokenizer.token_to_id(native)
}
pub(crate) fn canonical(mut text: String, native: bool) -> String {
    if !native {
        return text;
    }
    for tag in [
        "think",
        "tool_call",
        "arg_key",
        "arg_value",
        "tool_response",
    ] {
        for slash in ["", "/"] {
            text = text.replace(
                &format!("<{slash}{tag}:opensource>"),
                &format!("<{slash}{tag}>"),
            );
        }
    }
    for marker in [
        "<tool_calls:opensource>",
        "</tool_calls:opensource>",
        "<｜hy_end:opensource｜>",
        "<｜hy_pad:opensource｜>",
    ] {
        text = text.replace(marker, "");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        reasoning_parser::ReasoningFormat,
        tokenizer::{
            chat_render::{RenderFlags, render_chat},
            jinja_helpers::{ToolJsonStyle, build_jinja_env_with},
        },
    };
    use serde_json::json;

    #[test]
    fn native_markers_are_normalized_only_on_hy4_and_empty_text_stays_empty() {
        let s = "<think:opensource>x</think:opensource><tool_calls:opensource><tool_call:opensource>ping</tool_call:opensource></tool_calls:opensource>";
        assert_eq!(canonical(s.into(), false), s);
        assert_eq!(canonical(String::new(), true), "");
        assert_eq!(
            canonical("literal:opensource".into(), true),
            "literal:opensource"
        );
        let normalized = canonical(s.into(), true);
        let (reasoning, answer) = ReasoningFormat::Qwen
            .into_parser()
            .extract_thinking(&normalized, true);
        assert_eq!(reasoning.as_deref(), Some("x"));
        assert_eq!(answer, "<tool_call>ping</tool_call>");
    }
    #[test]
    fn hf_template_renders_native_think_modes_and_rejects_invalid_effort() {
        let env = build_jinja_env_with(
            include_str!("../../../../jinja-templates/hy_v4_flash.jinja"),
            ToolJsonStyle::HfSpaced,
        )
        .unwrap();
        let messages = [json!({"role":"user","content":"Hi"})];
        let enabled = render_chat(
            &env,
            &messages,
            None,
            RenderFlags {
                enable_thinking: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            enabled,
            "<｜hy_start:opensource｜>system<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:high<｜hy_end:opensource｜><｜hy_start:opensource｜>user<｜hy_middle:opensource｜>Hi<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource>"
        );
        let disabled = render_chat(&env, &messages, None, RenderFlags::default()).unwrap();
        assert!(disabled.contains("reasoning_effort:no_think"));
        assert!(disabled.ends_with("<think:opensource></think:opensource>"));
        assert!(
            render_chat(
                &env,
                &messages,
                None,
                RenderFlags {
                    enable_thinking: true,
                    reasoning_effort: Some("bogus"),
                    ..Default::default()
                }
            )
            .is_err()
        );
        let empty = render_chat(&env, &[], None, RenderFlags::default()).unwrap();
        assert!(empty.ends_with("<think:opensource></think:opensource>"));
    }
    #[test]
    fn hf_parallel_tool_history_and_typed_arguments_render_in_native_order() {
        let env = build_jinja_env_with(
            include_str!("../../../../jinja-templates/hy_v4_flash.jinja"),
            ToolJsonStyle::HfSpaced,
        )
        .unwrap();
        let messages = [
            json!({"role":"user","content":"Check both"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"a","function":{"name":"get","arguments":"{\"n\":2}"}},
                {"id":"b","function":{"name":"ping","arguments":{}}}]}),
            json!({"role":"tool","tool_call_id":"b","content":"second"}),
            json!({"role":"tool","tool_call_id":"a","content":"first"}),
        ];
        let tools = [
            json!({"type":"function","function":{"name":"get","parameters":{"type":"object","properties":{"n":{"type":"integer"}}}}}),
        ];
        let rendered = render_chat(&env, &messages, Some(&tools), RenderFlags::default()).unwrap();
        assert!(rendered.contains("<arg_key:opensource>n</arg_key:opensource><arg_value:opensource>2</arg_value:opensource>"));
        assert!(rendered.find("first").unwrap() < rendered.find("second").unwrap());
    }
}
