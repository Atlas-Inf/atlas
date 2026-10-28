// SPDX-License-Identifier: AGPL-3.0-only

//! Preserve literal GLM-5 and Qwen4_exp tool XML in incoming message prose.
//!
//! These models register their tool-envelope spellings as added tokens. A user
//! quoting `<tool_call>` would therefore put the native opener in the prompt
//! even when tools are disabled. This module uses the same split-encode
//! approach as the vision-markup path, with a separate tagged sentinel family
//! so the two marker-index spaces cannot overlap. The caller explicitly limits
//! the shared mechanism to model types with the verified added-token contract.

use super::super::ChatTokenizer;
use super::vision_markup::SentinelFamily;

/// Tool spellings that may be single added tokens in the supported GLM/Qwen
/// tokenizers. `markers` filters this list against the active tokenizer, so a
/// family only escapes spellings that are actually native tokens.
const GLM_TOOL_MARKUP: &[&str] = &[
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<tool_response>",
    "</tool_response>",
];

/// Distinct delimiters prevent marker index collisions with vision sentinels.
/// Original OPEN characters are escaped before markers, so caller text
/// cannot be confused with an internal sentinel, including tagged forms.
pub(super) const OPEN: char = '\u{FDD2}';
pub(super) const TAG: char = '\u{FDD4}';
pub(super) const CLOSE: char = '\u{FDD3}';

pub(super) fn contains_sentinel(tokenizer: &ChatTokenizer, rendered: &str) -> bool {
    tokenizer.glm_tool_markup_as_text && rendered.contains(OPEN)
}

pub(super) fn markers(tokenizer: &ChatTokenizer) -> Vec<String> {
    GLM_TOOL_MARKUP
        .iter()
        .filter(|marker| tokenizer.encode(marker).is_ok_and(|ids| ids.len() == 1))
        .map(|marker| (*marker).to_string())
        .chain(std::iter::once(OPEN.to_string()))
        .collect()
}

/// Escape native GLM/Qwen tool markup in incoming message prose. Structured
/// call names remain unchanged; literal argument values are escaped while
/// the template still emits real native call boundaries. Returns the number of escaped markers;
/// unsupported model types always return zero.
pub(crate) fn escape_glm_tool_markup<'a>(
    tokenizer: &ChatTokenizer,
    values: impl IntoIterator<Item = &'a mut serde_json::Value>,
) -> usize {
    if !tokenizer.glm_tool_markup_as_text {
        return 0;
    }
    let markup = markers(tokenizer);
    values
        .into_iter()
        .map(|value| {
            if value.get("role").is_some() {
                escape_message(value, &markup)
            } else {
                escape_value(value, &markup)
            }
        })
        .sum()
}

impl ChatTokenizer {
    pub(crate) fn escape_glm_tool_markup<'a>(
        &self,
        values: impl IntoIterator<Item = &'a mut serde_json::Value>,
    ) -> usize {
        escape_glm_tool_markup(self, values)
    }
}

fn replace_markers(s: &mut String, markup: &[String]) -> usize {
    // Reserve the last marker for the original delimiter itself. Escape it
    // first; inserted sentinels are not visited by this replacement pass.
    let escape_index = markup.len() - 1;
    *s = s.replace(OPEN, &format!("{OPEN}{TAG}{escape_index}{CLOSE}"));
    let mut n = 0;
    for (i, marker) in markup[..escape_index].iter().enumerate() {
        let hits = s.matches(marker.as_str()).count();
        if hits > 0 {
            *s = s.replace(marker.as_str(), &format!("{OPEN}{TAG}{i}{CLOSE}"));
            n += hits;
        }
    }
    n
}

/// Escape only text that the GLM template treats as visible message prose.
/// Call envelopes are generated later; media URLs are not prose.
fn escape_message(value: &mut serde_json::Value, markup: &[String]) -> usize {
    let serde_json::Value::Object(map) = value else {
        return 0;
    };
    let Some(role) = map.get("role").and_then(serde_json::Value::as_str) else {
        return 0;
    };
    if !matches!(role, "user" | "system" | "assistant" | "tool") {
        return 0;
    }
    let is_assistant = role == "assistant";

    let mut escaped = map
        .get_mut("content")
        .map(|content| escape_visible_content(content, markup))
        .unwrap_or(0);
    if is_assistant
        && let Some(reasoning) = map.get_mut("reasoning_content")
        && let serde_json::Value::String(text) = reasoning
    {
        escaped += replace_markers(text, markup);
    }
    if let Some(serde_json::Value::Array(calls)) = map.get_mut("tool_calls") {
        for call in calls {
            let function = if call.get("function").is_some() {
                &mut call["function"]
            } else {
                call
            };
            if let Some(arguments) = function.get_mut("arguments") {
                escaped += escape_value(arguments, markup);
            }
        }
    }
    escaped
}

// Literal strings in schemas and historical argument values are data too.
// The surrounding call envelope is generated by Jinja after this pass.
fn escape_value(value: &mut serde_json::Value, markup: &[String]) -> usize {
    match value {
        serde_json::Value::String(text) => replace_markers(text, markup),
        serde_json::Value::Array(values) => values
            .iter_mut()
            .map(|value| escape_value(value, markup))
            .sum(),
        serde_json::Value::Object(values) => values
            .values_mut()
            .map(|value| escape_value(value, markup))
            .sum(),
        _ => 0,
    }
}

fn escape_visible_content(value: &mut serde_json::Value, markup: &[String]) -> usize {
    match value {
        serde_json::Value::String(text) => replace_markers(text, markup),
        serde_json::Value::Array(parts) => parts
            .iter_mut()
            .map(|part| match part {
                serde_json::Value::String(text) => replace_markers(text, markup),
                serde_json::Value::Object(map)
                    if map.get("type").and_then(serde_json::Value::as_str) == Some("text") =>
                {
                    match map.get_mut("text") {
                        Some(serde_json::Value::String(text)) => replace_markers(text, markup),
                        _ => 0,
                    }
                }
                _ => 0,
            })
            .sum(),
        _ => 0,
    }
}

pub(super) fn sentinel_family<'a>(markup: &'a [String]) -> SentinelFamily<'a> {
    SentinelFamily {
        open: OPEN,
        tag: Some(TAG),
        close: CLOSE,
        markup,
    }
}

#[cfg(test)]
mod tests {
    use super::{CLOSE, OPEN, TAG, escape_message, sentinel_family};
    use crate::tokenizer::chat_impl::vision_markup::{
        SentinelFamily, encode_with_sentinel_families,
    };

    const IMAGE_PAD: u32 = 900;
    const TOOL_CALL: u32 = 901;

    fn fake_encode(s: &str) -> anyhow::Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut from = 0;
        while from < s.len() {
            let next = [
                (s[from..].find("<|image_pad|>"), "<|image_pad|>", IMAGE_PAD),
                (s[from..].find("<tool_call>"), "<tool_call>", TOOL_CALL),
            ]
            .into_iter()
            .filter_map(|(offset, marker, id)| offset.map(|offset| (from + offset, marker, id)))
            .min_by_key(|(offset, _, _)| *offset);
            let Some((start, marker, id)) = next else {
                out.extend(s[from..].bytes().map(u32::from));
                break;
            };
            out.extend(s[from..start].bytes().map(u32::from));
            out.push(id);
            from = start + marker.len();
        }
        Ok(out)
    }

    fn tool_markup() -> Vec<String> {
        super::GLM_TOOL_MARKUP
            .iter()
            .map(|marker| (*marker).to_owned())
            .chain(std::iter::once(OPEN.to_string()))
            .collect()
    }

    #[test]
    fn vision_and_glm_sentinels_keep_marker_indices_separate() {
        let vision = vec!["<|vision_start|>".into(), "<|image_pad|>".into()];
        let tool = tool_markup();
        let rendered = format!("\u{fdd0}1\u{fdd1}<|image_pad|>{OPEN}{TAG}0{CLOSE}<tool_call>");
        let ids = encode_with_sentinel_families(
            &rendered,
            &[
                SentinelFamily {
                    open: '\u{fdd0}',
                    tag: None,
                    close: '\u{fdd1}',
                    markup: &vision,
                },
                sentinel_family(&tool),
            ],
            fake_encode,
        )
        .unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == IMAGE_PAD).count(), 1);
        assert_eq!(ids.iter().filter(|&&id| id == TOOL_CALL).count(), 1);
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&id| id != IMAGE_PAD && id != TOOL_CALL)
            .map(|&id| id as u8)
            .collect();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "<|image_pad|><tool_call>"
        );
    }

    #[test]
    fn glm_message_escape_preserves_literal_sentinel_text() {
        let markup = tool_markup();
        let literal = format!("raw {OPEN}0{CLOSE} and {OPEN}{TAG}0{CLOSE} and <tool_call>");
        let mut message = serde_json::json!({"role": "user", "content": literal});
        assert_eq!(escape_message(&mut message, &markup), 1);
        let escaped = message["content"].as_str().unwrap();
        let rendered = format!("{escaped} <tool_call>");
        let ids =
            encode_with_sentinel_families(&rendered, &[sentinel_family(&markup)], fake_encode)
                .unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == TOOL_CALL).count(), 1);
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&id| id != TOOL_CALL)
            .map(|&id| id as u8)
            .collect();
        assert_eq!(String::from_utf8(bytes).unwrap(), format!("{literal} "));
    }

    #[test]
    fn glm_message_escape_preserves_structured_calls_and_media_values() {
        let markup = tool_markup();
        let mut messages = serde_json::json!([
            {"role": "user", "content": "literal <tool_call> and </tool_call>"},
            {
                "role": "assistant",
                "content": "prose <arg_key>",
                "reasoning_content": "reason <arg_value>",
                "tool_calls": [{
                    "function": {"name": "fn", "arguments": {"example": "<tool_call>"}}
                }]
            },
            {"role": "user", "content": [
                {"type": "text", "text": "part <arg_key>"},
                {"type": "image_url", "image_url": {"url": "<tool_call>"}}
            ]}
        ]);
        let escaped: usize = messages
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .map(|message| escape_message(message, &markup))
            .sum();
        assert_eq!(escaped, 6);
        assert_eq!(
            messages[0]["content"],
            "literal \u{fdd2}\u{fdd4}0\u{fdd3} and \u{fdd2}\u{fdd4}1\u{fdd3}"
        );
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"]["example"],
            format!("{OPEN}{TAG}0{CLOSE}")
        );
        assert_eq!(messages[2]["content"][1]["image_url"]["url"], "<tool_call>");
    }
}
