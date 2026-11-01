// SPDX-License-Identifier: AGPL-3.0-only

/// Tokenize stop sequence strings into single-token stop IDs.
/// Multi-token stop sequences are logged but excluded (require string matching).
pub(crate) fn tokenize_stop_sequences(
    tokenizer: &crate::tokenizer::ChatTokenizer,
    stops: &[String],
) -> Vec<u32> {
    let mut tokens = Vec::new();
    for s in stops {
        match tokenizer.encode(s) {
            Ok(ids) if ids.len() == 1 => tokens.push(ids[0]),
            Ok(ids) if ids.len() > 1 => {
                tracing::info!(
                    "Multi-token stop '{}' ({} tokens) — use string matching",
                    s,
                    ids.len()
                );
            }
            Ok(_) => {} // Empty encoding
            Err(e) => tracing::warn!("Failed to tokenize stop '{}': {e}", s),
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Strip any matching stop sequence from the end of the output text.
/// Per OpenAI spec, returned text must not contain the stop sequence.
pub(crate) fn strip_stop_sequences(text: String, stops: &[String]) -> String {
    strip_stop_sequences_matched(text, stops).0
}

/// [`strip_stop_sequences`], also reporting WHICH stop sequence matched
/// (feeds Anthropic's `stop_sequence` response field via the IR's
/// `matched_stop`).
pub(crate) fn strip_stop_sequences_matched(
    mut text: String,
    stops: &[String],
) -> (String, Option<String>) {
    // Try longest-first so overlapping prefixes (`["</answer", "</answer>"]`)
    // don't truncate at the shorter (wrong) match boundary. strip_suffix
    // is end-anchored, so usually only one of the two end-matches at a
    // time, but defensive ordering handles the cases where it doesn't.
    let mut sorted: Vec<&String> = stops.iter().collect();
    sorted.sort_by_key(|s| std::cmp::Reverse(s.len()));
    for s in sorted {
        if let Some(stripped) = text.strip_suffix(s.as_str()) {
            text.truncate(stripped.len());
            return (text, Some(s.clone()));
        }
    }
    (text, None)
}

/// Strip `<think>...</think>` reasoning content from model output.
///
/// Qwen3.5 models generate internal reasoning between `<think>` and `</think>` tags.
/// This must be removed from the API response so that:
/// 1. Clients don't see internal reasoning in the content field
/// 2. Multi-turn conversations aren't corrupted when clients echo assistant content back
///
/// Returns only the text after the final `</think>` tag (the actual response),
/// trimmed of leading whitespace.
/// Extract `<think>...</think>` reasoning from model output.
///
/// Returns `(reasoning_content, response_content)`.
/// - `enable_thinking=true`: reasoning extracted into first element, response in second.
/// - `enable_thinking=false`: reasoning discarded (None), only response returned.
pub(crate) fn extract_thinking(
    text: &str,
    enable_thinking: bool,
    force_nonempty_content: bool,
    parser: Option<&dyn crate::reasoning_parser::ReasoningParser>,
) -> (Option<String>, String) {
    let (reasoning, content) = if let Some(p) = parser {
        p.extract_thinking(text, enable_thinking)
    } else {
        (None, text.to_string())
    };
    // vLLM `force_nonempty_content` (deepseek_r1): an all-reasoning output
    // (unclosed think block) is returned as `content` so the response body
    // is never empty. Reasoning stays populated.
    if force_nonempty_content
        && content.is_empty()
        && let Some(r) = reasoning.clone()
    {
        return (reasoning, r);
    }
    (reasoning, content)
}
