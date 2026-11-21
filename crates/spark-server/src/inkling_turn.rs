// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2's turn format, mapped onto the generic reasoning and tool paths.
//!
//! An Inkling 2 turn is a run of messages, each opened by `<|message_model|>`,
//! an optional name and a content-type token, and closed by `<|end_message|>`:
//!
//! ```text
//! <|message_model|><|content_thinking|>…<|end_message|>
//! <|message_model|><|content_text|>…<|end_message|>
//! <|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>
//! <|content_model_end_sampling|>
//! ```
//!
//! Every marker is a special token, so a plain decode drops them, and
//! `<|end_message|>` closes every block, so it cannot be the scheduler's think
//! boundary. [`InklingTurn`] rewrites the generated ids before the chat paths
//! see them:
//!
//!  * headers (`<|message_model|>`, the name, the content-type token) go;
//!  * exactly one `<|end_message|>` survives, as the think end before the
//!    answer (the thinking block's own close, or a synthetic one ahead of the
//!    first answer block when the model did not think); with thinking off the
//!    thinking body goes too and no think end is emitted;
//!  * each tool block becomes a Hermes `<tool_call>` envelope with
//!    `"arguments"`, re-encoded once its close arrives (its arguments are not
//!    streamed piecewise);
//!  * a thinking block after the answer started goes (it has no channel).
//!
//! Reference: the checkpoint's `response_template` in `tokenizer_config.json`
//! (thinking blocks are reasoning; text, tool-error and tool-text blocks are
//! content; tool-json blocks are calls).

use crate::tokenizer::ChatTokenizer;

/// Text <-> ids, as the adapter needs them (the tokenizer, or a test fake).
pub(crate) trait TurnCodec {
    fn decode_ids(&self, ids: &[u32]) -> String;
    fn encode_text(&self, text: &str) -> Vec<u32>;
}

impl TurnCodec for ChatTokenizer {
    fn decode_ids(&self, ids: &[u32]) -> String {
        self.decode(ids).unwrap_or_default()
    }
    fn encode_text(&self, text: &str) -> Vec<u32> {
        self.encode(text).unwrap_or_default()
    }
}

/// The marker ids of an Inkling 2 tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InklingMarkers {
    pub message: u32,
    pub thinking: u32,
    pub text: u32,
    pub tool_error: u32,
    pub tool_text: u32,
    pub tool_json: u32,
    pub end: u32,
}

impl InklingMarkers {
    /// The markers for an `inkling2_loop` checkpoint; `None` for any other
    /// model or a tokenizer missing one of them.
    pub(crate) fn resolve(model_type: &str, tokenizer: &ChatTokenizer) -> Option<Self> {
        if model_type != "inkling2_loop" {
            return None;
        }
        let id = |s: &str| tokenizer.inner().token_to_id(s);
        Some(Self {
            message: id("<|message_model|>")?,
            thinking: id("<|content_thinking|>")?,
            text: id("<|content_text|>")?,
            tool_error: id("<|content_tool_error|>")?,
            tool_text: id("<|content_invoke_tool_text|>")?,
            tool_json: id("<|content_invoke_tool_json|>")?,
            end: id("<|end_message|>")?,
        })
    }

    /// A fresh turn; `keep_thinking` is the request's resolved thinking state.
    pub(crate) fn turn(self, keep_thinking: bool) -> InklingTurn {
        InklingTurn {
            m: self,
            keep_thinking,
            block: Block::Between,
            think_closed: false,
            tool: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Block {
    Between,
    Header,
    Thinking,
    Answer,
    Tool,
}

/// One generation's rewrite state.
#[derive(Clone, Debug)]
pub(crate) struct InklingTurn {
    m: InklingMarkers,
    keep_thinking: bool,
    block: Block,
    think_closed: bool,
    tool: Vec<u32>,
}

impl InklingTurn {
    /// Push the ids the chat paths see for one generated id onto `out`.
    pub(crate) fn feed(&mut self, tok: u32, codec: &dyn TurnCodec, out: &mut Vec<u32>) {
        let m = self.m;
        if tok == m.message {
            self.block = Block::Header;
            return;
        }
        if tok == m.end {
            match self.block {
                Block::Thinking => self.close_thinking(out),
                Block::Tool => self.emit_call(codec, out, true),
                Block::Between | Block::Header | Block::Answer => {}
            }
            self.block = Block::Between;
            return;
        }
        match self.block {
            Block::Header => {
                let next = if tok == m.thinking {
                    Block::Thinking
                } else if tok == m.text || tok == m.tool_error || tok == m.tool_text {
                    Block::Answer
                } else if tok == m.tool_json {
                    Block::Tool
                } else {
                    // The message's name: dropped (a call repeats it in its body).
                    return;
                };
                if next != Block::Thinking {
                    self.close_thinking(out);
                }
                self.block = next;
            }
            Block::Thinking => {
                if self.keep_thinking && !self.think_closed {
                    out.push(tok);
                }
            }
            Block::Tool => self.tool.push(tok),
            Block::Answer | Block::Between => out.push(tok),
        }
    }

    /// Flush what an unclosed block still holds (a length-cut tool call).
    pub(crate) fn finish(&mut self, codec: &dyn TurnCodec, out: &mut Vec<u32>) {
        if self.block == Block::Tool {
            self.emit_call(codec, out, false);
            self.block = Block::Between;
        }
    }

    /// The whole of a finished generation.
    pub(crate) fn map_all(mut self, ids: &[u32], codec: &dyn TurnCodec) -> Vec<u32> {
        let mut out = Vec::with_capacity(ids.len());
        for &tok in ids {
            self.feed(tok, codec, &mut out);
        }
        self.finish(codec, &mut out);
        out
    }

    fn close_thinking(&mut self, out: &mut Vec<u32>) {
        if !self.think_closed {
            self.think_closed = true;
            if self.keep_thinking {
                out.push(self.m.end);
            }
        }
    }

    fn emit_call(&mut self, codec: &dyn TurnCodec, out: &mut Vec<u32>, closed: bool) {
        let raw = codec.decode_ids(&std::mem::take(&mut self.tool));
        let text = if closed {
            format!("<tool_call>\n{}\n</tool_call>", hermes_body(&raw))
        } else {
            // Cut off: let the truncated-call salvage see the Hermes key.
            format!(
                "<tool_call>\n{}",
                raw.replacen("\"args\":", "\"arguments\":", 1)
            )
        };
        out.extend(codec.encode_text(&text));
    }
}

/// `{"name":…,"args":{…}}` as Hermes `{"name": …, "arguments": {…}}`; text
/// that is not such an object passes unchanged for the generic salvage. `<`
/// is written `\u003c` (it occurs only inside JSON strings), so an argument
/// quoting `</tool_call>` cannot close the envelope early.
fn hermes_body(raw: &str) -> String {
    let Ok(serde_json::Value::Object(obj)) = serde_json::from_str::<serde_json::Value>(raw.trim())
    else {
        return raw.to_string();
    };
    let (Some(name), Some(args)) = (obj.get("name"), obj.get("args")) else {
        return raw.to_string();
    };
    format!("{{\"name\": {name}, \"arguments\": {args}}}").replace('<', "\\u003c")
}

#[cfg(test)]
#[path = "inkling_turn_tests.rs"]
mod tests;
