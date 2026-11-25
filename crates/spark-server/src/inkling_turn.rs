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
//!    `"arguments"`: once the body reads `{"name":NAME,"args":` the envelope
//!    opens and the argument tokens pass through as they arrive, so the
//!    tool parser streams them; a body of any other shape is re-encoded
//!    whole when its close arrives;
//!  * a thinking block after the answer started goes (it has no channel);
//!  * a text block whose body is only a name followed by
//!    `<|content_invoke_tool_json|>` is a call: at thinking effort 0 the
//!    checkpoint opens calls as `<|content_text|>NAME<|content_invoke_tool_json|>`,
//!    so a text block's first tokens are held while they still read as a
//!    name (at most [`NAME_HOLD`] tokens) and flushed as text otherwise.
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
            call: Call::Prefix,
            held: None,
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

/// How far a tool block's body has been read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    /// Waiting for `{"name":NAME,"args":`.
    Prefix,
    /// The envelope is open; argument tokens pass through.
    Args,
    /// Another shape: held whole until the close.
    Whole,
}

/// One generation's rewrite state.
#[derive(Clone, Debug)]
pub(crate) struct InklingTurn {
    m: InklingMarkers,
    keep_thinking: bool,
    block: Block,
    think_closed: bool,
    /// The tool body's ids not yet emitted.
    tool: Vec<u32>,
    call: Call,
    /// A text block's opening tokens while they may still be a call's name.
    held: Option<Vec<u32>>,
}

/// Tokens a text block's opening may hold while it reads as a call's name.
const NAME_HOLD: usize = 16;

fn name_like(text: &str) -> bool {
    text.chars()
        .all(|c| c.is_ascii_alphanumeric() || "_.:-".contains(c))
}

impl InklingTurn {
    /// Push the ids the chat paths see for one generated id onto `out`.
    pub(crate) fn feed(&mut self, tok: u32, codec: &dyn TurnCodec, out: &mut Vec<u32>) {
        let m = self.m;
        if let Some(held) = self.held.as_mut() {
            if tok == m.tool_json {
                // NAME<|content_invoke_tool_json|>: the block was a call
                self.held = None;
                self.block = Block::Tool;
                return;
            }
            if tok != m.message && tok != m.end {
                held.push(tok);
                let text = codec.decode_ids(held);
                if held.len() <= NAME_HOLD && name_like(&text) {
                    return;
                }
            }
            self.flush_held(out);
            if tok != m.message && tok != m.end {
                return;
            }
        }
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
                if tok == m.text {
                    self.held = Some(Vec::new());
                }
                self.block = next;
            }
            Block::Thinking => {
                if self.keep_thinking && !self.think_closed {
                    out.push(tok);
                }
            }
            Block::Tool => self.call_token(tok, codec, out),
            Block::Answer | Block::Between => out.push(tok),
        }
    }

    /// Flush what an unclosed block still holds (a length-cut tool call).
    pub(crate) fn finish(&mut self, codec: &dyn TurnCodec, out: &mut Vec<u32>) {
        self.flush_held(out);
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

    /// A held text opening that turned out to be text.
    fn flush_held(&mut self, out: &mut Vec<u32>) {
        if let Some(held) = self.held.take() {
            out.extend(held);
        }
    }

    fn close_thinking(&mut self, out: &mut Vec<u32>) {
        if !self.think_closed {
            self.think_closed = true;
            if self.keep_thinking {
                out.push(self.m.end);
            }
        }
    }

    fn call_token(&mut self, tok: u32, codec: &dyn TurnCodec, out: &mut Vec<u32>) {
        self.tool.push(tok);
        match self.call {
            Call::Whole => {}
            Call::Args => self.pass_args(codec, out, false),
            Call::Prefix => {
                let text = codec.decode_ids(&self.tool);
                if text.ends_with(char::REPLACEMENT_CHARACTER) {
                    return; // a character split across tokens
                }
                match call_prefix(&text) {
                    Prefix::More => {}
                    Prefix::Other => self.call = Call::Whole,
                    Prefix::Done { name, end } => {
                        self.call = Call::Args;
                        self.tool.clear();
                        let open = format!(
                            "<tool_call>\n{{\"name\": {}, \"arguments\": {}",
                            escape_lt(&name),
                            escape_lt(&text[end..])
                        );
                        out.extend(codec.encode_text(&open));
                    }
                }
            }
        }
    }

    /// Argument tokens pass through unchanged unless they hold a `<`, which is
    /// re-encoded as `\u003c` (see [`hermes_body`]); `all` flushes a split
    /// character too. A special token decodes to nothing and goes, as it does
    /// from a call rewritten whole.
    fn pass_args(&mut self, codec: &dyn TurnCodec, out: &mut Vec<u32>, all: bool) {
        let text = codec.decode_ids(&self.tool);
        if text.is_empty() {
            self.tool.clear();
        } else if !text.contains('<') {
            out.append(&mut self.tool);
        } else if all || !text.ends_with(char::REPLACEMENT_CHARACTER) {
            out.extend(codec.encode_text(&escape_lt(&text)));
            self.tool.clear();
        }
    }

    fn emit_call(&mut self, codec: &dyn TurnCodec, out: &mut Vec<u32>, closed: bool) {
        if std::mem::replace(&mut self.call, Call::Prefix) == Call::Args {
            self.pass_args(codec, out, true);
            if closed {
                out.extend(codec.encode_text("\n</tool_call>"));
            }
            return;
        }
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

/// `s` with every `<` written `\u003c`.
fn escape_lt(s: &str) -> String {
    s.replace('<', "\\u003c")
}

enum Prefix {
    More,
    Other,
    /// `name` is the call's name as a JSON string; the arguments start at `end`.
    Done {
        name: String,
        end: usize,
    },
}

/// Longest body read for the name before giving up on streaming it.
const PREFIX_MAX: usize = 512;

/// Reads `{"name":NAME,"args":` (JSON whitespace allowed) off the start of a
/// tool body.
fn call_prefix(text: &str) -> Prefix {
    let b = text.as_bytes();
    let mut i = 0;
    let ws = |i: &mut usize| {
        while *i < b.len() && b[*i].is_ascii_whitespace() {
            *i += 1;
        }
    };
    // Each literal must match in full; a partial match at the end needs more.
    let lit = |i: &mut usize, want: &[u8]| -> Option<bool> {
        let have = &b[*i..b.len().min(*i + want.len())];
        if !want.starts_with(have) {
            return Some(false);
        }
        if have.len() < want.len() {
            return None;
        }
        *i += want.len();
        Some(true)
    };
    let mut name = String::new();
    for step in 0..5 {
        ws(&mut i);
        let ok = match step {
            0 => lit(&mut i, b"{"),
            1 => lit(&mut i, b"\"name\"").and_then(|ok| {
                ws(&mut i);
                if ok { lit(&mut i, b":") } else { Some(false) }
            }),
            2 => {
                let start = i;
                if i == b.len() {
                    None
                } else if b[i] != b'"' {
                    Some(false)
                } else {
                    let mut j = i + 1;
                    while j < b.len() && b[j] != b'"' {
                        j += if b[j] == b'\\' { 2 } else { 1 };
                    }
                    if j >= b.len() {
                        None
                    } else {
                        i = j + 1;
                        match serde_json::from_str::<String>(&text[start..i]) {
                            Ok(n) => {
                                name = serde_json::Value::String(n).to_string();
                                Some(true)
                            }
                            Err(_) => Some(false),
                        }
                    }
                }
            }
            3 => lit(&mut i, b","),
            _ => lit(&mut i, b"\"args\"").and_then(|ok| {
                ws(&mut i);
                if ok { lit(&mut i, b":") } else { Some(false) }
            }),
        };
        match ok {
            Some(true) => {}
            Some(false) => return Prefix::Other,
            None if text.len() > PREFIX_MAX => return Prefix::Other,
            None => return Prefix::More,
        }
    }
    Prefix::Done { name, end: i }
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
