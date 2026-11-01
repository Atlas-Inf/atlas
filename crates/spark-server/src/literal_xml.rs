// SPDX-License-Identifier: AGPL-3.0-only

//! Small lexical state machine shared by tool-boundary and content paths.
//!
//! A model can mention the tool envelope in ordinary prose, for example
//! ``The tag is `<tool_call>`.``. The parser must only treat the opener as
//! structural when it is outside a quoted or code-formatted span. This is
//! deliberately not a Markdown parser; it only tracks the delimiters that
//! can make an XML-looking marker literal.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LiteralXmlState {
    code_fence: bool,
    inline_ticks: u8,
    quote: Option<u8>,
    escaped: bool,
    previous: Option<u8>,
    pending_backticks: u8,
}

impl LiteralXmlState {
    pub(crate) fn is_literal(self) -> bool {
        self.code_fence
            || self.inline_ticks != 0
            || self.quote.is_some()
            // Models often nest unescaped quotation marks while quoting a
            // sentence. A marker immediately touching either quote boundary
            // is still a literal mention, even if that quote closed a span.
            || matches!(self.previous, Some(b'"' | b'\''))
            // A delimiter run at the end of a chunk is conservatively
            // literal until the next non-backtick byte identifies its size.
            || self.pending_backticks != 0
    }

    pub(crate) fn feed(&mut self, text: &str) {
        self.feed_bytes(text.as_bytes());
    }

    /// Feed already-decoded tokenizer bytes. Byte-fallback tokens can form
    /// invalid UTF-8, but delimiter tracking only needs the raw bytes.
    pub(crate) fn feed_bytes(&mut self, bytes: &[u8]) {
        let mut i = 0;
        while i < bytes.len() {
            let byte = bytes[i];

            if self.escaped {
                self.escaped = false;
                self.previous = Some(byte);
                i += 1;
                continue;
            }
            if byte == b'\\' && (self.inline_ticks != 0 || self.quote.is_some()) {
                self.escaped = true;
                self.previous = Some(byte);
                i += 1;
                continue;
            }

            // Quotes have precedence over inline/fenced backticks. A quoted
            // string may mention markdown fences as ordinary XML-looking
            // text; letting those bytes toggle code_fence would hide the
            // next real native opener.
            if byte == b'`' && self.quote.is_none() {
                let run = bytes[i..]
                    .iter()
                    .take_while(|&&candidate| candidate == b'`')
                    .count();
                self.pending_backticks = self
                    .pending_backticks
                    .saturating_add(run.min(u8::MAX as usize) as u8);
                i += run;
                continue;
            }

            self.finish_pending_backticks();
            if !self.code_fence {
                if let Some(quote) = self.quote {
                    if byte == quote {
                        self.quote = None;
                    }
                } else if self.inline_ticks == 0
                    && (byte == b'"' || byte == b'\'')
                    && self.previous.is_none_or(|previous| {
                        previous.is_ascii_whitespace()
                            || matches!(previous, b'(' | b'[' | b'{' | b':' | b'=' | b',')
                    })
                {
                    self.quote = Some(byte);
                }
            }

            self.previous = Some(byte);
            i += 1;
        }
    }

    fn finish_pending_backticks(&mut self) {
        let run = std::mem::take(&mut self.pending_backticks);
        if run == 0 {
            return;
        }
        if run >= 3 {
            if self.code_fence {
                self.code_fence = false;
                self.inline_ticks = 0;
                self.quote = None;
            } else {
                self.code_fence = true;
                self.inline_ticks = 0;
                self.quote = None;
            }
        } else if !self.code_fence && self.quote.is_none() {
            if self.inline_ticks == 0 {
                self.inline_ticks = run;
            } else if self.inline_ticks == run {
                self.inline_ticks = 0;
            }
        }
        self.previous = Some(b'`');
    }

    /// Finalize a delimiter run when the caller is about to inspect a marker
    /// that follows the supplied prefix. Streaming `feed` keeps a trailing run
    /// pending so a fence split across chunks remains a single delimiter.
    pub(crate) fn finish_pending(&mut self) {
        self.finish_pending_backticks();
    }
}

/// Return the lexical state after `prefix`, starting outside all delimiters.
pub(crate) fn state_after(prefix: &str) -> LiteralXmlState {
    let mut state = LiteralXmlState::default();
    state.feed(prefix);
    state.finish_pending();
    state
}

/// Whether a marker immediately following `prefix` is inside a literal span.
pub(crate) fn marker_is_literal(prefix: &str) -> bool {
    state_after(prefix).is_literal()
}

/// Find the first marker whose position is outside a literal span. `initial`
/// represents text already emitted before `text`; it is copied for each
/// candidate so markers inside a quote do not hide a later real marker.
pub(crate) fn first_marker_outside_literal(
    text: &str,
    initial: LiteralXmlState,
    markers: &[&str],
) -> Option<(usize, usize)> {
    markers
        .iter()
        .flat_map(|marker| {
            text.match_indices(marker).filter_map(|(position, _)| {
                let mut state = initial;
                state.feed(&text[..position]);
                state.finish_pending();
                (!state.is_literal()).then_some((position, marker.len()))
            })
        })
        .min_by_key(|(position, _)| *position)
}

/// Replace the opening byte of markers that occur inside a quoted or fenced
/// span. Parsers that scan their own format markers can then keep their normal
/// first-match logic without mistaking an example in prose for a call. The
/// replacement is reversible with [`unmask_literal_markers`].
pub(crate) fn mask_literal_markers(text: &str, markers: &[&str]) -> String {
    mask_literal_markers_with_state(text, LiteralXmlState::default(), markers)
}

pub(crate) fn mask_literal_markers_with_state(
    text: &str,
    mut state: LiteralXmlState,
    markers: &[&str],
) -> String {
    // The escape is a two-character code so original private-use text cannot
    // be mistaken for a masked `<`. Original SENTINEL bytes are doubled;
    // masked markers use SENTINEL + MASKED_LT. This remains reversible even
    // when the original sentinel is adjacent to a masked marker.
    const SENTINEL: char = '\u{e000}';
    const MASKED_LT: char = '\u{e001}';
    fn append_escaped(output: &mut String, text: &str) {
        for ch in text.chars() {
            if ch == '\u{e000}' {
                output.push('\u{e000}');
                output.push('\u{e000}');
            } else {
                output.push(ch);
            }
        }
    }
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;

    while let Some((relative, marker)) = markers
        .iter()
        .filter_map(|marker| {
            text[cursor..]
                .find(marker)
                .map(|position| (position, *marker))
        })
        .min_by_key(|(position, _)| *position)
    {
        let position = cursor + relative;
        let before = &text[cursor..position];
        append_escaped(&mut output, before);
        state.feed(before);
        state.finish_pending();
        if state.is_literal() {
            output.push(SENTINEL);
            output.push(MASKED_LT);
            append_escaped(&mut output, &text[position + 1..position + marker.len()]);
        } else {
            append_escaped(&mut output, marker);
        }
        state.feed(marker);
        cursor = position + marker.len();
    }
    append_escaped(&mut output, &text[cursor..]);
    output
}

/// Replace `<` bytes that occur inside a quoted or fenced span with a
/// matcher-safe space while preserving the caller's lexical state separately.
/// This byte-level form is used for tokenizer vocab entries, which may contain
/// invalid UTF-8 and therefore cannot go through the String masking helper.
pub(crate) fn mask_literal_lt_bytes_with_state(
    bytes: &[u8],
    mut state: LiteralXmlState,
) -> Vec<u8> {
    let mut output = Vec::with_capacity(bytes.len());
    for &byte in bytes {
        if byte == b'<' {
            state.finish_pending();
            if state.is_literal() {
                output.push(b' ');
                state.feed_bytes(&[byte]);
                continue;
            }
        }
        output.push(byte);
        state.feed_bytes(&[byte]);
    }
    output
}

pub(crate) fn unmask_literal_markers(text: &str) -> String {
    const SENTINEL: char = '\u{e000}';
    const MASKED_LT: char = '\u{e001}';
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != SENTINEL {
            output.push(ch);
            continue;
        }
        match chars.next() {
            Some(next) if next == SENTINEL => output.push(SENTINEL),
            Some(next) if next == MASKED_LT => output.push('<'),
            Some(next) => {
                output.push(SENTINEL);
                output.push(next);
            }
            None => output.push(SENTINEL),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        LiteralXmlState, marker_is_literal, mask_literal_lt_bytes_with_state, mask_literal_markers,
        state_after, unmask_literal_markers,
    };

    #[test]
    fn marker_touching_nested_quote_boundary_is_literal() {
        assert!(marker_is_literal("\"The opening tag is \""));
        assert!(!marker_is_literal("\"finished quote\" "));
    }

    #[test]
    fn tracks_inline_backticks() {
        assert!(marker_is_literal("The tag is `"));
        assert!(!marker_is_literal("The tag is `literal` "));
    }

    #[test]
    fn tracks_double_backtick_inline_code() {
        assert!(marker_is_literal("The tag is ``"));
        assert!(!marker_is_literal("The tag is ``literal`` "));
    }

    #[test]
    fn tracks_fenced_code() {
        assert!(marker_is_literal("```xml\n"));
        assert!(!marker_is_literal("```xml\ntext\n```\n"));
    }

    #[test]
    fn split_backtick_runs_are_delimiters() {
        for first_len in 1..=2 {
            for second_len in 1..=2 {
                let mut state = LiteralXmlState::default();
                state.feed(&"`".repeat(first_len));
                state.feed(&"`".repeat(second_len));
                state.feed("<tool_call>");
                assert!(state.is_literal(), "split run {first_len}+{second_len}");
            }
        }

        for split in 1..=2 {
            let mut state = LiteralXmlState::default();
            state.feed(&"`".repeat(split));
            state.feed(&"`".repeat(3 - split));
            state.feed("xml\n<tool_call>");
            assert!(state.is_literal(), "fence split {split}+{}", 3 - split);
            state.feed("\n```");
            state.feed(" ");
            state.feed("<tool_call>");
            assert!(!state.is_literal(), "closed fence must allow a real call");
        }
    }

    #[test]
    fn tracks_quotes_without_treating_apostrophes_as_quotes() {
        assert!(marker_is_literal("The string is \""));
        assert!(marker_is_literal("The string is '"));
        assert!(!marker_is_literal("I don't mean a quoted tag. "));
    }

    #[test]
    fn state_can_continue_across_chunks() {
        let mut state = LiteralXmlState::default();
        state.feed("The tag is `");
        assert!(state.is_literal());
        state.feed("<tool_call>");
        assert!(state.is_literal());
        state.feed("` ");
        assert!(!state.is_literal());
    }

    #[test]
    fn escaped_delimiter_does_not_change_state() {
        let state = state_after(r#"The string is \"quoted\" and `code`."#);
        assert!(!state.is_literal());
    }

    #[test]
    fn quotes_keep_fence_bytes_literal_without_hiding_later_call() {
        assert!(marker_is_literal("The example is \"``` <tool_call>"));
        assert!(!marker_is_literal(
            "The example is \"``` text\" <tool_call>"
        ));
    }

    #[test]
    fn masks_only_literal_markers_and_keeps_later_real_marker() {
        let text = "The example is `<function=demo>`; <function=real>";
        let masked = mask_literal_markers(text, &["<function="]);
        assert_eq!(unmask_literal_markers(&masked), text);
        assert!(!masked.starts_with("The example is <function="));
        assert!(masked.ends_with("<function=real>"));
    }

    #[test]
    fn masking_preserves_private_use_text_adjacent_to_literal_marker() {
        let text = "`\u{e000}<function=demo>`; <function=real>";
        let masked = mask_literal_markers(text, &["<function="]);
        assert_eq!(unmask_literal_markers(&masked), text);
    }

    #[test]
    fn byte_mask_handles_multibyte_text_and_real_marker_after_quote() {
        let state = state_after("The marker is `");
        let masked = mask_literal_lt_bytes_with_state("<tool_call>é`<tool_call>".as_bytes(), state);
        assert_eq!(masked, " tool_call>é`<tool_call>".as_bytes());
    }
}
