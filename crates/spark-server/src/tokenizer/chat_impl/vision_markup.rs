// SPDX-License-Identifier: AGPL-3.0-only

//! Vision markup written in message text is text.
//!
//! The prompt is rendered to one string and encoded with added-token matching
//! on, so `<|image_pad|>` typed into a message (pasted source code, a chat
//! template under discussion) became the placeholder token itself. The engine
//! attributes placeholders to media items by counting them: a stray one in a
//! request that carries an image put the splice past the image's encoded rows,
//! which on a two-rank world ended both ranks (a ~68K-token code paste with
//! three such strings did), and a stray one before the image would have moved
//! the image to the stray's place.
//!
//! For a request with media, [`ChatTokenizer::escape_vision_markup`] replaces every vision
//! marker inside the messages' strings with a sentinel before rendering, and
//! [`ChatTokenizer::encode_rendered`] encodes each sentinel as the marker's
//! plain-text spelling. Only the template's own markers stay placeholders. A
//! sentinel sits where an added-token match already split the string, so every
//! other token is exactly what the whole-string encode gives.
//! `ATLAS_VISION_MARKUP_AS_TEXT=0` turns this off (markers in text stay
//! special tokens, as before).

use std::sync::OnceLock;

use anyhow::Result;

use super::super::ChatTokenizer;

/// The vision markers the server's own placeholder expansion handles: Qwen's
/// (`expand_qwen_vision_placeholders`) and GLM-5's
/// (`expand_glm5_vision_placeholders`).
/// A tokenizer lacking one leaves it out ([`ChatTokenizer::vision_markup`]).
const VISION_MARKUP: &[&str] = &[
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|begin_of_image|>",
    "<|end_of_image|>",
    "<|begin_of_video|>",
    "<|end_of_video|>",
    "<|image|>",
    "<|video|>",
];

/// Sentinel delimiters: Unicode noncharacters, which never appear in
/// interchanged text.
const OPEN: char = '\u{FDD0}';
const CLOSE: char = '\u{FDD1}';

/// The most digits a sentinel's marker index has. Caller text can carry a bare
/// opener, so the search for its close must stop here, not at the prompt's end.
const MAX_INDEX_DIGITS: usize = 20;

/// Whether markers in the text of media requests are escaped. Read once.
fn markup_as_text() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_VISION_MARKUP_AS_TEXT").as_deref() != Ok("0"))
}

/// Replace every `markup[i]` inside `value`'s strings with sentinel `i`.
/// Returns how many it replaced.
fn escape_in_value(value: &mut serde_json::Value, markup: &[String]) -> usize {
    match value {
        serde_json::Value::String(s) => replace_markers(s, markup, OPEN, CLOSE),
        serde_json::Value::Array(items) => {
            items.iter_mut().map(|v| escape_in_value(v, markup)).sum()
        }
        serde_json::Value::Object(map) => {
            map.values_mut().map(|v| escape_in_value(v, markup)).sum()
        }
        _ => 0,
    }
}

fn replace_markers(s: &mut String, markup: &[String], open: char, close: char) -> usize {
    let mut n = 0;
    for (i, m) in markup.iter().enumerate() {
        let hits = s.matches(m.as_str()).count();
        if hits > 0 {
            *s = s.replace(m.as_str(), &format!("{open}{i}{close}"));
            n += hits;
        }
    }
    n
}

#[derive(Clone, Copy)]
pub(super) struct SentinelFamily<'a> {
    pub(super) open: char,
    pub(super) tag: Option<char>,
    pub(super) close: char,
    pub(super) markup: &'a [String],
}

/// Encode `rendered`, spelling each sentinel's marker as text. Without a
/// sentinel this is `encode(rendered)`. A malformed sentinel stays text.
fn encode_with_sentinels(
    rendered: &str,
    markup: &[String],
    encode: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<Vec<u32>> {
    encode_with_sentinel_families(
        rendered,
        &[SentinelFamily {
            open: OPEN,
            tag: None,
            close: CLOSE,
            markup,
        }],
        encode,
    )
}

pub(super) fn encode_with_sentinel_families(
    rendered: &str,
    families: &[SentinelFamily<'_>],
    mut encode: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<Vec<u32>> {
    // Each family's next opener: a family with none must not be searched for
    // again per sentinel.
    let mut next_open: Vec<_> = families.iter().map(|f| rendered.find(f.open)).collect();
    if next_open.iter().all(Option::is_none) {
        return encode(rendered);
    }
    let mut out = Vec::new();
    let mut text_from = 0usize;
    let mut search_from = 0usize;
    loop {
        for (family, next) in families.iter().zip(&mut next_open) {
            if next.is_some_and(|open| open < search_from) {
                *next = rendered[search_from..]
                    .find(family.open)
                    .map(|offset| search_from + offset);
            }
        }
        let Some((open, family_index)) = next_open
            .iter()
            .enumerate()
            .filter_map(|(index, next)| next.map(|open| (open, index)))
            .min_by_key(|(open, _)| *open)
        else {
            break;
        };
        let family = families[family_index];
        let body_start = open + family.open.len_utf8();
        let tag_len = family.tag.map_or(0, char::len_utf8);
        if let Some(tag) = family.tag
            && !rendered[body_start..].starts_with(tag)
        {
            search_from = body_start;
            continue;
        }
        let body = body_start + tag_len;
        let close = rendered[body..]
            .char_indices()
            .take(MAX_INDEX_DIGITS + 1)
            .find(|&(_, c)| c == family.close);
        let marker = close.and_then(|(len, _)| {
            let i: usize = rendered[body..body + len].parse().ok()?;
            Some((family.markup.get(i)?, body + len + family.close.len_utf8()))
        });
        let Some((marker, next)) = marker else {
            search_from = body;
            continue;
        };
        if open > text_from {
            out.extend(encode(&rendered[text_from..open])?);
        }
        // Split before the last character: neither piece is an added token,
        // so the marker comes out as ordinary text tokens.
        let cut = marker.len() - marker.chars().last().map_or(0, char::len_utf8);
        out.extend(encode(&marker[..cut])?);
        out.extend(encode(&marker[cut..])?);
        text_from = next;
        search_from = next;
    }
    if text_from < rendered.len() {
        out.extend(encode(&rendered[text_from..])?);
    }
    Ok(out)
}

impl ChatTokenizer {
    /// Escape the vision markup inside `values` (a media request's messages
    /// and tool definitions, before rendering) so [`Self::encode_rendered`]
    /// spells it as text. Returns how many markers it escaped; none with
    /// `ATLAS_VISION_MARKUP_AS_TEXT=0`.
    pub(crate) fn escape_vision_markup<'a>(
        &self,
        values: impl IntoIterator<Item = &'a mut serde_json::Value>,
    ) -> usize {
        if !markup_as_text() {
            return 0;
        }
        let markup = self.vision_markup();
        values
            .into_iter()
            .map(|v| escape_in_value(v, &markup))
            .sum()
    }

    /// The [`VISION_MARKUP`] markers that are single added tokens here.
    fn vision_markup(&self) -> Vec<String> {
        VISION_MARKUP
            .iter()
            .filter(|m| self.encode(m).is_ok_and(|ids| ids.len() == 1))
            .map(|m| (*m).to_string())
            .collect()
    }

    /// Encode a rendered prompt whose message text may carry escaped vision
    /// markup ([`Self::escape_vision_markup`]).
    pub(super) fn encode_rendered(&self, rendered: &str) -> Result<Vec<u32>> {
        let has_glm_tool_sentinel = super::glm_tool_markup::contains_sentinel(self, rendered);
        if !rendered.contains(OPEN) && !has_glm_tool_sentinel {
            return self.encode(rendered);
        }
        let vision_markup = self.vision_markup();
        let glm_tool_markup = if has_glm_tool_sentinel {
            super::glm_tool_markup::markers(self)
        } else {
            Vec::new()
        };
        let mut families = vec![SentinelFamily {
            open: OPEN,
            tag: None,
            close: CLOSE,
            markup: &vision_markup,
        }];
        if has_glm_tool_sentinel {
            families.push(SentinelFamily {
                open: super::glm_tool_markup::OPEN,
                tag: Some(super::glm_tool_markup::TAG),
                close: super::glm_tool_markup::CLOSE,
                markup: &glm_tool_markup,
            });
        }
        encode_with_sentinel_families(rendered, &families, |s| self.encode(s))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CLOSE, OPEN, SentinelFamily, encode_with_sentinel_families, encode_with_sentinels,
        escape_in_value,
    };

    const PAD: u32 = 900;

    /// One id per byte, except the added token `<|image_pad|>` matched
    /// anywhere in the input (as the real tokenizer does).
    fn fake_encode(s: &str) -> anyhow::Result<Vec<u32>> {
        let mut out = Vec::new();
        for (i, part) in s.split("<|image_pad|>").enumerate() {
            if i > 0 {
                out.push(PAD);
            }
            out.extend(part.bytes().map(u32::from));
        }
        Ok(out)
    }

    fn markup() -> Vec<String> {
        vec!["<|vision_start|>".into(), "<|image_pad|>".into()]
    }

    #[test]
    fn markup_in_message_text_encodes_as_text_and_template_markers_stay() {
        let mut msgs = serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "let pad = \"<|image_pad|>\";"},
                {"type": "image"},
            ]}
        ]);
        assert_eq!(escape_in_value(&mut msgs, &markup()), 1);
        let text = msgs[0]["content"][0]["text"].as_str().unwrap();
        // The template emits its own marker for the image part.
        let rendered = format!("{text}<|image_pad|>");
        let ids = encode_with_sentinels(&rendered, &markup(), fake_encode).unwrap();
        assert_eq!(ids.iter().filter(|&&t| t == PAD).count(), 1, "{ids:?}");
        // The escaped marker reads back as the text the user wrote.
        let bytes: Vec<u8> = ids
            .iter()
            .filter(|&&t| t != PAD)
            .map(|&t| t as u8)
            .collect();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "let pad = \"<|image_pad|>\";"
        );
        assert_eq!(*ids.last().unwrap(), PAD, "the real placeholder stays last");
    }

    #[test]
    fn a_prompt_without_sentinels_encodes_as_one_string() {
        let rendered = "a<|image_pad|>b";
        assert_eq!(
            encode_with_sentinels(rendered, &markup(), fake_encode).unwrap(),
            fake_encode(rendered).unwrap()
        );
    }

    #[test]
    fn a_malformed_sentinel_stays_text() {
        let rendered = "x\u{FDD0}7\u{FDD1}y\u{FDD0}z";
        assert_eq!(
            encode_with_sentinels(rendered, &markup(), fake_encode).unwrap(),
            fake_encode(rendered).unwrap()
        );
    }

    /// A rescan per sentinel takes tens of seconds here; the bound leaves
    /// room for a loaded machine.
    fn assert_quick(rendered: &str, families: &[SentinelFamily<'_>]) {
        let start = std::time::Instant::now();
        let ids =
            encode_with_sentinel_families(rendered, families, |s| Ok(vec![0; s.len().min(1)]))
                .unwrap();
        assert!(!ids.is_empty());
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "took {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn many_sentinels_do_not_rescan_a_family_with_none() {
        let markup = markup();
        let rendered = "\u{FDD2}\u{FDD4}0\u{FDD3}".repeat(1 << 18);
        let families = [
            SentinelFamily {
                open: '\u{FDD0}',
                tag: None,
                close: '\u{FDD1}',
                markup: &markup,
            },
            SentinelFamily {
                open: '\u{FDD2}',
                tag: Some('\u{FDD4}'),
                close: '\u{FDD3}',
                markup: &markup,
            },
        ];
        assert_quick(&rendered, &families);
    }

    #[test]
    fn unclosed_openers_do_not_rescan_for_a_close() {
        let markup = markup();
        let rendered = "\u{FDD0}".repeat(1 << 19);
        assert_quick(
            &rendered,
            &[SentinelFamily {
                open: '\u{FDD0}',
                tag: None,
                close: '\u{FDD1}',
                markup: &markup,
            }],
        );
    }

    #[test]
    fn a_body_too_long_for_an_index_stays_text() {
        let rendered = format!("x{OPEN}{}1{CLOSE}y", "0".repeat(32));
        assert_eq!(
            encode_with_sentinels(&rendered, &markup(), fake_encode).unwrap(),
            fake_encode(&rendered).unwrap()
        );
    }

    #[test]
    fn every_string_of_the_request_is_escaped() {
        let mut v = serde_json::json!({
            "messages": [{"role": "system", "content": "<|vision_start|><|image_pad|>"}],
            "tools": [{"description": "emits <|image_pad|>"}],
            "n": 3,
        });
        assert_eq!(escape_in_value(&mut v, &markup()), 3);
        assert!(!v.to_string().contains("<|"));
    }
}
