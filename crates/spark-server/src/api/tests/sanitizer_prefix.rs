// SPDX-License-Identifier: AGPL-3.0-only

//! Latency and boundary contracts through the production sanitizer and flush.

use super::harness::Stream;
use crate::tool_parser::{
    DeepseekV4DsmlParser, Gemma4Parser, LeakMarkers, MinimaxXmlParser, PoolsideV1Parser,
    Qwen3CoderParser, Qwen3XmlParser, ToolCallParser,
};

fn marker_sets() -> [LeakMarkers; 6] {
    [
        PoolsideV1Parser.leak_markers(),
        Qwen3CoderParser.leak_markers(),
        MinimaxXmlParser.leak_markers(),
        Gemma4Parser.leak_markers(),
        DeepseekV4DsmlParser.leak_markers(),
        Qwen3XmlParser.leak_markers(),
    ]
}

#[test]
fn sanitizer_prefix_emits_harmless_first_token_immediately() {
    for markers in marker_sets() {
        for text in [
            "1",
            " ",
            "é",
            "🦀",
            "1. Distributed systems",
            "<xyz",
            "a < b",
        ] {
            let mut stream = Stream::new(&markers);
            assert_eq!(stream.feed(text), text, "first token {text:?}");
            assert!(stream.buffered().is_empty());
            assert_eq!(stream.finish(), text);
        }
    }
}

#[test]
fn sanitizer_prefix_withholds_every_marker_split_with_utf8_prose() {
    for markers in marker_sets() {
        for marker in markers
            .orphan_open
            .iter()
            .chain(markers.close)
            .chain(markers.envelope_open)
            .chain(markers.envelope_close)
        {
            for split in 1..marker.len() {
                if !marker.is_char_boundary(split) {
                    continue;
                }
                let mut stream = Stream::new(&markers);
                let prefix = format!("é🦀{}", &marker[..split]);
                assert_eq!(stream.feed(&prefix), "é🦀", "{marker:?} split {split}");
                assert_eq!(stream.buffered(), &marker[..split]);
                stream.feed(&format!("{}tail", &marker[split..]));
                let mut whole = Stream::new(&markers);
                whole.feed(&format!("é🦀{marker}tail"));
                assert_eq!(stream.finish(), whole.finish(), "{marker:?} split {split}");
            }
        }
    }
}

#[test]
fn sanitizer_prefix_preserves_suppression_and_envelopes_at_every_utf8_boundary() {
    let markers = PoolsideV1Parser.leak_markers();
    for (text, expected) in [
        ("é<arg_key>secret</arg_key>🦀", "é🦀"),
        ("é</arg_value>🦀", "é🦀"),
        (
            "é<tool_call><arg_key>x</arg_key><arg_value>🦀</arg_value></tool_call>fin",
            "é<tool_call><arg_key>x</arg_key><arg_value>🦀</arg_value></tool_call>fin",
        ),
        ("é<arg_value>unfinished secret", "é"),
        ("é</arg_val", "é"),
    ] {
        for split in text.char_indices().map(|(i, _)| i).chain([text.len()]) {
            let mut stream = Stream::new(&markers);
            stream.feed(&text[..split]);
            stream.feed(&text[split..]);
            assert_eq!(stream.finish(), expected, "{text:?} split {split}");
        }
        let mut stream = Stream::new(&markers);
        stream.feed_chunked(text, 1);
        assert_eq!(stream.finish(), expected);
    }
}

#[test]
fn sanitizer_prefix_releases_disproved_prefix_without_waiting_for_done() {
    let markers = PoolsideV1Parser.leak_markers();
    let mut stream = Stream::new(&markers);
    assert_eq!(stream.feed("é<arg_"), "é");
    assert_eq!(stream.feed("x🦀"), "<arg_x🦀");
    assert!(stream.buffered().is_empty());
    assert_eq!(stream.finish(), "é<arg_x🦀");
}

#[test]
fn sanitizer_prefix_selects_longest_overlap_and_handles_unicode_markers() {
    let markers = LeakMarkers {
        orphan_open: &["abab!", "bé!", "é🦀!"],
        close: &["END"],
        envelope_open: &[],
        envelope_close: &[],
    };
    for prefix in ["aba", "bé", "é🦀"] {
        let mut stream = Stream::new(&markers);
        assert_eq!(stream.feed(&format!("hello{prefix}")), "hello");
        assert_eq!(stream.buffered(), prefix);
        assert_eq!(stream.feed("?"), format!("{prefix}?"));
        assert_eq!(stream.finish(), format!("hello{prefix}?"));
    }
}
