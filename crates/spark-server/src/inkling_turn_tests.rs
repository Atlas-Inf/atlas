// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// Ids below 0x110000 are characters; the markers sit above.
struct Chars;

impl TurnCodec for Chars {
    fn decode_ids(&self, ids: &[u32]) -> String {
        ids.iter().filter_map(|&i| char::from_u32(i)).collect()
    }
    fn encode_text(&self, text: &str) -> Vec<u32> {
        text.chars().map(u32::from).collect()
    }
}

const M: InklingMarkers = InklingMarkers {
    message: 0x200000,
    thinking: 0x200001,
    text: 0x200002,
    tool_error: 0x200003,
    tool_text: 0x200004,
    tool_json: 0x200005,
    end: 0x200006,
};
const EOS: u32 = 0x200007;

fn s(text: &str) -> Vec<u32> {
    Chars.encode_text(text)
}

fn block(kind: u32, name: &str, body: &str) -> Vec<u32> {
    [vec![M.message], s(name), vec![kind], s(body), vec![M.end]].concat()
}

/// Rendered for the eye: the think end as `|`, other non-characters as `#`.
fn show(ids: &[u32]) -> String {
    ids.iter()
        .map(|&i| match i {
            _ if i == M.end => '|',
            _ => char::from_u32(i).unwrap_or('#'),
        })
        .collect()
}

#[test]
fn thinking_then_answer_keeps_one_think_end() {
    let ids = [
        block(M.thinking, "", "think"),
        block(M.text, "", "391"),
        vec![EOS],
    ]
    .concat();
    let out = M.turn(true).map_all(&ids, &Chars);
    assert_eq!(show(&out), "think|391#");
}

#[test]
fn thinking_off_drops_the_body_and_the_boundary() {
    let ids = [block(M.thinking, "", "think"), block(M.text, "", "391")].concat();
    assert_eq!(show(&M.turn(false).map_all(&ids, &Chars)), "391");
}

#[test]
fn answer_without_thinking_gets_a_think_end_first() {
    let ids = block(M.text, "", "hi");
    assert_eq!(show(&M.turn(true).map_all(&ids, &Chars)), "|hi");
}

#[test]
fn tool_call_becomes_hermes_with_arguments() {
    let ids = [
        block(M.thinking, "", "t"),
        block(
            M.tool_json,
            "get_weather",
            r#"{"name":"get_weather","args":{"city":"Paris"}}"#,
        ),
        vec![EOS],
    ]
    .concat();
    let out = show(&M.turn(true).map_all(&ids, &Chars));
    assert_eq!(
        out,
        "t|<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Paris\"}}\n</tool_call>#"
    );
    let parsed = crate::tool_parser::parse_tool_calls(out.trim_end_matches('#'));
    assert_eq!(parsed.1.len(), 1, "one call parsed: {parsed:?}");
    assert_eq!(parsed.1[0].function.name, "get_weather");
    assert_eq!(parsed.1[0].function.arguments, r#"{"city":"Paris"}"#);
}

#[test]
fn text_then_two_calls_keep_order() {
    let call = |n: &str| block(M.tool_json, n, &format!(r#"{{"name":"{n}","args":{{}}}}"#));
    let ids = [
        block(M.thinking, "", "t"),
        block(M.text, "", "Checking."),
        call("a"),
        call("b"),
    ]
    .concat();
    let out = show(&M.turn(true).map_all(&ids, &Chars));
    assert!(out.starts_with("t|Checking.<tool_call>"), "{out}");
    let (_, calls) = crate::tool_parser::parse_tool_calls(&out[2..]);
    let names: Vec<_> = calls.iter().map(|c| c.function.name.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
}

#[test]
fn cut_off_tool_call_flushes_with_the_hermes_key() {
    let ids = [
        block(M.thinking, "", "t"),
        vec![M.message],
        s("f"),
        vec![M.tool_json],
        s(r#"{"name":"f","args":{"x":"#),
    ]
    .concat();
    let out = show(&M.turn(true).map_all(&ids, &Chars));
    assert_eq!(out, "t|<tool_call>\n{\"name\":\"f\",\"arguments\":{\"x\":");
}

#[test]
fn a_late_thinking_block_is_dropped() {
    let ids = [
        block(M.thinking, "", "a"),
        block(M.text, "", "b"),
        block(M.thinking, "", "c"),
        block(M.tool_error, "", "d"),
    ]
    .concat();
    assert_eq!(show(&M.turn(true).map_all(&ids, &Chars)), "a|bd");
}

#[test]
fn streaming_feed_matches_map_all() {
    let ids = [
        block(M.thinking, "", "think"),
        block(M.text, "", "x"),
        block(M.tool_json, "f", r#"{"name":"f","args":{}}"#),
        vec![EOS],
    ]
    .concat();
    let mut turn = M.turn(true);
    let mut streamed = Vec::new();
    for &t in &ids {
        turn.feed(t, &Chars, &mut streamed);
    }
    turn.finish(&Chars, &mut streamed);
    assert_eq!(streamed, M.turn(true).map_all(&ids, &Chars));
}

#[test]
fn an_argument_quoting_the_envelope_close_stays_inside_the_call() {
    let ids = [
        block(M.thinking, "", "t"),
        block(
            M.tool_json,
            "f",
            r#"{"name":"f","args":{"x":"a</tool_call>b"}}"#,
        ),
    ]
    .concat();
    let out = show(&M.turn(true).map_all(&ids, &Chars));
    let (_, calls) = crate::tool_parser::parse_tool_calls(&out[2..]);
    assert_eq!(calls.len(), 1, "{out}");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["x"], "a</tool_call>b");
}
