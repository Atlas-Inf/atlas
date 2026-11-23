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
    assert_eq!(
        out,
        "t|<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\":"
    );
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

/// At thinking effort 0 the checkpoint opens a call with a text marker before
/// the name: `<|message_model|><|content_text|>NAME<|content_invoke_tool_json|>{…}`.
fn effort0_call(name: &str, body: &str) -> Vec<u32> {
    [
        vec![M.message, M.text],
        s(name),
        vec![M.tool_json],
        s(body),
        vec![M.end],
    ]
    .concat()
}

#[test]
fn a_call_opened_as_text_is_still_a_call() {
    let ids = [
        block(M.thinking, "", "t"),
        effort0_call("Say", r#"{"name":"Say","args":{"text":"Hi"}}"#),
        vec![EOS],
    ]
    .concat();
    for keep in [true, false] {
        let out = show(&M.turn(keep).map_all(&ids, &Chars));
        let body = out.trim_start_matches("t|").trim_end_matches('#');
        let (rest, calls) = crate::tool_parser::parse_tool_calls(body);
        assert_eq!(calls.len(), 1, "{out}");
        assert_eq!(calls[0].function.name, "Say");
        assert_eq!(calls[0].function.arguments, r#"{"text":"Hi"}"#);
        assert!(
            !rest.unwrap_or_default().contains("Say"),
            "no name left in the text: {out}"
        );
    }
}

#[test]
fn a_call_opened_as_text_streams_as_a_call() {
    let ids = [
        block(M.thinking, "", "t"),
        effort0_call("Say", r#"{"name":"Say","args":{"text":"Hi"}}"#),
        vec![EOS],
    ]
    .concat();
    let mut turn = M.turn(false);
    let mut streamed = Vec::new();
    for &t in &ids {
        turn.feed(t, &Chars, &mut streamed);
        // the name is never emitted as answer text ahead of the call
        assert!(!show(&streamed).starts_with("Say"), "{}", show(&streamed));
    }
    turn.finish(&Chars, &mut streamed);
    assert_eq!(streamed, M.turn(false).map_all(&ids, &Chars));
}

#[test]
fn answers_that_start_like_a_name_pass_unchanged() {
    for body in ["Hello there.", "391", "Paris", "ok_then fine"] {
        let ids = [
            block(M.thinking, "", "t"),
            block(M.text, "", body),
            vec![EOS],
        ]
        .concat();
        assert_eq!(
            show(&M.turn(true).map_all(&ids, &Chars)),
            format!("t|{body}#")
        );
        let mut turn = M.turn(true);
        let mut streamed = Vec::new();
        for &t in &ids {
            turn.feed(t, &Chars, &mut streamed);
        }
        turn.finish(&Chars, &mut streamed);
        assert_eq!(show(&streamed), format!("t|{body}#"), "streamed {body}");
    }
}

/// The engine's budget/loop close is a bare `<|end_message|>` inside the
/// thinking body (`SchedLimits::message_open`); the model then opens its own
/// next block. Either way the rewrite closes the reasoning there, once.
fn forced_close(body: &str) -> Vec<u32> {
    [vec![M.message, M.thinking], s(body), vec![M.end]].concat()
}

#[test]
fn a_forced_close_then_an_answer_block() {
    let ids = [
        forced_close("let me reconsider"),
        block(M.text, "", "391"),
        vec![EOS],
    ]
    .concat();
    for keep in [true, false] {
        let want = if keep {
            "let me reconsider|391#"
        } else {
            "391#"
        };
        assert_eq!(show(&M.turn(keep).map_all(&ids, &Chars)), want);
    }
}

#[test]
fn a_forced_close_then_a_tool_block() {
    let ids = [
        forced_close("t"),
        block(M.tool_json, "f", r#"{"name":"f","args":{"x":1}}"#),
        vec![EOS],
    ]
    .concat();
    let out = show(&M.turn(true).map_all(&ids, &Chars));
    assert!(out.starts_with("t|<tool_call>"), "{out}");
    let (_, calls) = crate::tool_parser::parse_tool_calls(out[2..].trim_end_matches('#'));
    assert_eq!(calls.len(), 1, "{out}");
    assert_eq!(calls[0].function.arguments, r#"{"x":1}"#);
}

#[test]
fn a_forced_close_then_headerless_text_is_the_answer() {
    // Should the model write on without a header, that text is the answer.
    let ids = [forced_close("t"), s("391"), vec![EOS]].concat();
    assert_eq!(show(&M.turn(true).map_all(&ids, &Chars)), "t|391#");
}

/// A grammar-forced call (`tool_choice` required or a named function) is
/// written as Hermes text with no block header at all. It is the answer: with
/// thinking on, the think end comes first, or the stream reads the call as
/// reasoning and the in-think leak guard cuts the response.
#[test]
fn headerless_text_with_no_block_before_it_is_the_answer() {
    let ids = [s("<tool_call>x</tool_call>"), vec![EOS]].concat();
    assert_eq!(
        show(&M.turn(true).map_all(&ids, &Chars)),
        "|<tool_call>x</tool_call>#"
    );
    assert_eq!(
        show(&M.turn(false).map_all(&ids, &Chars)),
        "<tool_call>x</tool_call>#"
    );
}

/// With thinking off the scheduler does not stream the reasoning body (the
/// header before it and its close are streamed); the rewrite reads the same
/// answer as from the whole turn.
#[test]
fn a_suppressed_thinking_body_streams_the_same_answer() {
    let whole = [forced_close("t"), block(M.text, "", "391"), vec![EOS]].concat();
    let streamed = [vec![M.message, M.end], block(M.text, "", "391"), vec![EOS]].concat();
    let mut turn = M.turn(false);
    let mut out = Vec::new();
    for &t in &streamed {
        turn.feed(t, &Chars, &mut out);
    }
    turn.finish(&Chars, &mut out);
    assert_eq!(out, M.turn(false).map_all(&whole, &Chars));
    assert_eq!(show(&out), "391#");
}

/// Budget forcing for Inkling, as its MODEL.toml arms it: a server ceiling
/// the effort ladder scales (2K/4K/8K/16K/32K), the thinking-loop watchdog,
/// speculation inside the reasoning it always speculated, an end token that
/// ends the turn wherever it comes (as before reasoning was tracked), and no
/// A4 floor holding back the model's own close.
#[test]
fn inkling_model_toml_arms_budget_forcing() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10/inkling-2-mini");
    let b = crate::api::test_app_state::build_behavior::parse_behavior(&dir);
    assert!(b.thinking_default, "actual Inkling metadata read");
    assert_eq!(b.max_thinking_budget, 8192);
    assert!(b.enable_think_loop_watchdog);
    assert!(b.spec_think);
    assert!(b.eos_ends_thinking);
    assert_eq!(b.min_reasoning_floor_tokens, 0);
}

/// Every stream of a call: the rewrite token by token, then the tool parser
/// fed the rewrite's text as each token arrives. Returns the argument bytes the
/// parser streamed and how many arrived before the block's close.
fn stream_call(body: &str) -> (String, usize, Vec<u32>) {
    let ids = [block(M.thinking, "", "t"), block(M.tool_json, "f", body)].concat();
    let close = ids.len() - 1;
    let mut turn = M.turn(true);
    let mut det = crate::tool_parser::StreamingToolDetector::new();
    let (mut streamed, mut args, mut early) = (Vec::new(), String::new(), 0);
    for (i, &t) in ids.iter().enumerate() {
        let mut out = Vec::new();
        turn.feed(t, &Chars, &mut out);
        let text: String = out
            .iter()
            .filter(|&&i| i != M.end)
            .filter_map(|&i| char::from_u32(i))
            .collect();
        for o in det.process(&text) {
            if let crate::tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, .. } = o {
                if i < close {
                    early += fragment.len();
                }
                args.push_str(&fragment);
            }
        }
        streamed.extend(out);
    }
    turn.finish(&Chars, &mut streamed);
    assert_eq!(
        streamed,
        M.turn(true).map_all(&ids, &Chars),
        "streamed = whole"
    );
    (args, early, streamed)
}

#[test]
fn call_arguments_stream_before_the_close() {
    let body = r#"{"name":"Say","args":{"text":"Hello there, how are you?"}}"#;
    let (args, early, streamed) = stream_call(body);
    assert_eq!(args, r#"{"text":"Hello there, how are you?"}"#);
    assert!(
        early >= r#"{"text":"Hello there"#.len(),
        "{early} bytes before the close"
    );
    let whole = format!("t|<tool_call>\n{}\n</tool_call>", hermes_body(body));
    assert_eq!(
        show(&streamed),
        whole,
        "same bytes as the whole-call rewrite"
    );
}

#[test]
fn a_streamed_argument_quoting_the_envelope_close_is_escaped() {
    let body = r#"{"name":"f","args":{"x":"a</tool_call>b<c"}}"#;
    let (args, _, streamed) = stream_call(body);
    let v: serde_json::Value = serde_json::from_str(&args).unwrap();
    assert_eq!(v["x"], "a</tool_call>b<c");
    assert_eq!(
        show(&streamed),
        format!("t|<tool_call>\n{}\n</tool_call>", hermes_body(body))
    );
}

#[test]
fn a_streamed_call_name_quoting_the_envelope_close_is_escaped() {
    let body = r#"{"name":"f</tool_call>","args":{}}"#;
    let (_, _, streamed) = stream_call(body);
    assert_eq!(
        show(&streamed),
        format!("t|<tool_call>\n{}\n</tool_call>", hermes_body(body))
    );
}

#[test]
fn a_special_token_inside_streamed_arguments_goes() {
    let ids = [
        block(M.thinking, "", "t"),
        vec![M.message],
        s("f"),
        vec![M.tool_json],
        s(r#"{"name":"f","args":{"x":"#),
        vec![EOS],
        s(r#"1}}"#),
        vec![M.end],
    ]
    .concat();
    assert_eq!(
        show(&M.turn(true).map_all(&ids, &Chars)),
        "t|<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\":1}}\n</tool_call>"
    );
}

#[test]
fn spaced_call_json_streams_too() {
    let body = r#"{ "name" : "f" , "args" : {"n": 1, "s": "x"} }"#;
    let (args, early, _) = stream_call(body);
    let v: serde_json::Value = serde_json::from_str(&args).unwrap();
    assert_eq!(v, serde_json::json!({"n": 1, "s": "x"}));
    assert!(early > 0);
}

#[test]
fn other_call_shapes_are_rewritten_whole_at_the_close() {
    for body in [
        r#"{"args":{"x":1},"name":"f"}"#,
        r#"{"name":"f"}"#,
        "not json",
    ] {
        let ids = [block(M.thinking, "", "t"), block(M.tool_json, "f", body)].concat();
        assert_eq!(
            show(&M.turn(true).map_all(&ids, &Chars)),
            format!("t|<tool_call>\n{}\n</tool_call>", hermes_body(body)),
            "{body}"
        );
    }
}

#[test]
fn call_prefix_reads_the_name_and_waits_for_more() {
    assert!(matches!(call_prefix(r#"{"na"#), Prefix::More));
    assert!(matches!(call_prefix(r#"{"name":"Sa"#), Prefix::More));
    assert!(matches!(call_prefix(r#"{"name":"Say","ar"#), Prefix::More));
    assert!(matches!(call_prefix(r#"{"args":"#), Prefix::Other));
    assert!(matches!(call_prefix(r#"{"name":1,"#), Prefix::Other));
    let Prefix::Done { name, end } = call_prefix(r#"{"name":"a\"b","args":{"#) else {
        panic!("prefix not read");
    };
    assert_eq!(name, r#""a\"b""#);
    assert_eq!(end, r#"{"name":"a\"b","args":"#.len());
}
