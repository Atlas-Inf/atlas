// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;

fn run(chunks: &[&str], buffered: bool) -> Vec<DetectorOutput> {
    let mut detector = StreamingToolDetector::new();
    detector.set_promote_bare_names(true);
    detector.buffer_args = buffered;
    let mut outputs = Vec::new();
    for chunk in chunks {
        outputs.extend(detector.process(chunk));
    }
    outputs.extend(detector.flush());
    outputs
}

fn assert_call(outputs: &[DetectorOutput], name: &str, args: &serde_json::Value) {
    let mut names = Vec::new();
    let mut argument_text = String::new();
    let mut complete = 0;
    for output in outputs {
        match output {
            DetectorOutput::ToolCall(call, idx) => {
                assert_eq!(*idx, 0);
                names.push(call.function.name.as_str());
                argument_text.push_str(&call.function.arguments);
                complete += 1;
            }
            DetectorOutput::ToolCallStart { name, idx, .. } => {
                assert_eq!(*idx, 0);
                names.push(name.as_str());
            }
            DetectorOutput::ToolCallDelta { args, idx } => {
                assert_eq!(*idx, 0);
                argument_text.push_str(args);
            }
            DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                assert_eq!(*idx, 0);
                argument_text.push_str(fragment);
            }
            DetectorOutput::ToolCallEnd { idx } => {
                assert_eq!(*idx, 0);
                complete += 1;
            }
            DetectorOutput::Content(text) => assert!(text.trim().is_empty(), "{text:?}"),
        }
    }
    assert_eq!(names, [name]);
    assert_eq!(complete, 1, "one completed call, without duplicates");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&argument_text).unwrap(),
        *args
    );
}

#[test]
fn native_header_precedes_nested_attachment_name() {
    let body = "<tool_call>ManageMessages<arg_key>attachments</arg_key><arg_value>\
                [{\"path\":\"song.wav\",\"name\":\"evening-piece.wav\"}]</arg_value>";
    let mut detector = StreamingToolDetector::new();
    detector.set_promote_bare_names(true);
    let mut outputs = detector.process(body);
    assert!(
        matches!(outputs.as_slice(), [DetectorOutput::ToolCallStart { name, .. }] if name == "ManageMessages")
    );
    outputs.extend(detector.process("</tool_call>"));
    // Poolside values stay as written; the API layer types them from the
    // tool schema (`coerce_all`), which this schema-less detector lacks.
    assert_call(
        &outputs,
        "ManageMessages",
        &serde_json::json!({
            "attachments": "[{\"path\":\"song.wav\",\"name\":\"evening-piece.wav\"}]"
        }),
    );
}

#[test]
fn native_arguments_match_blocking_parser_at_every_split() {
    for value in [
        r#"[{"name":"evening-piece.wav","path":"song.wav"}]"#,
        r#"[{"path":"song.wav","name":"evening-piece.wav"}]"#,
        r#"[{"path":"song.wav"}]"#,
        r#"{"name":"nested","arguments":{"inner":true},"markup":"<parameter=x>y</parameter>"}"#,
    ] {
        let full = format!(
            "<tool_call>namespace:ManageMessages<arg_key>attachments</arg_key><arg_value>{value}</arg_value></tool_call>"
        );
        let (_, expected) = parse_tool_calls_promoting_bare_names(&full);
        assert_eq!(expected.len(), 1);
        let args = serde_json::from_str(&expected[0].function.arguments).unwrap();
        for buffered in [false, true] {
            for split in 0..=full.len() {
                assert_call(
                    &run(&[&full[..split], &full[split..]], buffered),
                    "ManageMessages",
                    &args,
                );
            }
            let bytes: Vec<_> = (0..full.len()).map(|i| &full[i..i + 1]).collect();
            assert_call(&run(&bytes, buffered), "ManageMessages", &args);
        }
    }
}

#[test]
fn partial_header_waits_for_complete_arg_key() {
    let mut detector = StreamingToolDetector::new();
    detector.set_promote_bare_names(true);
    for chunk in ["<tool_call>Manage", "Messages", "<arg_", "key"] {
        assert!(detector.process(chunk).is_empty());
    }
    let outputs = detector.process(">");
    assert!(
        matches!(outputs.as_slice(), [DetectorOutput::ToolCallStart { name, .. }] if name == "ManageMessages")
    );
}

#[test]
fn closed_zero_argument_call_matches_every_split() {
    let full = "<tool_call>get_status</tool_call>";
    for split in 0..=full.len() {
        assert_call(
            &run(&[&full[..split], &full[split..]], false),
            "get_status",
            &serde_json::json!({}),
        );
    }
}

#[test]
fn malformed_native_header_never_uses_nested_name() {
    for body in [
        r#"invalid header<arg_key>x</arg_key><arg_value>{"name":"nested"}</arg_value>"#,
        r#"<arg_value>{"name":"nested"}</arg_value>"#,
        r#"ManageMessages<arg_value>{"name":"nested"}</arg_value>"#,
    ] {
        assert!(extract_streaming_name(body, true).is_none());
    }
}

#[test]
fn incomplete_native_envelope_is_not_completed_at_eos() {
    for body in [
        "<tool_call>get_status",
        "<tool_call>ManageMessages<arg_key>x</arg_key><arg_value>{}",
    ] {
        assert!(run(&[body], false).iter().all(|o| !matches!(
            o,
            DetectorOutput::ToolCall(..)
                | DetectorOutput::ToolCallEnd { .. }
                | DetectorOutput::ToolCallDelta { .. }
                | DetectorOutput::ToolCallArgsFragment { .. }
        )));
    }
}

#[test]
fn other_format_headers_keep_existing_names() {
    for (body, name) in [
        (r#"{"name":"Read","arguments":{"name":"nested"}}"#, "Read"),
        ("<function=Read><parameter=path>x</parameter>", "Read"),
        ("call:Read{", "Read"),
        ("[TOOL_CALLS]Read[ARGS]{", "Read"),
    ] {
        assert_eq!(extract_streaming_name(body, false).as_deref(), Some(name));
    }
}
