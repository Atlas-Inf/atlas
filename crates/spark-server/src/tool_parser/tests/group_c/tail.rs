// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn streaming_detector_preserves_inline_literal_opener_split_across_chunks() {
    let mut detector = StreamingToolDetector::new();
    let mut outputs = Vec::new();
    for chunk in [
        "The opening tag is `",
        "<tool_",
        "call>",
        "` and this is prose.",
    ] {
        outputs.extend(detector.process(chunk));
    }
    outputs.extend(detector.flush());

    let content: String = outputs
        .into_iter()
        .filter_map(|output| match output {
            DetectorOutput::Content(text) => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(
        content,
        "The opening tag is `<tool_call>` and this is prose."
    );
    assert!(!detector.has_tool_calls());
}

#[test]
fn streaming_detector_preserves_fenced_literal_opener() {
    let mut detector = StreamingToolDetector::new();
    let mut outputs = detector.process("```xml\n<tool_call>\n```");
    outputs.extend(detector.flush());

    let content: String = outputs
        .into_iter()
        .filter_map(|output| match output {
            DetectorOutput::Content(text) => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(content, "```xml\n<tool_call>\n```");
    assert!(!detector.has_tool_calls());
}

#[test]
fn streaming_detector_falls_back_for_plain_prose_xml_marker() {
    let mut detector = StreamingToolDetector::new();
    let mut outputs = Vec::new();
    for chunk in [
        "The delimiter ",
        "<tool_",
        "call>",
        " opens a tool invocation. END.",
    ] {
        outputs.extend(detector.process(chunk));
    }
    outputs.extend(detector.flush());

    let content: String = outputs
        .into_iter()
        .filter_map(|output| match output {
            DetectorOutput::Content(text) => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(
        content,
        "The delimiter <tool_call> opens a tool invocation. END."
    );
    assert!(!detector.has_tool_calls());
}

#[test]
fn streaming_detector_preserves_closed_non_call_xml_block() {
    let mut detector = StreamingToolDetector::new();
    let outputs =
        detector.process("The literal block is <tool_call>ordinary prose</tool_call> END.");
    let mut outputs = outputs;
    outputs.extend(detector.flush());

    let content: String = outputs
        .into_iter()
        .filter_map(|output| match output {
            DetectorOutput::Content(text) => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(
        content,
        "The literal block is <tool_call>ordinary prose</tool_call> END."
    );
    assert!(!detector.has_tool_calls());
}

#[test]
fn streaming_detector_preserves_literal_bare_function_before_real_call() {
    let mut detector = StreamingToolDetector::new();
    let mut outputs = Vec::new();
    for chunk in [
        "The example is `",
        "<function=demo>",
        "`; then ",
        "<function=real><parameter=value>ok</parameter></function>",
    ] {
        outputs.extend(detector.process(chunk));
    }
    outputs.extend(detector.flush());

    let content: String = outputs
        .iter()
        .filter_map(|output| match output {
            DetectorOutput::Content(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(content.contains("The example is `<function=demo>`; then"));
    assert!(outputs.iter().any(|output| matches!(
        output,
        DetectorOutput::ToolCall(tc, _) if tc.function.name == "real"
    )));
}

#[test]
fn streaming_detector_keeps_chunked_registered_call_structural() {
    let tool = ToolDefinition {
        tool_type: "function".into(),
        function: FunctionDefinition {
            name: "get_weather".into(),
            description: None,
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
            })),
        },
    };
    let mut detector = StreamingToolDetector::new_with_tools(vec![tool]);
    detector.set_promote_bare_names(true);
    let mut outputs = Vec::new();
    for chunk in [
        "<tool_",
        "call>get_",
        "weather<arg_key>location</arg_key><arg_value>Paris",
        "</arg_value></tool_call>",
    ] {
        outputs.extend(detector.process(chunk));
    }
    outputs.extend(detector.flush());

    assert!(outputs.iter().any(|output| matches!(
        output,
        DetectorOutput::ToolCallStart { name, .. } if name == "get_weather"
    )));
    assert!(
        outputs
            .iter()
            .any(|output| matches!(output, DetectorOutput::ToolCallEnd { .. }))
    );
    assert!(detector.has_tool_calls());
}

#[test]
fn blocking_parser_preserves_prose_marker_and_scans_past_quoted_marker() {
    let prose = "The delimiter <tool_call> opens a tool invocation. END.";
    let (content, calls) = parse_tool_calls(prose);
    assert_eq!(content.as_deref(), Some(prose));
    assert!(calls.is_empty());

    let mixed = "The tag is `<tool_call>`; now call: <tool_call>\
                 {\"name\":\"get_weather\",\"arguments\":{}}</tool_call>";
    let (content, calls) = parse_tool_calls(mixed);
    assert!(
        content
            .as_deref()
            .is_some_and(|text| text.contains("`<tool_call>`"))
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
}

#[test]
fn bare_function_parser_skips_literal_marker_before_real_call() {
    let input = "The example is `<function=demo>`; then call: \
                 <function=real><parameter=value>ok</parameter></function>";
    let (content, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "real");
    assert_eq!(
        content.as_deref(),
        Some("The example is `<function=demo>`; then call:")
    );
}

#[test]
fn blocking_parser_scans_past_malformed_opener_to_later_call() {
    let input = "prefix <tool_call><arg_key>unfinished <tool_call>\
                 {\"name\":\"get_weather\",\"arguments\":{}}</tool_call>";
    let (content, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert!(
        content
            .as_deref()
            .is_some_and(|text| text.contains("<tool_call><arg_key>unfinished "))
    );
}

#[test]
fn blocking_malformed_opener_skips_quoted_marker_before_later_call() {
    let input = "prefix <tool_call><arg_key>unfinished \"<tool_call>\" then \\
                 <tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>";
    let (content, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert!(
        content
            .as_deref()
            .is_some_and(|text| text.contains("\"<tool_call>\" then"))
    );
}

#[test]
fn whitespace_before_ambiguous_markup_is_preserved_until_the_tag_is_known() {
    for text in [
        "The delimiter <tool_call> opens a tool invocation. END.",
        "```xml\n<tool_call>\n```\nEND",
    ] {
        let mut detector = StreamingToolDetector::new();
        let mut outputs = Vec::new();
        for ch in text.chars() {
            outputs.extend(detector.process(&ch.to_string()));
        }
        outputs.extend(detector.flush());
        let content: String = outputs
            .into_iter()
            .filter_map(|output| match output {
                DetectorOutput::Content(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(content, text);
        assert!(!detector.has_tool_calls());
    }
}
