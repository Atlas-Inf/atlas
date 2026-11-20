// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;

/// A completion as DeepSeek-V4.1 writes it: a space before each DSML tag name.
const CALL: &str = r#"answer

<｜DSML｜ calls>
<｜DSML｜ invoke name="files::read">
<｜DSML｜ parameter name="path" string="true">/tmp/x</｜DSML｜ parameter>
<｜DSML｜ parameter name="lines" string="false">[1, 2]</｜DSML｜ parameter>
</｜DSML｜ invoke>
<｜DSML｜ invoke name="now">

</｜DSML｜ invoke>
</｜DSML｜ calls>"#;

fn assert_calls(calls: &[ToolCall]) {
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "files::read");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args, serde_json::json!({"path": "/tmp/x", "lines": [1, 2]}));
    assert_eq!(calls[1].function.name, "now");
    assert_eq!(calls[1].function.arguments, "{}");
}

#[test]
fn blocking_parser_reads_spaced_tags() {
    let (content, calls) = parse_tool_calls(CALL);
    assert_eq!(content.as_deref(), Some("answer"));
    assert_calls(&calls);
}

#[test]
fn streaming_parser_reads_spaced_tags_split_anywhere() {
    for width in [1, 5, 7, 64] {
        let mut detector = StreamingToolDetector::new();
        let mut outputs = Vec::new();
        let chars: Vec<char> = CALL.chars().collect();
        for chunk in chars.chunks(width) {
            outputs.extend(detector.process(&chunk.iter().collect::<String>()));
        }
        outputs.extend(detector.flush());

        let mut content = String::new();
        let mut calls = Vec::new();
        for output in outputs {
            match output {
                DetectorOutput::Content(text) => content.push_str(&text),
                DetectorOutput::ToolCall(call, _) => calls.push(call),
                _ => {}
            }
        }
        assert_eq!(content.trim(), "answer", "width {width}");
        assert_calls(&calls);
    }
}

#[test]
fn the_parser_writes_history_in_its_own_dialect() {
    let parser = "deepseek_v41"
        .parse::<ToolCallFormat>()
        .expect("registered DSML parser")
        .into_parser();
    assert_eq!(parser.name(), "deepseek_v41");
    let (_, calls) = parse_tool_calls(CALL);
    let history: Vec<IncomingToolCall> = calls
        .iter()
        .map(|call| IncomingToolCall {
            id: Some(call.id.clone()),
            function: IncomingFunction {
                name: call.function.name.clone(),
                arguments: call.function.arguments.clone(),
            },
        })
        .collect();
    let written = parser.format_tool_calls(&history);
    assert!(written.starts_with("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"files::read\">\n"));
    assert_calls(&parse_tool_calls(&written).1);
    // V4's parser keeps V4's unspaced tags.
    let v4 = "deepseek_v4"
        .parse::<ToolCallFormat>()
        .unwrap()
        .into_parser();
    assert!(
        v4.format_tool_calls(&history)
            .starts_with("<｜DSML｜tool_calls>\n<｜DSML｜invoke ")
    );
}
