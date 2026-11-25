// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// One checkpoint family's spelling of DSML.
pub struct Dsml {
    name: &'static str,
    pub(super) open: &'static str,
    close: &'static str,
    invoke_open: &'static str,
    invoke_close: &'static str,
    param_open: &'static str,
    param_close: &'static str,
    leaks: LeakMarkers,
}

macro_rules! dsml {
    ($name:literal, $calls:literal, $invoke:literal, $parameter:literal) => {
        Dsml {
            name: $name,
            open: concat!("<｜DSML｜", $calls, ">"),
            close: concat!("</｜DSML｜", $calls, ">"),
            invoke_open: concat!("<｜DSML｜", $invoke, " name=\""),
            invoke_close: concat!("</｜DSML｜", $invoke, ">"),
            param_open: concat!("<｜DSML｜", $parameter, " name=\""),
            param_close: concat!("</｜DSML｜", $parameter, ">"),
            leaks: LeakMarkers {
                orphan_open: &[
                    concat!("<｜DSML｜", $invoke, " name=\""),
                    concat!("<｜DSML｜", $parameter, " name=\""),
                ],
                close: &[
                    concat!("</｜DSML｜", $parameter, ">"),
                    concat!("</｜DSML｜", $invoke, ">"),
                    concat!("</｜DSML｜", $calls, ">"),
                ],
                envelope_open: &[concat!("<｜DSML｜", $calls, ">")],
                envelope_close: &[concat!("</｜DSML｜", $calls, ">")],
            },
        }
    };
}

pub(super) static DSML_V4: Dsml = dsml!("deepseek_v4", "tool_calls", "invoke", "parameter");
/// DeepSeek-V4.1 writes each tag name after a space.
pub(super) static DSML_V41: Dsml = dsml!("deepseek_v41", " calls", " invoke", " parameter");
pub(super) static DSML_DIALECTS: [&Dsml; 2] = [&DSML_V4, &DSML_V41];

/// The first opening tag of either dialect in `text`, and where it starts.
fn find_dsml(text: &str) -> Option<(usize, &'static Dsml)> {
    DSML_DIALECTS
        .iter()
        .filter_map(|&dsml| Some((text.find(dsml.open)?, dsml)))
        .min_by_key(|&(start, _)| start)
}

pub struct DeepseekDsmlParser(&'static Dsml);

impl DeepseekDsmlParser {
    pub const V4: Self = Self(&DSML_V4);
    pub const V41: Self = Self(&DSML_V41);
}

impl ToolCallParser for DeepseekDsmlParser {
    fn name(&self) -> &str {
        self.0.name
    }

    fn system_prompt(
        &self,
        _tools: &[ToolDefinition],
        _tool_choice: &ToolChoice,
        _levers: &PromptLevers,
    ) -> String {
        // The checkpoint-native encoder renders the canonical DSML instructions
        // and schemas. A second generic parser prompt would change prompt bytes.
        String::new()
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let dsml = self.0;
        let mut invokes = Vec::with_capacity(calls.len());
        for call in calls {
            let args = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
            let mut params = Vec::new();
            if let Some(object) = args.as_object() {
                for (key, value) in object {
                    let (string, value) = match value {
                        serde_json::Value::String(value) => ("true", value.clone()),
                        value => (
                            "false",
                            serde_json::to_string(value).unwrap_or_else(|_| "null".into()),
                        ),
                    };
                    params.push(format!(
                        "{}{key}\" string=\"{string}\">{value}{}",
                        dsml.param_open, dsml.param_close,
                    ));
                }
            }
            invokes.push(format!(
                "{}{}\">\n{}\n{}",
                dsml.invoke_open,
                call.function.name,
                params.join("\n"),
                dsml.invoke_close,
            ));
        }
        format!("{}\n{}\n{}", dsml.open, invokes.join("\n"), dsml.close)
    }

    fn format_tool_response(&self, content: &str) -> String {
        format!("<tool_result>{content}</tool_result>")
    }

    fn leak_markers(&self) -> LeakMarkers {
        self.0.leaks
    }

    fn wants_typed_arguments(&self) -> bool {
        true
    }
}

pub(super) fn parse_dsml_tool_calls(text: &str) -> (Option<String>, Vec<ToolCall>) {
    let mut content = Vec::new();
    let mut calls = Vec::new();
    let mut rest = text;
    loop {
        let Some((start, dsml)) = find_dsml(rest) else {
            let tail = rest.trim();
            if !tail.is_empty() {
                content.push(tail.to_string());
            }
            break;
        };
        let before = rest[..start].trim();
        if !before.is_empty() {
            content.push(before.to_string());
        }
        let body = &rest[start + dsml.open.len()..];
        let Some(end) = body.find(dsml.close) else {
            content.push(rest[start..].to_string());
            break;
        };
        calls.extend(parse_dsml_invokes(&body[..end], dsml));
        rest = &body[end + dsml.close.len()..];
    }
    let content = (!content.is_empty()).then(|| content.join("\n"));
    (content, calls)
}

fn parse_dsml_invokes(body: &str, dsml: &Dsml) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find(dsml.invoke_open) {
        rest = &rest[start + dsml.invoke_open.len()..];
        let Some(name_end) = rest.find("\">\n") else {
            break;
        };
        let name = &rest[..name_end];
        let after_header = &rest[name_end + 3..];
        let Some(invoke_end) = after_header.find(dsml.invoke_close) else {
            break;
        };
        let params = after_header[..invoke_end].trim_matches('\n');
        if let Some(arguments) = parse_dsml_parameters(params, dsml) {
            calls.push(ToolCall {
                id: next_tool_call_id(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments: serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".into()),
                },
            });
        }
        rest = &after_header[invoke_end + dsml.invoke_close.len()..];
    }
    calls
}

fn parse_dsml_parameters(params: &str, dsml: &Dsml) -> Option<serde_json::Value> {
    let mut object = serde_json::Map::new();
    let mut rest = params.trim();
    while !rest.is_empty() {
        rest = rest.strip_prefix(dsml.param_open)?;
        let name_end = rest.find("\" string=\"")?;
        let name = &rest[..name_end];
        rest = &rest[name_end + "\" string=\"".len()..];
        let string_end = rest.find("\">")?;
        let is_string = &rest[..string_end];
        rest = &rest[string_end + 2..];
        let value_end = rest.find(dsml.param_close)?;
        let raw = &rest[..value_end];
        let value = match is_string {
            "true" => serde_json::Value::String(raw.to_string()),
            "false" => serde_json::from_str(raw).ok()?,
            _ => return None,
        };
        if object.insert(name.to_string(), value).is_some() {
            return None;
        }
        rest = rest[value_end + dsml.param_close.len()..].trim_start_matches('\n');
    }
    Some(serde_json::Value::Object(object))
}

pub(super) enum DsmlStreamAction {
    NotDsml,
    Wait,
    Continue,
}

impl StreamingToolDetector {
    pub(super) fn process_dsml(&mut self, outputs: &mut Vec<DetectorOutput>) -> DsmlStreamAction {
        if let Some(dsml) = self.inside_dsml {
            let Some(end) = self.buffer.find(dsml.close) else {
                return DsmlStreamAction::Wait;
            };
            let body = self.buffer[..end].to_string();
            self.buffer = self.buffer[end + dsml.close.len()..].to_string();
            self.inside_dsml = None;
            self.emit_dsml_calls(&body, dsml, outputs);
            return DsmlStreamAction::Continue;
        }
        let Some((start, dsml)) = find_dsml(&self.buffer) else {
            return DsmlStreamAction::NotDsml;
        };
        if start > 0 {
            let content = self.buffer[..start].to_string();
            if !content.trim().is_empty() {
                self.push_content(outputs, content);
            }
        }
        if self.marker_is_literal(start) {
            let marker = self.buffer[start..start + dsml.open.len()].to_string();
            self.buffer = self.buffer[start + dsml.open.len()..].to_string();
            self.push_content(outputs, marker);
            return DsmlStreamAction::Continue;
        }
        self.buffer = self.buffer[start + dsml.open.len()..].to_string();
        self.inside_dsml = Some(dsml);
        DsmlStreamAction::Continue
    }

    pub(super) fn flush_dsml(&mut self) -> Option<Vec<DetectorOutput>> {
        let dsml = self.inside_dsml.take()?;
        let body = std::mem::take(&mut self.buffer);
        let mut outputs = Vec::new();
        self.emit_dsml_calls(&body, dsml, &mut outputs);
        (!outputs.is_empty()).then_some(outputs)
    }

    fn emit_dsml_calls(&mut self, body: &str, dsml: &Dsml, outputs: &mut Vec<DetectorOutput>) {
        for call in parse_dsml_invokes(body, dsml) {
            let index = self.call_counter as usize;
            self.call_counter += 1;
            self.emitted_tool_calls = true;
            outputs.push(DetectorOutput::ToolCall(call, index));
        }
    }
}
