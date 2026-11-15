// SPDX-License-Identifier: AGPL-3.0-only

//! What only DeepSeek-V4.1's `encoding.py` does: image blocks become a
//! placeholder token, and tools may be namespaced.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use super::preprocess::role;

/// One image in the prompt. The vision path replaces it with the image's
/// tokens; text may not spell it.
const IMAGE: &str = "<｜deepseek_image｜>";

/// Before any merging: refuse text that spells the image token, and lower
/// OpenAI's `developer` role to the system message V4.1 accepts anywhere.
pub(super) fn prepare(messages: &mut [Value]) -> Result<()> {
    for message in messages {
        for field in ["content", "reasoning_content"] {
            if message
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(|text| text.contains(IMAGE))
            {
                bail!("DeepSeek-V4.1 message {field} contains the image token '{IMAGE}'");
            }
        }
        if role(message) == Some("developer") {
            message["role"] = json!("system");
        }
    }
    Ok(())
}

/// After tool results are merged and ordered: replace every image block with
/// the placeholder, and give a message whose content was a list of parts the
/// text those parts hold.
pub(super) fn images_to_placeholders(messages: &mut [Value]) -> Result<()> {
    for message in messages {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if !object.contains_key("content_blocks")
            && object.get("content").is_some_and(Value::is_array)
            && let Some(parts) = object.shift_remove("content")
        {
            object.insert("content_blocks".into(), parts);
        }
        let Some(blocks) = object
            .get_mut("content_blocks")
            .and_then(Value::as_array_mut)
            .filter(|blocks| !blocks.is_empty())
        else {
            continue;
        };
        replace_image_blocks(blocks)?;
        if !object.get("content").is_some_and(Value::is_string) {
            let text: Vec<&str> = object["content_blocks"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                .map(|block| {
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                })
                .collect();
            let text = text.join("\n\n");
            object.insert("content".into(), Value::String(text));
        }
    }
    Ok(())
}

fn replace_image_blocks(blocks: &mut [Value]) -> Result<()> {
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some(kind @ ("image" | "image_url")) => {
                let has_source = if kind == "image_url" {
                    let url = block.get("image_url");
                    truthy(
                        url.filter(|url| url.is_string())
                            .or_else(|| url.and_then(|url| url.get("url"))),
                    )
                } else {
                    ["source", "url", "data"]
                        .iter()
                        .any(|key| truthy(block.get(*key)))
                };
                if !has_source {
                    bail!("DeepSeek-V4.1 image block does not contain a valid source");
                }
                *block = json!({"type": "text", "text": IMAGE});
            }
            Some("tool_result") => {
                if let Some(nested) = block.get_mut("content").and_then(Value::as_array_mut) {
                    replace_image_blocks(nested)?;
                }
            }
            Some("text") => {
                if block
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.contains(IMAGE))
                {
                    bail!("DeepSeek-V4.1 text block contains the image token '{IMAGE}'");
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Python's truthiness, which the reference applies to an image's source and
/// to a tool call's namespace.
pub(super) fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(fields)) => !fields.is_empty(),
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64() != Some(0.0),
    }
}

/// `namespace::name`, or `name` alone. A name may already be qualified; it
/// must then agree with any namespace given beside it. `namespace` is a
/// string, or an object with a `name`.
pub(super) fn qualified_tool_name(name: &str, namespace: Option<&Value>) -> Result<String> {
    let namespace = match namespace {
        None | Some(Value::Null) => None,
        Some(Value::String(namespace)) => Some(namespace.as_str()),
        Some(namespace) => Some(
            namespace
                .get("name")
                .and_then(Value::as_str)
                .context("DeepSeek-V4.1 tool namespace is missing its name")?,
        ),
    };
    let (namespace, name) = match name.split_once("::") {
        Some((prefix, bare)) => {
            if namespace.is_some_and(|namespace| namespace != prefix) {
                bail!(
                    "conflicting DeepSeek-V4.1 tool namespaces: {} != {prefix}",
                    namespace.unwrap_or_default()
                );
            }
            (Some(prefix), bare)
        }
        None => (namespace, name),
    };
    if name.contains("::") {
        bail!("DeepSeek-V4.1 tool name must not contain '::': {name}");
    }
    match namespace {
        Some(namespace) if namespace.contains("::") => {
            bail!("DeepSeek-V4.1 tool namespace must not contain '::': {namespace}")
        }
        Some(namespace) => Ok(format!("{namespace}::{name}")),
        None => Ok(name.to_string()),
    }
}

/// The schema the system prompt lists for one tool: its `function`, named with
/// its namespace, and described under the namespace's own description.
pub(super) fn tool_schema(tool: &Value) -> Result<Value> {
    let mut function = tool
        .get("function")
        .and_then(Value::as_object)
        .context("DeepSeek-V4.1 tools must use OpenAI function format")?
        .clone();
    let namespace = tool
        .get("namespace")
        .filter(|namespace| !namespace.is_null())
        .cloned()
        .or_else(|| function.shift_remove("namespace"));
    function.shift_remove("namespace");
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .context("DeepSeek-V4.1 tool is missing function.name")?;
    let name = qualified_tool_name(name, namespace.as_ref())?;
    function.insert("name".into(), Value::String(name));
    if let Some(about) = namespace
        .as_ref()
        .and_then(|namespace| namespace.get("description"))
        .and_then(Value::as_str)
        .filter(|about| !about.is_empty())
    {
        let own = function
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default();
        function.insert(
            "description".into(),
            Value::String(format!("{about}\n{own}")),
        );
    }
    Ok(Value::Object(function))
}
