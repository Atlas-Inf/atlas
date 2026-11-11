// SPDX-License-Identifier: AGPL-3.0-only

//! Closed argument values for a forced Poolside v1 (GLM) call.
//!
//! The Poolside v1 body grammar constrains argument names only: a value is
//! any text up to `</arg_value>`. When `tool_choice` forces the call, a
//! property whose schema closes its values with `enum` or `const` is decoded
//! from that set instead, the way a `response_format` schema bounds an
//! answer. A string is written raw, as `format_tool_calls` renders it; any
//! other value as its JSON with an optional space around each separator, so
//! both the template's `json.dumps` spacing and compact JSON decode. Open
//! properties keep the shared free-text value rule: typing them would need a
//! raw-string/JSON XML schema converter for this format and live A/B
//! evidence, which the Qwen XML history shows is not free (`compile_tools`).

use serde_json::Value;

/// `pair_body` rules for a forced call, or `None` when no named property is
/// closed. `pair_body` covers one `NAME</arg_key><arg_value>VALUE{value_close}`;
/// an open property refers to the caller's `value` rule.
pub(super) fn closed_pair_rules(
    schema: &Value,
    param_names: &[String],
    value_close: &str,
) -> Option<String> {
    let properties = schema.get("properties")?;
    let values: Vec<_> = param_names
        .iter()
        .map(|name| properties.get(name).and_then(closed_value_ebnf))
        .collect();
    if values.iter().all(Option::is_none) {
        return None;
    }
    let mut rules = vec![format!(
        "pair_body ::= {}",
        (0..param_names.len())
            .map(|i| format!("key_{i}"))
            .collect::<Vec<_>>()
            .join(" | ")
    )];
    for (i, (name, value)) in param_names.iter().zip(values).enumerate() {
        let value = value.map_or_else(|| "value".to_string(), |arms| format!("({arms})"));
        rules.push(format!(
            "key_{i} ::= {} \"</arg_key><arg_value>\" {value} \"{value_close}\"",
            literal(name)
        ));
    }
    rules.push("ws ::= [ ]?".to_string());
    Some(rules.join("\n"))
}

/// The `enum`/`const` set as an EBNF alternation. An empty string keeps the
/// property open: an empty literal is not a value the grammar can require.
fn closed_value_ebnf(property: &Value) -> Option<String> {
    let values: Vec<&Value> = match property.get("const") {
        Some(value) => vec![value],
        None => property.get("enum")?.as_array()?.iter().collect(),
    };
    if values.is_empty() || values.iter().any(|v| v.as_str() == Some("")) {
        return None;
    }
    let arms: Vec<String> = values
        .into_iter()
        .map(|value| match value {
            Value::String(text) => literal(text),
            value => json_ebnf(value),
        })
        .collect();
    Some(arms.join(" | "))
}

fn json_ebnf(value: &Value) -> String {
    let (open, close, items): (&str, &str, Vec<String>) = match value {
        Value::Array(items) => ("[", "]", items.iter().map(json_ebnf).collect()),
        Value::Object(map) => (
            "{",
            "}",
            map.iter()
                .map(|(key, value)| {
                    format!(
                        "{} ws \":\" ws {}",
                        literal(&Value::from(key.as_str()).to_string()),
                        json_ebnf(value)
                    )
                })
                .collect(),
        ),
        scalar => return literal(&scalar.to_string()),
    };
    if items.is_empty() {
        return format!("\"{open}\" ws \"{close}\"");
    }
    format!(
        "\"{open}\" ws {} ws \"{close}\"",
        items.join(" ws \",\" ws ")
    )
}

/// An EBNF string literal; xgrammar reads JSON string escapes.
fn literal(text: &str) -> String {
    serde_json::to_string(text).unwrap_or_else(|_| "\"\"".into())
}
