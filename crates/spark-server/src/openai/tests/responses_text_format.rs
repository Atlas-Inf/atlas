// SPDX-License-Identifier: AGPL-3.0-only

use crate::openai::*;
use serde_json::{Value, json};

fn lower(text: Value) -> Result<ChatCompletionRequest, LowerResponsesError> {
    let req = serde_json::from_value(json!({
        "model": "test", "input": "Return JSON", "text": text,
    }))
    .unwrap();
    lower_responses_to_chat(req, |_| None)
}

#[test]
fn responses_text_format_json_schema() {
    let schema = json!({
        "type": "object", "properties": {"answer": {"type": "integer"}},
        "required": ["answer"], "additionalProperties": false,
    });
    for strict in [true, false] {
        let chat = lower(json!({"format": {
            "type": "json_schema", "name": "answer", "description": "An answer",
            "schema": schema, "strict": strict,
        }}))
        .unwrap();
        match chat.response_format {
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                assert_eq!(json_schema.name, "answer");
                assert_eq!(json_schema.description.as_deref(), Some("An answer"));
                assert_eq!(json_schema.schema, schema);
                assert_eq!(json_schema.strict, strict);
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }
}

#[test]
fn responses_text_format_json_object() {
    let chat = lower(json!({"format": {"type": "json_object"}})).unwrap();
    assert!(matches!(
        chat.response_format,
        Some(ResponseFormat::JsonObject)
    ));
}

#[test]
fn responses_text_format_unconstrained() {
    for text in [Value::Null, json!({}), json!({"format": {"type": "text"}})] {
        assert!(lower(text).unwrap().response_format.is_none());
    }
}

#[test]
fn responses_text_format_invalid_rejected() {
    for format in [
        json!({"type": "unknown"}),
        json!({"type": "json_schema", "name": "missing_schema"}),
        json!({"type": "json_schema", "schema": {}}),
        json!({"type": "json_schema", "name": "bad_strict", "schema": {}, "strict": "yes"}),
        json!({}),
        json!("json_object"),
    ] {
        let err = lower(json!({"format": format})).unwrap_err();
        assert!(matches!(err, LowerResponsesError::BadRequest(_)));
        assert!(err.message().contains("text.format"));
    }
}
