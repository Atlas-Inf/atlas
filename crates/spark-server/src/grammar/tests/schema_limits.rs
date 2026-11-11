// SPDX-License-Identifier: AGPL-3.0-only

//! Large response_format schemas: a 96-key strict object compiles quickly
//! and small, repeated ones keep the compiler cache within its bound, and
//! a schema past the limits is refused (HTTP 400 upstream), never OOMs.

use super::*;

/// `{"<p>000": {"keep": bool, "reason": str}, ...}`: n required keys.
fn keyed_object(n: usize, prefix: &str) -> String {
    let keys: Vec<String> = (0..n).map(|i| format!("{prefix}{i:03}")).collect();
    let props: serde_json::Map<String, serde_json::Value> = keys
        .iter()
        .map(|k| {
            let value = serde_json::json!({
                "type": "object", "additionalProperties": false,
                "required": ["keep", "reason"],
                "properties": {"keep": {"type": "boolean"}, "reason": {"type": "string"}}
            });
            (k.clone(), value)
        })
        .collect();
    serde_json::json!({
        "type": "object", "additionalProperties": false,
        "required": keys, "properties": props
    })
    .to_string()
}

fn engine() -> GrammarEngine {
    GrammarEngine::new(&test_vocab(), &[130]).unwrap()
}

#[test]
fn a_96_key_strict_object_compiles_fast_and_small() {
    let mut engine = engine();
    let t = std::time::Instant::now();
    let compiled = engine.compile_json_schema(&keyed_object(96, "m")).unwrap();
    // Was O(keys^2) in time and memory (gigabytes at 96 keys in production).
    assert!(
        t.elapsed() < std::time::Duration::from_secs(10),
        "{:?}",
        t.elapsed()
    );
    assert!(compiled.grammar_memory_size_bytes() < 16 << 20);
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();
    for b in br#"{"m000": {"keep": true, "reason": "ok"}, "m001""# {
        assert!(state.accept_token(*b as u32), "{}", *b as char);
    }
}

#[test]
fn repeated_large_schemas_stay_within_the_cache_bound() {
    let mut engine = engine();
    let limit = engine.compiler.cache_limit_bytes();
    assert!(limit > 0, "the cache is bounded");
    for round in 0..12 {
        let compiled = engine
            .compile_json_schema(&keyed_object(96, &format!("r{round}_")))
            .unwrap();
        let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();
        state.fill_bitmask();
        drop(state);
        drop(compiled);
        assert!(
            engine.compiler.get_cache_size_bytes() <= limit,
            "round {round}"
        );
    }
}

#[test]
fn a_schema_past_the_limits_is_refused() {
    let mut engine = engine();
    engine.compiler.set_max_schema_grammar_bytes(64 << 10);
    let err = engine
        .compile_json_schema(&keyed_object(96, "m"))
        .unwrap_err();
    assert!(err.to_string().contains("limit"), "{err}");
    let huge = format!(
        "{{\"type\":\"string\",\"description\":\"{}\"}}",
        "x".repeat(super::super::compile_misc::MAX_SCHEMA_TEXT_BYTES)
    );
    let err = engine.compile_json_schema(&huge).unwrap_err();
    assert!(err.to_string().contains("limit"), "{err}");
}

/// ~40 KB of schema whose `[^"]{min,max}` strings expand past the 32,768
/// rules an FSM edge can name: refused, never a wrapped id or a panic.
pub(crate) fn too_many_rules_schema() -> String {
    let props: Vec<String> = (0..760)
        .map(|i| {
            let min = i % 64;
            format!(
                "\"p{i}\":{{\"type\":\"string\",\"minLength\":{min},\"maxLength\":{}}}",
                min + 50 + i % 13
            )
        })
        .collect();
    format!(
        "{{\"type\":\"object\",\"properties\":{{{}}}}}",
        props.join(",")
    )
}

#[test]
fn a_schema_past_the_fsm_encoding_is_refused() {
    let err = engine()
        .compile_json_schema(&too_many_rules_schema())
        .unwrap_err();
    assert!(err.to_string().contains("grammar too large"), "{err}");
}
