// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::tool_parser::ToolCallParser;

fn auto_poolside_state() -> GrammarState {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), true, "</arg_value>")
        .expect("Poolside auto grammar must compile");
    GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_auto_literal_dispatch(true)
}

fn accept_ascii(state: &mut GrammarState, text: &str) {
    for (offset, byte) in text.bytes().enumerate() {
        assert!(
            state.accept_token(u32::from(byte)),
            "byte {offset} ({byte:?}) must be accepted while feeding {text:?}"
        );
    }
}

fn accept_real_poolside_call(state: &mut GrammarState) {
    assert!(state.accept_token(128), "real <tool_call> opener");
    accept_ascii(state, "get_weather");
    assert!(state.accept_token(b'<' as u32), "argument opener");
    accept_ascii(
        state,
        "arg_key>location</arg_key><arg_value>Boston</arg_value>",
    );
    assert!(state.accept_token(129), "real </tool_call> closer");
}

#[test]
fn auto_grammar_keeps_known_poolside_xml_inside_fence_then_accepts_call() {
    let mut state = auto_poolside_state();
    accept_ascii(&mut state, "ordinary prose ```xml\n");

    // The opener token is a registered tool prefix, so the normal matcher
    // masks it. The literal-aware adapter must restore it while the fence is
    // open, and must preserve its bytes in the emitted history.
    state.fill_bitmask();
    assert!(
        state.is_token_allowed(128),
        "known tool opener must remain legal inside a fenced literal"
    );
    assert!(state.accept_token(128));
    accept_ascii(
        &mut state,
        "get_weather<arg_key>not_a_parameter</arg_key><arg_value>BAD</arg_value>",
    );
    state.fill_bitmask();
    assert!(
        state.is_token_allowed(129),
        "known tool closer must remain legal inside a fenced literal"
    );
    assert!(state.accept_token(129));
    accept_ascii(&mut state, "\n```\n");

    accept_real_poolside_call(&mut state);
}

#[test]
fn auto_grammar_keeps_known_prefixes_inside_inline_and_quoted_literals() {
    for (opening, closing) in [
        (r#"The marker is `"#, r#"` "#),
        (r#"The marker is ``"#, r#"`` "#),
        ("The marker is \"", "\" "),
    ] {
        let mut state = auto_poolside_state();
        accept_ascii(&mut state, opening);
        state.fill_bitmask();
        assert!(
            state.is_token_allowed(128),
            "known opener must remain legal inside literal {opening:?}"
        );
        assert!(state.accept_token(128));
        accept_ascii(&mut state, "get_weather");
        assert!(state.accept_token(129));
        accept_ascii(&mut state, closing);

        accept_real_poolside_call(&mut state);
    }
}

#[test]
fn auto_literal_grammar_restores_lexical_state_on_rollback_and_reset() {
    for reset in [false, true] {
        let mut state = auto_poolside_state();
        let before = state.num_history_steps();
        accept_ascii(&mut state, "The marker is `");
        assert!(state.accept_token(128));
        let accepted = state.num_history_steps() - before;
        assert!(accepted > 0);

        if reset {
            state.reset();
        } else {
            state.rollback(accepted);
            assert_eq!(state.num_history_steps(), before);
        }
        accept_real_poolside_call(&mut state);
    }
}

#[test]
fn required_poolside_grammar_does_not_enable_literal_bypass() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), false, "</arg_value>")
        .expect("Poolside required grammar must compile");
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    state.fill_bitmask();
    assert!(
        !state.is_token_allowed(b'T' as u32),
        "required mode must still reject leading prose"
    );
    assert!(!state.accept_token(b'T' as u32));
}

#[test]
fn auto_literal_dispatch_keeps_compiled_stop_tokens_out_of_history() {
    let mut state = auto_poolside_state();
    accept_ascii(&mut state, "text `");
    let before = state.num_history_steps();
    assert!(state.accept_token(130));
    assert_eq!(state.num_history_steps(), before);
    state.rollback(before);
    accept_real_poolside_call(&mut state);
}

#[test]
fn quotes_inside_a_real_tool_body_do_not_enable_literal_dispatch() {
    let mut state = auto_poolside_state();
    assert!(state.accept_token(128));
    accept_ascii(
        &mut state,
        "get_weather<arg_key>location</arg_key><arg_value>\"Boston\"</arg_value>",
    );
    state.fill_bitmask();
    assert!(!state.is_token_allowed(b'X' as u32));
    assert!(!state.accept_token(b'X' as u32));
    assert!(state.accept_token(129));
}

/// qwen4_exp spark config: temp 0, tools armed (`use_triggers`), the model
/// shows a fenced ```xml <tool_call> example. Inside the fence the grammar
/// must not let the literal opener count as a real call (mask side), and
/// must not forbid the fence's bytes (allow side). Discriminates the
/// literal-dispatch mask shape from plain dispatch.
#[test]
fn qwen_xml_fence_keeps_literal_tool_call_text() {
    let parser = crate::tool_parser::Qwen3XmlParser;
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let compiled = parser
        .compile_tool_grammar(&mut engine, &test_tool_defs(), true)
        .expect("qwen3_xml tool grammar must compile")
        .expect("qwen3_xml tool grammar must compile");
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_auto_literal_dispatch(true);
    accept_ascii(&mut state, "example:\n```xml\n");
    state.fill_bitmask();
    // Inside the fence, a literal `<tool_call>` token must still be
    // sampleable — the masked-view mask is what lets literal prose through.
    let lt = b'<' as u32;
    assert!(
        state.is_token_allowed(lt),
        "literal `<` must remain legal inside a fenced literal"
    );
}
