// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use xgrammar::{CompiledGrammar, GrammarMatcher};

const VALUE_CLOSE: &str = "</arg_value:opensource>";

fn hy4_vocab() -> Vec<String> {
    let mut vocab = test_vocab();
    for tag in [
        "tool_call:opensource",
        "tool_calls:opensource",
        "arg_key:opensource",
        "arg_value:opensource",
    ] {
        vocab.push(format!("<{tag}>"));
        vocab.push(format!("</{tag}>"));
    }
    vocab
}

fn compile(use_triggers: bool) -> CompiledGrammar {
    GrammarEngine::new(&hy4_vocab(), &[130])
        .unwrap()
        .compile_poolside_family_tool_grammar(
            &test_tool_defs(),
            use_triggers,
            &HY4_TAGS,
            VALUE_CLOSE,
        )
        .expect("compile must succeed")
}

fn accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    matcher.accept_string(input, false) && matcher.is_terminated()
}

fn call(location: &str) -> String {
    format!(
        "<tool_call:opensource>get_weather<arg_key:opensource>location</arg_key:opensource>\
         <arg_value:opensource>{location}{VALUE_CLOSE}</tool_call:opensource>"
    )
}

#[test]
fn hy4_auto_grammar_accepts_native_parallel_envelope() {
    let block = format!(
        "<tool_calls:opensource>{}{}</tool_calls:opensource>",
        call("Paris"),
        call("Tokyo")
    );

    assert!(accepts(&compile(true), &block));
    assert!(accepts(&compile(true), "Hey! How's it going?"));
}

#[test]
fn hy4_auto_grammar_rejects_unknown_argument_key() {
    let bad = "<tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource>\
               <arg_value:opensource>Paris</arg_value:opensource></tool_call:opensource>";

    assert!(!accepts(&compile(true), bad));
}

#[test]
fn hy4_forced_grammar_requires_enveloped_call() {
    let forced = compile(false);
    let block = format!(
        "<tool_calls:opensource>{}</tool_calls:opensource>",
        call("Paris")
    );

    assert!(accepts(&forced, &block));
    assert!(!accepts(&forced, "Hey! How's it going?"));
    assert!(!accepts(&forced, &call("Paris")));
}

#[test]
fn poolside_forced_grammar_keeps_bare_call_frame() {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).unwrap();
    let forced = engine
        .compile_poolside_v1_tool_grammar(&test_tool_defs(), false, "</arg_value>")
        .expect("compile must succeed");
    let call = "<tool_call>get_weather<arg_key>location</arg_key>\
                <arg_value>Boston</arg_value></tool_call>";

    assert!(accepts(&forced, call));
}
