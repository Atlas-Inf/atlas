// SPDX-License-Identifier: AGPL-3.0-only

use super::apply_stop_string_holdback;

#[test]
fn empty_stop_sequences_never_match() {
    for (text, stops, expected, should_trigger) in [
        ("hello", vec![""], "hello", false),
        ("", vec![""], "", false),
        ("hello", vec!["", "STOP"], "hello", false),
        ("helloSTOPtail", vec!["", "STOP"], "hello", true),
        ("helloSTOPtail", vec!["STOP", ""], "hello", true),
    ] {
        let stops: Vec<String> = stops.into_iter().map(String::from).collect();
        let mut accumulated = String::new();
        let mut emitted = 0;
        let mut triggered = false;
        let output = apply_stop_string_holdback(
            text,
            &stops,
            0,
            &mut accumulated,
            &mut emitted,
            &mut triggered,
        );
        assert_eq!(output, expected, "text={text:?}, stops={stops:?}");
        assert_eq!(triggered, should_trigger);
        assert_eq!(accumulated, expected);
    }
}
