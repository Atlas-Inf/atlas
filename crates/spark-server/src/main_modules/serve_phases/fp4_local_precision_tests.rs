// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn lookup_from<'a>(slice: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |key| {
        slice
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v.to_string())
    }
}

#[test]
fn not_fp4_local_says_nothing() {
    let env: &[(&str, &str)] = &[];
    let unset = unset_native_precision_switches(false, None, lookup_from(env));
    assert!(unset.is_empty());
}

#[test]
fn a_serving_mode_validates_the_switches_itself() {
    let env: &[(&str, &str)] = &[];
    let unset = unset_native_precision_switches(true, Some("spark"), lookup_from(env));
    assert!(unset.is_empty());
}

#[test]
fn set_switches_say_nothing_whatever_the_value() {
    let all_one: &[(&str, &str)] = &[
        ("ATLAS_QWEN4EXP_MXFP8_DECODE", "1"),
        ("ATLAS_QWEN4EXP_PREFILL_W8A8", "1"),
        ("ATLAS_QWEN4EXP_MXFP8_SHARED", "1"),
        ("ATLAS_QWEN4EXP_PREFILL_W4A4", "1"),
    ];
    assert!(unset_native_precision_switches(true, None, lookup_from(all_one)).is_empty());
    let all_zero: &[(&str, &str)] = &[
        ("ATLAS_QWEN4EXP_MXFP8_DECODE", "0"),
        ("ATLAS_QWEN4EXP_PREFILL_W8A8", "0"),
        ("ATLAS_QWEN4EXP_MXFP8_SHARED", "0"),
        ("ATLAS_QWEN4EXP_PREFILL_W4A4", "0"),
    ];
    assert!(unset_native_precision_switches(true, None, lookup_from(all_zero)).is_empty());
}

#[test]
fn lists_exactly_the_unset_switches_in_order() {
    let env: &[(&str, &str)] = &[
        ("ATLAS_QWEN4EXP_PREFILL_W8A8", "1"),
        ("ATLAS_QWEN4EXP_MXFP8_SHARED", "1"),
    ];
    let unset = unset_native_precision_switches(true, None, lookup_from(env));
    assert_eq!(
        unset,
        ["ATLAS_QWEN4EXP_MXFP8_DECODE", "ATLAS_QWEN4EXP_PREFILL_W4A4"]
    );
}

#[test]
fn nothing_set_lists_all_four() {
    let env: &[(&str, &str)] = &[];
    let unset = unset_native_precision_switches(true, None, lookup_from(env));
    assert_eq!(unset, NATIVE_PRECISION_SWITCHES);
}

#[test]
fn warning_names_every_unset_switch_and_is_none_for_none() {
    assert!(warning(&[]).is_none());
    let line = warning(&["ATLAS_QWEN4EXP_MXFP8_DECODE", "ATLAS_QWEN4EXP_PREFILL_W4A4"]).unwrap();
    assert!(line.contains("ATLAS_QWEN4EXP_MXFP8_DECODE"));
    assert!(line.contains("ATLAS_QWEN4EXP_PREFILL_W4A4"));
    assert!(line.contains("unset"));
    assert!(line.contains("ATLAS_QWEN4EXP_SERVING_MODE"));
}
