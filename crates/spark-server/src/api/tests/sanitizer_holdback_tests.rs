// SPDX-License-Identifier: AGPL-3.0-only

//! The streaming sanitizer holds back only a suffix that could still grow
//! into a leak marker — not a flat `tag_max - 1` bytes. Plain prose must
//! reach the client in the same delta that produced it; a partial marker
//! must still be held; and chunking must never change the total output.

use super::{flush_content_sanitizer, qwen, sanitize_content_chunk};
use crate::api::sanitizer::marker_prefix_hold_start;

fn feed(chunks: &[&str]) -> (Vec<String>, String) {
    let m = qwen();
    let (mut buf, mut sup, mut env) = (String::new(), false, false);
    let outs: Vec<String> = chunks
        .iter()
        .map(|c| sanitize_content_chunk(c, &mut buf, &mut sup, &mut env, &m))
        .collect();
    let tail = flush_content_sanitizer(&mut buf, &mut sup, &m);
    (outs, tail)
}

#[test]
fn prose_is_emitted_in_the_same_delta() {
    let (outs, tail) = feed(&["The prov", "ided text consis", "ts of a r"]);
    assert_eq!(outs, vec!["The prov", "ided text consis", "ts of a r"]);
    assert_eq!(tail, "");
}

#[test]
fn partial_marker_suffix_is_held_then_resolved() {
    let (outs, tail) = feed(&["see <tool_", "response> hidden </tool_response> ok"]);
    assert_eq!(outs[0], "see ", "only the marker-prefix suffix may be held");
    assert_eq!(format!("{}{}{}", outs[0], outs[1], tail), "see  ok");
}

#[test]
fn a_bare_angle_bracket_waits_but_non_marker_text_does_not() {
    let m = qwen();
    let tag_max = 32;
    assert_eq!(marker_prefix_hold_start("a <", tag_max, &m), 2);
    assert_eq!(marker_prefix_hold_start("a < b", tag_max, &m), 5);
    assert_eq!(marker_prefix_hold_start("x </tool_ca", tag_max, &m), 2);
    assert_eq!(
        marker_prefix_hold_start("héllo wörld", tag_max, &m),
        "héllo wörld".len()
    );
}

/// Chunk invariance: for every text and many chunkings, chunked output
/// (deltas + end-of-stream flush) equals feeding the text whole.
#[test]
fn chunking_never_changes_the_total_output() {
    let texts = [
        "Plain prose with no markup at all, just words and punctuation.",
        "Use a < b and c > d, then <b>bold</b> is not a marker.",
        "before <parameter=path>/etc/x</parameter> after",
        "leak <tool_response>fake result</tool_response> real text continues",
        "trailing partial close </tool_ca",
        "unicode ✓ ünïcödé <function_results><result>1</result></function_results> tail",
        "ends with a bare <",
    ];
    for t in texts {
        let (whole, wt) = feed(&[t]);
        let want = format!("{}{}", whole.concat(), wt);
        let bytes: Vec<usize> = t.char_indices().map(|(i, _)| i).chain([t.len()]).collect();
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200 {
            let mut cuts = vec![0usize];
            for &b in &bytes[1..bytes.len() - 1] {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if s % 4 == 0 {
                    cuts.push(b);
                }
            }
            cuts.push(t.len());
            let chunks: Vec<&str> = cuts.windows(2).map(|w| &t[w[0]..w[1]]).collect();
            let (outs, tail) = feed(&chunks);
            assert_eq!(
                format!("{}{}", outs.concat(), tail),
                want,
                "text {t:?} chunks {chunks:?}"
            );
        }
    }
}
