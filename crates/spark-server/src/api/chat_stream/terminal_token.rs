// SPDX-License-Identifier: AGPL-3.0-only

use super::{DeltaVec, StreamState};

/// Keep the terminal check ahead of all token/ID accumulation and detokenization.
/// The triggering event itself may emit a sanitized client-stop prefix. Later
/// queued events must not run the body, even if scheduler cancellation races.
pub(super) fn while_open(
    state: &mut StreamState,
    process: impl FnOnce(&mut StreamState) -> DeltaVec,
) -> DeltaVec {
    if state.stop_string_triggered || state.guard_stop.is_some() {
        Vec::new()
    } else {
        process(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::StreamDelta;
    use std::sync::{Arc, atomic::AtomicBool};

    fn state() -> StreamState {
        StreamState::new(false, false, Arc::new(AtomicBool::new(false)), Vec::new())
    }

    #[test]
    fn terminal_guard_drops_late_content_reasoning_and_token_ids_before_body() {
        for guard_only in [false, true] {
            let mut s = state();
            s.stop_string_triggered = !guard_only;
            s.guard_stop = guard_only.then_some("simhash");
            s.all_toks = vec![1];
            s.pending_token_ids = vec![1];
            let result = while_open(&mut s, |s| {
                s.all_toks.push(2);
                s.pending_token_ids.push(2);
                vec![
                    StreamDelta::Content {
                        text: "late".into(),
                        token_ids: vec![2],
                    },
                    StreamDelta::Reasoning {
                        text: "late reasoning".into(),
                        token_ids: vec![2],
                    },
                ]
            });
            assert!(result.is_empty(), "terminal event must not emit any delta");
            assert_eq!(s.all_toks, vec![1]);
            assert_eq!(
                s.pending_token_ids,
                vec![1],
                "late IDs cannot reach Done's drain"
            );
        }
    }

    #[test]
    fn open_stream_preserves_normal_delivery() {
        let mut s = state();
        let result = while_open(&mut s, |s| {
            s.all_toks.push(2);
            vec![StreamDelta::Content {
                text: "normal".into(),
                token_ids: vec![2],
            }]
        });
        assert_eq!(s.all_toks, vec![2]);
        assert!(
            matches!(&result[..], [StreamDelta::Content { text, token_ids }]
            if text == "normal" && token_ids == &[2])
        );
    }

    #[test]
    fn explicit_stop_preserves_same_event_prefix_but_not_following_tokens() {
        let mut s = state();
        let prefix = while_open(&mut s, |s| {
            let text = super::super::apply_stop_string_holdback(
                "hello STOP trailing",
                &["STOP".into()],
                3,
                &mut s.accumulated_content,
                &mut s.stop_string_emitted_len,
                &mut s.stop_string_triggered,
            );
            assert!(s.stop_string_triggered);
            s.note_stop_string_match();
            vec![StreamDelta::Content {
                text,
                token_ids: vec![1],
            }]
        });
        assert!(matches!(&prefix[..], [StreamDelta::Content { text, .. }] if text == "hello "));
        let later = while_open(&mut s, |_| {
            vec![StreamDelta::Content {
                text: "must not follow STOP".into(),
                token_ids: vec![2],
            }]
        });
        assert!(later.is_empty());
        assert_eq!(s.accumulated_content, "hello ");
        assert!(s.stop_string_matched);
    }
}
