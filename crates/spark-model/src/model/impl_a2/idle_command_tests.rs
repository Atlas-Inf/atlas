// SPDX-License-Identifier: AGPL-3.0-only
#[path = "idle_command_fixture.rs"]
mod fixture;
// The host command channel's tests share the fixture.
#[path = "cmd_words_tests.rs"]
mod cmd_words;
use crate::traits::Model;
use fixture::{Event, Failure, Fixture, STREAM};

#[test]
fn actual_worker_entry_idle_is_first_word_only_v1_v2_and_repeated() {
    for v2 in [false, true] {
        let words = if v2 {
            vec![19, u32::MAX, 23, u32::MAX]
        } else {
            vec![u32::MAX, u32::MAX]
        };
        let f = Fixture::new(1, v2, &words);
        let allocs = f.gpu.alloc_count();
        for _ in 0..2 {
            assert!(!Model::ep_worker_step(&f.model, &mut []).unwrap());
        }
        let mut one = f.word_events(true);
        if v2 {
            one.extend(f.word_events(false));
        }
        let expected: Vec<_> = one.iter().cloned().chain(one.iter().cloned()).collect();
        assert_eq!(f.events(), expected, "v2={v2}");
        assert_eq!(f.gpu.alloc_count(), allocs);
        assert!(f.record.words.lock().is_empty());
    }
}

#[test]
fn actual_nonzero_preamble_arguments_and_bulk_payload_keep_timing() {
    for v2 in [false, true] {
        let mut words = if v2 {
            vec![0x12345678, 0x89abcdef]
        } else {
            vec![0x89abcdef]
        };
        words.extend([0xdeadbeef, 17, 19, 23]);
        let f = Fixture::new(1, v2, &words);
        assert_eq!(
            f.model.ep_recv_seq_and_cmd(v2).unwrap(),
            (if v2 { 0x12345678 } else { 0 }, 0x89abcdef)
        );
        assert_eq!(f.model.ep_broadcast_u32(0).unwrap(), 0xdeadbeef);
        assert_eq!(
            f.model.ep_broadcast_tokens(&[0; 3]).unwrap(),
            vec![17, 19, 23]
        );
        let mut expected = f.word_events(true);
        if v2 {
            expected.extend(f.word_events(false));
        }
        expected.extend(f.word_events(false));
        let scratch = f.model.buffers.scratch().0;
        expected.extend([
            Event::Timed(scratch, 12, 0),
            Event::Sync(STREAM),
            Event::D2h(scratch, 12),
        ]);
        assert_eq!(f.events(), expected);
    }
}

#[test]
fn actual_head_preamble_and_payload_never_use_idle_entry() {
    for v2 in [false, true] {
        let f = Fixture::new(0, v2, &[]);
        f.model.ep_broadcast_seq_and_cmd(19, 23, v2).unwrap();
        f.model.ep_broadcast_u32(29).unwrap();
        f.model.ep_broadcast_tokens(&[31, 37]).unwrap();
        let ptr = f.model.ep_cmd_buf.0;
        let mut expected = vec![];
        for word in if v2 {
            vec![19u32, 23, 29]
        } else {
            vec![23u32, 29]
        } {
            expected.extend([
                Event::H2d(ptr, word.to_le_bytes().to_vec()),
                Event::Timed(ptr, 4, 0),
            ]);
        }
        let scratch = f.model.buffers.scratch().0;
        expected.extend([
            Event::H2d(
                scratch,
                [31u32, 37].into_iter().flat_map(u32::to_le_bytes).collect(),
            ),
            Event::Timed(scratch, 8, 0),
        ]);
        assert_eq!(f.events(), expected);
    }
}

#[test]
fn actual_worker_errors_short_circuit_before_later_word_or_dispatch() {
    for v2 in [false, true] {
        for failure in [Failure::Idle, Failure::Sync, Failure::D2h] {
            let f = Fixture::new(1, v2, &[19, u32::MAX]);
            *f.record.failure.lock() = Some(failure);
            let error = Model::ep_worker_step(&f.model, &mut [])
                .unwrap_err()
                .to_string();
            assert!(error.contains(&format!("injected {failure:?}")), "{error}");
            let len = match failure {
                Failure::Idle => 1,
                Failure::Sync => 2,
                Failure::D2h => 3,
                _ => unreachable!(),
            };
            assert_eq!(f.events(), f.word_events(true)[..len]);
        }
    }
    let f = Fixture::new(1, true, &[19, u32::MAX]);
    *f.record.failure.lock() = Some(Failure::Timed);
    assert!(
        Model::ep_worker_step(&f.model, &mut [])
            .unwrap_err()
            .to_string()
            .contains("injected Timed")
    );
    let mut expected = f.word_events(true);
    expected.push(Event::Timed(f.model.ep_cmd_buf.0, 4, 0));
    assert_eq!(f.events(), expected);
    assert_eq!(f.record.words.lock().len(), 1);
}

#[test]
fn actual_worker_dispatch_prefill_and_verify_follow_ons_are_timed() {
    for v2 in [false, true] {
        // Each prefill header word and the bulk token payload, then the first
        // argument of all supported verify widths. Stop before numerical work.
        for (cmd, stop) in [
            (0xfffffff0, 1),
            (0xfffffff0, 2),
            (0xfffffff0, 3),
            (0xfffffff0, 4),
            (0xfffffff2, 1),
            (0xfffffff3, 1),
            (0xfffffff4, 1),
            (0xfffffff5, 1),
        ] {
            let mut words = if v2 { vec![0, cmd] } else { vec![cmd] };
            words.extend([2, 0, 2, 101, 103]);
            let f = Fixture::new(1, v2, &words);
            let at = stop + usize::from(v2);
            *f.record.timed_failure_at.lock() = Some(at);
            let mut slots = [Some(crate::traits::SequenceState::host_only(0))];
            let error = Model::ep_worker_step(&f.model, &mut slots)
                .unwrap_err()
                .to_string();
            assert_eq!(error, format!("injected timed payload {at}"));
            let events = f.events();
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::Idle(_)))
                    .count(),
                1
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::Timed(..)))
                    .count(),
                at
            );
            let (ptr, bytes) = if cmd == 0xfffffff0 && stop == 4 {
                (f.model.buffers.scratch().0, 8)
            } else {
                (f.model.ep_cmd_buf.0, 4)
            };
            assert_eq!(events.last(), Some(&Event::Timed(ptr, bytes, 0)));
            assert!(!events.contains(&Event::Alloc));
        }
    }
}
