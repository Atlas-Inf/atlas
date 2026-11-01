// SPDX-License-Identifier: AGPL-3.0-only

//! The host command channel carries the broadcast path's word stream: what
//! the head sends, what the worker reads, and which of the two paths each
//! message takes are the same on both ranks and on both paths.

use super::fixture::{Event, Failure, Fixture, STREAM};
use crate::traits::Model;

/// The channel's message size in these tests (the RDMA pair's is 64).
const MAX: usize = 8;

fn fixture(rank: usize, v2: bool, host: bool, words: &[u32]) -> Fixture {
    let f = Fixture::new(rank, v2, words);
    *f.record.host_words_max.lock() = if host { MAX } else { 0 };
    f
}

/// One DFlash verify step and a prefill start, as the scheduler issues them:
/// slot + command, width, tokens, verdict; then a prompt too long for a
/// command message.
fn head_script(f: &Fixture, v2: bool) -> Vec<u32> {
    let tokens = [101u32, 103, 107, 109, 113];
    let prompt: Vec<u32> = (0..MAX as u32 + 1).map(|i| 1000 + i).collect();
    f.model
        .ep_broadcast_seq_and_cmd(3, 0xffff_fff5, v2)
        .unwrap();
    f.model.ep_broadcast_u32(tokens.len() as u32).unwrap();
    assert_eq!(f.model.ep_broadcast_tokens(&tokens).unwrap(), tokens);
    f.model.ep_broadcast_u32(2).unwrap();
    f.model
        .ep_broadcast_seq_and_cmd(3, 0xffff_fff0, v2)
        .unwrap();
    assert_eq!(f.model.ep_broadcast_tokens(&prompt).unwrap(), prompt);
    let mut sent = if v2 { vec![3] } else { vec![] };
    sent.extend([0xffff_fff5, tokens.len() as u32]);
    sent.extend(tokens);
    sent.push(2);
    sent.extend(v2.then_some(3));
    sent.push(0xffff_fff0);
    sent.extend(prompt);
    sent
}

/// The words a head put on the wire, in order, from either path's events.
fn wire_words(events: &[Event]) -> Vec<u32> {
    let mut words = Vec::new();
    for event in events {
        match event {
            Event::HostSend(sent) => words.extend(sent),
            Event::H2d(_, bytes) => words.extend(
                bytes
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
            ),
            _ => {}
        }
    }
    words
}

#[test]
fn both_ranks_pick_the_channel_by_word_count_alone() {
    for v2 in [false, true] {
        for host in [false, true] {
            let ranks = [fixture(0, v2, host, &[]), fixture(1, v2, host, &[])];
            for n in 0..=3 * MAX {
                let want = host && (1..=MAX).contains(&n);
                for f in &ranks {
                    assert_eq!(f.model.ep_cmd_words_on_host(n), want, "n={n} host={host}");
                }
            }
        }
    }
}

#[test]
fn the_head_sends_the_same_word_stream_on_either_path() {
    for v2 in [false, true] {
        let broadcast = fixture(0, v2, false, &[]);
        let sent = head_script(&broadcast, v2);
        assert_eq!(wire_words(&broadcast.events()), sent);
        assert!(
            !broadcast
                .events()
                .iter()
                .any(|e| matches!(e, Event::HostSend(_) | Event::HostRecv(_)))
        );

        let host = fixture(0, v2, true, &[]);
        assert_eq!(head_script(&host, v2), sent);
        let events = host.events();
        assert_eq!(wire_words(&events), sent, "v2={v2}");
        // Every message that fits is one host send behind a stream drain;
        // only the long prompt is still an upload and a broadcast.
        let scratch = host.model.buffers.scratch().0;
        let prompt = &sent[sent.len() - (MAX + 1)..];
        let mut expected = Vec::new();
        let mut messages: Vec<Vec<u32>> = if v2 { vec![vec![3]] } else { vec![] };
        messages.extend([
            vec![0xffff_fff5],
            vec![5],
            vec![101, 103, 107, 109, 113],
            vec![2],
        ]);
        messages.extend(v2.then(|| vec![3]));
        messages.push(vec![0xffff_fff0]);
        for message in messages {
            expected.extend([Event::Sync(STREAM), Event::HostSend(message)]);
        }
        expected.extend([
            Event::H2d(
                scratch,
                prompt.iter().flat_map(|w| w.to_le_bytes()).collect(),
            ),
            Event::Timed(scratch, 4 * (MAX + 1), 0),
        ]);
        assert_eq!(events, expected, "v2={v2}");
    }
}

/// What the worker decodes from the head's stream does not depend on the
/// path, and equals what the head sent.
#[test]
fn the_worker_reads_what_the_head_sent_on_either_path() {
    for v2 in [false, true] {
        let sent = head_script(&fixture(0, v2, true, &[]), v2);
        let mut decoded = Vec::new();
        for host in [false, true] {
            let f = fixture(1, v2, host, &sent);
            let mut got = Vec::new();
            let (slot, cmd) = f.model.ep_recv_seq_and_cmd(v2).unwrap();
            got.extend(v2.then_some(slot));
            got.push(cmd);
            let k = f.model.ep_broadcast_u32(0).unwrap();
            got.push(k);
            got.extend(f.model.ep_broadcast_tokens(&vec![0; k as usize]).unwrap());
            got.push(f.model.ep_broadcast_u32(0).unwrap());
            let (slot, cmd) = f.model.ep_recv_seq_and_cmd(v2).unwrap();
            got.extend(v2.then_some(slot));
            got.push(cmd);
            got.extend(f.model.ep_broadcast_tokens(&[0; MAX + 1]).unwrap());
            assert_eq!(got, sent, "v2={v2} host={host}");
            assert!(f.record.words.lock().is_empty());
            let events = f.events();
            let count = |pick: fn(&Event) -> bool| events.iter().filter(|e| pick(e)).count();
            let small = sent.len() - (MAX + 1) - 5; // words outside the two token payloads
            if host {
                // Two idle words, their follow-ons and the verify tokens are
                // host receives, each followed by the stream drain.
                assert_eq!(count(|e| matches!(e, Event::HostRecv(1))), small);
                assert_eq!(count(|e| matches!(e, Event::HostRecv(5))), 1);
                assert_eq!(count(|e| matches!(e, Event::Idle(_))), 0);
                assert_eq!(count(|e| matches!(e, Event::Timed(..))), 1);
                assert_eq!(count(|e| matches!(e, Event::D2h(..))), 1);
                for pair in events.chunks(2).take(small + 1) {
                    assert!(matches!(pair, [Event::HostRecv(_), Event::Sync(STREAM)]));
                }
            } else {
                assert_eq!(count(|e| matches!(e, Event::HostRecv(_))), 0);
                assert_eq!(count(|e| matches!(e, Event::Idle(_))), 2);
            }
            decoded.push(got);
        }
        assert_eq!(decoded[0], decoded[1]);
    }
}

#[test]
fn the_worker_entry_takes_its_idle_word_from_the_channel() {
    for v2 in [false, true] {
        let words = if v2 {
            vec![19, u32::MAX]
        } else {
            vec![u32::MAX]
        };
        let f = fixture(1, v2, true, &words);
        assert!(!Model::ep_worker_step(&f.model, &mut []).unwrap());
        let expected: Vec<Event> = (0..words.len())
            .flat_map(|_| [Event::HostRecv(1), Event::Sync(STREAM)])
            .collect();
        assert_eq!(f.events(), expected, "v2={v2}");
    }
}

/// `ep_min_u32` is rooted at each rank in turn, so it stays two broadcasts.
#[test]
fn the_min_reduction_stays_on_broadcasts() {
    for rank in 0..2 {
        let f = fixture(rank, true, true, &[7, 8]);
        f.model.ep_min_u32(9).unwrap();
        let events = f.events();
        let roots: Vec<usize> = events
            .iter()
            .filter_map(|e| match e {
                Event::Timed(_, 4, root) => Some(*root),
                _ => None,
            })
            .collect();
        assert_eq!(roots, [0, 1], "rank {rank}");
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Event::HostSend(_) | Event::HostRecv(_)))
        );
    }
}

#[test]
fn a_channel_error_ends_the_command_before_the_next_word() {
    let head = fixture(0, true, true, &[]);
    *head.record.failure.lock() = Some(Failure::Host);
    let error = head
        .model
        .ep_broadcast_seq_and_cmd(3, 0xffff_fff5, true)
        .unwrap_err();
    assert!(error.to_string().contains("injected Host"), "{error}");
    assert_eq!(
        head.events(),
        [Event::Sync(STREAM), Event::HostSend(vec![3])]
    );

    let worker = fixture(1, true, true, &[19, u32::MAX]);
    *worker.record.failure.lock() = Some(Failure::Host);
    let error = Model::ep_worker_step(&worker.model, &mut []).unwrap_err();
    assert!(error.to_string().contains("injected Host"), "{error}");
    assert_eq!(worker.events(), [Event::HostRecv(1)]);
    assert_eq!(worker.record.words.lock().len(), 2);
}
