// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Model entry points: target/capture/primer ordering, not GPU numerics.

#[path = "prefill_stream_commit_order_tests.rs"]
mod commit_order;
#[path = "prefill_stream_test_fixture.rs"]
mod fixture;
use crate::model::kv_admission::kv_admission_refusal;
use crate::traits::{Model, SequenceState};
use fixture::*;

/// Whether `e` is the agreed refusal a victim's blocks can cure.
fn retryable(e: &anyhow::Error) -> bool {
    kv_admission_refusal(e).is_some_and(|r| r.retryable)
}

fn isolated(name: &str) -> bool {
    isolated_with(name, None)
}

/// Re-run `name` in a child process with `ATLAS_GLM_DET_TRACE` set to `det`.
fn isolated_with(name: &str, det: Option<&str>) -> bool {
    if std::env::var("ATLAS_PREFILL_STREAM_TEST_CHILD").as_deref() == Ok("1") {
        return false;
    }
    // This module's path inside the test binary (it is mounted under `entry`).
    let path = module_path!().split_once("::").unwrap().1;
    let mut child = std::process::Command::new(std::env::current_exe().unwrap());
    match det {
        Some(level) => child.env("ATLAS_GLM_DET_TRACE", level),
        None => child.env_remove("ATLAS_GLM_DET_TRACE"),
    };
    let output = child
        .env_remove("ATLAS_GLM_DET_TRACE_STAGES")
        .args(["--exact", &format!("{path}::{name}"), "--nocapture"])
        .env("ATLAS_PREFILL_STREAM_TEST_CHILD", "1")
        .env("ATLAS_GLM_MTP_HIDDEN_TRACE", "0")
        .env_remove("ATLAS_NO_MTP_EAGER_DRAFTER")
        .env_remove("ATLAS_MTP_CARRY_DRAFTER")
        .env_remove("ATLAS_MTP_ACCEPT_DEBUG")
        .env_remove("ATLAS_DIAG_GEMMA4")
        .env_remove("ATLAS_NEMO_DUMP")
        .env_remove("ATLAS_SSM_SAVE_DUMP")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    // A filter that matches nothing also exits 0: require the one test.
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed"),
        "actual entry-point child failed: {name}\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

#[derive(Clone, Copy, Debug)]
enum Entry {
    Full,
    Chunk,
    TwoPhase,
}

fn invoke(f: &mut Fixture, entry: Entry) -> anyhow::Result<spark_runtime::gpu::DevicePtr> {
    let tokens = [1, 2, 3, 4];
    match entry {
        Entry::Full => f.model.prefill(&tokens, &mut f.seq, CALLER),
        Entry::Chunk => f
            .model
            .prefill_chunk(&tokens, &mut f.seq, 0, 4, true, CALLER),
        Entry::TwoPhase => f.model.prefill_twophase(&tokens, &mut f.seq, 4, CALLER),
    }
}

#[test]
fn actual_multi_rank_entries_keep_target_capture_and_primer_on_one_stream() {
    if isolated("actual_multi_rank_entries_keep_target_capture_and_primer_on_one_stream") {
        return;
    }
    for (tp, ep, rank) in [(2, 2, 0), (2, 2, 1), (2, 1, 0), (2, 1, 1)] {
        for entry in [Entry::Full, Entry::Chunk, Entry::TwoPhase] {
            let mut f = Fixture::new(tp, ep, rank);
            invoke(&mut f, entry).unwrap();
            assert_eq!(
                f.events(),
                vec![
                    Event::Target(4, DEFAULT),
                    Event::Capture(4, DEFAULT),
                    Event::Primer(3, DEFAULT),
                ],
                "{entry:?}, TP{tp}/EP{ep}, rank{rank}"
            );
            assert_eq!(f.seq.tokens, [1, 2, 3, 4]);
            assert_eq!(f.seq.seq_len, 4);
        }
    }
}

#[test]
fn actual_single_rank_preserves_full_default_and_chunk_caller_streams() {
    if isolated("actual_single_rank_preserves_full_default_and_chunk_caller_streams") {
        return;
    }
    for (entry, expected) in [
        (Entry::Full, DEFAULT),
        (Entry::Chunk, CALLER),
        (Entry::TwoPhase, CALLER),
    ] {
        let mut f = Fixture::new(1, 1, 0);
        invoke(&mut f, entry).unwrap();
        assert_eq!(
            f.events(),
            vec![
                Event::Target(4, expected),
                Event::Capture(4, expected),
                Event::Primer(3, expected),
            ],
            "{entry:?}"
        );
    }
}

#[test]
fn actual_nonfinal_chunks_capture_but_consume_only_after_final_chunk() {
    if isolated("actual_nonfinal_chunks_capture_but_consume_only_after_final_chunk") {
        return;
    }
    for (tp, ep, stream) in [(2, 2, DEFAULT), (1, 1, CALLER)] {
        let mut f = Fixture::new(tp, ep, 0);
        let tokens = [1, 2, 3, 4];
        f.model
            .prefill_chunk(&tokens, &mut f.seq, 0, 2, false, CALLER)
            .unwrap();
        assert_eq!(
            f.events(),
            vec![Event::Target(2, stream), Event::Capture(2, stream)]
        );
        f.model
            .prefill_chunk(&tokens, &mut f.seq, 2, 2, true, CALLER)
            .unwrap();
        assert_eq!(
            f.events(),
            vec![
                Event::Target(2, stream),
                Event::Capture(2, stream),
                Event::Target(2, stream),
                Event::Capture(2, stream),
                Event::Primer(3, stream),
            ]
        );
    }
}

#[test]
fn actual_target_or_capture_errors_never_reach_primer() {
    if isolated("actual_target_or_capture_errors_never_reach_primer") {
        return;
    }
    for entry in [Entry::Full, Entry::Chunk, Entry::TwoPhase] {
        for capture_failure in [false, true] {
            let mut f = Fixture::new(2, 2, 0);
            f.fail(capture_failure);
            assert!(invoke(&mut f, entry).is_err());
            let mut expected = vec![Event::Target(4, DEFAULT)];
            if capture_failure {
                expected.push(Event::Capture(4, DEFAULT));
            }
            assert_eq!(
                f.events(),
                expected,
                "{entry:?}, capture_failure={capture_failure}"
            );
        }
    }
}

#[test]
fn actual_disabled_capture_owner_keeps_target_path_without_primer() {
    if isolated("actual_disabled_capture_owner_keeps_target_path_without_primer") {
        return;
    }
    for entry in [Entry::Full, Entry::Chunk, Entry::TwoPhase] {
        let mut f = Fixture::new(2, 2, 0);
        f.disable_capture();
        invoke(&mut f, entry).unwrap();
        assert_eq!(f.events(), vec![Event::Target(4, DEFAULT)]);
    }
}

#[test]
fn det_trace_numbers_requests_and_labels_each_chunk_and_layer() {
    let name = "det_trace_numbers_requests_and_labels_each_chunk_and_layer";
    if isolated_with(name, Some("1")) {
        return;
    }
    let tokens = [1, 2, 3, 4];
    let mut hashes = Vec::new();
    for request in 1..=2 {
        // Rank 1 of the pair; every request is numbered once, at chunk 0.
        let mut f = Fixture::new(2, 2, 1);
        f.model
            .prefill_chunk(&tokens, &mut f.seq, 0, 2, false, CALLER)
            .unwrap();
        f.model
            .prefill_chunk(&tokens, &mut f.seq, 2, 2, true, CALLER)
            .unwrap();
        // Each line is its key, then the hash.
        let (keys, hash): (Vec<String>, Vec<String>) = crate::det_trace::take_lines()
            .iter()
            .map(|line| line.rsplit_once(" h=").unwrap())
            .map(|(key, hash)| (key.to_owned(), hash.to_owned()))
            .unzip();
        hashes.push(hash);
        assert_eq!(
            keys,
            [
                format!("DET r=1 q={request} c=0 L=0 s=emb r0=0 n=2 b=16384"),
                format!("DET r=1 q={request} c=0 L=1 s=final r0=0 n=2 b=16384"),
                format!("DET r=1 q={request} c=2 L=0 s=emb r0=0 n=2 b=16384"),
                format!("DET r=1 q={request} c=2 L=1 s=final r0=0 n=2 b=16384"),
                format!("DET r=1 q={request} c=2 L=1 s=logits r0=1 n=1 b=16"),
            ]
        );
        // The events the untraced entry point produces, unchanged.
        assert_eq!(
            f.events(),
            vec![
                Event::Target(2, DEFAULT),
                Event::Capture(2, DEFAULT),
                Event::Target(2, DEFAULT),
                Event::Capture(2, DEFAULT),
                Event::Primer(3, DEFAULT),
            ]
        );
    }
    // Same bytes, same hashes; every hash is a value, not an error.
    assert_eq!(hashes[0], hashes[1]);
    assert!(hashes[0].iter().all(|h| h.len() == 16));
}

#[test]
fn det_trace_is_silent_when_unset() {
    if isolated("det_trace_is_silent_when_unset") {
        return;
    }
    let mut f = Fixture::new(2, 2, 0);
    invoke(&mut f, Entry::Chunk).unwrap();
    assert!(crate::det_trace::take_lines().is_empty());
    assert!(!crate::det_trace::on());
}

/// Take every free KV block but `keep` (stand-ins for other sequences).
fn hold_all_but(f: &Fixture, keep: usize) -> Vec<u32> {
    let mut kv = f.model.kv_cache.lock();
    (keep..kv.num_free_blocks())
        .map(|_| kv.alloc_block().unwrap())
        .collect()
}

#[test]
fn actual_refused_chunk_runs_no_layer_and_its_retry_completes() {
    if isolated("actual_refused_chunk_runs_no_layer_and_its_retry_completes") {
        return;
    }
    let tokens: Vec<u32> = (1..=24).collect();
    for (tp, ep, rank) in [(1, 1, 0), (2, 2, 0), (2, 2, 1), (2, 1, 1)] {
        let mut f = Fixture::new(tp, ep, rank);
        f.disable_capture();
        // [0,8) and [8,16) fit the one free block; [16,24) needs a second.
        let held = hold_all_but(&f, 1);
        for start in [0, 8] {
            f.model
                .prefill_chunk(&tokens, &mut f.seq, start, 8, false, CALLER)
                .unwrap();
        }
        let ran = f.events().len();
        let e = f
            .model
            .prefill_chunk(&tokens, &mut f.seq, 16, 8, true, CALLER)
            .unwrap_err();
        assert!(retryable(&e), "TP{tp}/EP{ep}/rank{rank}: {e:#}");
        assert_eq!(f.events().len(), ran, "a layer ran for the refused chunk");
        let s = &f.seq;
        assert_eq!(
            (s.seq_len, s.tokens.len(), s.block_table.len()),
            (16, 16, 1)
        );
        // A preempted victim's block comes back; the same chunk now runs once.
        f.model.kv_cache.lock().free_blocks(&held[..1]);
        f.model
            .prefill_chunk(&tokens, &mut f.seq, 16, 8, true, CALLER)
            .unwrap();
        assert_eq!(f.events().len(), ran + 1);
        assert_eq!(f.seq.tokens, tokens);
        assert_eq!((f.seq.seq_len, f.seq.block_table.len()), (24, 2));
    }
}

#[test]
fn actual_peer_refusal_is_agreed_and_rolled_back_before_any_layer() {
    if isolated("actual_peer_refusal_is_agreed_and_rolled_back_before_any_layer") {
        return;
    }
    // The peer ran out of blocks (1: retryable) or failed otherwise (0).
    for (tp, ep, rank, peer) in [(2, 2, 0, 1), (2, 2, 1, 1), (2, 1, 0, 1), (2, 2, 1, 0)] {
        let mut f = Fixture::new(tp, ep, rank);
        f.disable_capture();
        let free = f.model.kv_cache.lock().num_free_blocks();
        f.set_peer_word(peer);
        let e = invoke(&mut f, Entry::Chunk).unwrap_err();
        let r = kv_admission_refusal(&e).map(|r| (r.by_peer, r.retryable));
        assert_eq!(
            r,
            Some((true, peer == 1)),
            "TP{tp}/EP{ep}/rank{rank}: {e:#}"
        );
        assert!(f.events().is_empty(), "a layer ran for the refused chunk");
        assert!(f.seq.block_table.is_empty() && f.seq.tokens.is_empty());
        assert_eq!(f.model.kv_cache.lock().num_free_blocks(), free);
        f.set_peer_word(u32::MAX);
        invoke(&mut f, Entry::Chunk).unwrap();
        assert_eq!(f.events(), vec![Event::Target(4, DEFAULT)]);
        assert_eq!(f.seq.seq_len, 4);
    }
}

/// The real tail-checkpoint split (24 tokens, 4-token blocks: the last chunk
/// [8,24) cuts at 16) with its second half refused, by this rank running out
/// of blocks or by the peer's vote. The completed first half stays (tokens
/// and progress at the cut, its layer pass not repeated), and the head's
/// resumed [16,24) runs exactly the pass an uninterrupted split runs, ending
/// in the same sequence state.
#[test]
fn actual_split_refused_second_half_keeps_the_first_and_resumes_at_the_cut() {
    if isolated("actual_split_refused_second_half_keeps_the_first_and_resumes_at_the_cut") {
        return;
    }
    let tokens: Vec<u32> = (1..=24).collect();
    let ranks = [(1, 1, 0), (2, 2, 0), (2, 2, 1), (2, 1, 1)];
    for ((tp, ep, rank), by_peer) in ranks.into_iter().flat_map(|r| [(r, false), (r, true)]) {
        if by_peer && tp == 1 {
            continue; // a single rank has no peer
        }
        let ctx = format!("TP{tp}/EP{ep}/rank{rank}, refused by peer: {by_peer}");
        let fixture = || {
            let mut f = Fixture::with_tail_split(tp, ep, rank);
            f.disable_capture();
            f.model
                .prefill_chunk(&tokens, &mut f.seq, 0, 8, false, CALLER)
                .unwrap();
            f
        };
        let mut whole = fixture();
        whole
            .model
            .prefill_chunk(&tokens, &mut whole.seq, 8, 16, true, CALLER)
            .unwrap();
        let halves = &whole.events()[1..];
        assert!(
            matches!(halves, [Event::Target(8, _), Event::Target(8, _)]),
            "{ctx}: {halves:?}"
        );

        let mut f = fixture();
        let held = if by_peer {
            // The peer admits [8,16), then runs out of blocks for [16,24).
            f.script(&[u32::MAX, 1]);
            vec![]
        } else {
            // [8,16) takes two of the three free blocks; [16,24) gets one of
            // its two.
            hold_all_but(&f, 3)
        };
        let free = |f: &Fixture| f.model.kv_cache.lock().num_free_blocks();
        let free_before = free(&f);
        let e = f
            .model
            .prefill_chunk(&tokens, &mut f.seq, 8, 16, true, CALLER)
            .unwrap_err();
        let r = kv_admission_refusal(&e).map(|r| (r.by_peer, r.retryable));
        assert_eq!(r, Some((by_peer, true)), "{ctx}: {e:#}");
        assert_eq!(
            f.events(),
            whole.events()[..2],
            "{ctx}: first half ran once"
        );
        let s = &f.seq;
        assert_eq!((s.seq_len, s.block_table.len()), (16, 4), "{ctx}");
        assert_eq!(s.tokens, tokens[..16], "{ctx}");
        // Only the first half's two blocks stay taken.
        assert_eq!(free(&f), free_before - 2, "{ctx}");

        f.model.kv_cache.lock().free_blocks(&held);
        f.model
            .prefill_chunk(&tokens, &mut f.seq, 16, 8, true, CALLER)
            .unwrap();
        assert_eq!(f.events(), whole.events(), "{ctx}");
        assert_eq!(f.seq.tokens, tokens, "{ctx}");
        let (s, w) = (&f.seq, &whole.seq);
        assert_eq!(
            (s.seq_len, s.block_table.len()),
            (w.seq_len, w.block_table.len()),
            "{ctx}"
        );
    }
}

/// The worker's 0xFFFFFFF0 handler, driven through `ep_worker_step` by the
/// head's scripted words: an agreed refusal (retryable or not) keeps it
/// serving with every command word consumed and the slot where the refusal
/// left it (untouched, or at the cut when the real tail split had completed
/// its first half); the head's re-sent chunk from there then runs; a chunk
/// away from the slot's progress fails, after its words are read.
#[test]
fn actual_worker_survives_a_refused_chunk_and_runs_the_resend() {
    if isolated("actual_worker_survives_a_refused_chunk_and_runs_the_resend") {
        return;
    }
    let tokens: Vec<u32> = (1..=24).collect();
    let cases = [(2, 2, 1, false), (2, 1, 1, false), (2, 2, 0, false)];
    for (tp, ep, peer, split) in cases.into_iter().chain([(2, 2, 1, true), (2, 1, 0, true)]) {
        let ctx = format!("TP{tp}/EP{ep}, head vote {peer}, split {split}");
        let mut f = if split {
            Fixture::with_tail_split(tp, ep, 1)
        } else {
            Fixture::new(tp, ep, 1)
        };
        f.disable_capture();
        let v2_slot: &[u32] = if f.model.ep_protocol_v2 { &[0] } else { &[] };
        let chunk =
            |len: u32, start: u32| [v2_slot, &[0xFFFF_FFF0, len, start, 24], &tokens].concat();
        let mut slots = [Some(std::mem::replace(
            &mut f.seq,
            SequenceState::host_only(0),
        ))];
        let mut step = |f: &Fixture, words: &[u32]| {
            f.script(words);
            let r = f.model.ep_worker_step(&mut slots);
            assert_eq!(f.unread(), 0, "{ctx}: command words left unread");
            let s = slots[0].as_ref().unwrap();
            (r, s.seq_len, s.tokens.len(), f.events().len())
        };
        let (r, ..) = step(&f, &chunk(8, 0));
        assert!(r.unwrap(), "{ctx}");

        // The head refuses [8,16); or, for the final chunk [8,24) that splits
        // at 16, admits the first half and refuses the second.
        let (end, kept, votes): (u32, u32, &[u32]) = if split {
            (24, 16, &[u32::MAX, peer])
        } else {
            (16, 8, &[peer])
        };
        let ran_at = |at: u32| (at as usize, at as usize, at as usize / 8);
        let (r, seq_len, n, ran) = step(&f, &[&chunk(end - 8, 8)[..], votes].concat());
        assert!(r.unwrap(), "{ctx}: the worker keeps serving");
        assert_eq!((seq_len, n, ran), ran_at(kept), "{ctx}");

        // The head's re-send starts at the recorded progress and is admitted.
        let (r, seq_len, n, ran) = step(&f, &chunk(end - kept, kept));
        assert!(r.unwrap(), "{ctx}");
        assert_eq!((seq_len, n, ran), ran_at(end), "{ctx}");

        let (r, ..) = step(&f, &chunk(8, 0));
        let e = r.unwrap_err();
        assert!(format!("{e:#}").contains("diverged"), "{ctx}: {e:#}");
    }
}
