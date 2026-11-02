// SPDX-License-Identifier: AGPL-3.0-only

//! A chunk the cache covers completely: which chunk that is, and on the real
//! `prefill_chunk` that the switch changes nothing but the zero and embed of
//! those chunks. Then the deep tail cut, the trace modes and the zero modes
//! of `ATLAS_GLM_ZERO_ROWS`.

// The real model and recording layer of `prefill_stream_tests`.
#[allow(clippy::duplicate_mod)]
#[path = "../prefill_stream_test_fixture.rs"]
mod fixture;

use crate::layer::EmptyLayerState;
use crate::model::warm_turn::{TraceMode, WarmTurn, ZeroRows};
use crate::traits::{Model, SequenceState};
use fixture::*;

#[test]
fn a_chunk_is_cached_when_the_restore_covers_all_of_it_and_it_is_not_the_last() {
    let mut seq = SequenceState::host_only(0);
    // Chunk [8, 16) against restore depths around it; 0 is "nothing
    // restored", which a prefix hit without a snapshot also leaves.
    for (skip_to, want) in [(0, false), (8, false), (15, false), (16, true), (40, true)] {
        seq.marconi_skip_to = skip_to;
        assert_eq!(seq.prefill_chunk_cached(16, false), want, "{skip_to}");
        // The last chunk always runs its final row.
        assert!(!seq.prefill_chunk_cached(16, true));
    }
    // Chunk 0.
    seq.marconi_skip_to = 8;
    assert!(seq.prefill_chunk_cached(8, false));
    assert!(!seq.prefill_chunk_cached(9, false));
}

/// Re-run `name` in a child process with the Marconi restore floor lifted
/// (the fixture's prompts are tens of tokens) and `env` set. `true` in the
/// parent.
fn in_child(name: &str, env: &[(&str, &str)]) -> bool {
    const CHILD: &str = "ATLAS_WARM_TEST_CHILD";
    if std::env::var(CHILD).as_deref() == Ok("1") {
        return false;
    }
    let path = module_path!().split_once("::").unwrap().1;
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &format!("{path}::{name}"), "--nocapture"])
        .env(CHILD, "1")
        .env("ATLAS_MARCONI_MIN_TOKENS", "0")
        // As served: the prefill checkpoints are the only restore points.
        .env("ATLAS_MARCONI_PREFILL_ONLY", "1")
        .env_remove("ATLAS_GLM_DET_TRACE")
        .env_remove("ATLAS_NO_TAIL_SPLIT")
        .env_remove("ATLAS_MARCONI_EXACT")
        .env_remove("ATLAS_SSM_SAVE_DUMP")
        .env_remove("ATLAS_GLM_TAIL_CUT_DEEP")
        .envs(env.iter().copied())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    // A filter that matches nothing also exits 0: require the one test.
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed"),
        "child failed: {name}\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

const SENTINEL: [u8; 8] = [0xAB; 8];

/// Run the chunks of `tokens` (8-row chunks, the last one final) on `seq`.
/// Returns, per chunk, whether it zeroed the arena: a sentinel written to
/// the residual buffer before the chunk is gone after it.
fn run_chunks(f: &Fixture, seq: &mut SequenceState, tokens: &[u32]) -> Vec<bool> {
    let residual = f.model.buffers.residual();
    (0..tokens.len())
        .step_by(8)
        .map(|start| {
            f.model.gpu.copy_h2d(&SENTINEL, residual).unwrap();
            let last = start + 8 >= tokens.len();
            f.model
                .prefill_chunk(
                    tokens,
                    seq,
                    start,
                    tokens.len().min(start + 8) - start,
                    last,
                    CALLER,
                )
                .unwrap();
            let mut now = [0u8; 8];
            f.model.gpu.copy_d2h(residual, &mut now).unwrap();
            now == [0u8; 8]
        })
        .collect()
}

/// A fresh sequence for the conversation's next turn, as the fixture builds
/// its own (it takes over `first`'s proposer state).
fn next_sequence(first: &mut SequenceState) -> SequenceState {
    let mut seq = SequenceState::host_only(0);
    seq.layer_states = vec![Box::new(EmptyLayerState)];
    seq.disk_last_offloaded_per_layer = vec![0];
    seq.proposer_state = first.proposer_state.take();
    seq
}

/// What a turn leaves: the layer passes, the sequence, the zeroed chunks.
type Turn = (Vec<Event>, (usize, usize, usize, usize, usize), Vec<bool>);

/// A cold 24-token turn, then the 28-token next turn of the conversation on
/// a fresh sequence (the pool's last free block), with the switch `on`.
/// Returns the second turn.
fn warm_turn(tp: usize, ep: usize, rank: usize, on: bool) -> Turn {
    let mut f = Fixture::with_tail_split(tp, ep, rank);
    f.disable_capture();
    f.model.warm.skip_cached = on;
    let tokens: Vec<u32> = (1..=28).collect();
    // Cold: [0,8), then [8,24) split at the tail cut 16, where the
    // checkpoint the next turn restores is saved.
    let mut first = std::mem::replace(&mut f.seq, SequenceState::host_only(0));
    f.model
        .prefill_chunk(&tokens[..24], &mut first, 0, 8, false, CALLER)
        .unwrap();
    f.model
        .prefill_chunk(&tokens[..24], &mut first, 8, 16, true, CALLER)
        .unwrap();
    let cold = f.events().len();
    assert_eq!(cold, 3, "three passes of 8 rows");

    let mut seq = next_sequence(&mut first);
    let zeroed = run_chunks(&f, &mut seq, &tokens);
    assert_eq!(seq.tokens, tokens);
    let state = (
        seq.seq_len,
        seq.block_table.len(),
        seq.cached_prefix_tokens,
        seq.marconi_skip_to,
        seq.kv_valid_tokens,
    );
    (f.events()[cold..].to_vec(), state, zeroed)
}

/// The next turn matches the 24 cached tokens and restores at 16: chunks
/// [0,8) and [8,16) compute nothing, [16,24) replays and [24,28) is new.
/// With the switch a multi-rank world leaves the first two unzeroed and runs
/// the same two passes to the same sequence state; a single rank, which
/// zeroes once per request, is unchanged.
#[test]
fn actual_cached_chunks_skip_the_zero_and_nothing_else_changes() {
    if in_child(
        "actual_cached_chunks_skip_the_zero_and_nothing_else_changes",
        &[],
    ) {
        return;
    }
    for (tp, ep, rank) in [(2, 2, 0), (2, 2, 1), (2, 1, 0), (2, 1, 1), (1, 1, 0)] {
        let ctx = format!("TP{tp}/EP{ep}/rank{rank}");
        let (off, on) = (
            warm_turn(tp, ep, rank, false),
            warm_turn(tp, ep, rank, true),
        );
        let passes = &off.0;
        assert!(
            matches!(passes[..], [Event::Target(8, _), Event::Target(4, _)]),
            "{ctx}: {passes:?}"
        );
        assert_eq!(off.1, (28, 7, 24, 16, 28), "{ctx}");
        assert_eq!((&on.0, on.1), (passes, off.1), "{ctx}");
        if tp > 1 {
            assert_eq!(off.2, [true, true, true, true], "{ctx}");
            assert_eq!(on.2, [false, false, true, true], "{ctx}");
        } else {
            assert_eq!(off.2, [true, false, false, false], "{ctx}");
            assert_eq!(on.2, off.2, "{ctx}");
        }
    }
}

/// The cold turn has no chunk to skip: with the switch every chunk still
/// zeroes, and the lookup moving ahead of the zero changes no pass.
#[test]
fn actual_cold_chunks_all_zero_with_the_switch() {
    let tokens: Vec<u32> = (1..=24).collect();
    let cold = |on: bool| {
        let mut f = Fixture::with_tail_split(2, 2, 0);
        f.disable_capture();
        f.model.warm.skip_cached = on;
        let mut seq = std::mem::replace(&mut f.seq, SequenceState::host_only(0));
        let zeroed = run_chunks(&f, &mut seq, &tokens);
        (f.events(), seq.seq_len, seq.block_table.len(), zeroed)
    };
    let (off, on) = (cold(false), cold(true));
    assert_eq!(off.3, [true, true, true]);
    assert_eq!(on, off);
    assert_eq!(off.0.len(), 3);
}

/// `ATLAS_GLM_TAIL_CUT_DEEP` on the real chunk path: a cold 24-token turn
/// splits its last chunk at 20, the last block boundary under its end, and
/// saves its checkpoint there instead of at 16. The next turn restores at 20
/// and replays 4 rows where the base cut replays 8 (the test above).
#[test]
fn actual_deep_tail_cut_restores_one_block_deeper() {
    let name = "actual_deep_tail_cut_restores_one_block_deeper";
    if in_child(name, &[("ATLAS_GLM_TAIL_CUT_DEEP", "1")]) {
        return;
    }
    let tokens: Vec<u32> = (1..=28).collect();
    for (tp, ep, rank) in [(2, 2, 0), (2, 2, 1), (2, 1, 1), (1, 1, 0)] {
        let ctx = format!("TP{tp}/EP{ep}/rank{rank}");
        let mut f = Fixture::with_tail_split(tp, ep, rank);
        f.disable_capture();
        let mut first = std::mem::replace(&mut f.seq, SequenceState::host_only(0));
        run_chunks(&f, &mut first, &tokens[..24]);
        let cold = f.events();
        assert!(
            matches!(
                cold[..],
                [
                    Event::Target(8, _),
                    Event::Target(8, _),
                    Event::Target(4, _),
                    Event::Target(4, _)
                ]
            ),
            "{ctx}: {cold:?}"
        );
        let mut seq = next_sequence(&mut first);
        run_chunks(&f, &mut seq, &tokens);
        let warm = &f.events()[cold.len()..];
        assert!(
            matches!(warm, [Event::Target(4, _), Event::Target(4, _)]),
            "{ctx}: {warm:?}"
        );
        assert_eq!(
            (seq.cached_prefix_tokens, seq.marconi_skip_to, seq.seq_len),
            (24, 20, 28),
            "{ctx}"
        );
    }
}

/// Only GLM-5's template makes the deep cut restorable: another model
/// refuses to load with the switch.
#[test]
fn actual_deep_tail_cut_is_refused_off_glm5() {
    let name = "actual_deep_tail_cut_is_refused_off_glm5";
    if in_child(name, &[("ATLAS_GLM_TAIL_CUT_DEEP", "1")]) {
        return;
    }
    assert!(WarmTurn::from_env("glm5_next").is_ok());
    let e = WarmTurn::from_env("qwen3_next").err().unwrap();
    assert!(format!("{e:#}").contains("requires glm5_next"), "{e:#}");
}

/// `ATLAS_GLM_WARM_TRACE` logs the request's line after the last chunk (and
/// with `1` syncs the stream at every step of a chunk): the passes, the
/// sequence and the zeroed chunks are those of a run without it.
#[test]
fn actual_trace_changes_no_pass() {
    let tokens: Vec<u32> = (1..=24).collect();
    let cold = |trace: TraceMode| {
        let mut f = Fixture::with_tail_split(2, 2, 0);
        f.disable_capture();
        f.model.warm.trace = trace;
        let mut seq = std::mem::replace(&mut f.seq, SequenceState::host_only(0));
        let zeroed = run_chunks(&f, &mut seq, &tokens);
        (f.events(), seq.seq_len, seq.block_table.len(), zeroed)
    };
    let off = cold(TraceMode::Off);
    for mode in [TraceMode::Hash, TraceMode::Spans] {
        assert_eq!(cold(mode), off, "{mode:?}");
    }
}

/// The request's line carries the hash of the logits row its last chunk
/// returned, read from the device: another row, another hash.
#[test]
fn actual_trace_line_fingerprints_the_logits_row() {
    use crate::model::warm_turn::logits_hash;
    let f = Fixture::with_tail_split(2, 2, 0);
    let (gpu, logits) = (f.model.gpu.as_ref(), f.model.buffers.logits());
    let width = f.model.config.vocab_size * 2;
    let line = |fill: u8| {
        gpu.memset(logits, fill, width).unwrap();
        let (zero, now) = (std::time::Duration::ZERO, std::time::Instant::now());
        f.model
            .warm_trace_line(
                &f.seq,
                24,
                now,
                (0, 8),
                [zero; 2],
                [zero; 5],
                Some(logits),
                CALLER,
            )
            .unwrap()
            .unwrap()
    };
    for fill in [0x11u8, 0x12] {
        let want = format!(" logits={:016x}", logits_hash(&vec![fill; width]));
        assert!(line(fill).ends_with(&want), "{fill:#x}");
    }
    assert_ne!(
        logits_hash(&vec![0x11; width]),
        logits_hash(&vec![0x12; width])
    );
}

/// `ATLAS_GLM_ZERO_ROWS` on the real chunk path. The fixture's buffers are
/// all small, so a trimmed zero covers each of them whole and the check
/// finds nothing: every mode zeroes every chunk and runs the same passes.
/// (What a trimmed zero covers of a large buffer is the arena's own test.)
#[test]
fn actual_zero_rows_modes_zero_every_chunk_of_a_small_arena() {
    let tokens: Vec<u32> = (1..=24).collect();
    let cold = |mode: ZeroRows| {
        let mut f = Fixture::with_tail_split(2, 2, 0);
        f.disable_capture();
        f.model.warm.zero_rows = mode;
        let mut seq = std::mem::replace(&mut f.seq, SequenceState::host_only(0));
        let zeroed = run_chunks(&f, &mut seq, &tokens);
        (f.events(), seq.seq_len, seq.block_table.len(), zeroed)
    };
    let off = cold(ZeroRows::Off);
    assert_eq!((off.0.len(), &off.3[..]), (3, &[true, true, true][..]));
    for mode in [ZeroRows::Trim(256), ZeroRows::Check(256)] {
        assert_eq!(cold(mode), off, "{mode:?}");
    }
}
