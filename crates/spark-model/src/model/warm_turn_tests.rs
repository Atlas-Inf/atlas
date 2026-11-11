// SPDX-License-Identifier: AGPL-3.0-only

//! The warm-turn switches: the deep tail cut, the trace line and its logits
//! hash, and what each environment value parses to.

use super::*;

const BS: usize = 16;

/// A `WarmTurn` with every switch off but the trace `mode`.
fn tracing(mode: TraceMode) -> WarmTurn {
    let mut warm = WarmTurn::from_env("glm5_next").unwrap();
    warm.trace = mode;
    warm
}

/// The base cut is what it was before the switch existed: one block below
/// the last block boundary strictly under the prompt end (`pc_policy`'s own
/// test pins `tail_cut`, which runs this with the switch off).
#[test]
fn the_base_tail_cut_is_unchanged() {
    for n in 0..2_000usize {
        let base = ((n.saturating_sub(1) / BS) * BS).saturating_sub(BS);
        assert_eq!(tail_cut_at(n, BS, false), base, "n={n}");
    }
}

/// The deep cut is the last block boundary strictly under the prompt end
/// (`ssm_tail_boundary`): one block above the base cut, 2 to 16 rows under
/// the end. It is at or below the match of a next turn whose history
/// reproduces this prompt (`floor(n/16)*16`), and above the match of one
/// that does not reproduce a suffix that crosses the boundary.
#[test]
fn the_deep_tail_cut_is_the_last_boundary_under_the_end() {
    assert_eq!(tail_cut_at(40_000, BS, true), 39_984);
    assert_eq!(tail_cut_at(40_002, BS, true), 40_000);
    assert_eq!(tail_cut_at(40_016, BS, true), 40_000);
    for n in 34..2_000 {
        let (deep, base) = (tail_cut_at(n, BS, true), tail_cut_at(n, BS, false));
        if n % BS == 1 {
            continue;
        }
        assert_eq!(Some(deep), spark_runtime::ssm_tail_boundary(n, BS), "n={n}");
        assert_eq!(deep, base + BS, "n={n}");
        assert!(n - deep >= 2 && n - deep <= BS);
        // A next turn that reproduces all `n` tokens matches this far.
        assert!(deep <= n / BS * BS);
        // One that diverges `suffix` tokens before the end matches less when
        // the suffix crosses the boundary; the base cut is still under it.
        for suffix in 1..=BS {
            let matched = (n - suffix) / BS * BS;
            assert_eq!(deep > matched, suffix > n - deep, "n={n} suffix={suffix}");
            assert!(base <= matched);
        }
    }
}

/// Two prompts keep the base cut under the switch: one that ends one row
/// past a boundary (a one-row final pass would take the decode path), and
/// one the base cut does not split (32 tokens or fewer: no second pass and no
/// checkpoint at token 16 appear that base does not have).
#[test]
fn the_deep_tail_cut_keeps_the_base_cut_where_it_would_add_a_pass_or_a_one_row_pass() {
    for n in (0..=33).chain([49, 40_001]) {
        let base = tail_cut_at(n, BS, false);
        assert_eq!(tail_cut_at(n, BS, true), base, "n={n}");
        if n <= 32 {
            assert_eq!(base, 0, "n={n}: one pass");
        } else {
            assert_eq!(n - base, BS + 1, "n={n}: a 17-row final pass");
        }
    }
    assert_eq!(tail_cut_at(34, BS, true), 32);
}

/// Only GLM-5's template makes the deep cut restorable; this process runs
/// without the switch, which any model accepts.
#[test]
fn the_switches_load_for_any_model_while_the_deep_cut_is_off() {
    assert!(!tail_cut_deep());
    for model in ["glm5_next", "qwen3_next"] {
        let warm = WarmTurn::from_env(model).unwrap();
        assert!(!warm.skip_cached);
        assert_eq!(
            (warm.trace, warm.zero_rows),
            (TraceMode::Off, ZeroRows::Off)
        );
    }
}

#[test]
fn the_trace_is_off_a_hash_line_or_a_synced_line_and_refuses_anything_else() {
    use TraceMode::*;
    assert_eq!(TraceMode::parse(None).unwrap(), Off);
    assert_eq!(TraceMode::parse(Some("0")).unwrap(), Off);
    assert_eq!(TraceMode::parse(Some("1")).unwrap(), Spans);
    assert_eq!(TraceMode::parse(Some("hash")).unwrap(), Hash);
    for bad in ["true", "2", "", "HASH"] {
        let e = TraceMode::parse(Some(bad)).unwrap_err();
        assert!(
            format!("{e:#}").contains("ATLAS_GLM_WARM_TRACE must"),
            "{e:#}"
        );
    }
}

/// The fingerprint tells apart rows that differ in any byte, in length, or
/// only past the last whole word.
#[test]
fn the_logits_hash_covers_every_byte_and_the_length() {
    let row: Vec<u8> = (0..=255u8).cycle().take(1003).collect();
    let h = logits_hash(&row);
    assert_eq!(h, logits_hash(&row.clone()));
    for at in [0, 7, 8, 500, 999, 1000, 1002] {
        let mut other = row.clone();
        other[at] ^= 1;
        assert_ne!(logits_hash(&other), h, "byte {at}");
    }
    assert_ne!(logits_hash(&row[..1002]), h);
    assert_ne!(logits_hash(&[]), logits_hash(&[0]));
    assert_ne!(logits_hash(&[0; 8]), logits_hash(&[0; 9]));
}

fn shape() -> RequestShape {
    RequestShape {
        rank: 1,
        prompt: 40,
        matched: 32,
        restored: 16,
        logits: 0xabc,
    }
}

#[test]
fn the_trace_line_sums_a_requests_chunks_and_forgets_them() {
    let ms = Duration::from_millis;
    let warm = tracing(TraceMode::Spans);
    let began = Instant::now();
    *warm.transfer.lock() += ms(2);
    // Two cached chunks: no rows, a lookup and a block vote.
    let cached = [ms(0), ms(0), ms(3), ms(1), ms(0), ms(0), ms(0)];
    assert_eq!(warm.note_chunk(7, began, true, 0, cached, None), None);
    assert_eq!(warm.note_chunk(7, began, false, 0, cached, None), None);
    // Another slot's chunk in between stays out of this request's line.
    assert_eq!(warm.note_chunk(2, began, true, 99, [ms(50); 7], None), None);
    let pass = [ms(17), ms(2), ms(0), ms(1), ms(1), ms(100), ms(5)];
    assert_eq!(warm.note_chunk(7, began, false, 8, pass, None), None);
    let line = warm
        .note_chunk(7, began, false, 16, pass, Some(shape()))
        .unwrap();
    assert!(
        line.starts_with(
            "warm-turn rank=1 slot=7 tokens=40 matched=32 restored=16 chunks=4 \
             cached_chunks=2 rows=24 ms: transfer=2.0 zero=34.0 embed=4.0 lookup=6.0 \
             blocks=4.0 meta=2.0 forward=200.0 finish=10.0 wall="
        ),
        "{line}"
    );
    assert!(line.ends_with(" logits=0000000000000abc"), "{line}");
    // The wall clock starts with the first chunk's transfer.
    let at = line.find("wall=").unwrap() + 5;
    let wall: f64 = line[at..].split(' ').next().unwrap().parse().unwrap();
    assert!(wall >= 2.0, "{line}");
    // The request is gone; the other slot's chunk is still pending.
    let next = warm
        .note_chunk(7, began, false, 1, pass, Some(shape()))
        .unwrap();
    assert!(next.contains("chunks=1 cached_chunks=0 rows=1 "), "{next}");
    let other = warm
        .note_chunk(2, began, false, 1, pass, Some(shape()))
        .unwrap();
    assert!(
        other.contains("chunks=2 cached_chunks=0 rows=100 "),
        "{other}"
    );
}

/// A request that failed or was preempted before its last chunk leaves a
/// trace behind; the slot's next request starts over at its first chunk.
#[test]
fn a_first_chunk_drops_what_an_unfinished_request_left() {
    let ms = Duration::from_millis;
    let warm = tracing(TraceMode::Hash);
    let pass = [ms(1); 7];
    let long_ago = Instant::now() - ms(60_000);
    assert_eq!(warm.note_chunk(3, long_ago, true, 8, pass, None), None);
    let line = warm
        .note_chunk(3, Instant::now(), true, 4, pass, Some(shape()))
        .unwrap();
    assert!(line.contains("chunks=1 cached_chunks=0 rows=4 "), "{line}");
    let at = line.find("wall=").unwrap() + 5;
    let wall: f64 = line[at..].split(' ').next().unwrap().parse().unwrap();
    assert!(wall < 30_000.0, "{line}");
}

/// Bulk token broadcasts are timed only while the trace is on, and charged
/// to the next chunk noted.
#[test]
fn a_transfer_is_timed_only_with_the_trace_on() {
    let off = tracing(TraceMode::Off);
    assert!(off.transfer_span().is_none());
    for mode in [TraceMode::Hash, TraceMode::Spans] {
        let warm = tracing(mode);
        let span = warm.transfer_span().unwrap();
        std::thread::sleep(Duration::from_millis(2));
        drop(span);
        assert!(*warm.transfer.lock() >= Duration::from_millis(2));
        let line = warm
            .note_chunk(
                0,
                Instant::now(),
                true,
                1,
                [Duration::ZERO; 7],
                Some(shape()),
            )
            .unwrap();
        assert!(!line.contains("transfer=0.0 "), "{line}");
        assert_eq!(*warm.transfer.lock(), Duration::ZERO);
    }
}

#[test]
fn zero_rows_is_off_trim_or_check_and_refuses_anything_else() {
    use ZeroRows::*;
    assert_eq!(ZeroRows::parse(None, None).unwrap(), Off);
    assert_eq!(ZeroRows::parse(Some("0"), None).unwrap(), Off);
    assert_eq!(ZeroRows::parse(Some("1"), None).unwrap(), Trim(2048));
    assert_eq!(ZeroRows::parse(Some("check"), None).unwrap(), Check(2048));
    assert_eq!(
        ZeroRows::parse(Some("1"), Some("4096")).unwrap(),
        Trim(4096)
    );
    assert_eq!(
        ZeroRows::parse(Some("check"), Some("256")).unwrap(),
        Check(256)
    );
    // A floor without the switch is only read.
    assert_eq!(ZeroRows::parse(None, Some("512")).unwrap(), Off);
    for bad in ["true", "2", "", "CHECK"] {
        let e = ZeroRows::parse(Some(bad), None).unwrap_err();
        assert!(
            format!("{e:#}").contains("ATLAS_GLM_ZERO_ROWS must"),
            "{e:#}"
        );
    }
    // A floor under a batched verify's rows could not cover a decode step.
    for bad in ["255", "0", "rows", "-1"] {
        let e = ZeroRows::parse(Some("1"), Some(bad)).unwrap_err();
        assert!(format!("{e:#}").contains("_FLOOR must"), "{e:#}");
    }
}
