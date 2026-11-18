// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn the_pair_step_costs_track_its_measured_steps() {
    // Median step by (owners, rows per owner), a range by its middle
    // (0.3.4 engine, 2026-11-14).
    for (owners, rows, ms) in [
        (1, 2, 46.8),
        (1, 3, 53.5),
        (1, 4, 61.0),
        (1, 5, 65.9),
        (1, 6, 69.9),
        (1, 8, 82.5),
        (2, 2, 66.3),
        (2, 3, 79.5),
        (2, 4, 89.0),
        (2, 5, 101.0),
        (2, 7, 117.0),
        (2, 8, 128.0),
        (4, 2, 100.6),
        (4, 3, 123.5),
        (4, 4, 136.5),
        (4, 5, 155.0),
        (4, 6, 166.0),
        (4, 7, 179.0),
        (4, 8, 189.0),
    ] {
        let fit = uniform_step_ms(owners, rows);
        assert!(
            (fit - ms).abs() / ms < 0.10,
            "{owners} x {rows}: fit {fit} vs measured {ms}"
        );
    }
    // A step costs more with every owner and every row, and a row less with
    // every owner up to four.
    for n in 1..8 {
        assert!(uniform_step_ms(n + 1, 3) > uniform_step_ms(n, 3));
        assert!(uniform_step_ms(n, 4) > uniform_step_ms(n, 3));
        assert!(live_row_ms(n + 1) <= live_row_ms(n));
    }
    assert_eq!(live_row_ms(3), 4.2);
    assert_eq!(live_row_ms(8), live_row_ms(4));
    // Never past the 32-row verify budget.
    assert!(uniform_step_ms(4, 8).is_finite() && uniform_step_ms(4, 9).is_infinite());
    assert!(uniform_step_ms(8, 4).is_finite() && uniform_step_ms(8, 5).is_infinite());
}

#[test]
fn a_dead_row_saves_a_live_rows_cost_less_its_own() {
    for owners in 1..=8 {
        for rows in 2..=(32 / owners).min(8) {
            let total = owners * rows;
            let uniform = uniform_step_ms(owners, rows);
            // No dead row: the uniform fit.
            assert_eq!(step_ms_live(owners, rows, total), uniform);
            assert_eq!(step_ms_live_at(owners, rows, total, 0.0), uniform);
            // Each dead row saves a live row's cost less the dead row's.
            let saved = live_row_ms(owners) - PAIR_DEAD_ROW_MS;
            for dead in 1..=total - owners {
                let live = total - dead;
                let want = uniform - dead as f32 * saved;
                let got = step_ms_live_at(owners, rows, live, PAIR_DEAD_ROW_MS);
                assert!(
                    (got - want).abs() < 1e-3,
                    "{owners} x {rows}, {dead} dead: {got} vs {want}"
                );
                assert!(got < step_ms_live_at(owners, rows, live + 1, PAIR_DEAD_ROW_MS));
            }
        }
    }
    // More live rows than the batch holds are the batch's rows.
    assert_eq!(step_ms_live(2, 3, 9), uniform_step_ms(2, 3));
    // A dead row never costs more than a live one, and past the budget the
    // step never runs however many of its rows are dead.
    assert_eq!(step_ms_live_at(4, 4, 8, 9.0), uniform_step_ms(4, 4));
    assert!(step_ms_live(4, 9, 8).is_infinite());
}

#[test]
fn the_dead_row_cost_override_takes_only_a_cost() {
    assert_eq!(parse_dead_row_ms(None), PAIR_DEAD_ROW_MS);
    assert_eq!(parse_dead_row_ms(Some(" 1.25 ")), 1.25);
    assert_eq!(parse_dead_row_ms(Some("0")), 0.0);
    for bad in ["", "fast", "-1", "nan", "inf"] {
        assert_eq!(parse_dead_row_ms(Some(bad)), PAIR_DEAD_ROW_MS, "{bad:?}");
    }
}
