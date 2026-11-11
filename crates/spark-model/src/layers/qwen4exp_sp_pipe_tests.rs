// SPDX-License-Identifier: AGPL-3.0-only

use super::{PIECE, Pieces};

fn pieces(m: usize, lead: usize, rows: usize) -> Pieces {
    Pieces {
        m,
        lead,
        rows,
        sent: 0,
        done: 0,
    }
}

/// Walk the slabs: every piece goes out exactly once, in order, and only
/// once all the local rows its window part covers are done.
#[test]
fn pieces_wait_for_their_rows_and_cover_the_window() {
    for (m, lead, rows) in [
        (8192, 0, 8192),
        (8192, 338, 7854),
        (8192, 0, 7854),
        (2048, 0, 2048),
        (7000, 0, 7000),
        (9000, 1000, 8000),
    ] {
        let mut p = pieces(m, lead, rows);
        let mut covered = 0;
        let mut t = 0;
        while t < rows {
            t = (t + PIECE).min(rows);
            p.done = t;
            while let Some((w0, n)) = p.next_ready() {
                assert_eq!(w0, covered, "{m}/{lead}/{rows}: in order");
                // Window rows [w0, w0 + n) are local rows [w0 - lead, ..).
                assert!((w0 + n).saturating_sub(lead).min(rows) <= p.done);
                covered += n;
                p.sent += 1;
            }
        }
        assert_eq!(covered, m, "{m}/{lead}/{rows}: the whole window");
        assert_eq!(p.sent, p.count());
    }
}

/// An offer is visible only while the caller holds it, and says whether the
/// site took it.
#[test]
fn a_reduce_scatter_offer_lives_with_its_caller() {
    use super::{RsOffer, rs_offered, rs_took};
    assert!(!rs_offered(), "no caller, no offer");
    let offer = RsOffer::new();
    assert!(rs_offered() && !offer.taken());
    rs_took();
    assert!(offer.taken() && !rs_offered(), "taken once");
    drop(offer);
    assert!(!rs_offered());
    let untaken = RsOffer::new();
    assert!(!untaken.taken());
}
