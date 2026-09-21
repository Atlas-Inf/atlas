// SPDX-License-Identifier: AGPL-3.0-only

//! Keep every row of a speculative verify below the served context ceiling.
//!
//! The serial path stops a sequence one token BEFORE the ceiling
//! (`helpers::seqlen_force_stop`: `position + 1 >= max_seq_len`), so a live
//! sequence can sit at `seq_len = max_seq_len - 2`. A verify runs
//! `num_drafts + 1` rows at positions `seq_len ..= seq_len + num_drafts`. One
//! draft ends at `max_seq_len - 1` and fits; two or more can put a row AT the
//! ceiling — a position the serve never promised. The KV pool carries
//! speculative headroom, but per-sequence state sized to `--max-seq-len` does
//! not (the QSA indexer refuses `pos >= capacity` mid-forward), and a
//! mid-forward refusal ends the request with `finish_reason: "error"` on what
//! should have been its last `"length"` step.
//!
//! Declining the speculative step there costs at most the final `num_drafts`
//! steps of a sequence that is about to be cut off anyway: it decodes
//! serially and reaches the existing force-stop. Verify shapes are fixed per
//! draft count, so declining is simpler and safer than shrinking one step.

/// Whether every row of a verify with `num_drafts` drafts, for a sequence at
/// `seq_len`, lands below `max_seq_len` (`0` = no ceiling).
pub fn verify_rows_fit(seq_len: usize, num_drafts: usize, max_seq_len: usize) -> bool {
    max_seq_len == 0 || seq_len + num_drafts < max_seq_len
}

#[cfg(test)]
mod tests {
    use super::verify_rows_fit;

    #[test]
    fn rows_fit_table_around_the_ceiling() {
        const MAX: usize = 32768;
        // The serial force-stop keeps a live sequence at or below MAX - 2.
        // Expected, written out by hand: [num_drafts 1, 2, 3] per row.
        let table: [(usize, [bool; 3]); 5] = [
            (MAX - 5, [true, true, true]),    // last rows at MAX-4, MAX-3, MAX-2
            (MAX - 4, [true, true, true]),    // num_drafts 3 ends at MAX-1
            (MAX - 3, [true, true, false]),   // num_drafts 3 would run a row AT MAX
            (MAX - 2, [true, false, false]),  // one draft ends at MAX-1: still fits
            (MAX - 1, [false, false, false]), // not a live position; never fits
        ];
        for (seq_len, expected) in table {
            for (j, &num_drafts) in [1usize, 2, 3].iter().enumerate() {
                assert_eq!(
                    verify_rows_fit(seq_len, num_drafts, MAX),
                    expected[j],
                    "seq_len={seq_len} num_drafts={num_drafts}"
                );
            }
        }
    }

    #[test]
    fn the_default_draft_depth_never_loses_a_live_step() {
        // One draft: every position the force-stop leaves alive still fits.
        const MAX: usize = 4096;
        for seq_len in 0..=MAX - 2 {
            assert!(verify_rows_fit(seq_len, 1, MAX), "seq_len={seq_len}");
        }
    }

    #[test]
    fn no_ceiling_means_no_clamp() {
        assert!(verify_rows_fit(usize::MAX / 2, 7, 0));
    }
}
