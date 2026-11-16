// SPDX-License-Identifier: AGPL-3.0-only

//! The late layers of a prefill on the rows that matter.
//!
//! After the last layer that produces compressed keys, a layer leaves two
//! things behind: its window, which is the keys of the chunk's last 128
//! positions, and the stream of the rows the layers after it read. The last
//! layer's stream is read at one row, the chunk's last. So layer `L` of
//! `layers` needs correct outputs for the rows the next layer runs, and
//! keys for 128 rows before those, which come from its inputs and not from
//! its outputs: it runs the last `128 * (layers - L)` rows of a chunk, and
//! the first 128 of them, which attend over keys it never made, leave
//! outputs nothing reads. Nothing that is read changes.
//!
//! That is true only while nothing reads the last layer's other rows. A
//! request that wants the logits of every prompt position does, so the
//! prefill that serves it runs every row ([`whole`]); and the layers that
//! feed something else per row (key sources and the layers before them,
//! Engram layers) are never cut short ([`rows_of_layer`]).
//!
//! Only the rank that serves the request knows which kind it is, and the
//! ranks must run the same rows: the first layer of every chunk has rank 0
//! tell the others (`Layer::agree_on_rows`), and the layers after it go by
//! what was said.

use std::cell::Cell;

use super::cache_recipe::WINDOW;

thread_local! {
    static WHOLE: Cell<bool> = const { Cell::new(false) };
}

/// Keeps every row of the prefills on this thread until dropped.
pub struct Whole {
    was: bool,
}

/// Run every row of every layer for the prefills on this thread until the
/// guard drops: for a pass whose every row's final stream is read.
pub fn whole() -> Whole {
    Whole {
        was: WHOLE.with(|w| w.replace(true)),
    }
}

impl Drop for Whole {
    fn drop(&mut self) {
        WHOLE.with(|w| w.set(self.was));
    }
}

/// The rows of a chunk's end layer `layer` of `layers` may run alone, if it
/// may: `per_row` are the layers something reads at every row.
/// `ATLAS_DSV41_NO_TAIL` runs every row everywhere.
pub fn rows_of_layer(layer: usize, layers: usize, per_row: &[usize]) -> Option<usize> {
    let free = per_row.iter().all(|&l| l < layer);
    (free && std::env::var_os("ATLAS_DSV41_NO_TAIL").is_none()).then_some(WINDOW * (layers - layer))
}

/// Whether a pass on this thread is one that reads every row ([`whole`]).
pub fn whole_here() -> bool {
    WHOLE.with(Cell::get)
}

/// Where in a chunk of `tokens` rows a layer that may run its last `tail`
/// alone starts, in a pass that reads every row (`whole`) or does not.
pub fn first_row(tail: Option<usize>, tokens: usize, whole: bool) -> usize {
    match tail {
        Some(tail) if !whole => tokens.saturating_sub(tail),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{first_row, rows_of_layer, whole, whole_here};

    #[test]
    fn a_late_layer_runs_the_rows_the_layers_after_it_read() {
        let per_row = [1, 2, 8, 14, 20];
        // Up to the last layer something reads per row, every row.
        assert_eq!(rows_of_layer(20, 40, &per_row), None);
        assert_eq!(rows_of_layer(3, 40, &per_row), None);
        // After it, a window more for each layer still to come.
        assert_eq!(rows_of_layer(39, 40, &per_row), Some(128));
        assert_eq!(rows_of_layer(21, 40, &per_row), Some(19 * 128));
        let tail = rows_of_layer(38, 40, &per_row);
        assert_eq!(first_row(tail, 8192, false), 8192 - 256);
        // A chunk shorter than the tail runs whole.
        assert_eq!(first_row(tail, 100, false), 0);
        assert_eq!(first_row(None, 8192, false), 0);
        // And a pass that reads every row runs every row.
        assert_eq!(first_row(tail, 8192, true), 0);
    }

    #[test]
    fn a_pass_that_reads_every_row_says_so_until_it_ends() {
        assert!(!whole_here());
        {
            let _all = whole();
            assert!(whole_here());
            {
                let _nested = whole();
            }
            assert!(whole_here());
        }
        assert!(!whole_here());
    }
}
