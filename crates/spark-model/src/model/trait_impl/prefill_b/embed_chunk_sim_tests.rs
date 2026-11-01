// SPDX-License-Identifier: AGPL-3.0-only

//! Every computed placeholder reads its own item's own row, whatever the
//! cache covers and wherever the chunks cut.
//!
//! The per-chunk encode (`vision_cache`) and the splice (`vision_splice_plan`)
//! agree on a window only through arithmetic: which items a chunk encodes,
//! the window's virtual base, the rows a cached prefix skips. This drives the
//! real selection (`items_in_range`, `cached_items`, `encode_window`) and the
//! real plan over random prompts, cached prefixes and chunkings (cuts inside
//! images, chunks wholly in the cache, a straddling cached item, several
//! items per chunk), and checks each computed placeholder against the item
//! and offset it belongs to, as if the window held each item encoded alone.

use super::vision_splice_plan;
use crate::model::vision_cache::{
    cached_items, encode_window, first_pad_positions, items_in_range, last_pad_positions,
};

const PAD: u32 = 10;
const TEXT: u32 = 1;

fn is_pad(t: u32) -> bool {
    t == PAD
}

/// Tiny deterministic generator (xorshift).
struct Rng(u64);
impl Rng {
    fn next(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// One prompt: `counts[i]` pads per item, `gaps[i]` text tokens before item i
/// (and `gaps[n]` after the last).
fn prompt(counts: &[usize], gaps: &[usize]) -> (Vec<u32>, Vec<Option<(usize, usize)>>) {
    let (mut tokens, mut truth) = (Vec::new(), Vec::new());
    for (i, &c) in counts.iter().enumerate() {
        tokens.extend(std::iter::repeat_n(TEXT, gaps[i]));
        truth.extend(std::iter::repeat_n(None, gaps[i]));
        for k in 0..c {
            tokens.push(PAD);
            truth.push(Some((i, k)));
        }
    }
    tokens.extend(std::iter::repeat_n(TEXT, gaps[counts.len()]));
    truth.extend(std::iter::repeat_n(None, gaps[counts.len()]));
    (tokens, truth)
}

/// Run every chunk as the prefill does and check each computed placeholder.
fn check(counts: &[usize], gaps: &[usize], computed_from: usize, chunks: &[(usize, usize)]) {
    let (tokens, truth) = prompt(counts, gaps);
    let first = first_pad_positions(&tokens, is_pad, counts);
    let last = last_pad_positions(&tokens, is_pad, counts);
    let (_, skip_rows) = cached_items(&tokens, is_pad, counts, computed_from);
    let (mut rows_base, mut pending, mut window): (usize, usize, Vec<usize>) = (0, 0, Vec::new());
    let ctx = || format!("counts={counts:?} gaps={gaps:?} from={computed_from} chunks={chunks:?}");
    for (ci, &(s, e)) in chunks.iter().enumerate() {
        // Chunk 0 encodes inside its lookup from the computed position; a
        // later chunk from its own start (`vision_resolve_deferred`,
        // `vision_encode_chunk_items`).
        let from = if ci == 0 { computed_from } else { s };
        let items = items_in_range(&first, &last, computed_from, from, e);
        // A chunk with no computed placeholder encodes nothing.
        let lo = from.max(computed_from);
        if !(lo..e).any(|p| truth[p].is_some()) {
            assert!(
                items.is_empty(),
                "chunk [{s},{e}) encodes {items:?}: {}",
                ctx()
            );
        }
        let (a, rows) = encode_window(counts, &items, rows_base, skip_rows);
        if rows > 0 {
            (rows_base, pending, window) = (a, rows, items);
        }
        // The full-chunk splice, then the uncached re-embed's.
        let cs = s.max(computed_from);
        for (ss, ee) in [(s, e), (cs, e)] {
            if ss >= ee {
                continue;
            }
            let plan = vision_splice_plan(
                &tokens,
                ss,
                ee - ss,
                is_pad,
                0,
                skip_rows,
                computed_from,
                rows_base,
                pending,
            )
            .unwrap_or_else(|e| panic!("{e:#} at chunk [{ss},{ee}): {}", ctx()));
            let want: Vec<usize> = (ss.max(computed_from)..ee)
                .filter(|&p| truth[p].is_some())
                .collect();
            let got: Vec<usize> = plan.iter().map(|&(i, _)| ss + i).collect();
            assert_eq!(got, want, "computed pads of [{ss},{ee}): {}", ctx());
            for &(i, physical) in &plan {
                // The window holds `window`'s items packed in order from
                // virtual row `rows_base`.
                let (mut at, mut found) = (0usize, None);
                for &j in &window {
                    if physical < at + counts[j] {
                        found = Some((j, physical - at));
                        break;
                    }
                    at += counts[j];
                }
                assert_eq!(found, truth[ss + i], "position {}: {}", ss + i, ctx());
            }
        }
    }
}

#[test]
fn the_issue_180_warm_sequences() {
    // The GPU window's requests scaled down 64x: a 2048^2 image is 64 rows,
    // a 16384-token chunk 256 tokens, the 20480 cache snapshot 320. a5, then
    // b16680 (3 of 5 cached, computed from 256, inside item 3), d8 (4 of 8
    // cached, computed from 320, inside item 4: the sixth image read the
    // fifth's rows) and mem4 (7 of 8 cached, from 512).
    for size in [256, 64] {
        let chunks = |total: usize| -> Vec<(usize, usize)> {
            (0..total)
                .step_by(size)
                .map(|s| (s, (s + size).min(total)))
                .collect()
        };
        let gaps5 = [8, 2, 2, 2, 2, 6];
        check(&[64; 5], &gaps5, 0, &chunks(64 * 5 + 22));
        check(&[64, 64, 64, 64, 5], &gaps5, 256, &chunks(4 * 64 + 5 + 22));
        let gaps8 = [8, 2, 2, 2, 2, 2, 2, 2, 6];
        for from in [0, 256, 320, 448, 512] {
            check(&[64; 8], &gaps8, from, &chunks(64 * 8 + 28));
        }
    }
}

#[test]
fn random_prompts_caches_and_chunkings() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..4000 {
        let n = 1 + rng.next(7);
        let counts: Vec<usize> = (0..n).map(|_| 1 + rng.next(40)).collect();
        let gaps: Vec<usize> = (0..=n).map(|_| rng.next(4)).collect();
        let total: usize = counts.iter().sum::<usize>() + gaps.iter().sum::<usize>();
        if total == 0 {
            continue;
        }
        let computed_from = match rng.next(3) {
            0 => 0,
            1 => rng.next(total),
            _ => (rng.next(total) / 16) * 16,
        };
        let max_chunk = 1 + rng.next(60);
        let mut chunks = Vec::new();
        let mut s = 0;
        while s < total {
            let e = (s + 1 + rng.next(max_chunk)).min(total);
            chunks.push((s, e));
            s = e;
        }
        check(&counts, &gaps, computed_from, &chunks);
    }
}
