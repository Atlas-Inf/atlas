// SPDX-License-Identifier: AGPL-3.0-only

//! Which placeholders belong to which vision item, and which items a chunk
//! encodes (`vision_cache`).

/// Leading items whose every placeholder sits before `computed_from`, and
/// their placeholder (= encoder row) count. `counts` holds each item's
/// placeholders in prompt order.
pub(crate) fn cached_items(
    tokens: &[u32],
    is_pad: impl Fn(u32) -> bool,
    counts: &[usize],
    computed_from: usize,
) -> (usize, usize) {
    let (mut item, mut rows, mut seen) = (0usize, 0usize, 0usize);
    for (pos, &t) in tokens.iter().enumerate() {
        if item == counts.len() || pos >= computed_from {
            break;
        }
        if is_pad(t) {
            seen += 1;
            if seen == counts[item] {
                rows += seen;
                item += 1;
                seen = 0;
            }
        }
    }
    (item, rows)
}

/// The items a chunk must encode: those with a placeholder AT OR AFTER
/// `computed_from` inside `[from, to)` (the chunk's computed range), as
/// indices into the item list. An item straddling a chunk boundary is
/// RE-encoded at every chunk holding one of its placeholders: the encoder
/// writes its packed output at `buf_out` row 0, so a later chunk could not
/// index the earlier chunk's resident tail under one window base — and the
/// re-encode makes the resident window exactly this chunk's splice range.
/// The re-encoded head rows were already spliced; only the pads this chunk
/// computes are read. An item straddling `computed_from` (its cached head is
/// skipped) is encoded at the chunk that first computes one of its
/// placeholders.
///
/// `first_pad[i]`/`last_pad[i]` are item i's first/last placeholder positions
/// in the prompt; `computed_from` is the first prompt position the prefill
/// computes.
pub(crate) fn items_in_range(
    first_pad: &[usize],
    last_pad: &[usize],
    computed_from: usize,
    from: usize,
    to: usize,
) -> Vec<usize> {
    first_pad
        .iter()
        .zip(last_pad)
        .enumerate()
        .filter(|&(_, (&first, &last))| {
            let first = first.max(computed_from);
            first < to && last >= from
        })
        .map(|(i, _)| i)
        .collect()
}

/// First placeholder position of every item, in prompt order. Items' pad runs
/// may be split by text (a GLM video's timestamp tokens), so the first pad
/// after the previous item's last one is the item's own first — the same walk
/// [`cached_items`] uses to attribute pads to items. An item with no pad in
/// `tokens` gets `usize::MAX` (never inside any chunk range).
pub(crate) fn first_pad_positions(
    tokens: &[u32],
    is_pad: impl Fn(u32) -> bool,
    counts: &[usize],
) -> Vec<usize> {
    let mut out = Vec::with_capacity(counts.len());
    let (mut item, mut seen) = (0usize, 0usize);
    for (pos, &t) in tokens.iter().enumerate() {
        if item == counts.len() {
            break;
        }
        if is_pad(t) {
            if seen == 0 {
                out.push(pos);
            }
            seen += 1;
            if seen == counts[item] {
                item += 1;
                seen = 0;
            }
        }
    }
    out.resize(counts.len(), usize::MAX);
    out
}

/// Last placeholder position of every item, in prompt order — the same walk
/// [`first_pad_positions`] uses, recording the position each item's run ends
/// on. An item with no pad in `tokens` gets 0 (never passes a `>= from` test
/// against a real chunk start).
pub(crate) fn last_pad_positions(
    tokens: &[u32],
    is_pad: impl Fn(u32) -> bool,
    counts: &[usize],
) -> Vec<usize> {
    let mut out = vec![0; counts.len()];
    let (mut item, mut seen) = (0usize, 0usize);
    for (pos, &t) in tokens.iter().enumerate() {
        if item == counts.len() {
            break;
        }
        if is_pad(t) {
            seen += 1;
            if seen == counts[item] {
                out[item] = pos;
                item += 1;
                seen = 0;
            }
        }
    }
    out
}
