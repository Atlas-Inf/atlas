// SPDX-License-Identifier: AGPL-3.0-only

//! System-content and dedup helpers (extracted from `msg_entry.rs`).

use super::*;

/// True when a system message carries no actual instruction and should
/// be treated as absent. Conservative by design — a substantive prompt
/// must never be stripped:
///   * empty / whitespace-only body, OR
///   * a single short bare label line ending in ':' with nothing after
///     it (e.g. `User Context:`, `Context:`, `System:`) — the residue
///     of an empty client template (Open WebUI's RAG/context block).
/// Anything multi-line, or with any text past the colon, is a real
/// prompt and returns false.
pub(super) fn is_vacuous_system_content(content: &str) -> bool {
    let t = content.trim();
    if t.is_empty() {
        return true;
    }
    if !t.contains('\n') && t.len() <= 32 && t.ends_with(':') {
        let label = &t[..t.len() - 1];
        return !label.is_empty()
            && label
                .chars()
                .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '_' || c == '-');
    }
    false
}

/// P1-6 (2026-07-09): kill-switch — `ATLAS_NO_ERROR_DEDUP=1` restores
/// verbatim duplicate-error history (disables the masking pass).
pub(super) fn error_dedup_disabled() -> bool {
    std::env::var("ATLAS_NO_ERROR_DEDUP").as_deref() == Ok("1")
}

/// P1-6 (2026-07-09): duplicate-error observation masking.
///
/// Input: `(message_index, original_pre_hint_text)` for every
/// tool-result entry, in conversation order. Output: `(message_index,
/// replacement_text)` for the OLDER members of each duplicate-error
/// group; the newest member of each group stays verbatim. Two
/// tool results are duplicates when BOTH are error-shaped
/// (`crate::hint_injector::looks_like_error`) AND either equal after
/// trim or Jaccard >= 0.9 over 4-gram shingles (the loop_detector
/// measure — SSOT). Successful outputs never participate: identical
/// success observations (e.g. repeated `ls`) are legitimate.
pub(super) fn duplicate_error_masks(tool_results: &[(usize, String)]) -> Vec<(usize, String)> {
    const NEAR_DUP_JACCARD: f64 = 0.9;
    let errors: Vec<(usize, &str)> = tool_results
        .iter()
        .filter(|(_, t)| crate::hint_injector::looks_like_error(t))
        .map(|(i, t)| (*i, t.trim()))
        .collect();
    if errors.len() < 2 {
        return Vec::new();
    }
    let shingle_sets: Vec<_> = errors
        .iter()
        .map(|(_, t)| crate::loop_detector::shingle_set(t))
        .collect();
    // First-match grouping against each group's first member. Short
    // errors (< 4 tokens) have empty shingle sets — jaccard() returns
    // 0.0 for those, so they group via exact-trim match only.
    let mut groups: Vec<Vec<usize>> = Vec::new(); // indices into `errors`
    for i in 0..errors.len() {
        let group = groups.iter_mut().find(|g| {
            let rep = g[0];
            errors[rep].1 == errors[i].1
                || crate::loop_detector::jaccard(&shingle_sets[rep], &shingle_sets[i])
                    >= NEAR_DUP_JACCARD
        });
        match group {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    let mut masks = Vec::new();
    for g in &groups {
        let n = g.len();
        if n < 2 {
            continue;
        }
        // Keep the LAST (newest) verbatim; mask the earlier ones.
        for (k, &ei) in g.iter().take(n - 1).enumerate() {
            masks.push((
                errors[ei].0,
                format!("[same error as below, attempt {} of {}]", k + 1, n),
            ));
        }
    }
    masks
}
