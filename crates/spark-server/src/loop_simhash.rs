// SPDX-License-Identifier: AGPL-3.0-only

//! F4 (2026-04-26): Jaccard sentence-level loop guard.
//!
//! Catches the agentic-failure mode where a model emits the same
//! sentence (semantically — modulo case, punctuation, and minor
//! word-substitution paraphrase) over and over within a single
//! response. Token-level period detection (`detect_token_loop` in
//! scheduler.rs) cannot catch this because:
//!
//! 1. Tool calls between repeated paragraphs push the period beyond
//!    `CONTENT_LOOP_PERIOD_MAX=64`.
//! 2. Surface variations ("Rust Axum"/"Rust axum", "the /echo"/"an
//!    /echo") break exact-token matching.
//!
//! Algorithm:
//!
//! - Normalize input (lowercase ASCII, strip punctuation, collapse
//!   whitespace) so case- and punctuation-paraphrases collide.
//! - Build a `BTreeSet<u64>` of word-bigram hashes — order-preserving,
//!   dedup'd within a sentence.
//! - Maintain a ring buffer of the last `cap` bigram-sets.
//! - On each new sentence, return `true` if the Jaccard similarity
//!   to any prior set in the ring is at or above `jaccard_threshold`.
//!
//! Empirically (per the test fixtures from the user's failed Claude
//! Code session): paraphrased restarts give Jaccard ≈ 0.55-0.70,
//! while legitimately distinct prose at typical sentence lengths
//! gives Jaccard < 0.10. Default threshold 0.55 sits comfortably
//! between the two.
//!
//! `check` is `O(|A| + |B|)` per ring entry — for typical 10-15
//! bigrams per sentence and ring=16, ~3-5 µs per call.
//!
//! Fenced code blocks (triple-backtick): sentence splitting inside code
//! is a false-positive source by construction — docstring conventions
//! make sibling methods near-identical bigram-for-bigram, and code
//! lines trip the `.`/`:`/blank-line boundaries constantly. The
//! streaming consumer therefore tracks fence state (`simhash_step`):
//! inside an unclosed fence no boundary or force-flush is evaluated
//! and the whole block — up to `FENCE_PENDING_CAP` — is hashed once at
//! its closing fence. Two identical blocks emitted back to back still
//! trip, since each close is its own `check`.

use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

const DEFAULT_RING_CAP: usize = 16;
const DEFAULT_JACCARD_THRESHOLD_PCT: u32 = 55; // i.e. 0.55
const WORD_SHINGLE_LEN: usize = 2;
const MIN_NORMALIZED_LEN: usize = 24; // skip very short sentences
const MIN_WORDS: usize = 3; // need ≥ WORD_SHINGLE_LEN+1 words

/// Sentence-level loop guard.
#[derive(Debug)]
pub struct SimHashLoopGuard {
    ring: VecDeque<BTreeSet<u64>>,
    cap: usize,
    threshold_pct: u32,
}

impl Default for SimHashLoopGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl SimHashLoopGuard {
    /// Construct with default cap=16, Jaccard threshold=0.55.
    pub fn new() -> Self {
        Self::with_params(DEFAULT_RING_CAP, DEFAULT_JACCARD_THRESHOLD_PCT)
    }

    /// Construct with explicit ring capacity and Jaccard threshold
    /// expressed as a percentage 0..=100 (e.g. 55 = 0.55).
    pub fn with_params(cap: usize, threshold_pct: u32) -> Self {
        Self {
            ring: VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
            threshold_pct,
        }
    }

    /// Check whether `sentence` is a near-duplicate of any sentence
    /// already in the ring. Returns `true` on duplicate. Always
    /// pushes the new bigram-set into the ring (so subsequent calls
    /// see it). Sentences whose normalized form is shorter than
    /// `MIN_NORMALIZED_LEN` are accepted but not counted as a
    /// duplicate.
    pub fn check(&mut self, sentence: &str) -> bool {
        let normalized = normalize(sentence);
        if normalized.len() < MIN_NORMALIZED_LEN {
            return false;
        }
        let shingles = bigram_set(&normalized);
        if shingles.is_empty() {
            return false;
        }
        let dup = self
            .ring
            .iter()
            .any(|prev| jaccard_pct(&shingles, prev) >= self.threshold_pct);
        if self.ring.len() >= self.cap {
            self.ring.pop_front();
        }
        self.ring.push_back(shingles);
        dup
    }

    /// Drop all stored signatures.
    pub fn reset(&mut self) {
        self.ring.clear();
    }

    /// Number of signatures currently stored.
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    /// True iff the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

/// Lowercase, strip non-alphanumeric, collapse whitespace.
fn normalize(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut last_was_space = true; // suppress leading whitespace
    for &b in s.as_bytes() {
        let c = b.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_was_space = false;
        } else if (c == b' ' || c == b'\t' || c == b'\n' || c == b'\r') && !last_was_space {
            out.push(b' ');
            last_was_space = true;
        }
        // skip punctuation entirely
    }
    if out.last() == Some(&b' ') {
        out.pop();
    }
    out
}

/// Build a deduplicated set of word-bigram hashes from the
/// normalized byte sequence. Each bigram is the joined pair of
/// adjacent words separated by a single space.
fn bigram_set(bytes: &[u8]) -> BTreeSet<u64> {
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return BTreeSet::new(),
    };
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < MIN_WORDS {
        return BTreeSet::new();
    }
    let mut set = BTreeSet::new();
    for window in words.windows(WORD_SHINGLE_LEN) {
        let mut h = DefaultHasher::new();
        for (i, w) in window.iter().enumerate() {
            if i > 0 {
                h.write_u8(b' ');
            }
            h.write(w.as_bytes());
        }
        set.insert(h.finish());
    }
    set
}

/// Jaccard similarity expressed as a percentage 0..=100.
/// `|A ∩ B| / |A ∪ B| * 100`.
fn jaccard_pct(a: &BTreeSet<u64>, b: &BTreeSet<u64>) -> u32 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let inter = a.intersection(b).count() as u64;
    let union = (a.len() + b.len()) as u64 - inter;
    if union == 0 {
        return 0;
    }
    ((inter * 100) / union) as u32
}

/// Detect a sentence boundary at the END of `buffer`. Returns the
/// byte offset (exclusive) of the boundary if the buffer ends with
/// any of: `[.!?]` (with or without trailing whitespace), `\n\n`,
/// or a double-newline anywhere in the trailing-whitespace tail.
/// Returns `None` otherwise. Used by the streaming consumer to
/// decide when to flush a sentence into the guard.
pub fn ends_at_sentence_boundary(buffer: &str) -> Option<usize> {
    let bytes = buffer.as_bytes();
    let n = bytes.len();
    if n == 0 {
        return None;
    }
    let mut tail = n;
    let mut newlines = 0u32;
    while tail > 0 {
        let b = bytes[tail - 1];
        if b == b'\n' {
            newlines += 1;
            tail -= 1;
        } else if b == b' ' || b == b'\t' || b == b'\r' {
            tail -= 1;
        } else {
            break;
        }
    }
    if newlines >= 2 {
        return Some(n);
    }
    // F22 (2026-04-26): triple-backtick fence close as a sentence
    // boundary. Code blocks (Cargo.toml, main.rs etc.) have no
    // `[.!?]` punctuation and rarely have `\n\n`, so without this
    // hook the SimHash buffer never gets to hash them. Recognise
    // ` ``` ` immediately before the trailing-whitespace tail as
    // the end of an emitted code block.
    if tail >= 3 && &bytes[tail - 3..tail] == b"```" {
        return Some(n);
    }
    if tail == 0 {
        return None;
    }
    let last = bytes[tail - 1];
    // F47 (2026-04-27): colon recognised as a sentence boundary.
    // cc-session-fix34 showed the model emit
    //   "I see the issue. Let me try a different approach - let
    //    me use the cargo bin directly:"
    // 23 times in a row; SimHash never ran because the boundary
    // detector returned None on `:`. Adding `:` is safe because
    // SimHash also enforces MIN_NORMALIZED_LEN=24 and MIN_WORDS=3,
    // so trivial enumerations like "Steps:" don't trigger.
    if last == b'.' || last == b'!' || last == b'?' || last == b':' {
        Some(n)
    } else {
        None
    }
}

/// Hard bound on `simhash_pending` while inside an unclosed fence —
/// sentence flush is suspended there, so an over-long (or never-closed)
/// block would otherwise grow without limit.
pub const FENCE_PENDING_CAP: usize = 16 * 1024;

/// Count non-overlapping "```" fence markers in `s`, scanning
/// left-to-right three bytes at a time: a run of four backticks is one
/// marker plus a stray byte, a run of six is two markers. Deliberately
/// simple — no line anchoring or language tag handling.
pub fn count_fence_markers(s: &str) -> usize {
    let bytes = s.as_bytes();
    let mut n = 0;
    let mut i = 0;
    while i + 3 <= bytes.len() {
        if &bytes[i..i + 3] == b"```" {
            n += 1;
            i += 3;
        } else {
            i += 1;
        }
    }
    n
}

/// Drop roughly the older half of `pending`, rounded up to a char
/// boundary so `drain` never splits a multibyte sequence.
fn drain_older_half(pending: &mut String) -> usize {
    let mut drop_to = pending.len() / 2;
    while !pending.is_char_boundary(drop_to) {
        drop_to += 1;
    }
    pending.drain(..drop_to);
    drop_to
}

/// One chunk of the streaming fence-aware sentence-loop state machine.
/// Appends `chunk` to `pending`, tracks ` ``` ` fences, and runs
/// `guard.check` at the right flush points. Returns the guard's
/// duplicate verdict (`true` = semantic loop trip).
///
/// - Fence CLOSE this chunk: the whole pending buffer — prose before the
///   open marker plus the block itself — is hashed once as a single
///   unit (the F22 design), then cleared. Prose boundaries inside the
///   same chunk are not evaluated separately; folding the leading prose
///   into the block's hash is intentional (the alternative — flushing
///   `pending[..open]` at the open marker — would split a sentence that
///   runs straight into the fence, a worse false-positive trade).
/// - Inside an unclosed fence: no `ends_at_sentence_boundary`, no 1KB
///   force-flush. Past `FENCE_PENDING_CAP` the older half is dropped
///   (a long block loses part of its hash; the fence state is kept).
/// - Outside a fence: unchanged F4 behaviour — flush at a sentence
///   boundary or at 1024 bytes, with the >4096 drain backstop.
///
/// `scan` is the caller-held byte offset into `pending` up to which
/// fence markers have been counted; it is rewound to `len - 2` after
/// each pass so a marker split across chunks ("`" + "``") is still
/// counted once it completes.
pub fn simhash_step(
    pending: &mut String,
    in_fence: &mut bool,
    scan: &mut usize,
    guard: &mut SimHashLoopGuard,
    chunk: &str,
) -> bool {
    pending.push_str(chunk);
    let markers = count_fence_markers(&pending[*scan..]);
    *scan = pending.len().saturating_sub(2);
    // The rewind target is a byte offset and may land inside a multibyte
    // char at the chunk tail — walk back so the next call's slice never
    // panics.
    while !pending.is_char_boundary(*scan) {
        *scan -= 1;
    }
    let mut closed = false;
    for _ in 0..markers {
        *in_fence = !*in_fence;
        if !*in_fence {
            closed = true;
        }
    }

    if closed {
        let dup = guard.check(pending);
        pending.clear();
        *scan = 0;
        return dup;
    }
    if *in_fence {
        if pending.len() > FENCE_PENDING_CAP {
            *scan = scan.saturating_sub(drain_older_half(pending));
            // `drain_older_half` removes a whole number of chars from the
            // front, so a boundary stays a boundary (or saturates to 0).
            debug_assert!(pending.is_char_boundary(*scan));
        }
        return false;
    }
    let mut dup = false;
    if ends_at_sentence_boundary(pending).is_some() || pending.len() >= 1024 {
        dup = guard.check(pending);
        pending.clear();
        *scan = 0;
    }
    if pending.len() > 4096 {
        *scan = scan.saturating_sub(drain_older_half(pending));
        debug_assert!(pending.is_char_boundary(*scan));
    }
    dup
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paraphrase_with_case_change_triggers_on_third_repeat() {
        let mut g = SimHashLoopGuard::new();
        let s1 = "I'll create a proper Rust Axum server with the /echo endpoint.";
        let s2 = "I'll create a proper Rust axum server with an /echo endpoint and tests.";
        let s3 = "I'll create a proper Rust axum server with the /echo endpoint.";
        assert!(!g.check(s1), "first sentence is novel");
        assert!(
            g.check(s2),
            "second sentence (case+wording paraphrase) duplicates first"
        );
        assert!(g.check(s3), "third sentence duplicates first/second");
    }

    #[test]
    fn legitimate_distinct_prose_does_not_trigger() {
        let mut g = SimHashLoopGuard::new();
        let sentences = [
            "First, we set up the Rust workspace and add axum as a dependency.",
            "Then we define a handler function that echoes the request body.",
            "After that we wire the handler into the router at /echo.",
            "Finally we add an integration test using tower's ServiceExt.",
            "Run the test suite with cargo test to confirm everything passes.",
        ];
        let mut any_dup = false;
        for s in sentences {
            if g.check(s) {
                any_dup = true;
            }
        }
        assert!(!any_dup, "five distinct prose sentences must not collide");
    }

    #[test]
    fn short_sentences_do_not_trigger() {
        let mut g = SimHashLoopGuard::new();
        for _ in 0..10 {
            assert!(!g.check("Yes."));
            assert!(!g.check("Done."));
        }
    }

    #[test]
    fn enumeration_with_template_does_not_trigger_immediately() {
        let mut g = SimHashLoopGuard::new();
        assert!(!g.check("Test 1: verify the echo handler returns the body unchanged."));
        assert!(!g.check("Test 2: verify the router rejects malformed POST bodies."));
        assert!(!g.check("Test 3: verify the server gracefully shuts down on signal."));
    }

    #[test]
    fn reset_clears_ring() {
        let mut g = SimHashLoopGuard::new();
        let s = "I'll create a proper Rust Axum server with the /echo endpoint.";
        g.check(s);
        assert_eq!(g.len(), 1);
        g.reset();
        assert_eq!(g.len(), 0);
        assert!(!g.check(s), "after reset the same sentence is novel again");
    }

    #[test]
    fn boundary_detection_period_then_space() {
        assert!(ends_at_sentence_boundary("Hello world. ").is_some());
        assert!(ends_at_sentence_boundary("Hello world! ").is_some());
        assert!(ends_at_sentence_boundary("Hello world? ").is_some());
        assert!(ends_at_sentence_boundary("Hello world").is_none());
        assert!(ends_at_sentence_boundary("Hello world.").is_some());
        assert!(ends_at_sentence_boundary("Hello world,").is_none());
    }

    #[test]
    fn f47_colon_is_sentence_boundary() {
        // F47 (2026-04-27): colon-terminated sentences must be
        // recognised as boundaries so SimHash can hash and dedup
        // them. The cc-session-fix34 23x repetition ended with
        // `directly:` and SimHash never ran because of this.
        assert!(
            ends_at_sentence_boundary(
                "Let me try a different approach - let me use the cargo bin directly:"
            )
            .is_some()
        );
        assert!(ends_at_sentence_boundary("Step one:").is_some());
        assert!(ends_at_sentence_boundary("Step one: ").is_some());
    }

    #[test]
    fn f47_cc_session_23x_phrase_loop_trips() {
        // The cc-session-fix34 sentence repeated 23 times must
        // trip on the SECOND check (first novel, second duplicate).
        let mut g = SimHashLoopGuard::new();
        let s =
            "I see the issue. Let me try a different approach - let me use the cargo bin directly:";
        assert!(!g.check(s), "first emit must be novel");
        assert!(g.check(s), "exact repeat must trip");
    }

    #[test]
    fn boundary_detection_double_newline() {
        assert!(ends_at_sentence_boundary("paragraph\n\n").is_some());
        assert!(ends_at_sentence_boundary("paragraph\n").is_none());
    }

    #[test]
    fn ring_capacity_is_respected() {
        let mut g = SimHashLoopGuard::with_params(4, 55);
        for i in 0..10 {
            g.check(&format!(
                "this is sentence number {} with sufficiently long content for hashing",
                i
            ));
        }
        assert_eq!(g.len(), 4, "ring length capped at cap=4");
    }

    #[test]
    fn identical_sentences_trip_immediately() {
        let mut g = SimHashLoopGuard::new();
        let s = "I will create the Rust axum server with proper tests.";
        assert!(!g.check(s), "first emit is novel");
        assert!(g.check(s), "exact repeat must trip");
    }

    #[test]
    fn related_topics_with_distinct_action_do_not_trip() {
        // Both sentences talk about Rust+axum but the verbs/nouns
        // differ enough to keep Jaccard below threshold. Guards
        // against "topic = duplicate" false positives.
        let mut g = SimHashLoopGuard::new();
        assert!(!g.check("The axum router accepts incoming HTTP requests."));
        assert!(!g.check("The axum handler returns a JSON response body."));
        assert!(!g.check("The axum extractor parses the query parameters."));
    }
}
