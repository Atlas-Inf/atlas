// SPDX-License-Identifier: AGPL-3.0-only
//! Fence-awareness tests for `simhash_step` (the streaming state machine
//! behind the F4 SimHash semantic-loop guard). Kept out of
//! `loop_simhash.rs` to stay under the 500-LoC file cap.

use crate::loop_simhash::{FENCE_PENDING_CAP, SimHashLoopGuard, count_fence_markers, simhash_step};

/// Bundles the per-stream state `process_detector_content` threads
/// through `simhash_step`, so tests can drive the machine chunk by chunk.
struct Stream {
    pending: String,
    in_fence: bool,
    scan: usize,
    guard: SimHashLoopGuard,
}

impl Stream {
    fn new() -> Self {
        Self {
            pending: String::new(),
            in_fence: false,
            scan: 0,
            guard: SimHashLoopGuard::new(),
        }
    }

    fn step(&mut self, chunk: &str) -> bool {
        simhash_step(
            &mut self.pending,
            &mut self.in_fence,
            &mut self.scan,
            &mut self.guard,
            chunk,
        )
    }

    /// Feed `text` in ~`chunk_size`-byte pieces (never splitting a
    /// multibyte char); returns the chunk index that tripped, if any.
    fn feed(&mut self, text: &str, chunk_size: usize) -> Option<usize> {
        let mut i = 0;
        let mut n = 0;
        while i < text.len() {
            let mut end = (i + chunk_size).min(text.len());
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            if self.step(&text[i..end]) {
                return Some(n);
            }
            i = end;
            n += 1;
        }
        None
    }
}

/// A MinHeap-style method with a prompt-mandated docstring — the
/// `_left_child`/`_right_child` twin shape that false-tripped the
/// sentence splitter (near-identical bigrams by convention).
fn method(name: &str, relation: &str, body: &str) -> String {
    format!(
        "    def {name}(self, index):\n\
         \x20       \"\"\"Return the index of the {relation} child of the node at the given index.\n\
         \n\
         \x20       Args:\n\
         \x20           index: The index of the node.\n\
         \n\
         \x20       Returns:\n\
         \x20           The index of the {relation} child.\n\
         \n\
         \x20       Time Complexity: O(1)\n\
         \x20       \"\"\"\n\
         \x20       {body}\n\n"
    )
}

fn docstring_fence() -> String {
    let mut s = String::from("```python\nclass MinHeap:\n");
    s.push_str(&method("_left_child", "left", "return 2 * index + 1"));
    s.push_str(&method("_right_child", "right", "return 2 * index + 2"));
    s.push_str(&method("_parent", "parent", "return (index - 1) // 2"));
    s.push_str(&method(
        "_grandparent",
        "grandparent",
        "return self._parent(self._parent(index))",
    ));
    s
}

/// Docstring twins inside ONE never-closed fence must never trip the
/// guard — this is the 660-token decode-floor cut.
#[test]
fn docstring_twins_inside_open_fence_never_trip() {
    let mut st = Stream::new();
    let text = docstring_fence(); // fence opens, never closes
    assert!(text.contains("```"), "fixture opens a fence");
    let trip = st.feed(&text, 40);
    assert!(
        trip.is_none(),
        "guard tripped at chunk {:?} inside an open fence",
        trip
    );
    assert!(st.in_fence, "fence must still be open at end of stream");
}

/// The fence closes, then prose paraphrase-loops → the prose loop trips
/// exactly like unfenced F4 behaviour.
#[test]
fn prose_loop_after_fence_close_still_trips() {
    let mut st = Stream::new();
    let fenced = format!("{}\n```\n", docstring_fence());
    assert!(st.feed(&fenced, 40).is_none(), "the block itself is novel");
    assert!(!st.in_fence, "fence closed");
    assert!(st.pending.is_empty(), "block hashed and flushed at close");

    // Reuse the F47 fixture: the 23x paraphrased-restart sentence.
    let s = "I see the issue. Let me try a different approach - let me use the cargo bin directly:";
    assert!(st.feed(s, 40).is_none(), "first prose emit is novel");
    assert!(
        st.feed(s, 40).is_some(),
        "second identical prose sentence must trip"
    );
}

/// Two IDENTICAL fenced blocks back to back must trip at the second
/// closing fence — F22's one-hash-per-block design preserved.
#[test]
fn two_identical_fenced_blocks_trip_at_second_close() {
    let mut st = Stream::new();
    let block = format!("{}\n```\n\n", docstring_fence());
    assert!(
        st.feed(&block, 40).is_none(),
        "first block is novel (its close is the first check)"
    );
    assert!(
        st.feed(&block, 40).is_some(),
        "second identical block must trip at its closing fence"
    );
}

/// A ``` marker split across chunk boundaries ("``" | "`python\n") is
/// still recognised as an open.
#[test]
fn fence_marker_split_across_chunks_detected() {
    let mut st = Stream::new();
    assert!(!st.step("``"), "two backticks are not a marker");
    assert!(!st.in_fence);
    assert!(!st.step("`python\n"), "the split marker completes here");
    assert!(st.in_fence, "split marker must open the fence");
    // …and docstring twins inside it stay quiet.
    let rest = docstring_fence();
    let rest = rest.strip_prefix("```python\n").unwrap();
    assert!(st.feed(rest, 40).is_none());
    assert!(st.in_fence);
}

/// `count_fence_markers` consumes non-overlapping triples left-to-right.
#[test]
fn fence_marker_counting() {
    assert_eq!(count_fence_markers(""), 0);
    assert_eq!(count_fence_markers("``"), 0);
    assert_eq!(count_fence_markers("```"), 1);
    assert_eq!(count_fence_markers("````"), 1); // 3 + stray
    assert_eq!(count_fence_markers("``````"), 2);
    assert_eq!(count_fence_markers("```python\nx\n```"), 2);
    assert_eq!(count_fence_markers("a``b``c``"), 0);
}

/// An unclosed fence longer than FENCE_PENDING_CAP drains the older
/// half without tripping and keeps the fence state — and keeps
/// recognising the eventual close.
#[test]
fn in_fence_cap_drains_without_tripping() {
    let mut st = Stream::new();
    assert!(!st.step("```python\n"));
    assert!(st.in_fence);
    // Push well past the cap in chunks; no sentence boundaries needed.
    let filler = "x".repeat(FENCE_PENDING_CAP + 4096);
    for chunk in filler.as_bytes().chunks(4096) {
        assert!(
            !st.step(std::str::from_utf8(chunk).unwrap()),
            "in-fence filler must never trip"
        );
    }
    assert!(
        st.pending.len() <= FENCE_PENDING_CAP + 4096,
        "cap drain bound the buffer (len {})",
        st.pending.len()
    );
    assert!(st.in_fence, "fence state survives the drain");
    // The close is still recognised even though part of the block
    // was drained away.
    let _ = st.step("\n```\n");
    assert!(!st.in_fence, "closing fence detected after drain");
    assert!(st.pending.is_empty(), "close flushed the remainder");
}

/// Multibyte char at a chunk tail: the `len - 2` scan rewind can land
/// inside it; the next call must not panic slicing `pending` there.
#[test]
fn multibyte_tail_then_split_fence_marker() {
    let mut st = Stream::new();
    assert!(!st.step("abc é"), "prose, tail is a 2-byte char");
    assert!(!st.step("``"), "two backticks are not a marker");
    assert!(!st.step("`python\n"), "split marker completes here");
    assert!(st.in_fence, "fence opened across multibyte-split chunks");
}

/// A chunk ending in a 3-byte char, then a chunk opening with a fence —
/// the rewind walks back over the whole char without panicking.
#[test]
fn three_byte_char_tail_before_fence_open() {
    let mut st = Stream::new();
    assert!(!st.step("some prose —"), "tail is U+2014 (3 bytes)");
    assert!(!st.step("```python\n"), "fence opens on the next chunk");
    assert!(st.in_fence);
}

/// Chunk by CHARS (1..7 chars per step, deterministic LCG) through 20 KB
/// of mixed multibyte text with embedded fences: no panic, and the final
/// fence state matches the parity of markers actually emitted.
#[test]
fn fuzz_multibyte_chunks_fence_parity() {
    let mut st = Stream::new();
    // Mixed-width payload: ASCII, 2-byte (é), 3-byte (—), 4-byte (🙂).
    let unit = "alpha é—🙂 beta `x` code line\n";
    let mut text = String::new();
    let mut markers = 0usize;
    while text.len() < 20 * 1024 {
        text.push_str(unit);
        if text.len() % 3000 < unit.len() {
            text.push_str("```rust\nlet x = 1;\n```\n");
            markers += 2;
        }
    }
    if markers % 2 == 1 {
        text.push_str("```\n"); // keep total parity even for the assert
        markers += 1;
    }
    assert!(markers >= 2);

    let chars: Vec<char> = text.chars().collect();
    // xorshift64 — deterministic 1..7-char chunk sizes.
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut i = 0;
    while i < chars.len() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let take = 1 + (seed % 7) as usize;
        let end = (i + take).min(chars.len());
        let chunk: String = chars[i..end].iter().collect();
        let _ = st.step(&chunk); // must never panic
        i = end;
    }
    assert_eq!(
        st.in_fence,
        markers % 2 == 1,
        "fence state must match emitted marker parity ({markers})"
    );
}

/// A fence that closes mid-chunk hashes prose + block as one unit and
/// leaves the machine back in prose mode for the next chunk.
#[test]
fn close_mid_chunk_flushes_and_resumes_prose() {
    let mut st = Stream::new();
    // Open marker plus a full block plus close all in one chunk.
    let block = format!("{}\n```\n", docstring_fence());
    assert!(!st.step(&block), "open+close in one chunk is one unit");
    assert!(!st.in_fence);
    assert!(st.pending.is_empty());
    // Prose boundaries work again immediately after.
    assert!(!st.step("We wired the handler into the router at /echo."));
}
