// SPDX-License-Identifier: AGPL-3.0-only

//! `zero_dirty`: the prefix it zeroes, that it leaves the arena `zero_all`
//! leaves when no pass wrote past that prefix, and that the check finds the
//! byte it would otherwise have left.

use super::accessors::zero_dirty::dirty_prefix;
use super::*;
use crate::gpu::mock::MockGpuBackend;

const ROWS: usize = 128;
/// Every buffer of this arena that holds at least a page a row is trimmed.
const MIN: usize = ROWS * 4096;
const FLOOR: usize = 8;

fn arena(gpu: &MockGpuBackend) -> BufferArena {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    BufferArena::new(&cfg, ROWS, 4096, 16, 32, gpu).unwrap()
}

/// Whether every byte of every buffer `zero_all` zeroes is zero.
fn all_zero(arena: &BufferArena, gpu: &MockGpuBackend) -> bool {
    arena.zeroed().iter().all(|&(_, ptr, bytes)| {
        let mut host = vec![0u8; bytes];
        gpu.copy_d2h(ptr, &mut host).unwrap();
        host.iter().all(|&b| b == 0)
    })
}

/// What a trimmed zero would leave behind now.
fn stale(arena: &BufferArena, gpu: &MockGpuBackend) -> Vec<(&'static str, usize, usize)> {
    arena.stale_past_dirty_over(gpu, 0, FLOOR, MIN).unwrap()
}

/// The largest zeroed buffer (trimmed in this arena): name, pointer, bytes.
fn largest(arena: &BufferArena) -> (&'static str, DevicePtr, usize) {
    let big = arena
        .zeroed()
        .into_iter()
        .max_by_key(|&(_, _, bytes)| bytes);
    big.filter(|&(_, _, bytes)| bytes >= MIN && bytes % ROWS == 0)
        .expect("the fixture arena must have a row-major trimmed buffer")
}

fn fill(arena: &BufferArena, gpu: &MockGpuBackend, value: u8) {
    for (_, ptr, bytes) in arena.zeroed() {
        gpu.memset(ptr, value, bytes).unwrap();
    }
}

#[test]
fn the_prefix_is_a_row_share_rounded_up_to_a_page() {
    let bytes = 1 << 20;
    // A small buffer, or rows that fill the arena: all of it.
    assert_eq!(dirty_prefix(bytes, 1, ROWS, bytes + 1), bytes);
    assert_eq!(dirty_prefix(bytes, ROWS, ROWS, 0), bytes);
    assert_eq!(dirty_prefix(bytes, usize::MAX, ROWS, 0), bytes);
    // 16 of 128 rows: an eighth.
    assert_eq!(dirty_prefix(bytes, 16, ROWS, 0), bytes / 8);
    // A share that is not a whole page rounds up, never past the buffer.
    assert_eq!(dirty_prefix(10_000, 1, ROWS, 0), 4096);
    assert_eq!(dirty_prefix(5_000, 127, ROWS, 0), 5_000);
    assert_eq!(dirty_prefix(bytes, 0, ROWS, 0), 0);
    // More rows never zero less.
    let prefixes: Vec<usize> = (0..=ROWS)
        .map(|r| dirty_prefix(bytes + 12, r, ROWS, 0))
        .collect();
    assert!(prefixes.windows(2).all(|w| w[0] <= w[1]));
}

/// Until the arena has been zeroed whole once nothing is known about it.
#[test]
fn an_arena_never_zeroed_is_zeroed_whole() {
    let gpu = MockGpuBackend::new();
    let arena = arena(&gpu);
    fill(&arena, &gpu, 0xAB);
    arena.zero_dirty_over(&gpu, 0, FLOOR, MIN).unwrap();
    assert!(all_zero(&arena, &gpu));
}

/// After a whole zero, a pass of 20 rows and a later one of 5: the trimmed
/// zero clears what they wrote, and then covers only the floor.
#[test]
fn a_trimmed_zero_clears_what_the_noted_passes_wrote() {
    let gpu = MockGpuBackend::new();
    let arena = arena(&gpu);
    let before = gpu.memset_count();
    arena.zero_all(&gpu, 0).unwrap();
    let fills = gpu.memset_count() - before;
    let (_, big, bytes) = largest(&arena);
    let row = bytes / ROWS;
    // The passes fill their rows of every buffer; the floor's rows hold what
    // a decode step left.
    arena.note_rows(20);
    arena.note_rows(5);
    for (_, ptr, bytes) in arena.zeroed() {
        let dirty = (bytes / ROWS * 28).min(bytes);
        gpu.memset(ptr, 0xAB, dirty).unwrap();
    }
    assert!(!all_zero(&arena, &gpu));
    assert!(stale(&arena, &gpu).is_empty());
    let before = gpu.memset_count();
    arena.zero_dirty_over(&gpu, 0, FLOOR, MIN).unwrap();
    assert_eq!(gpu.memset_count() - before, fills, "the fills of zero_all");
    assert!(all_zero(&arena, &gpu));

    // Nothing noted since: only the floor's share of a large buffer is
    // zeroed, so a byte past it survives.
    let floor = dirty_prefix(bytes, FLOOR, ROWS, MIN);
    assert!(floor >= row * FLOOR && floor < row * (FLOOR + 1));
    gpu.memset(big.offset(floor - 1), 0xCD, 2).unwrap();
    arena.zero_dirty_over(&gpu, 0, FLOOR, MIN).unwrap();
    let mut host = vec![0u8; 2];
    gpu.copy_d2h(big.offset(floor - 1), &mut host).unwrap();
    assert_eq!(host, [0, 0xCD]);
}

/// A pass that wrote past its noted rows plus the floor: the check names
/// the buffer and the byte, where a trimmed zero would have left it.
#[test]
fn the_check_finds_a_byte_past_the_prefix() {
    let gpu = MockGpuBackend::new();
    let arena = arena(&gpu);
    arena.zero_all(&gpu, 0).unwrap();
    let (name, big, bytes) = largest(&arena);
    let row = bytes / ROWS;
    arena.note_rows(20);
    let prefix = dirty_prefix(bytes, 20 + FLOOR, ROWS, MIN);
    assert_eq!(prefix, (row * 28).next_multiple_of(4096));
    gpu.memset(big.offset(prefix - 1), 0xEE, 1).unwrap();
    assert!(stale(&arena, &gpu).is_empty());
    gpu.memset(big.offset(row * 60 + 3), 0xEE, 1).unwrap();
    assert_eq!(stale(&arena, &gpu), [(name, row * 60 + 3, prefix)]);
    // The check reads; it does not zero or forget the noted rows.
    assert_eq!(stale(&arena, &gpu).len(), 1);
    // A whole zero forgets them: the arena is clean again.
    arena.zero_all(&gpu, 0).unwrap();
    assert!(all_zero(&arena, &gpu) && stale(&arena, &gpu).is_empty());
    // A path that cannot say what it writes marks everything dirty.
    arena.note_rows(usize::MAX);
    fill(&arena, &gpu, 0x11);
    assert!(stale(&arena, &gpu).is_empty());
    arena.zero_dirty_over(&gpu, 0, FLOOR, MIN).unwrap();
    assert!(all_zero(&arena, &gpu));
}

/// A chunk embeds all its rows into `hidden_states` while its pass notes
/// only the rows it computes: that buffer is zeroed whole, and the check
/// does not report what the embed left in it.
#[test]
fn the_embedded_buffer_is_zeroed_whole() {
    let gpu = MockGpuBackend::new();
    let arena = arena(&gpu);
    arena.zero_all(&gpu, 0).unwrap();
    let (hidden, bytes) = (arena.hidden_states(), arena.sizes().hidden_states);
    assert!(bytes >= MIN, "large enough to be trimmed in this arena");
    // A full chunk embedded, five rows computed.
    gpu.memset(hidden, 0xAB, bytes).unwrap();
    arena.note_rows(5);
    assert!(stale(&arena, &gpu).is_empty());
    arena.zero_dirty_over(&gpu, 0, FLOOR, MIN).unwrap();
    assert!(all_zero(&arena, &gpu));
}
