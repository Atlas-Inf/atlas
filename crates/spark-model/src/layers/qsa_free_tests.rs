// SPDX-License-Identifier: AGPL-3.0-only

//! The pooled-key carry grows on demand and keeps every live byte.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

pub(in crate::layers::qsa) const HD: usize = 8;

pub(in crate::layers::qsa) fn indexer(gpu: &MockGpuBackend, max_seq_len: usize) -> QsaIndexer {
    QsaIndexer::new(
        DevicePtr::NULL,
        DevicePtr::NULL,
        DevicePtr::NULL,
        /*n_heads*/ 2,
        HD,
        /*ratio*/ 4,
        /*budget*/ 64,
        max_seq_len,
        /*rot*/ 8,
        /*theta*/ 1e5,
        /*eps*/ 1e-5,
        /*hidden*/ 128,
        /*nkv_attn*/ 2,
        /*hd_attn*/ 16,
        gpu,
    )
    .unwrap()
}

pub(in crate::layers::qsa) fn bytes(n: usize, seed: u32) -> Vec<u8> {
    (0..n as u32)
        .map(|v| ((v * 7 + seed) % 251) as u8)
        .collect()
}

#[test]
fn a_fresh_carry_holds_nothing_until_reserved() {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 1 << 16);
    let allocs = gpu.alloc_count();
    let mut st = qsa.new_seq_state(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), allocs, "no up-front max_seq_len buffers");
    qsa.reserve(&mut st, 100, &gpu, 0).unwrap();
    assert_eq!(st.cap, 4096, "rounded up to the granule");
    // Within capacity: no new allocation.
    let allocs = gpu.alloc_count();
    qsa.reserve(&mut st, 4096, &gpu, 0).unwrap();
    assert_eq!(gpu.alloc_count(), allocs);
}

#[test]
fn growth_keeps_the_pooled_blocks() {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 1 << 16);
    let mut st = qsa.new_seq_state(&gpu).unwrap();
    qsa.reserve(&mut st, 4096, &gpu, 0).unwrap();
    let pooled = 1000usize;
    let block = bytes(pooled * HD * 2, 5);
    gpu.copy_h2d(&block, st.block_keys).unwrap();
    (st.ingested, st.pooled) = (4000, pooled);

    let old = st.block_keys;
    qsa.reserve(&mut st, 4097, &gpu, 0).unwrap();
    assert_eq!(st.cap, 8192, "at least half again, granule-aligned");
    assert_ne!(st.block_keys, old, "moved");
    assert!(gpu.read_alloc(old).is_none(), "old freed");
    assert_eq!(
        &gpu.read_alloc(st.block_keys).unwrap()[..block.len()],
        &block[..]
    );
    assert_eq!(
        gpu.read_alloc(st.block_keys).unwrap().len(),
        8192 / 4 * HD * 2
    );
    assert_eq!((st.ingested, st.pooled), (4000, pooled));
}

#[test]
fn growth_stops_at_the_served_capacity() {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 10_000);
    let mut st = qsa.new_seq_state(&gpu).unwrap();
    qsa.reserve(&mut st, 9_000, &gpu, 0).unwrap();
    assert_eq!(st.cap, 10_000, "clamped to max_seq_len");
    let e = qsa.reserve(&mut st, 10_001, &gpu, 0).unwrap_err();
    assert!(
        format!("{e:#}").contains("exceeds the indexer capacity"),
        "{e:#}"
    );
}

#[test]
fn free_releases_everything_and_is_idempotent() {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 1 << 16);
    let mut st = qsa.new_seq_state(&gpu).unwrap();
    qsa.free_seq_state(&mut st, &gpu).unwrap(); // nothing allocated yet
    qsa.reserve(&mut st, 10, &gpu, 0).unwrap();
    qsa.raw_room(&mut st, 10, &gpu, 0).unwrap();
    let held = [st.block_keys, st.raw.bufs[0], st.raw.bufs[1]];
    qsa.free_seq_state(&mut st, &gpu).unwrap();
    assert!(held.iter().all(|&p| gpu.read_alloc(p).is_none()));
    assert_eq!((st.cap, st.raw.cap), (0, 0));
    qsa.free_seq_state(&mut st, &gpu).unwrap();
}

#[test]
fn bytes_per_token_is_the_pooled_share() {
    let gpu = MockGpuBackend::new();
    // hd 8, ratio 4: a 16 B pooled row per 4 tokens.
    assert_eq!(indexer(&gpu, 256).bytes_per_token(), 4);
}
