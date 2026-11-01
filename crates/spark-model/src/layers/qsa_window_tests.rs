// SPDX-License-Identifier: AGPL-3.0-only

//! The raw-key window keeps every key a later pool or rewind can read, at
//! the address the pooling kernel computes from `raw_origin`.

use super::super::qsa_free::tests::{HD, bytes, indexer};
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

const ROW: usize = HD * 2;

/// Write the key of position `pos` (its bytes are a function of `pos`).
fn ingest(qsa: &QsaIndexer, st: &mut QsaSeqState, gpu: &MockGpuBackend, n: usize) {
    qsa.raw_room(st, n, gpu, 0).unwrap();
    for pos in st.ingested..st.ingested + n {
        gpu.copy_h2d(&key(pos), qsa.raw_slot(st, pos)).unwrap();
    }
    st.ingested += n;
    st.pooled = st.ingested / 4; // what `pool_new_blocks` leaves
}

fn key(pos: usize) -> Vec<u8> {
    bytes(ROW, pos as u32)
}

/// The bytes the pooling kernel would read for `pos`: `raw_origin + pos`.
fn kernel_read(st: &QsaSeqState, qsa: &QsaIndexer, gpu: &MockGpuBackend, pos: usize) -> Vec<u8> {
    let addr = qsa.raw_origin(st).0 + (pos * ROW) as u64;
    let buf = st.raw.bufs[st.raw.cur];
    let all = gpu.read_alloc(buf).unwrap();
    let at = (addr - buf.0) as usize;
    all[at..at + ROW].to_vec()
}

#[test]
fn every_unpooled_key_and_the_margin_survive_a_long_decode() {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 1 << 16);
    let mut st = qsa.new_seq_state(&gpu).unwrap();
    ingest(&qsa, &mut st, &gpu, 3000); // a prefill
    let allocs = gpu.alloc_count();
    for _ in 0..20_000 {
        ingest(&qsa, &mut st, &gpu, 1);
        let keep_from = (st.pooled * 4).saturating_sub(REWIND_MARGIN);
        assert!(
            st.raw.base <= keep_from,
            "base {} > {keep_from}",
            st.raw.base
        );
        let tail = (st.pooled * 4).min(st.ingested - 1); // first unpooled key, if any
        for pos in [keep_from, tail, st.ingested - 1] {
            assert_eq!(kernel_read(&st, &qsa, &gpu, pos), key(pos), "pos {pos}");
        }
    }
    assert_eq!(
        gpu.alloc_count(),
        allocs,
        "slides swap buffers, never allocate"
    );
    assert_eq!(
        st.raw.cap, 4096,
        "two 4096-position buffers whatever the length"
    );
}

#[test]
fn a_slab_wider_than_the_window_grows_it_and_keeps_the_tail() {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 1 << 16);
    let mut st = qsa.new_seq_state(&gpu).unwrap();
    ingest(&qsa, &mut st, &gpu, 4095);
    let old = st.raw.bufs;
    ingest(&qsa, &mut st, &gpu, 6000);
    assert!(
        old.iter().all(|&p| gpu.read_alloc(p).is_none()),
        "old freed"
    );
    assert!(st.raw.cap >= 6000 + 3);
    for pos in st.raw.base..st.ingested {
        assert_eq!(kernel_read(&st, &qsa, &gpu, pos), key(pos), "pos {pos}");
    }
}

/// A verify of up to 128 rows may be rewound to its first row; the block it
/// un-pools must still find its raw keys, even when the window slid between
/// two of its rows.
#[test]
fn a_rewind_after_a_mid_verify_slide_finds_its_keys() {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 1 << 16);
    for start in 4096 - 140..4096 + 4 {
        let mut st = qsa.new_seq_state(&gpu).unwrap();
        ingest(&qsa, &mut st, &gpu, start);
        let before = st.ingested;
        for _ in 0..128 {
            ingest(&qsa, &mut st, &gpu, 1);
        }
        qsa.rewind_verify(&mut st, 127).unwrap();
        assert_eq!(st.ingested, before + 1);
        for pos in st.pooled * 4..st.ingested {
            assert_eq!(kernel_read(&st, &qsa, &gpu, pos), key(pos), "start {start}");
        }
        qsa.free_seq_state(&mut st, &gpu).unwrap();
    }
}

#[test]
fn a_rewind_below_the_window_is_refused() {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 1 << 16);
    let mut st = qsa.new_seq_state(&gpu).unwrap();
    ingest(&qsa, &mut st, &gpu, 4000);
    ingest(&qsa, &mut st, &gpu, 1000); // slides past 3000 - margin
    let e = qsa.rewind_verify(&mut st, 2000).unwrap_err();
    assert!(format!("{e:#}").contains("raw-key window"), "{e:#}");
    assert_eq!(st.ingested, 5000, "state untouched");
}
