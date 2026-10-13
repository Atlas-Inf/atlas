// SPDX-License-Identifier: AGPL-3.0-only

//! `commit_staged_rows`: the pitched form leaves the per-row commit's state.

use super::super::qsa_free::tests::{HD, bytes, indexer};
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

/// Commit `runs` (`(first_row, count)`, consecutive positions from 0) on a
/// fresh sequence; return its raw keys at `0..end`, its counters, and the
/// copy / launch counts the commit issued.
fn commit(pitched: bool, runs: &[(usize, usize)]) -> (Vec<Vec<u8>>, (usize, usize), usize, usize) {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 1 << 16);
    let mut st = qsa.new_seq_state(&gpu).unwrap();
    // Staging rows hold distinct keys (the graph half's output).
    for row in 0..32 {
        gpu.copy_h2d(&bytes(HD * 2, 1 + row as u32), qsa.staged_key(row))
            .unwrap();
    }
    let (d2d, launches) = (gpu.d2d_count() + gpu.d2d_2d_count(), gpu.launch_count());
    let mut pos = 0;
    for &(row, count) in runs {
        qsa.commit_staged_rows(&mut st, row, pos, count, pitched, &gpu, 0)
            .unwrap();
        pos += count;
    }
    let keys = (0..pos)
        .map(|p| {
            let at = qsa.raw_slot(&st, p);
            let buf = st.raw.bufs[st.raw.cur];
            let all = gpu.read_alloc(buf).unwrap();
            let off = (at.0 - buf.0) as usize;
            all[off..off + HD * 2].to_vec()
        })
        .collect();
    (
        keys,
        (st.ingested, st.pooled),
        gpu.d2d_count() + gpu.d2d_2d_count() - d2d,
        gpu.launch_count() - launches,
    )
}

#[test]
fn a_run_commits_what_its_rows_commit() {
    // A K=4 verify window, a K=3 one and a decode row, then a window that
    // closes two blocks: ragged runs crossing 4-token block boundaries.
    let runs = [(0, 4), (4, 3), (7, 1), (8, 8)];
    let (k_loop, c_loop, copies_loop, pools_loop) = commit(false, &runs);
    let (k_run, c_run, copies_run, pools_run) = commit(true, &runs);
    assert_eq!(k_loop, k_run, "raw keys");
    assert_eq!(c_loop, c_run, "ingested / pooled");
    assert_eq!(c_run, (16, 4));
    assert_eq!(copies_loop, 16, "a copy a row");
    assert_eq!(copies_run, 4, "a copy a run");
    // One pool launch per block closed vs one per run that closes any.
    assert_eq!(pools_loop, 4);
    assert_eq!(pools_run, 3);
}

#[test]
fn a_run_past_the_inert_bound_is_refused() {
    let gpu = MockGpuBackend::new();
    let qsa = indexer(&gpu, 1 << 16);
    let mut st = qsa.new_seq_state(&gpu).unwrap();
    // budget 64 + ratio 4 - 1: position 67 is the first active one.
    let bound = qsa.inert_bound();
    st.ingested = bound - 2;
    st.pooled = st.ingested / 4;
    assert!(
        qsa.commit_staged_rows(&mut st, 0, bound - 2, 4, true, &gpu, 0)
            .is_err()
    );
}
