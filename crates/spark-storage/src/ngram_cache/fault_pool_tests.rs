// SPDX-License-Identifier: AGPL-3.0-only

//! Pool mechanics only: a fake `FetchFn` marks its slot in a plain byte
//! "arena", so these need no pinned arena, GPU, or NVMe.

use std::sync::Weak;
use std::sync::mpsc::{RecvTimeoutError, channel};
use std::time::Duration;

use super::*;

/// The id `mark` panics on.
const PANIC_ID: u64 = u64::MAX;

fn mark(_: &RowSource<'_>, p: &ArenaPtrs, _: &mut AlignedBlock, id: u64, slot: u32) -> Result<()> {
    assert_ne!(id, PANIC_ID, "injected fetch panic");
    // SAFETY: every test sizes its arena past each slot it schedules.
    unsafe { *p.rows.add(slot as usize) = 1 };
    Ok(())
}

/// Never read: `mark` ignores the source.
fn src() -> OwnedSource {
    OwnedSource {
        file: File::open(std::env::current_exe().expect("test exe")).expect("open test exe"),
        base_offset: 0,
        segments: None,
        row_stride: 1,
        scale_file: None,
    }
}

/// Fault `slot = 0..n` (id = slot), with row `panic_at` swapped for `PANIC_ID`.
fn faults(n: u32, panic_at: Option<u32>) -> Vec<Fault> {
    (0..n)
        .map(|slot| Fault {
            id: if Some(slot) == panic_at {
                PANIC_ID
            } else {
                slot as u64
            },
            slot,
        })
        .collect()
}

/// Run `f` on its own thread, so a pool hang fails the test instead of
/// wedging the suite.
fn within_deadline(f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = channel();
    let h = std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(()) => {}
        Err(RecvTimeoutError::Timeout) => panic!("fault pool hung"),
        Err(RecvTimeoutError::Disconnected) => {
            std::panic::resume_unwind(h.join().expect_err("sender dropped only on panic"))
        }
    }
}

/// A spawn failing part-way must not leak the workers already started: each
/// holds a `Shared`, so a parked survivor keeps it alive.
#[test]
fn failed_spawn_joins_the_started_workers() {
    let mut shared: Option<Weak<Shared>> = None;
    let r = FaultPool::start(src(), mark, 4, |i, sh| {
        shared.get_or_insert_with(|| Arc::downgrade(&sh));
        if i == 2 {
            return Err(std::io::Error::other("injected spawn failure"));
        }
        spawn_worker(i, sh)
    });
    let e = r.err().expect("a failed spawn must fail the build");
    assert!(format!("{e:#}").contains("injected spawn failure"), "{e:#}");
    let shared = shared.expect("spawn was called");
    assert!(
        shared.upgrade().is_none(),
        "started workers outlived the failed build"
    );
}

/// A panicking fetch, on the caller's thread (0 workers) or a worker's, is
/// that batch's error rather than a hang; every other row still lands, and
/// the workers keep serving later batches.
#[test]
fn panicking_fetch_errors_the_batch_and_workers_survive() {
    within_deadline(|| {
        for workers in [0, 3] {
            let pool = FaultPool::start(src(), mark, workers, spawn_worker).expect("pool");
            for at in [0u32, 31, 63] {
                let mut arena = vec![0u8; 64];
                let ptrs = ArenaPtrs {
                    rows: arena.as_mut_ptr(),
                    scales: None,
                };
                let r = pool.run(&faults(64, Some(at)), &ptrs);
                let e = r.expect_err("a panicked row must fail the batch");
                assert!(format!("{e:#}").contains("panicked"), "{e:#}");
                for (slot, &b) in arena.iter().enumerate() {
                    assert_eq!(b == 1, slot != at as usize, "slot {slot}, panic at {at}");
                }
            }
            assert!(
                pool.workers.iter().all(|w| !w.is_finished()),
                "a worker died"
            );
            let mut arena = vec![0u8; 64];
            let ptrs = ArenaPtrs {
                rows: arena.as_mut_ptr(),
                scales: None,
            };
            pool.run(&faults(64, None), &ptrs)
                .expect("pool still serves");
            assert!(arena.iter().all(|&b| b == 1));
        }
    });
}
