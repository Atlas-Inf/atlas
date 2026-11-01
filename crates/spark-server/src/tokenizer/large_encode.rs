// SPDX-License-Identifier: AGPL-3.0-only

//! Bound temporary HF encoding allocations without limiting model requests.
use std::sync::{Arc, Condvar, Mutex};

pub(super) const LARGE_INPUT_BYTES: usize = 1024 * 1024;

pub(super) struct Gate {
    limit: usize,
    active: Mutex<usize>,
    available: Condvar,
}

pub(super) fn configured(raw: Option<&str>) -> Result<Option<Arc<Gate>>, String> {
    raw.map(|raw| {
        let limit = raw
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                "ATLAS_LARGE_ENCODE_CONCURRENCY must be a positive integer".to_string()
            })?;
        Ok(Arc::new(Gate {
            limit,
            active: Mutex::new(0),
            available: Condvar::new(),
        }))
    })
    .transpose()
}

impl Gate {
    pub(super) fn acquire(&self, bytes: usize) -> Option<Permit<'_>> {
        if bytes < LARGE_INPUT_BYTES {
            return None;
        }
        let mut active = self.active.lock().unwrap_or_else(|e| e.into_inner());
        while *active >= self.limit {
            active = self
                .available
                .wait(active)
                .unwrap_or_else(|e| e.into_inner());
        }
        *active += 1;
        Some(Permit(self))
    }
}

pub(super) struct Permit<'a>(&'a Gate);

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut active = self.0.active.lock().unwrap_or_else(|e| e.into_inner());
        *active -= 1;
        self.0.available.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn clones_share_the_large_encode_bound_and_small_inputs_bypass_it() {
        let gate = configured(Some("2")).unwrap().unwrap();
        let first = gate.acquire(LARGE_INPUT_BYTES).expect("first permit");
        let second = gate.acquire(LARGE_INPUT_BYTES).expect("second permit");
        assert!(gate.acquire(LARGE_INPUT_BYTES - 1).is_none());
        let clone = Arc::clone(&gate);
        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let _permit = clone.acquire(LARGE_INPUT_BYTES).expect("waiting permit");
            acquired_tx.send(()).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(acquired_rx.recv_timeout(Duration::from_millis(30)).is_err());
        drop(first);
        acquired_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        waiter.join().unwrap();
        drop(second);
        assert_eq!(*gate.active.lock().unwrap(), 0);
    }

    #[test]
    fn failed_or_panicking_encode_releases_its_permit() {
        let gate = configured(Some("1")).unwrap().unwrap();
        let result: Result<(), ()> = {
            let _permit = gate.acquire(LARGE_INPUT_BYTES).expect("error permit");
            Err(())
        };
        assert!(result.is_err());
        let result = std::panic::catch_unwind(|| {
            let _permit = gate.acquire(LARGE_INPUT_BYTES).expect("panic permit");
            panic!("simulated tokenizer panic");
        });
        assert!(result.is_err());
        assert_eq!(*gate.active.lock().unwrap(), 0);
        assert!(gate.acquire(LARGE_INPUT_BYTES).is_some());
    }

    #[test]
    fn only_an_explicit_positive_limit_enables_the_gate() {
        assert!(configured(None).unwrap().is_none());
        for raw in ["", "0", "-1", "two"] {
            assert!(configured(Some(raw)).is_err());
        }
        assert_eq!(configured(Some("2")).unwrap().unwrap().limit, 2);
    }
}
