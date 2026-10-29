// SPDX-License-Identifier: AGPL-3.0-only

//! A server-wide bound on the vision rows whose pixels are held at once.
//!
//! Rank 0 keeps a request's preprocessed pixels (f32) from admission until its
//! prefill ends, in the scheduler's queue as well as while it prefills. The
//! per-request limit (`vision_limits`) does not bound the sum across requests,
//! and the queue does not bound their number, so the API acquires a
//! [`VisionRowPermit`] for every item before decoding it. The permit rides on
//! the [`crate::VisionItem`] (shared by its clones) and returns its rows when
//! the last copy drops: when the prefill clears its items, or when a queued
//! request is cancelled or refused. An exhausted budget is a transient
//! condition (the client should retry later), not a malformed request.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The budget: `limit` rows, `used` of them held by live permits.
#[derive(Debug)]
pub struct VisionRowBudget {
    limit: usize,
    used: AtomicUsize,
}

impl VisionRowBudget {
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
        })
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Rows currently held.
    pub fn in_use(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }

    /// `rows` more, if they fit; never blocks.
    pub fn try_acquire(self: &Arc<Self>, rows: usize) -> Option<VisionRowPermit> {
        let mut used = self.used.load(Ordering::Acquire);
        loop {
            let next = used.checked_add(rows).filter(|&n| n <= self.limit)?;
            match self
                .used
                .compare_exchange_weak(used, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    return Some(VisionRowPermit {
                        budget: self.clone(),
                        rows,
                    });
                }
                Err(now) => used = now,
            }
        }
    }
}

/// Rows held against a [`VisionRowBudget`], returned on drop.
#[derive(Debug)]
pub struct VisionRowPermit {
    budget: Arc<VisionRowBudget>,
    rows: usize,
}

impl VisionRowPermit {
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Return unused rows to the budget before this permit is shared.
    ///
    /// A video has no cheap header equivalent to a still image, so the API
    /// may reserve a conservative upper bound before decoding it. Once the
    /// actual groups are known, shrink that reservation before wrapping the
    /// permit in an `Arc` and attaching it to the request's items.
    pub fn shrink_to(&mut self, rows: usize) {
        assert!(
            rows <= self.rows,
            "a vision permit cannot grow via shrink_to"
        );
        let refund = self.rows - rows;
        if refund != 0 {
            self.rows = rows;
            self.budget.used.fetch_sub(refund, Ordering::AcqRel);
        }
    }

    /// Grow a pre-decode reservation to the actual request total.
    ///
    /// This keeps the same permit in place, so the API never has to drop a
    /// temporary reservation and race another request before it acquires the
    /// final amount.
    pub fn try_grow(&mut self, rows: usize) -> bool {
        if rows == 0 {
            return true;
        }
        let mut used = self.budget.used.load(Ordering::Acquire);
        loop {
            let Some(next) = used.checked_add(rows).filter(|&n| n <= self.budget.limit) else {
                return false;
            };
            match self.budget.used.compare_exchange_weak(
                used,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.rows += rows;
                    return true;
                }
                Err(now) => used = now,
            }
        }
    }
}

impl Drop for VisionRowPermit {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.rows, Ordering::AcqRel);
    }
}

/// The permit an item holds. Accounting, not content: two items compare equal
/// whatever they hold, and clones share the one permit.
#[derive(Clone, Debug, Default)]
pub struct HeldRows(pub(crate) Option<Arc<VisionRowPermit>>);

impl HeldRows {
    /// Rows held (0 for an item admitted without a budget).
    pub fn rows(&self) -> usize {
        self.0.as_ref().map_or(0, |p| p.rows)
    }
}

impl PartialEq for HeldRows {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permits_return_their_rows_when_the_last_holder_drops() {
        let b = VisionRowBudget::new(100);
        let p = b.try_acquire(60).unwrap();
        assert_eq!(b.in_use(), 60);
        assert!(b.try_acquire(41).is_none(), "over the limit");
        let q = b.try_acquire(40).unwrap();
        assert_eq!(b.in_use(), 100);
        let held = HeldRows(Some(Arc::new(p)));
        assert_eq!(held.rows(), 60);
        let clone = held.clone();
        drop(held);
        assert_eq!(b.in_use(), 100, "a clone still holds the rows");
        drop(clone);
        drop(q);
        assert_eq!(b.in_use(), 0);
        assert!(b.try_acquire(100).is_some());
        assert_eq!(b.in_use(), 0, "the unbound permit dropped at once");
    }

    #[test]
    fn concurrent_acquires_never_exceed_the_limit() {
        let b = VisionRowBudget::new(1000);
        let held: Vec<_> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..8)
                .map(|_| s.spawn(|| (0..50).filter_map(|_| b.try_acquire(7)).collect::<Vec<_>>()))
                .collect();
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(held.len(), 1000 / 7);
        assert_eq!(b.in_use(), 7 * held.len());
    }

    #[test]
    fn a_conservative_reservation_can_be_shrunk_or_grown_in_place() {
        let b = VisionRowBudget::new(100);
        let mut p = b.try_acquire(80).unwrap();
        assert_eq!(b.in_use(), 80);
        p.shrink_to(50);
        assert_eq!(p.rows(), 50);
        assert_eq!(b.in_use(), 50);
        assert!(p.try_grow(25));
        assert_eq!(p.rows(), 75);
        assert_eq!(b.in_use(), 75);
        assert!(!p.try_grow(30));
        assert_eq!(p.rows(), 75);
        assert_eq!(b.in_use(), 75);
    }
}
