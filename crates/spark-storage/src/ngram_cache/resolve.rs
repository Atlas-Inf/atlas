// SPDX-License-Identifier: AGPL-3.0-only

//! The decide+fault cluster: `resolve`/`prefetch` -> `resolve_inner`
//! (decide pass 1, fault pass 2), `drop_reservations`, `fetch_many`,
//! `end_batch`, `victim`. Split out of `ngram_cache.rs` for the <=500 LoC
//! cap; the pin/fault invariants are documented on the methods.

use anyhow::{Result, bail};

use super::fault::*;
use super::{AlignedBlock, NgramRowCache};

impl NgramRowCache {
    /// Resolve `row_ids` to slot indices, faulting misses in from NVMe.
    ///
    /// Every returned slot is PINNED for the caller's batch: the gather runs
    /// after this returns, so a later resolve in the same batch must not
    /// evict a row the kernel is about to read. Call [`Self::end_batch`] once
    /// the gather has been issued.
    pub fn resolve(&mut self, row_ids: &[u64], out_slots: &mut Vec<u32>) -> Result<()> {
        self.resolve_inner(row_ids, Some(out_slots), true)?;
        Ok(())
    }

    /// Warm `row_ids` into the cache WITHOUT pinning or returning slots —
    /// the prefetch half of `resolve`, for a caller that knows its ids ahead
    /// of time (the PLE row ids are a pure function of the prompt, so a warm
    /// thread can run them while earlier layers compute). Same decide+fault,
    /// so the caller's mutex must stay held across it: residency is published
    /// at decision time, and the lock is what keeps "resident" synonymous
    /// with "bytes landed". Rows stay evictable — a consume that arrives
    /// after eviction just faults again — which is also what makes prefetch
    /// unable to starve `victim`: it never adds to the pinned set.
    pub fn prefetch(&mut self, row_ids: &[u64]) -> Result<usize> {
        self.resolve_inner(row_ids, None, false)
    }

    /// The shared decide+fault. With `pin`, every touched slot joins the
    /// batch in flight; `out_slots` receives the per-id slot when asked.
    /// Returns the number of rows faulted in.
    fn resolve_inner(
        &mut self,
        row_ids: &[u64],
        out_slots: Option<&mut Vec<u32>>,
        pin: bool,
    ) -> Result<usize> {
        let mut out_slots = out_slots;
        if let Some(out) = out_slots.as_deref_mut() {
            out.clear();
            out.reserve(row_ids.len());
        }

        // PASS 1 -- DECIDE. Serial, and deliberately so: the CLOCK hand, the
        // eviction order and the slot handed to each id are exactly what the
        // old single-pass loop produced, in the same order. Only the I/O moves.
        //
        // The residency bookkeeping that `fetch_into` used to do at its END now
        // happens HERE, at decision time. That is what makes a repeated id
        // inside one batch resolve as a hit to the slot already scheduled for
        // it, instead of being faulted twice into two slots.
        let mut faults: Vec<Fault> = Vec::new();
        for &id in row_ids {
            if id >= self.rows_total {
                bail!(
                    "NgramRowCache: row id {id} >= table rows {} (hash/table mismatch)",
                    self.rows_total
                );
            }
            let slot = match self.map.get(&id) {
                Some(&s) => {
                    self.hits += 1;
                    self.refbit[s as usize] = true;
                    if pin {
                        self.pinned[s as usize] = true;
                    }
                    s
                }
                None => {
                    self.misses += 1;
                    // Oversubscription still REFUSES here, before a single byte
                    // of I/O is issued -- `victim` bails when every slot is
                    // pinned by the batch in flight. That refusal must UNDO the
                    // reservations already made: publishing residency at
                    // decision time is what makes duplicates cheap, and it is
                    // also what would leave rows claimed but never read if this
                    // returned straight out of the loop.
                    let s = match self.victim() {
                        Ok(s) => s,
                        Err(e) => {
                            self.drop_reservations(&faults);
                            return Err(e);
                        }
                    };
                    self.map.insert(id, s);
                    self.slot_row[s as usize] = id;
                    self.refbit[s as usize] = true;
                    if pin {
                        self.pinned[s as usize] = true;
                    }
                    faults.push(Fault { id, slot: s });
                    s
                }
            };
            if let Some(out) = out_slots.as_deref_mut() {
                out.push(slot);
            }
        }

        // PASS 2 -- FAULT. The misses are independent: distinct slots (every
        // slot handed out above is pinned, and `victim` skips pinned slots), so
        // the arena writes are disjoint, and `read_exact_at` is positional so a
        // shared `&File` has no cursor to race on.
        if let Err(e) = self.fetch_many(&faults) {
            // A partially-written slot must not stay claimed as resident: a
            // wrong n-gram row reads as fluent output with wrong logits, which
            // is the failure mode this cache exists to avoid. Drop the WHOLE
            // batch's claims rather than only the failed worker's -- the
            // workers share no progress record, so which rows landed is not
            // knowable here, and over-dropping only costs a refetch.
            self.drop_reservations(&faults);
            return Err(e);
        }
        Ok(faults.len())
    }

    /// Un-publish reservations whose bytes never landed.
    ///
    /// Conditional on both sides: a claim is only withdrawn if it is still the
    /// one this batch made. Nothing in the current code can have replaced it —
    /// the slots stay pinned for the batch — but an unconditional `remove`
    /// would delete a LIVE mapping the moment that stops being true, and the
    /// symptom would be a wrong n-gram row rather than a crash.
    fn drop_reservations(&mut self, faults: &[Fault]) {
        for f in faults {
            if self.map.get(&f.id) == Some(&f.slot) {
                self.map.remove(&f.id);
            }
            if self.slot_row[f.slot as usize] == f.id {
                self.slot_row[f.slot as usize] = u64::MAX;
            }
        }
    }

    /// Issue `faults` concurrently. `&self`: the residency bookkeeping is
    /// already done (pass 1) and every write below goes through a raw pointer
    /// into a slot this batch owns exclusively.
    fn fetch_many(&self, faults: &[Fault]) -> Result<()> {
        if faults.is_empty() {
            return Ok(());
        }
        let src = RowSource {
            file: &self.file,
            base_offset: self.base_offset,
            segments: self.segments.as_ref(),
            row_stride: self.row_stride,
            scale_file: self.scales.as_ref().and_then(|sc| sc.file.as_ref()),
        };
        let ptrs = ArenaPtrs {
            rows: self.arena.slot_host_ptr(0, 0)?,
            scales: match self.scales.as_ref().filter(|sc| sc.file.is_some()) {
                Some(sc) => Some(sc.arena.slot_host_ptr(0, 0)?),
                None => None,
            },
        };

        let want = fault_threads();
        if want <= 1 || faults.len() == 1 {
            let mut bounce = AlignedBlock::new();
            for f in faults {
                fetch_row(&src, &ptrs, &mut bounce, f.id, f.slot)?;
            }
            return Ok(());
        }

        // One chunk per worker rather than a shared queue: the faults cost the
        // same each (one or two 4 KiB O_DIRECT reads), so static partitioning
        // needs no synchronisation and leaves no tail worth stealing.
        let nthreads = want.min(faults.len());
        let chunk = faults.len().div_ceil(nthreads);
        let first_err: std::sync::Mutex<Option<anyhow::Error>> = std::sync::Mutex::new(None);
        std::thread::scope(|scope| {
            for part in faults.chunks(chunk) {
                let (src, ptrs, first_err) = (&src, &ptrs, &first_err);
                scope.spawn(move || {
                    let mut bounce = AlignedBlock::new();
                    for f in part {
                        if let Err(e) = fetch_row(src, ptrs, &mut bounce, f.id, f.slot) {
                            let mut g = first_err.lock().expect("fault error mutex");
                            if g.is_none() {
                                *g = Some(e);
                            }
                            return;
                        }
                    }
                });
            }
        });
        match first_err.into_inner().expect("fault error mutex") {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Release the batch's pins (call after the gather kernels are issued).
    pub fn end_batch(&mut self) {
        self.pinned.fill(false);
    }

    /// CLOCK second-chance victim among the unpinned slots.
    fn victim(&mut self) -> Result<u32> {
        for _ in 0..(self.slots * 2) {
            let s = self.hand;
            self.hand = (self.hand + 1) % self.slots;
            if self.pinned[s] {
                continue;
            }
            if self.refbit[s] {
                self.refbit[s] = false;
                continue;
            }
            if self.slot_row[s] != u64::MAX {
                let old = self.slot_row[s];
                self.map.remove(&old);
                self.evictions += 1;
            }
            return Ok(s as u32);
        }
        bail!(
            "NgramRowCache: every one of {} slots is pinned by the batch in flight — \
             raise the cache size or lower max-prefill-tokens",
            self.slots
        )
    }
}
