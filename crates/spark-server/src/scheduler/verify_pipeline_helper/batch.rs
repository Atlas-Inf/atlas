// SPDX-License-Identifier: AGPL-3.0-only

//! The batched verify's Phase-1 picks with the host-pipelined sequences
//! fanned out across the rayon pool (`ATLAS_VERIFY_PICK_PAR=1`).
//!
//! Serial Phase 1 runs `verify_pick_all_with_pipeline` per sequence: the GPU
//! argmax arms, else a D2H of the span's rows and the host pipeline over
//! them. At C8 with several sequences inside `<think>` that host work (~0.7
//! ms a row, F2's `exp` sum most of it) ran back to back, ~6.8 ms of a step
//! the GPU waits through. Here every sequence's GPU-side work (the argmax
//! arms' probes, the row copy) stays on the scheduler thread, in order, and
//! only the host pipeline over the copied rows runs in parallel.
//!
//! Each sequence's pipeline reads its own copied rows and its own
//! `ActiveSeq`, and writes only that `ActiveSeq` (plus the run's atomic
//! timing/stat counters): the same function on the same inputs as the
//! serial loop, so every pick is the serial loop's. Off while a logits or
//! AdaDec dump is active (their records would interleave).

use rayon::prelude::*;

use super::selection::{copy_rows, fast, slow};
use super::selection_io::CopyFailurePolicy;
use super::{ActiveSeq, LogitsContext, Model};
use crate::scheduler::sched_ctx::DecodeScratch;

thread_local! {
    /// Per-sequence row buffers, kept warm on the scheduler thread.
    static ROW_BUFS: std::cell::RefCell<Vec<Vec<u8>>> = const { std::cell::RefCell::new(Vec::new()) };
    /// A rayon worker's own scratch for the context it binds.
    static WORKER_SCRATCH: DecodeScratch = DecodeScratch::default();
}

/// Whether [`verify_pick_batch`] serves this step.
pub(in crate::scheduler) fn applies(ctx: &LogitsContext, n: usize) -> bool {
    n > 1 && super::prepick::enabled() && ctx.dumps.logits.is_none() && ctx.dumps.adadec.is_none()
}

/// `verify_pick_all_with_pipeline` for every member of a batched verify:
/// member `i`'s GPU argmax row ids are `results[off[i]..off[i + 1]]`, which
/// is also its row range in the logits buffer. Returns each member's picks.
pub(in crate::scheduler) fn verify_pick_batch(
    model: &dyn Model,
    results: &[u32],
    off: &[usize],
    batch: &mut [&mut ActiveSeq],
    ctx: &LogitsContext,
) -> Vec<Vec<u32>> {
    let n = batch.len();
    let vocab = model.vocab_size();
    let mut picks: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut pool = ROW_BUFS.with(|b| std::mem::take(&mut *b.borrow_mut()));
    let mut jobs: Vec<(usize, Vec<u8>, &mut ActiveSeq)> = Vec::new();
    for (i, a) in batch.iter_mut().enumerate() {
        let ids = &results[off[i]..off[i + 1]];
        // `verify_pick_all_with_pipeline`'s legacy policy: any failure picks
        // the raw argmax.
        match fast(
            model,
            ids,
            a,
            ctx,
            off[i],
            CopyFailurePolicy::LegacyFallback,
        ) {
            Ok(Some(p)) => picks[i] = p,
            Err(_) => picks[i] = ids.to_vec(),
            Ok(None) => {
                let mut buf = pool.pop().unwrap_or_default();
                buf.resize(ids.len() * vocab * 2, 0);
                if copy_rows(model, off[i], ctx, &mut buf).is_ok() {
                    jobs.push((i, buf, &mut **a));
                } else {
                    picks[i] = ids.to_vec();
                    pool.push(buf);
                }
            }
        }
    }
    let run = |(i, buf, a): &mut (usize, Vec<u8>, &mut ActiveSeq), ctx: &LogitsContext| {
        let k = off[*i + 1] - off[*i];
        (*i, slow(buf, k, vocab, a, ctx))
    };
    let done: Vec<(usize, Vec<u32>)> = if jobs.len() > 1 {
        let parts = ctx.parts();
        jobs.par_iter_mut()
            .map(|job| WORKER_SCRATCH.with(|scratch| run(job, &parts.bind(scratch))))
            .collect()
    } else {
        jobs.iter_mut().map(|job| run(job, ctx)).collect()
    };
    for (i, p) in done {
        picks[i] = p;
    }
    pool.extend(jobs.into_iter().map(|(_, buf, _)| buf));
    ROW_BUFS.with(|b| *b.borrow_mut() = pool);
    picks
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod tests;
