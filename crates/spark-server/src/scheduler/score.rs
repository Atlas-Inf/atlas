// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-only score jobs (`POST /v1/score`).
//!
//! A job is one shared prefix plus N branches, each scored over its own
//! single-token options by [`spark_model::traits::Model::score_requests`].
//! Jobs travel on their own channel (like LoRA rotations) and never enter the
//! sequence machinery: they allocate, use and free their own sequences inside
//! one call, so they run between scheduler steps whenever nothing else is in
//! flight. Jobs queued together share ONE forward pass (and one read of the
//! weights), packed up to the row arena and the SSM slot count; if a merged
//! pass fails, each of its jobs is retried alone so one bad request cannot
//! fail its neighbours. A job too big for one pass runs as several passes,
//! each over the prefix and a subset of its branches, reassembled in order.

use std::time::Instant;

use spark_model::traits::{Model, ScoreInput};

/// One `/v1/score` request.
pub struct ScoreJob {
    pub prefix: Vec<u32>,
    pub branches: Vec<Vec<u32>>,
    pub options: Vec<Vec<u32>>,
    /// Read the prefix as a bidirectional state block (`"prefix_bidir"`).
    pub prefix_bidir: bool,
    /// When the HTTP handler queued the job (for `queue_us`).
    pub enqueued: Instant,
    pub reply: tokio::sync::oneshot::Sender<Result<ScoreOutput, String>>,
}

/// Per-branch option probabilities plus the engine time for the job.
pub struct ScoreOutput {
    pub probs: Vec<Vec<f32>>,
    pub engine_us: u64,
    /// Time from the handler queuing the job to the scheduler starting it.
    pub queue_us: u64,
    /// Jobs that shared this job's forward pass (1 = ran alone).
    pub batched: usize,
}

/// Limits a merged pass must respect.
pub(super) struct ScoreBudget {
    /// Rows per pass (the prefill arena).
    pub rows: usize,
    /// Sequences per pass (the SSM state pool).
    pub streams: usize,
}

/// Upper bound on a job's rows and sequences (a branch replays < one block).
fn cost(job: &ScoreJob, block: usize) -> (usize, usize) {
    let head = usize::from(job.prefix.len() >= block);
    let rows = job.prefix.len() + job.branches.iter().map(|b| b.len() + block - 1).sum::<usize>();
    (rows, head + job.branches.len())
}

/// Run queued jobs in arrival order on `stream`, merging neighbours.
pub(super) fn run_jobs(model: &dyn Model, jobs: Vec<ScoreJob>, stream: u64, budget: &ScoreBudget) {
    const BLOCK: usize = 16;
    let mut batch: Vec<ScoreJob> = Vec::new();
    let (mut rows, mut streams) = (0, 0);
    for job in jobs {
        let (r, s) = cost(&job, BLOCK);
        if job.branches.len() > 1 && (r > budget.rows || s > budget.streams) {
            run_split(model, job, stream, budget, BLOCK);
            continue;
        }
        if !batch.is_empty() && (rows + r > budget.rows || streams + s > budget.streams) {
            run_batch(model, std::mem::take(&mut batch), stream);
            (rows, streams) = (0, 0);
        }
        rows += r;
        streams += s;
        batch.push(job);
    }
    if !batch.is_empty() {
        run_batch(model, batch, stream);
    }
}

/// Branch groups of `job` that each fit one pass (at least one branch each).
fn split_branches(job: &ScoreJob, budget: &ScoreBudget, block: usize) -> Vec<std::ops::Range<usize>> {
    let head = usize::from(job.prefix.len() >= block);
    let mut groups = Vec::new();
    let (mut lo, mut rows, mut streams) = (0, job.prefix.len(), head);
    for (i, b) in job.branches.iter().enumerate() {
        let r = b.len() + block - 1;
        if i > lo && (rows + r > budget.rows || streams + 1 > budget.streams) {
            groups.push(lo..i);
            (lo, rows, streams) = (i, job.prefix.len(), head);
        }
        rows += r;
        streams += 1;
    }
    groups.push(lo..job.branches.len());
    groups
}

/// Run one oversized job as several passes and reply once.
fn run_split(model: &dyn Model, job: ScoreJob, stream: u64, budget: &ScoreBudget, block: usize) {
    let t = Instant::now();
    let mut probs = Vec::with_capacity(job.branches.len());
    let groups = split_branches(&job, budget, block);
    for g in &groups {
        let input = ScoreInput {
            prefix: &job.prefix,
            branches: &job.branches[g.clone()],
            options: &job.options[g.clone()],
            prefix_bidir: job.prefix_bidir,
        };
        match model.score_requests(std::slice::from_ref(&input), stream) {
            Ok(mut out) if out.len() == 1 => probs.append(&mut out[0]),
            Ok(_) => {
                let _ = job.reply.send(Err("score: engine returned the wrong number of results".into()));
                return;
            }
            Err(e) => {
                tracing::warn!("split score job failed: {e:#}");
                let _ = job.reply.send(Err(format!("{e:#}")));
                return;
            }
        }
    }
    let queue_us = t.duration_since(job.enqueued).as_micros() as u64;
    let _ = job.reply.send(Ok(ScoreOutput {
        probs,
        engine_us: t.elapsed().as_micros() as u64,
        queue_us,
        batched: 1,
    }));
}

fn run_batch(model: &dyn Model, batch: Vec<ScoreJob>, stream: u64) {
    let t = Instant::now();
    let n = batch.len();
    let inputs: Vec<ScoreInput<'_>> = batch
        .iter()
        .map(|j| ScoreInput {
            prefix: &j.prefix,
            branches: &j.branches,
            options: &j.options,
            prefix_bidir: j.prefix_bidir,
        })
        .collect();
    let res = model.score_requests(&inputs, stream).and_then(|r| {
        anyhow::ensure!(r.len() == n, "score: engine returned {} results for {n} jobs", r.len());
        Ok(r)
    });
    drop(inputs);
    match res {
        Ok(per_job) => {
            let engine_us = t.elapsed().as_micros() as u64;
            for (job, probs) in batch.into_iter().zip(per_job) {
                let queue_us = t.duration_since(job.enqueued).as_micros() as u64;
                // The client may have gone away; nothing to do then.
                let _ = job.reply.send(Ok(ScoreOutput {
                    probs,
                    engine_us,
                    queue_us,
                    batched: n,
                }));
            }
        }
        Err(e) if n > 1 => {
            tracing::warn!("merged score pass of {n} jobs failed, retrying each alone: {e:#}");
            for job in batch {
                run_batch(model, vec![job], stream);
            }
        }
        Err(e) => {
            tracing::warn!("score job failed: {e:#}");
            let msg = format!("{e:#}");
            for job in batch {
                let _ = job.reply.send(Err(msg.clone()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(prefix: usize, branches: &[usize]) -> ScoreJob {
        let (reply, _rx) = tokio::sync::oneshot::channel();
        ScoreJob {
            prefix: vec![1; prefix],
            branches: branches.iter().map(|&n| vec![2; n]).collect(),
            options: branches.iter().map(|_| vec![3, 4]).collect(),
            prefix_bidir: false,
            enqueued: Instant::now(),
            reply,
        }
    }

    #[test]
    fn oversized_jobs_split_into_passes_that_fit() {
        let budget = ScoreBudget { rows: 100, streams: 4 };
        // prefix 40; each branch costs 10 + 15 = 25 rows: two fit per pass.
        let groups = split_branches(&job(40, &[10, 10, 10, 10, 10]), &budget, 16);
        assert_eq!(groups, vec![0..2, 2..4, 4..5]);
        // A single branch that is too big still gets its own pass.
        assert_eq!(split_branches(&job(40, &[90, 10]), &budget, 16), vec![0..1, 1..2]);
        // The stream budget splits too (head + 3 branches).
        let tight = ScoreBudget { rows: 1000, streams: 4 };
        assert_eq!(split_branches(&job(40, &[1; 5]), &tight, 16), vec![0..3, 3..5]);
    }

    #[test]
    fn cost_bounds_rows_and_streams() {
        assert_eq!(cost(&job(40, &[10, 5]), 16), (40 + 25 + 20, 3));
        // A prefix under one block has no head stream.
        assert_eq!(cost(&job(3, &[4]), 16), (3 + 19, 1));
    }
}
