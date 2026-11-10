// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-only score jobs (`POST /v1/score`).
//!
//! A job is one shared prefix plus N branches, each scored over its own
//! single-token options by [`spark_model::traits::Model::score_requests`].
//! Jobs travel on their own channel (like LoRA rotations) and never enter the
//! sequence machinery: they allocate, use and free their own sequences inside
//! one call, so they run between scheduler steps whenever nothing else is in
//! flight (a pass shares the prefill arena, the KV pool and the SSM slots
//! with chat requests). At most [`MAX_PENDING_JOBS`] wait; more are refused
//! as busy, and a job whose client has gone is dropped unrun. Each quiescent
//! tick runs ONE pass: the front jobs that fit it together share that forward
//! pass (and one read of the weights), packed up to the row arena and the SSM
//! slot count, and the rest stay queued so chat admission gets a turn between
//! passes. If a merged pass fails, each of its jobs is retried alone so one
//! bad request cannot fail its neighbours. A job too big for one pass runs as
//! several passes in its tick, each over the prefix and a subset of its
//! branches, reassembled in order.

use std::time::Instant;

use spark_model::traits::{Model, ScoreInput};

/// Most jobs waiting for a quiescent tick; further jobs are refused as busy.
pub const MAX_PENDING_JOBS: usize = 64;

/// Why a job produced no probabilities.
#[derive(Debug, PartialEq, Eq)]
pub enum ScoreError {
    /// The pending backlog is full ([`MAX_PENDING_JOBS`]).
    Busy,
    /// The engine refused or failed the job.
    Failed(String),
}

/// The handlers' end of the score queue, present only when the loaded model
/// scores (`Model::score_capacity` is not `(0, 0)`).
#[derive(Clone)]
pub struct ScoreQueue {
    pub tx: tokio::sync::mpsc::Sender<ScoreJob>,
    /// The scheduler's pass limits for this model.
    pub budget: ScoreBudget,
}

impl ScoreQueue {
    /// Whether every branch fits a pass of its own, the least any split
    /// needs: the prefix plus that branch in rows, and its head (if any)
    /// plus itself in sequences.
    pub fn check_fits(&self, prefix: &[u32], branches: &[Vec<u32>]) -> Result<(), String> {
        let ScoreBudget {
            rows,
            streams,
            block,
        } = self.budget;
        let need_streams = 1 + usize::from(prefix.len() >= block);
        if need_streams > streams {
            return Err(format!(
                "score: a branch needs {need_streams} sequences, one pass holds {streams}"
            ));
        }
        let longest = branches.iter().map(Vec::len).max().unwrap_or(0);
        if prefix.len() + longest > rows {
            return Err(format!(
                "score: prefix {} + branch {longest} tokens exceed one pass ({rows} rows: \
                 the prefill arena and SSM staging buffers; raise --max-prefill-tokens \
                 and --max-seq-len)",
                prefix.len()
            ));
        }
        Ok(())
    }
}

/// One `/v1/score` request.
pub struct ScoreJob {
    pub prefix: Vec<u32>,
    pub branches: Vec<Vec<u32>>,
    pub options: Vec<Vec<u32>>,
    /// Read the prefix as a bidirectional state block (`"prefix_bidir"`).
    pub prefix_bidir: bool,
    /// When the HTTP handler queued the job (for `queue_us`).
    pub enqueued: Instant,
    pub reply: tokio::sync::oneshot::Sender<Result<ScoreOutput, ScoreError>>,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoreBudget {
    /// Rows per pass (the prefill arena).
    pub rows: usize,
    /// Sequences per pass (the SSM state pool).
    pub streams: usize,
    /// KV block size (a branch replays under one block of its prefix).
    pub block: usize,
}

impl ScoreBudget {
    /// `capacity` is `Model::score_capacity`; sequences are also capped by
    /// the scheduler's batch size.
    pub fn new((rows, streams): (usize, usize), max_batch: usize, block: usize) -> Self {
        Self {
            rows: rows.max(1),
            streams: streams.clamp(1, max_batch.max(1)),
            block: block.max(1),
        }
    }
}

/// Queue `job` unless the backlog is full, in which case it is refused as
/// busy. Jobs whose client has gone are dropped first. Returns whether the
/// job was queued.
pub(super) fn admit(queue: &mut Vec<ScoreJob>, job: ScoreJob) -> bool {
    queue.retain(|j| !j.reply.is_closed());
    if queue.len() >= MAX_PENDING_JOBS {
        let _ = job.reply.send(Err(ScoreError::Busy));
        return false;
    }
    queue.push(job);
    true
}

/// One tick's work: the front jobs (arrival order) that share one pass, at
/// least one; an oversized job is taken alone (it runs split). Jobs whose
/// client has gone are dropped unrun.
pub(super) fn take_tick(queue: &mut Vec<ScoreJob>, budget: &ScoreBudget) -> Vec<ScoreJob> {
    queue.retain(|j| !j.reply.is_closed());
    let (mut n, mut rows, mut streams) = (0, 0, 0);
    for job in queue.iter() {
        let (r, s) = cost(job, budget.block);
        let fits = rows + r <= budget.rows && streams + s <= budget.streams;
        if n > 0 && !fits {
            break;
        }
        n += 1;
        (rows, streams) = (rows + r, streams + s);
        if !fits {
            break;
        }
    }
    queue.drain(..n).collect()
}

/// Upper bound on a job's rows and sequences (a branch replays < one block).
fn cost(job: &ScoreJob, block: usize) -> (usize, usize) {
    let head = usize::from(job.prefix.len() >= block);
    let rows = job.prefix.len()
        + job
            .branches
            .iter()
            .map(|b| b.len() + block - 1)
            .sum::<usize>();
    (rows, head + job.branches.len())
}

/// Run queued jobs in arrival order on `stream`, merging neighbours.
pub(super) fn run_jobs(model: &dyn Model, jobs: Vec<ScoreJob>, stream: u64, budget: &ScoreBudget) {
    let block = budget.block;
    let mut batch: Vec<ScoreJob> = Vec::new();
    let (mut rows, mut streams) = (0, 0);
    for job in jobs {
        let (r, s) = cost(&job, block);
        if job.branches.len() > 1 && (r > budget.rows || s > budget.streams) {
            run_split(model, job, stream, budget, block);
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
fn split_branches(
    job: &ScoreJob,
    budget: &ScoreBudget,
    block: usize,
) -> Vec<std::ops::Range<usize>> {
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
                let _ = job.reply.send(Err(ScoreError::Failed(
                    "score: engine returned the wrong number of results".into(),
                )));
                return;
            }
            Err(e) => {
                tracing::warn!("split score job failed: {e:#}");
                let _ = job.reply.send(Err(ScoreError::Failed(format!("{e:#}"))));
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
        anyhow::ensure!(
            r.len() == n,
            "score: engine returned {} results for {n} jobs",
            r.len()
        );
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
                let _ = job.reply.send(Err(ScoreError::Failed(msg.clone())));
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

    /// A job whose client is still waiting (its receiver kept alive).
    fn live(prefix: usize, branches: &[usize]) -> (ScoreJob, ScoreRx) {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let j = job(prefix, branches);
        (ScoreJob { reply, ..j }, rx)
    }
    type ScoreRx = tokio::sync::oneshot::Receiver<Result<ScoreOutput, ScoreError>>;

    fn budget(rows: usize, streams: usize) -> ScoreBudget {
        ScoreBudget {
            rows,
            streams,
            block: 16,
        }
    }

    #[test]
    fn admit_refuses_a_full_backlog_and_drops_abandoned_jobs() {
        let mut queue = Vec::new();
        let mut rxs = Vec::new();
        for _ in 0..MAX_PENDING_JOBS {
            let (j, rx) = live(4, &[2]);
            assert!(admit(&mut queue, j));
            rxs.push(rx);
        }
        let (j, mut rx) = live(4, &[2]);
        assert!(!admit(&mut queue, j));
        assert_eq!(rx.try_recv().unwrap().err(), Some(ScoreError::Busy));
        // One client goes away: its job is dropped and the next one fits.
        drop(rxs.pop());
        let (j, _rx) = live(4, &[2]);
        assert!(admit(&mut queue, j));
        assert_eq!(queue.len(), MAX_PENDING_JOBS);
    }

    #[test]
    fn a_tick_takes_one_pass_and_leaves_the_rest_queued() {
        // Each job costs 4 + 17 = 21 rows and one sequence.
        let (mut queue, mut rxs) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            let (j, rx) = live(4, &[2]);
            queue.push(j);
            rxs.push(rx);
        }
        let b = budget(50, 8);
        assert_eq!(take_tick(&mut queue, &b).len(), 2);
        assert_eq!(queue.len(), 3);
        // Abandoned jobs are dropped unrun.
        drop(rxs.remove(2));
        drop(rxs.remove(2));
        assert_eq!(take_tick(&mut queue, &b).len(), 1);
        assert!(queue.is_empty());
        // An oversized job runs alone (split), its neighbours wait.
        let (big, _a) = live(40, &[10, 10, 10]);
        let (small, _b) = live(4, &[2]);
        queue.extend([big, small]);
        let tick = take_tick(&mut queue, &b);
        assert_eq!((tick.len(), tick[0].branches.len(), queue.len()), (1, 3, 1));
    }

    #[test]
    fn check_fits_needs_each_branch_to_fit_a_pass_alone() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let q = ScoreQueue {
            tx,
            budget: budget(50, 2),
        };
        let b = |n: usize| vec![vec![1u32; n]];
        assert!(q.check_fits(&[1; 40], &b(10)).is_ok());
        assert!(q.check_fits(&[1; 40], &b(11)).is_err());
        let one = ScoreQueue {
            budget: budget(50, 1),
            ..q.clone()
        };
        // A prefix of a whole block needs a head sequence too.
        assert!(one.check_fits(&[1; 15], &b(1)).is_ok());
        assert!(one.check_fits(&[1; 16], &b(1)).is_err());
    }

    #[test]
    fn oversized_jobs_split_into_passes_that_fit() {
        let b = budget(100, 4);
        // prefix 40; each branch costs 10 + 15 = 25 rows: two fit per pass.
        let groups = split_branches(&job(40, &[10, 10, 10, 10, 10]), &b, 16);
        assert_eq!(groups, vec![0..2, 2..4, 4..5]);
        // A single branch that is too big still gets its own pass.
        assert_eq!(
            split_branches(&job(40, &[90, 10]), &b, 16),
            vec![0..1, 1..2]
        );
        // The stream budget splits too (head + 3 branches).
        let tight = budget(1000, 4);
        assert_eq!(
            split_branches(&job(40, &[1; 5]), &tight, 16),
            vec![0..3, 3..5]
        );
    }

    #[test]
    fn cost_bounds_rows_and_streams() {
        assert_eq!(cost(&job(40, &[10, 5]), 16), (40 + 25 + 20, 3));
        // A prefix under one block has no head stream.
        assert_eq!(cost(&job(3, &[4]), 16), (3 + 19, 1));
    }
}
