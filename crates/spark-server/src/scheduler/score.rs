// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-only score jobs (`POST /v1/score`).
//!
//! A job is one shared prefix plus N branches, each scored over its own
//! single-token options by [`spark_model::traits::Model::score_branches`] in one
//! forward pass. Jobs travel on their own channel (like LoRA rotations) and
//! never enter the sequence machinery: they allocate, use and free their own
//! sequences inside one call, so they run between scheduler steps whenever
//! nothing else is in flight.

use std::time::Instant;

use spark_model::traits::Model;

/// One `/v1/score` request.
pub struct ScoreJob {
    pub prefix: Vec<u32>,
    pub branches: Vec<Vec<u32>>,
    pub options: Vec<Vec<u32>>,
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
}

/// Run queued jobs in arrival order on `stream`.
pub(super) fn run_jobs(model: &dyn Model, jobs: Vec<ScoreJob>, stream: u64) {
    for job in jobs {
        let t = Instant::now();
        let queue_us = t.duration_since(job.enqueued).as_micros() as u64;
        let res = model
            .score_branches(&job.prefix, &job.branches, &job.options, stream)
            .map(|probs| ScoreOutput {
                probs,
                engine_us: t.elapsed().as_micros() as u64,
                queue_us,
            })
            .map_err(|e| format!("{e:#}"));
        if let Err(ref e) = res {
            tracing::warn!("score job failed: {e}");
        }
        // The client may have gone away; nothing to do then.
        let _ = job.reply.send(res);
    }
}
