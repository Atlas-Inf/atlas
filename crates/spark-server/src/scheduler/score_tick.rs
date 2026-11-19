// SPDX-License-Identifier: AGPL-3.0-only

//! Which queued score jobs a tick runs: the backlog bound, the packs' read
//! state, and the jobs that share one pass (`score.rs`).

use spark_model::traits::{Model, ScorePackKey, ScorePackSpec};

use super::{MAX_PENDING_JOBS, PackRef, ScoreBudget, ScoreError, ScoreJob, cost};

/// Queue `job` unless the backlog is full, in which case it is refused as
/// busy. Jobs whose client has gone are dropped first. Returns whether the
/// job was queued.
pub(in crate::scheduler) fn admit(queue: &mut Vec<ScoreJob>, job: ScoreJob) -> bool {
    queue.retain(|j| !j.reply.is_closed());
    if queue.len() >= MAX_PENDING_JOBS {
        let _ = job.reply.send(Err(ScoreError::Busy));
        return false;
    }
    queue.push(job);
    true
}

/// Drop the queued jobs whose client has gone.
pub(in crate::scheduler) fn prune(queue: &mut Vec<ScoreJob>) {
    queue.retain(|j| !j.reply.is_closed());
}

/// Whether each queued job's pack is ready for a pass (held, in device
/// memory, or nothing to read), and whether reads are still running, from
/// the model; a read is started for each pack that is not.
pub(in crate::scheduler) fn readiness(model: &dyn Model, queue: &[ScoreJob]) -> (Vec<bool>, bool) {
    let keyed: Vec<(usize, ScorePackKey<'_>)> = queue
        .iter()
        .enumerate()
        .filter_map(|(i, job)| match &job.pack {
            None => None,
            Some(PackRef::Id(id)) => Some((i, ScorePackKey::Id(*id))),
            Some(PackRef::Tokens(t)) => Some((
                i,
                ScorePackKey::Spec(ScorePackSpec {
                    tokens: t,
                    images: &job.images,
                    prefix_bidir: job.prefix_bidir,
                }),
            )),
        })
        .collect();
    let mut ready = vec![true; queue.len()];
    if keyed.is_empty() {
        return (ready, false);
    }
    let keys: Vec<ScorePackKey<'_>> = keyed.iter().map(|(_, k)| *k).collect();
    let (flags, reading) = model.score_packs_prefetch(&keys);
    for ((i, _), r) in keyed.iter().zip(flags) {
        ready[*i] = r;
    }
    (ready, reading)
}

/// One tick's work: jobs in arrival order that share one pass, at least
/// one, with no more on packs than the model holds (`pack_limit`: a tick's
/// packs are all held through its pass); an oversized job is taken alone (it
/// runs split). `ready` flags each
/// queued job's pack and `reading` says reads are running (`readiness`).
/// With every pack ready the front jobs are taken; with at least half ready
/// (or some, and nothing reading) only the ready ones, the rest staying
/// queued; with none ready and nothing reading the front jobs again (their
/// restores read in the pass). `None`: too few ready while reads run, so
/// wait for them.
pub(in crate::scheduler) fn take_tick(
    queue: &mut Vec<ScoreJob>,
    budget: &ScoreBudget,
    ready: &[bool],
    reading: bool,
) -> Option<Vec<ScoreJob>> {
    let n_ready = ready.iter().filter(|&&r| r).count();
    let only_ready = match n_ready {
        r if r == queue.len() => false,
        r if 2 * r >= queue.len() || (r > 0 && !reading) => true,
        _ if reading => return None,
        _ => false,
    };
    let (mut take, mut rows, mut streams, mut vision, mut packs) = (Vec::new(), 0, 0, 0, 0);
    for (i, job) in queue.iter().enumerate() {
        if only_ready && !ready[i] {
            continue;
        }
        let (r, s) = cost(job, budget.block);
        let on_pack = usize::from(job.pack.is_some());
        let fits = rows + r <= budget.rows
            && streams + s <= budget.streams
            && budget.vision_fits(vision, job)
            && (job.pack_limit == 0 || packs + on_pack <= job.pack_limit);
        if !take.is_empty() && !fits {
            break;
        }
        take.push(i);
        (rows, streams) = (rows + r, streams + s);
        vision += job.vision_rows;
        packs += on_pack;
        if !fits {
            break;
        }
    }
    let mut jobs: Vec<ScoreJob> = take.iter().rev().map(|&i| queue.remove(i)).collect();
    jobs.reverse();
    Some(jobs)
}
