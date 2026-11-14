// SPDX-License-Identifier: AGPL-3.0-only

//! Tests of `score.rs`.

use super::*;

fn job(prefix: usize, branches: &[usize]) -> ScoreJob {
    let (reply, _rx) = tokio::sync::oneshot::channel();
    ScoreJob {
        prefix: vec![1; prefix],
        branches: branches.iter().map(|&n| vec![2; n]).collect(),
        options: branches.iter().map(|_| vec![3, 4]).collect(),
        prefix_bidir: false,
        pack: None,
        pack_limit: 0,
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
        packs: 0,
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
    // On a pack it may have one, and the pack's rows are never counted.
    let on_pack = ScoreJob {
        pack: Some(PackRef::Id(7)),
        ..job(3, &[4])
    };
    assert_eq!(cost(&on_pack, 16), (3 + 19, 2));
}

#[test]
fn an_unknown_pack_error_is_its_own_kind() {
    let e: anyhow::Error = UnknownScorePack(0xab).into();
    assert_eq!(job_error(&e), ScoreError::UnknownPack(0xab));
    let other = anyhow::anyhow!("boom");
    assert_eq!(job_error(&other), ScoreError::Failed("boom".into()));
}
