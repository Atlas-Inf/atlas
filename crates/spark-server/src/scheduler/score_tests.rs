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
        images: Vec::new(),
        vision_rows: 0,
        readouts: None,
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

/// A tick with every pack ready (`take_tick` after `prune`).
fn tick(queue: &mut Vec<ScoreJob>, b: &ScoreBudget) -> Vec<ScoreJob> {
    prune(queue);
    let ready = vec![true; queue.len()];
    take_tick(queue, b, &ready, false).expect("all ready")
}

fn budget(rows: usize, streams: usize) -> ScoreBudget {
    ScoreBudget {
        rows,
        streams,
        block: 16,
        vision_rows: 0,
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
    assert_eq!(tick(&mut queue, &b).len(), 2);
    assert_eq!(queue.len(), 3);
    // Abandoned jobs are dropped unrun.
    drop(rxs.remove(2));
    drop(rxs.remove(2));
    assert_eq!(tick(&mut queue, &b).len(), 1);
    assert!(queue.is_empty());
    // An oversized job runs alone (split), its neighbours wait.
    let (big, _a) = live(40, &[10, 10, 10]);
    let (small, _b) = live(4, &[2]);
    queue.extend([big, small]);
    let tick = tick(&mut queue, &b);
    assert_eq!((tick.len(), tick[0].branches.len(), queue.len()), (1, 3, 1));
}

fn with_image(job: ScoreJob, vision_rows: usize) -> ScoreJob {
    ScoreJob {
        images: vec![spark_model::VisionItem::image(vec![0.0; 12], 2, 2)],
        vision_rows,
        ..job
    }
}

#[test]
fn image_jobs_share_passes_up_to_the_vision_row_budget() {
    let b = budget(1000, 8).with_vision_rows(10);
    let (text, _t) = live(4, &[2]);
    let (a, _a) = live(4, &[2]);
    let (c, _c) = live(4, &[2]);
    let (d, _d) = live(4, &[2]);
    let mut queue = vec![text, with_image(a, 4), with_image(c, 6), with_image(d, 1)];
    // Text plus two images (4 + 6 = 10 rows); the third would make 11.
    let first = tick(&mut queue, &b);
    assert_eq!(first.len(), 3);
    let second = tick(&mut queue, &b);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].vision_rows, 1);
}

#[test]
fn image_jobs_are_taken_in_passes_of_their_own() {
    let b = budget(1000, 8);
    let (text, _text_rx) = live(4, &[2]);
    let (image, _image_rx) = live(4, &[2]);
    let image = with_image(image, 1);
    let (tail, _tail_rx) = live(4, &[2]);
    let mut queue = vec![text, image, tail];
    let first = tick(&mut queue, &b);
    assert_eq!(first.len(), 1);
    assert!(first[0].images.is_empty());
    let second = tick(&mut queue, &b);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].images.len(), 1);
    assert_eq!(tick(&mut queue, &b).len(), 1);
}

/// Queued jobs `0..n`, told apart by their prefix length (10 + i).
fn numbered(n: usize) -> (Vec<ScoreJob>, Vec<ScoreRx>) {
    (0..n).map(|i| live(10 + i, &[2])).unzip()
}

fn names(jobs: &[ScoreJob]) -> Vec<usize> {
    jobs.iter().map(|j| j.prefix.len() - 10).collect()
}

#[test]
fn a_tick_runs_the_ready_packs_once_half_the_queue_is_ready() {
    let b = budget(1000, 8);
    let (mut queue, _rxs) = numbered(4);
    // Under half ready while reads run: wait for them.
    assert!(take_tick(&mut queue, &b, &[false, true, false, false], true).is_none());
    assert_eq!(queue.len(), 4);
    // Half ready: those run, in arrival order; the others keep reading.
    let tick = take_tick(&mut queue, &b, &[false, true, false, true], true).unwrap();
    assert_eq!(names(&tick), vec![1, 3]);
    assert_eq!(names(&queue), vec![0, 2]);
}

#[test]
fn a_tick_waits_for_no_read_that_is_not_running() {
    let b = budget(1000, 8);
    // Nothing reading: the ready jobs run however few they are...
    let (mut queue, _rxs) = numbered(4);
    let tick = take_tick(&mut queue, &b, &[false, false, true, false], false).unwrap();
    assert_eq!(names(&tick), vec![2]);
    // ...and with none ready the front jobs run (their reads happen in the
    // pass's restore, as without read-ahead).
    let tick = take_tick(&mut queue, &b, &[false, false, false], false).unwrap();
    assert_eq!(names(&tick), vec![0, 1, 3]);
    assert!(queue.is_empty());
}

#[test]
fn ready_jobs_still_share_a_pass_only_within_its_budget() {
    // Each job costs 10 + i + 17 rows: two of the ready ones fit 60.
    let b = budget(60, 8);
    let (mut queue, _rxs) = numbered(5);
    let tick = take_tick(&mut queue, &b, &[true, false, true, true, false], true).unwrap();
    assert_eq!(names(&tick), vec![0, 2]);
    assert_eq!(names(&queue), vec![1, 3, 4]);
}

#[test]
fn a_tick_takes_no_more_pack_jobs_than_the_model_holds() {
    let b = budget(1000, 16);
    let (mut queue, _rxs) = numbered(5);
    for (k, job) in queue.iter_mut().enumerate() {
        job.pack = Some(PackRef::Id(k as u64));
        job.pack_limit = 2;
    }
    let tick = take_tick(&mut queue, &b, &[true; 5], false).unwrap();
    assert_eq!(names(&tick), vec![0, 1]);
    // A job without a pack still fits beside them.
    queue[0].pack = None;
    let tick = take_tick(&mut queue, &b, &[true; 3], false).unwrap();
    assert_eq!(names(&tick), vec![2, 3, 4]);
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
