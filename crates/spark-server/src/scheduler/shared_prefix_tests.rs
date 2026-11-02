// SPDX-License-Identifier: AGPL-3.0-only

//! The shared-prefix admission plan (`shared_prefix::plan`): who leads, who
//! waits, and where the checkpoints go.

use super::{Admit, Leader, Plan, Prompt, plan, shared_boundary};

const BS: usize = 16;
const MIN: usize = 2048;
/// The measured agentic probe: a 22.5K-token shared system prompt.
const SYSTEM: usize = 22_500;
/// Its block-aligned boundary, where the leader plants.
const AT: usize = SYSTEM / BS * BS;

/// A conversation: the shared `system` prefix, then a suffix unique to `id`.
fn convo(system: usize, id: u32, suffix: usize) -> Vec<u32> {
    let mut t: Vec<u32> = (0..system as u32).map(|i| i % 50_000 + 7).collect();
    t.extend((0..suffix as u32).map(|i| 1_000_000 * (id + 1) + i));
    t
}

fn prompt(t: &[u32]) -> Option<Prompt<'_>> {
    Some(Prompt {
        tokens: t,
        adapter: 0,
    })
}

fn leader(t: &[u32], done: usize, plant: Option<usize>) -> Leader<'_> {
    Leader {
        prompt: prompt(t).unwrap(),
        done,
        plant,
    }
}

fn plan_new(inflight: &[Leader<'_>], new: &[Vec<u32>]) -> Plan {
    let new: Vec<_> = new.iter().map(|t| prompt(t)).collect();
    plan(inflight, &new, &[], MIN, BS)
}

#[test]
fn shared_boundary_is_whole_blocks_and_leaves_a_row() {
    let a = convo(100, 1, 50);
    let b = convo(100, 2, 50);
    assert_eq!(shared_boundary(&a, &b, BS), 96);
    assert_eq!(shared_boundary(&a, &a, BS), 144, "identical: below the end");
    assert_eq!(shared_boundary(&a[..96], &a, BS), 80);
    assert_eq!(shared_boundary(&a, &[], BS), 0);
    assert_eq!(shared_boundary(&[1, 2, 3], &[9, 2, 3], BS), 0);
}

/// The probe's burst: eight new conversations at once. The first leads and
/// plants at the shared boundary; the other seven wait for it.
#[test]
fn a_burst_of_new_conversations_has_one_leader_and_seven_followers() {
    let burst: Vec<_> = (0..8).map(|id| convo(SYSTEM, id, 100)).collect();
    let p = plan_new(&[], &burst);
    assert_eq!(p.admit[0], Admit::Now(Some(AT)));
    assert!(p.admit[1..].iter().all(|a| *a == Admit::Hold));
    assert!(p.replant.is_empty());
}

#[test]
fn followers_wait_on_a_prefill_in_flight_and_plant_it_once() {
    let lead = convo(SYSTEM, 0, 100);
    let new: Vec<_> = (1..4).map(|id| convo(SYSTEM, id, 100)).collect();
    // Not planted yet (it started alone): plant it between its chunks.
    let p = plan_new(&[leader(&lead, 16_384, None)], &new);
    assert_eq!(p.admit, vec![Admit::Hold; 3]);
    assert_eq!(p.replant, vec![(0, AT)]);
    // Already planted: nothing to send.
    let p = plan_new(&[leader(&lead, 16_384, Some(AT))], &new);
    assert_eq!(p.admit, vec![Admit::Hold; 3]);
    assert!(p.replant.is_empty());
}

/// A leader already past the shared boundary without a checkpoint there
/// cannot help: the request starts at once (and may lead others).
#[test]
fn a_leader_past_the_boundary_holds_nobody() {
    let lead = convo(SYSTEM, 0, 100);
    let new = vec![convo(SYSTEM, 1, 100)];
    for done in [AT, AT + 64] {
        let p = plan_new(&[leader(&lead, done, None)], &new);
        assert_eq!(p.admit, vec![Admit::Now(None)]);
        assert!(p.replant.is_empty());
    }
    // Planted and past it: the checkpoint is there, nothing to wait for.
    let p = plan_new(&[leader(&lead, AT, Some(AT))], &new);
    assert_eq!(p.admit, vec![Admit::Now(None)]);
}

#[test]
fn short_shared_prefixes_and_unrelated_traffic_are_untouched() {
    let short: Vec<_> = (0..3).map(|id| convo(MIN - BS, id, 5_000)).collect();
    let p = plan_new(&[], &short);
    assert_eq!(p.admit, vec![Admit::Now(None); 3]);
    // An unrelated request between two followers starts at once.
    let a = convo(SYSTEM, 0, 100);
    let other: Vec<u32> = (0..30_000).map(|i| 900_000 + i).collect();
    let b = convo(SYSTEM, 1, 100);
    let p = plan_new(&[], &[a, other, b]);
    assert_eq!(
        p.admit,
        vec![Admit::Now(Some(AT)), Admit::Now(None), Admit::Hold]
    );
}

/// Once the leader is past its checkpoint the held cohort is released
/// together: a checkpoint already computed for them leaves no gain in
/// waiting on each other, so none of them leads the rest.
#[test]
fn a_released_cohort_does_not_chain_behind_itself() {
    let lead = convo(SYSTEM, 0, 100);
    let cohort: Vec<_> = (1..8).map(|id| convo(SYSTEM, id, 100)).collect();
    let passed = [(prompt(&lead).unwrap(), AT)];
    let new: Vec<_> = cohort.iter().map(|t| prompt(t)).collect();
    let p = plan(&[], &new, &passed, MIN, BS);
    assert_eq!(p.admit, vec![Admit::Now(None); 7]);
    // Without that memory the first would lead the other six.
    let p = plan(&[], &new, &[], MIN, BS);
    assert_eq!(p.admit[0], Admit::Now(Some(AT)));
}

/// A computed checkpoint shallower than the shared prefix by at least the
/// minimum still makes waiting worthwhile; within it, it does not.
#[test]
fn waiting_must_gain_the_minimum_over_what_is_computed() {
    let lead = convo(SYSTEM, 0, 100);
    let r = convo(SYSTEM, 1, 100);
    let inflight = [leader(&lead, 0, Some(AT))];
    let gain = |known: usize| {
        let passed = [(prompt(&lead).unwrap(), known)];
        plan(&inflight, &[prompt(&r)], &passed, MIN, BS).admit[0]
    };
    assert_eq!(gain(16_384), Admit::Hold);
    assert_eq!(gain(AT - MIN), Admit::Hold);
    assert_eq!(gain(AT - MIN + BS), Admit::Now(None));
}

/// A later follower that shares less than the planted point moves the plant
/// down to its boundary while the leader has not reached it; the earlier
/// followers restore there too.
#[test]
fn a_shorter_follower_moves_the_plant_down() {
    let lead = convo(SYSTEM, 0, 100);
    let shorter = convo(10_000, 9, 20_000);
    let p = plan_new(
        &[leader(&lead, 8_192, Some(AT))],
        std::slice::from_ref(&shorter),
    );
    assert_eq!(p.admit, vec![Admit::Hold]);
    assert_eq!(p.replant, vec![(0, 10_000 / BS * BS)]);
    // Past that point already: it cannot be planted, so no wait.
    let p = plan_new(&[leader(&lead, 12_288, Some(AT))], &[shorter]);
    assert_eq!(p.admit, vec![Admit::Now(None)]);
    assert!(p.replant.is_empty());
}

/// A near-identical prompt (shared up to the leader's tail) needs no plant:
/// it waits for the leader to finish and restores at its tail checkpoint.
#[test]
fn a_prompt_sharing_the_leaders_tail_waits_for_it_to_finish() {
    let lead = convo(SYSTEM, 0, 4);
    let mut twin = lead.clone();
    twin.push(5);
    let p = plan_new(&[leader(&lead, 16_384, None)], std::slice::from_ref(&twin));
    assert_eq!(p.admit, vec![Admit::Hold]);
    assert!(p.replant.is_empty());
    let p = plan_new(&[leader(&lead, lead.len(), None)], &[twin]);
    assert_eq!(p.admit, vec![Admit::Now(None)]);
}

#[test]
fn other_adapters_and_vision_prompts_never_follow() {
    let lead = convo(SYSTEM, 0, 100);
    let r = convo(SYSTEM, 1, 100);
    let other = Prompt {
        tokens: &r,
        adapter: 3,
    };
    let p = plan(
        &[leader(&lead, 0, None)],
        &[Some(other), None],
        &[],
        MIN,
        BS,
    );
    assert_eq!(p.admit, vec![Admit::Now(None); 2]);
    assert!(p.replant.is_empty());
}

/// Of two leaders, a request follows the one that leaves it the deepest
/// restore.
#[test]
fn a_request_follows_the_deepest_shared_prefix() {
    let shallow = convo(4_096, 0, 30_000);
    let deep = convo(SYSTEM, 1, 100);
    let r = convo(SYSTEM, 2, 100);
    let p = plan_new(&[leader(&shallow, 0, None), leader(&deep, 0, None)], &[r]);
    assert_eq!(p.admit, vec![Admit::Hold]);
    assert_eq!(p.replant, vec![(1, AT)]);
}
