// SPDX-License-Identifier: AGPL-3.0-only

//! A two-rank pair on the host. Each rank has its own KV pool and prefix
//! cache. Both look the prompt up, agree on the smaller match and cap to it
//! (`cap_prefix_match`, the production step). The head caches a sequence when
//! it retires; the worker only frees it, so its cache holds prompt blocks
//! alone and the two trees differ.

use super::*;

struct Pair {
    head: World,
    worker: World,
}

/// One sequence on both ranks.
struct PairSeq {
    head: SequenceState,
    worker: SequenceState,
}

impl Pair {
    fn new() -> Self {
        Self {
            head: World::new(),
            worker: World::new(),
        }
    }

    /// Prefill on both ranks. Returns the sequence and each rank's own match
    /// before the cap, `[head, worker]`.
    fn prefill(&mut self, prompt: &[u32]) -> (PairSeq, [usize; 2]) {
        let mut local = [&self.head, &self.worker]
            .map(|rank| Some(rank.cache.lookup_whole_blocks(prompt, BS, 0, 0)));
        let matched = [0, 1].map(|r| local[r].as_ref().unwrap().matched_tokens);
        let agreed = matched[0].min(matched[1]);
        let mut on = |rank: &mut World, r: usize| {
            let own = local[r].take().unwrap();
            let capped = cap_prefix_match(&rank.cache, prompt, BS, 0, 0, own, agreed);
            assert_eq!(capped.matched_tokens, agreed, "rank {r} after the cap");
            rank.prefill_matched(prompt, capped)
        };
        let seq = PairSeq {
            head: on(&mut self.head, 0),
            worker: on(&mut self.worker, 1),
        };
        assert_eq!(
            seq.head.cached_prefix_blocks,
            seq.worker.cached_prefix_blocks
        );
        assert_eq!(seq.head.block_table.len(), seq.worker.block_table.len());
        (seq, matched)
    }

    fn decode(&mut self, seq: &mut PairSeq, tokens: &[u32]) {
        self.head.decode(&mut seq.head, tokens);
        self.worker.decode(&mut seq.worker, tokens);
    }

    fn verify(&mut self, seq: &mut PairSeq, drafts: &[u32], accepted: usize) {
        self.head.verify(&mut seq.head, drafts, accepted);
        self.worker.verify(&mut seq.worker, drafts, accepted);
    }

    /// The head caches and frees now. The worker's copy is returned: the
    /// worker frees it when the head tells it to, which a test may delay.
    fn retire_head(&mut self, seq: PairSeq) -> SequenceState {
        self.head.retire(seq.head);
        seq.worker
    }

    fn retire(&mut self, seq: PairSeq) {
        let worker = self.retire_head(seq);
        self.worker.free(worker);
    }

    fn drain(&mut self) {
        self.head.drain();
        self.worker.drain();
    }
}

/// A three-turn conversation. The worker caches prompt blocks only, so the
/// pair's agreed match is the previous prompt's whole blocks even though the
/// head also cached the response.
#[test]
fn pair_multi_turn_conversation_agrees_on_the_previous_prompt() {
    let mut p = Pair::new();
    let turn1 = toks(0..PROMPT as u32);
    let answer1 = toks(500..530);
    let (mut seq, matched) = p.prefill(&turn1);
    assert_eq!(matched, [0, 0]);
    p.verify(&mut seq, &[answer1[0], 9001, 9002], 1);
    p.decode(&mut seq, &answer1[1..]);
    p.retire(seq);

    let turn2 = join(&[&turn1, &answer1, &toks(900..925)]);
    let answer2 = toks(600..640);
    let (mut seq, matched) = p.prefill(&turn2);
    assert_eq!(matched, [2 * BS, 2 * BS]);
    p.decode(&mut seq, &answer2);
    p.retire(seq);

    let turn3 = join(&[&turn2, &answer2, &toks(950..975)]);
    let (mut seq, matched) = p.prefill(&turn3);
    assert_eq!(matched[1], turn2.len() / BS * BS);
    assert_eq!(seq.head.cached_prefix_tokens, turn2.len() / BS * BS);
    p.decode(&mut seq, &toks(980..990));
    p.retire(seq);
    p.drain();
}

/// The probe's cells on a pair: a padding send, turn 2, the strict-prefix
/// retry while the worker still holds turn 2, turn 2 again, and a live twin.
#[test]
fn pair_strict_prefix_retry_and_live_twin_share_whole_blocks_only() {
    let mut p = Pair::new();
    let turn1 = toks(0..PROMPT as u32);
    let (mut seq, _) = p.prefill(&turn1);
    p.decode(&mut seq, &[77]);
    p.retire(seq);

    let turn2 = join(&[&turn1, &toks(500..520), &toks(900..930)]);
    let (mut conversation, matched) = p.prefill(&turn2);
    assert_eq!(matched, [2 * BS, 2 * BS]);
    p.decode(&mut conversation, &toks(600..625));
    let conversation_on_worker = p.retire_head(conversation);

    let (mut retry, matched) = p.prefill(&turn1);
    assert_eq!(matched, [2 * BS, 2 * BS]);
    p.verify(&mut retry, &toks(7000..7008), 2);
    p.decode(&mut retry, &toks(7100..7140));
    let retry_on_worker = p.retire_head(retry);
    p.worker.free(conversation_on_worker);

    let turn2_blocks = turn2.len() / BS * BS;
    let (mut again, matched) = p.prefill(&turn2);
    assert_eq!(matched, [turn2_blocks, turn2_blocks]);
    p.worker.free(retry_on_worker);
    let (mut twin, matched) = p.prefill(&turn2);
    assert_eq!(matched, [turn2_blocks, turn2_blocks]);
    p.decode(&mut again, &toks(600..610));
    p.verify(&mut twin, &toks(600..606), 3);
    p.retire(again);
    p.decode(&mut twin, &toks(603..640));
    p.retire(twin);
    p.drain();
}

/// The ranks' caches diverge (the worker evicted its deepest blocks). The
/// head matched more, so it releases and takes the agreed blocks again; the
/// next turn finds both caches whole.
#[test]
fn pair_caps_the_deeper_rank_and_recovers_on_the_next_turn() {
    let mut p = Pair::new();
    let turn1 = toks(0..PROMPT as u32);
    let answer1 = toks(500..530);
    let (mut seq, _) = p.prefill(&turn1);
    p.decode(&mut seq, &answer1);
    p.retire(seq);
    let turn2 = join(&[&turn1, &answer1, &toks(900..925)]);
    let answer2 = toks(600..640);
    let (mut seq, _) = p.prefill(&turn2);
    p.decode(&mut seq, &answer2);
    p.retire(seq);

    let evicted = p.worker.cache.evict(2);
    assert_eq!(evicted.physical.len(), 2);
    apply_evicted_blocks(evicted, &mut p.worker.kv);

    let turn2_blocks = turn2.len() / BS * BS;
    let turn3 = join(&[&turn2, &answer2, &toks(950..975)]);
    let (mut seq, matched) = p.prefill(&turn3);
    assert!(matched[0] >= turn2_blocks, "{matched:?}");
    assert_eq!(matched[1], turn2_blocks - 2 * BS);
    assert_eq!(seq.head.cached_prefix_tokens, matched[1]);
    let answer3 = toks(980..990);
    p.decode(&mut seq, &answer3);
    p.retire(seq);

    let turn4 = join(&[&turn3, &answer3, &toks(1000..1020)]);
    let (mut seq, matched) = p.prefill(&turn4);
    assert_eq!(matched[1], turn3.len() / BS * BS);
    assert_eq!(seq.head.cached_prefix_tokens, turn3.len() / BS * BS);
    p.decode(&mut seq, &toks(1100..1105));
    p.retire(seq);
    p.drain();
}
