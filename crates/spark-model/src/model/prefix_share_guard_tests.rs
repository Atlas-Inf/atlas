// SPDX-License-Identifier: AGPL-3.0-only

//! The consumer side of the whole-block rule, checked without relying on the
//! radix tree to refuse a partly covered block: what the cache is offered,
//! what adoption accepts, and which lookup the model code may call.

use super::*;
use spark_runtime::prefix_cache::EvictedBlocks;
use std::sync::Mutex;

/// A cache that matches every token it is offered, one block per started
/// block, and records each offered length.
#[derive(Default)]
struct MatchesEverything {
    offered: Mutex<Vec<usize>>,
    blocks: Vec<u32>,
}

impl PrefixCache for MatchesEverything {
    fn lookup(&self, tokens: &[u32], block_size: usize, _: u64, _: u64) -> PrefixMatch {
        self.offered.lock().unwrap().push(tokens.len());
        PrefixMatch {
            matched_blocks: self.blocks[..tokens.len().div_ceil(block_size)].to_vec(),
            matched_tokens: tokens.len(),
            ..PrefixMatch::empty()
        }
    }

    fn insert(
        &self,
        _: &[u32],
        _: &[u32],
        _: &[u32],
        _: usize,
        _: usize,
        _: u64,
    ) -> InsertAcquired {
        InsertAcquired::default()
    }

    fn insert_with_snapshot(
        &self,
        _: &[u32],
        _: &[u32],
        _: &[u32],
        _: usize,
        _: usize,
        _: u64,
        _: usize,
        _: u64,
    ) -> (Option<usize>, InsertAcquired) {
        (None, InsertAcquired::default())
    }

    fn insert_intermediate_snapshot(
        &self,
        _: &[u32],
        _: &[u32],
        _: &[u32],
        _: usize,
        _: usize,
        _: u64,
        _: usize,
        _: u64,
    ) -> Option<usize> {
        None
    }

    fn insert_tail_snapshot(&self, _: &[u32], _: usize, _: u64, _: u64) -> Vec<usize> {
        Vec::new()
    }

    fn insert_tail_sibling_snapshot(&self, _: &[u32], _: usize, _: u64, _: u64) -> Option<usize> {
        None
    }

    fn release(&self, _: &[u32], _: usize, _: u64) {}

    fn release_matched(&self, _: &[u32], _: usize, _: usize, _: u64) {}

    fn evict(&self, _: usize) -> EvictedBlocks {
        EvictedBlocks::default()
    }

    fn evict_snapshot_lru(&self) -> Option<usize> {
        None
    }

    fn snapshot_count(&self) -> usize {
        0
    }

    fn stats(&self) -> (usize, usize) {
        (0, 0)
    }
}

/// Whatever the cache would match, a sequence offers it the whole blocks of
/// its prompt only, so the match it adopts covers whole blocks. The rank cap
/// offers the agreed whole blocks only, too.
#[test]
fn a_sequence_offers_the_cache_whole_blocks_only() {
    let mut w = World::new();
    let blocks: Vec<u32> = (0..6).map(|_| w.kv.alloc_block().unwrap()).collect();
    let cache = MatchesEverything {
        blocks: blocks.clone(),
        ..Default::default()
    };
    for len in 0..=5 * BS {
        let prompt = toks(0..len as u32);
        let whole = len / BS * BS;
        let matched = cache.lookup_whole_blocks(&prompt, BS, 0, 0);
        assert_eq!(cache.offered.lock().unwrap().pop(), Some(whole), "{len}");
        assert_eq!(matched.matched_tokens, whole);

        let agreed = whole.saturating_sub(BS);
        let capped = cap_prefix_match(&cache, &prompt, BS, 0, 0, matched, agreed);
        let offered = cache.offered.lock().unwrap().pop();
        assert_eq!(offered, (agreed > 0).then_some(agreed), "{len}");

        let mut seq = SequenceState::host_only(0);
        adopt_prefix_match(&mut seq, &capped, &mut w.kv).unwrap();
        assert_eq!(seq.cached_prefix_tokens, agreed);
        assert_eq!(seq.block_table, blocks[..agreed / BS]);
        w.kv.free_blocks(&seq.block_table);
    }
}

/// A match that ends inside a block is refused, in release builds too, and
/// no reference is taken on any of its blocks.
#[test]
fn a_match_that_ends_inside_a_block_is_not_adopted() {
    let mut w = World::new();
    let blocks = [w.kv.alloc_block().unwrap(), w.kv.alloc_block().unwrap()];
    for matched_tokens in [5, BS, BS + 5, 2 * BS + 1] {
        let inside = PrefixMatch {
            matched_blocks: blocks.to_vec(),
            matched_tokens,
            ..PrefixMatch::empty()
        };
        let mut seq = SequenceState::host_only(0);
        let refused = adopt_prefix_match(&mut seq, &inside, &mut w.kv).unwrap_err();
        assert!(refused.to_string().contains("whole blocks"), "{refused}");
        assert!(seq.block_table.is_empty());
        assert_eq!(seq.cached_prefix_tokens, 0);
        assert_eq!(blocks.map(|b| w.kv.ref_count(b)), [1, 1]);
    }
}

/// A rank that matched deeper than the pair agreed releases the nodes its
/// lookup acquired and no others. The block that straddles a retired prompt's
/// end is dead until the next insert revives it, and the response blocks
/// behind it must keep the cache's reference through the cap.
#[test]
fn the_rank_cap_releases_only_what_the_lookup_acquired() {
    let tree = RadixTree::new();
    let retired = toks(0..(5 * BS) as u32);
    tree.insert(&retired[..PROMPT], &[10, 11, 12], &[], BS, 0, 0);
    tree.insert(&retired, &[10, 11, 12, 13, 14], &[], BS, PROMPT, 0);
    tree.release(&retired, BS, 0);

    let turn2 = toks(0..(6 * BS) as u32);
    let local = tree.lookup_whole_blocks(&turn2, BS, 0, 0);
    assert_eq!(local.matched_tokens, 2 * BS);
    let capped = cap_prefix_match(&tree, &turn2, BS, 0, 0, local, BS);
    assert_eq!(capped.matched_blocks, vec![10]);
    assert_eq!(capped.matched_tokens, BS);

    // Another request revives the straddling block's node: the response
    // blocks behind it are matchable again.
    tree.insert(&retired[..3 * BS], &[10, 11, 40], &[], BS, 0, 0);
    tree.release(&retired[..3 * BS], BS, 0);
    assert_eq!(tree.peek_matched_tokens(&retired, BS, 0), 5 * BS);
}

/// Model code acquires a cached prefix through `lookup_whole_blocks` only.
/// A direct `prefix_cache.lookup(` would offer the cache the prompt's partly
/// filled last block.
#[test]
fn model_code_never_calls_the_unfloored_lookup() {
    fn scan(dir: &std::path::Path, offenders: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            if path.is_dir() {
                scan(&path, offenders);
            } else if name.ends_with(".rs") && !name.ends_with("tests.rs") {
                let mut source = std::fs::read_to_string(&path).unwrap();
                source.retain(|c| !c.is_whitespace());
                if source.contains("prefix_cache.lookup(") {
                    offenders.push(path.display().to_string());
                }
            }
        }
    }
    let mut offenders = Vec::new();
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    scan(&src, &mut offenders);
    assert!(offenders.is_empty(), "{offenders:?}");
}
