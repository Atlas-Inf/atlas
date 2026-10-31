// SPDX-License-Identifier: AGPL-3.0-only

//! Issue #58: vision prompts keyed by content. The placeholder run of two
//! different images is the same token stream, so the radix tree is fed the
//! content-derived key from `vision_cache_key` instead. These tests pin the
//! properties the model layer relies on when it admits vision prefills.

use crate::prefix_cache::PrefixCache;
use crate::prefix_cache::vision_key::vision_cache_key;
use crate::radix_tree::RadixTree;

const BS: usize = 16;
const IMAGE_PAD: u32 = 151655; // <|image_pad|>

fn pad(t: u32) -> bool {
    t == IMAGE_PAD
}

/// `text` tokens, then each image as a run of `count` placeholders followed
/// by a 3-token text separator.
fn prompt(text: u32, images: &[usize]) -> Vec<u32> {
    let mut t: Vec<u32> = (1..=text).collect();
    for (i, &n) in images.iter().enumerate() {
        t.extend(std::iter::repeat_n(IMAGE_PAD, n));
        t.extend([900 + i as u32, 901, 902]);
    }
    t
}

fn key(tokens: &[u32], items: &[(u128, usize)]) -> Vec<u32> {
    vision_cache_key(tokens, pad, items).expect("prompt and items agree")
}

fn blocks(tokens: &[u32]) -> Vec<u32> {
    (0..(tokens.len() / BS) as u32).map(|b| 100 + b).collect()
}

/// The old raw-token key: two different pages with the same prompt collide.
/// Kept as the reason the content key exists.
#[test]
fn raw_placeholder_streams_collide() {
    let page = prompt(3, &[29]);
    let tree = RadixTree::new();
    tree.insert(&page, &blocks(&page), &[], BS, 0, 0);
    assert_eq!(tree.lookup(&page, BS, 0, 0).matched_tokens, 32);
}

#[test]
fn different_images_miss() {
    let page = prompt(3, &[29]); // block 0 already holds placeholders
    let tree = RadixTree::new();
    let a = key(&page, &[(0xA, 29)]);
    tree.insert(&a, &blocks(&a), &[], BS, 0, 0);
    let b = key(&page, &[(0xB, 29)]);
    let m = tree.lookup_whole_blocks(&b, BS, 0, 0);
    assert_eq!(m.matched_tokens, 0, "a different image must not reuse KV");
}

#[test]
fn same_image_resent_hits() {
    let page = prompt(20, &[40]);
    let tree = RadixTree::new();
    let a = key(&page, &[(0xA, 40)]);
    tree.insert(&a, &blocks(&a), &[], BS, 0, 0);
    let again = key(&page, &[(0xA, 40)]);
    assert_eq!(a, again, "the key is a pure function of tokens and hashes");
    let m = tree.lookup_whole_blocks(&again, BS, 0, 0);
    assert_eq!(m.matched_tokens, page.len() / BS * BS);
    assert_eq!(m.matched_blocks, blocks(&page));
}

#[test]
fn swapped_second_image_misses_from_its_first_block() {
    // 20 text, image 1 (40), 3 sep, image 2 (40), 3 sep: image 2 starts at 63,
    // inside block 3 ([48, 64)).
    let page = prompt(20, &[40, 40]);
    let tree = RadixTree::new();
    let first = key(&page, &[(0xA, 40), (0xB, 40)]);
    tree.insert(&first, &blocks(&first), &[], BS, 0, 0);

    let swapped = key(&page, &[(0xA, 40), (0xC, 40)]);
    let m = tree.lookup_whole_blocks(&swapped, BS, 0, 0);
    assert_eq!(m.matched_tokens, 48, "match stops at image 2's first block");

    // Order swap: image 1's first block (block 1, holding 16..32) already
    // differs, so only the pure-text block 0 survives.
    let reordered = key(&page, &[(0xB, 40), (0xA, 40)]);
    let m = tree.lookup_whole_blocks(&reordered, BS, 0, 0);
    assert_eq!(
        m.matched_tokens, 16,
        "text before the first image still hits"
    );
}

#[test]
fn text_before_the_first_image_keys_like_text() {
    let page = prompt(40, &[30]);
    let text_only: Vec<u32> = page[..40].to_vec();
    let tree = RadixTree::new();
    tree.insert(&text_only, &blocks(&text_only), &[], BS, 0, 0);
    let m = tree.lookup_whole_blocks(&key(&page, &[(0xA, 30)]), BS, 0, 0);
    assert_eq!(
        m.matched_tokens, 32,
        "a text-only prefix serves the image prompt"
    );
}

#[test]
fn ranks_rebuilding_the_key_agree_on_the_match() {
    // Rank 0 and rank 1 each rebuild the key from the broadcast tokens and
    // item hashes; their trees hold the same entries, so the matched lengths
    // agree before the EP min-sync ever has to cap.
    let page = prompt(20, &[40, 40]);
    let items = [(0x1234_5678_9abc_def0_u128 << 64 | 7, 40), (u128::MAX, 40)];
    let (rank0, rank1) = (RadixTree::new(), RadixTree::new());
    for tree in [&rank0, &rank1] {
        let k = key(&page, &items);
        tree.insert(&k, &blocks(&k), &[], BS, 0, 0);
    }
    let next_turn: Vec<u32> = page.iter().copied().chain(500..540).collect();
    let m0 = rank0.lookup_whole_blocks(&key(&next_turn, &items), BS, 0, 0);
    let m1 = rank1.lookup_whole_blocks(&key(&next_turn, &items), BS, 0, 0);
    assert_eq!(m0.matched_tokens, m1.matched_tokens);
    assert_eq!(m0.matched_tokens, page.len() / BS * BS);
}
