// SPDX-License-Identifier: AGPL-3.0-only

//! Content-addressed prefix-cache keys for vision prompts (issue #58).
//!
//! A vision prompt carries one placeholder token per encoder row, and the
//! placeholder id is the same for every image, so two prompts that differ only
//! in pixels are byte-identical token streams. Keying the radix tree on them
//! would hand one image's KV to another. The fix is the vLLM `mm_hash` /
//! SGLang `pad_value` one: the CACHE key replaces each placeholder with an id
//! derived from the item's content hash and the placeholder's index inside the
//! item, while the model still embeds the real tokens. Two different images
//! then diverge at their first placeholder, a re-sent image matches in full,
//! and text before the first image is keyed exactly as before.
//!
//! Derived ids set the top bit, so they can never equal a real vocabulary id.

/// High bit set on every derived placeholder key. Real vocabularies are far
/// below 2^31 ids.
pub const VISION_KEY_BIT: u32 = 0x8000_0000;

/// 64-bit finaliser (splitmix64): full avalanche, so neighbouring indices and
/// hashes that differ in one bit give unrelated keys.
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Cache-key token for placeholder `index` of the item whose content hash is
/// `item_hash`.
pub fn vision_key_token(item_hash: u128, index: u32) -> u32 {
    let lo = item_hash as u64;
    let hi = (item_hash >> 64) as u64;
    let h = mix64(lo ^ mix64(hi ^ mix64(index as u64)));
    VISION_KEY_BIT | (h as u32 & !VISION_KEY_BIT)
}

/// The prefix-cache key stream for `tokens`: a copy in which the n-th
/// placeholder of item i (in prompt order) becomes
/// `vision_key_token(items[i].0, n)`. `items` holds `(content_hash,
/// placeholder_count)` in prompt order.
///
/// `None` when the placeholders do not account for the items exactly (a
/// mismatch means the prompt and its media disagree, so nothing about it may
/// be cached) or when the stream already holds an id with the key bit set.
pub fn vision_cache_key(
    tokens: &[u32],
    is_placeholder: impl Fn(u32) -> bool,
    items: &[(u128, usize)],
) -> Option<Vec<u32>> {
    let total: usize = items.iter().map(|it| it.1).sum();
    let mut key = Vec::with_capacity(tokens.len());
    // `seen` placeholders so far; item `item` owns `[start, start + count)`.
    let (mut seen, mut item, mut start) = (0usize, 0usize, 0usize);
    for &t in tokens {
        if t & VISION_KEY_BIT != 0 {
            return None;
        }
        if !is_placeholder(t) {
            key.push(t);
            continue;
        }
        if seen == total {
            return None;
        }
        while seen >= start + items[item].1 {
            start += items[item].1;
            item += 1;
        }
        key.push(vision_key_token(items[item].0, (seen - start) as u32));
        seen += 1;
    }
    (seen == total).then_some(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: u32 = 7;
    fn pad(t: u32) -> bool {
        t == PAD
    }

    #[test]
    fn placeholders_take_item_derived_ids_and_text_is_unchanged() {
        let key = vision_cache_key(&[1, PAD, PAD, 2, PAD], pad, &[(11, 2), (22, 1)]).unwrap();
        assert_eq!(key[0], 1);
        assert_eq!(key[3], 2);
        assert_eq!(key[1], vision_key_token(11, 0));
        assert_eq!(key[2], vision_key_token(11, 1));
        assert_eq!(key[4], vision_key_token(22, 0));
        assert!(key[1] & VISION_KEY_BIT != 0 && key[1] != key[2]);
    }

    #[test]
    fn mismatched_placeholder_counts_are_not_keyable() {
        // Too few placeholders for the items, too many, and an empty item list.
        assert!(vision_cache_key(&[PAD, PAD], pad, &[(1, 3)]).is_none());
        assert!(vision_cache_key(&[PAD, PAD, PAD], pad, &[(1, 2)]).is_none());
        assert!(vision_cache_key(&[PAD], pad, &[]).is_none());
        assert!(vision_cache_key(&[PAD, PAD], pad, &[(1, 1), (2, 2)]).is_none());
        assert!(vision_cache_key(&[1, 2], pad, &[(1, 1)]).is_none());
        // Text-only with no items is its own key.
        assert_eq!(vision_cache_key(&[1, 2], pad, &[]).unwrap(), vec![1, 2]);
    }

    #[test]
    fn split_runs_follow_the_item_count_not_contiguity() {
        // A GLM video's frames are separated by timestamp text: one item,
        // placeholders in several runs.
        let key = vision_cache_key(&[PAD, 5, PAD, PAD], pad, &[(9, 3)]).unwrap();
        assert_eq!(key[2], vision_key_token(9, 1));
        assert_eq!(key[3], vision_key_token(9, 2));
    }
}
