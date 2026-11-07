// SPDX-License-Identifier: AGPL-3.0-only

//! Prompt-end tails that keep their sliding-window slots for prefix reuse.
//!
//! A sequence resuming from a cached prefix of `b` blocks reads, in each
//! slotted layer, only the window behind position `b * block_size`: blocks
//! `[b - window_blocks, b)`. When a prompt finishes prefill, its last
//! `tail_blocks` whole blocks are pinned, so the next turn that matches up to
//! (or a little short of) that prompt's end resumes there instead of
//! re-prefilling everything. Pins form an LRU of `max_tails` tails.
//!
//! A pin holds one KV reference on each block (the block cannot be freed and
//! re-lent under it) and keeps its slot: releasing a pinned block's slot only
//! parks it, and the last unpin returns a parked slot.

use std::collections::VecDeque;

use super::PagedKvCache;

pub(super) struct SwaPins {
    count: Vec<u16>,
    parked: Vec<bool>,
    tails: VecDeque<Vec<u32>>,
    max_tails: usize,
    tail_blocks: usize,
}

impl SwaPins {
    pub(super) fn new(num_blocks: usize, max_tails: usize, tail_blocks: usize) -> Self {
        let blocks = if max_tails == 0 { 0 } else { num_blocks };
        Self {
            count: vec![0; blocks],
            parked: vec![false; blocks],
            tails: VecDeque::with_capacity(max_tails),
            max_tails,
            tail_blocks,
        }
    }

    fn pinned(&self, block: u32) -> bool {
        self.count.get(block as usize).is_some_and(|&c| c > 0)
    }

    /// Defer releasing a pinned block's slot to its last unpin. `false` when
    /// no pin holds `block`: release it now.
    pub(super) fn park(&mut self, block: u32) -> bool {
        if !self.pinned(block) {
            return false;
        }
        self.parked[block as usize] = true;
        true
    }

    pub(super) fn forget(&mut self, block: u32) {
        debug_assert!(!self.pinned(block), "freed block {block} is still pinned");
        if let Some(p) = self.parked.get_mut(block as usize) {
            *p = false;
        }
    }

    /// Drop one pin; `true` when that was the last one and the slot was
    /// parked, so it must be released now.
    fn unpin(&mut self, block: u32) -> bool {
        let b = block as usize;
        self.count[b] -= 1;
        self.count[b] == 0 && std::mem::take(&mut self.parked[b])
    }

    fn touch(&mut self, block: u32) {
        if let Some(i) = self.tails.iter().rposition(|t| t.contains(&block))
            && let Some(tail) = self.tails.remove(i)
        {
            self.tails.push_back(tail);
        }
    }
}

impl PagedKvCache {
    /// Whether finished prompts pin their sliding-window tails.
    pub fn swa_prefix_tails_enabled(&self) -> bool {
        self.swa.as_ref().is_some_and(|s| s.pins.max_tails > 0)
    }

    /// Pin the slotted blocks among the last `tail_blocks` of `prompt_blocks`
    /// (a prompt's whole blocks, in order), evicting the least recently used
    /// tail when the budget is full. Pinning the same tail again refreshes it.
    pub fn pin_swa_tail(&mut self, prompt_blocks: &[u32]) {
        let Some(swa) = self.swa.as_mut().filter(|s| s.pins.max_tails > 0) else {
            return;
        };
        let from = prompt_blocks.len().saturating_sub(swa.pins.tail_blocks);
        let tail: Vec<u32> = prompt_blocks[from..]
            .iter()
            .copied()
            .filter(|&b| swa.slots.holds(b))
            .collect();
        if tail.is_empty() {
            return;
        }
        if let Some(i) = swa.pins.tails.iter().position(|t| *t == tail) {
            let same = swa.pins.tails.remove(i).expect("position is in range");
            swa.pins.tails.push_back(same);
            return;
        }
        let mut evicted = Vec::new();
        while swa.pins.tails.len() >= swa.pins.max_tails {
            let Some(old) = swa.pins.tails.pop_front() else {
                break;
            };
            for &b in &old {
                if swa.pins.unpin(b) {
                    swa.slots.release(b);
                }
            }
            evicted.extend(old);
        }
        for &b in &tail {
            swa.pins.count[b as usize] += 1;
        }
        swa.pins.tails.push_back(tail.clone());
        // Take the new references before dropping the old ones, so a block in
        // both tails never transiently reaches zero.
        tail.iter().for_each(|&b| self.inc_ref(b));
        evicted.iter().for_each(|&b| {
            self.dec_ref(b);
        });
    }

    /// Deepest prefix of `matched` (whole blocks, in order) a sequence can
    /// resume from: every block a slotted layer reads behind its end holds a
    /// slot. `None` when no layer is slotted. Refreshes the tail serving it.
    pub fn swa_adoptable_blocks(&mut self, matched: &[u32]) -> Option<usize> {
        let swa = self.swa.as_mut()?;
        let window = swa.window_blocks;
        let mut run = 0usize;
        let mut depth = 0usize;
        for (i, &block) in matched.iter().enumerate() {
            run = if swa.slots.holds(block) { run + 1 } else { 0 };
            if run >= window.min(i + 1) {
                depth = i + 1;
            }
        }
        if depth > 0 {
            swa.pins.touch(matched[depth - 1]);
        }
        Some(depth)
    }
}
