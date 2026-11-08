// SPDX-License-Identifier: AGPL-3.0-only

//! The DFlash2 selector's lattice for an exact tree verify
//! (`ATLAS_DFLASH_TREE`, `speculative::draft_tree`).
//!
//! Each single-sequence propose records the buffers its selector read
//! ([`LastSelect`]). The tree builder reruns the selector on them (one
//! ~0.05 ms launch; its scratch then holds every row's exact top-K list,
//! which the batched selector does not leave behind), reads back those
//! lists, the projected hidden rows, the block anchor and the min_tokens ban
//! depth, and scores children on the host from the selector's codebook
//! copies (`Dflash2CandidateSelector::predecessor_host`): the same bilinear
//! score the selector's walk computes, so the tree's first branch is, up to
//! rounding, the chain the propose drafted. The proposal need not be exact:
//! the verify is.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::OnceLock;

use super::BlockDiffusionDraftHead;
use crate::layers::ops;
use crate::speculative::draft_tree::{DraftTree, Lattice, best_first};

/// The buffers one single-sequence selector launch read.
#[derive(Clone, Copy, Debug)]
pub struct LastSelect {
    pub logits: DevicePtr,
    pub projected: DevicePtr,
    pub scratch: DevicePtr,
    pub anchor: DevicePtr,
    pub ban_depth: DevicePtr,
}

/// A process-lifetime `[2 * MAX_GAMMA]` word buffer for the rerun's picks.
fn picks_buffer(gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    static BUF: OnceLock<u64> = OnceLock::new();
    if let Some(&p) = BUF.get() {
        return Ok(DevicePtr(p));
    }
    let p = gpu.alloc(4 * 64)?;
    Ok(DevicePtr(*BUF.get_or_init(|| p.0)))
}

fn words(b: &[u8]) -> impl Iterator<Item = [u8; 4]> + '_ {
    b.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]])
}

impl BlockDiffusionDraftHead {
    /// Record (or forget) the buffers of this propose's selector launch.
    pub(super) fn note_select(&self, select: Option<LastSelect>) {
        *self.last_select.lock() = select;
    }

    /// A best-first draft tree of at most `rows` rows over the last
    /// single-sequence propose's lattice, or `None` where there is none (no
    /// DFlash2 selector, its device kernel or host codebooks; a batched or
    /// rank-split propose last).
    pub(super) fn tree_from_last_select(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
    ) -> Result<Option<DraftTree>> {
        let Some(sel) = *self.last_select.lock() else {
            return Ok(None);
        };
        let Some(selector) = self.candidate_selector.as_ref() else {
            return Ok(None);
        };
        let (Some(pred_cb), Some(succ_cb)) = (
            selector.predecessor_host.as_ref(),
            selector.successor_host.as_ref(),
        ) else {
            return Ok(None);
        };
        let Some(kernel) = self.kernels.dflash2_candidate_selector.filter(|k| k.0 != 0) else {
            return Ok(None);
        };
        let (gamma, rank, vocab, top_k) =
            (self.gamma, selector.rank, self.vocab_size, selector.top_k);
        if !(2..=32).contains(&gamma) {
            return Ok(None);
        }
        let stream = gpu.default_stream();
        let end_ids = crate::traits::EosBan::model_end_ids();
        ops::dflash2_candidate_selector(
            gpu,
            kernel,
            sel.logits,
            sel.projected,
            selector.predecessor_codebook.weight,
            selector.successor_codebook.weight,
            picks_buffer(gpu)?,
            sel.anchor,
            sel.ban_depth,
            end_ids,
            gamma as u32,
            vocab as u32,
            rank as u32,
            top_k as u32,
            sel.scratch,
            stream,
        )?;
        gpu.synchronize(stream)?;

        // The rows' merged lists: f32 values [gamma, K], then u32 ids.
        let kmax = ops::DFLASH2_SELECTOR_MAX_TOP_K;
        let lists_at =
            ops::dflash2_selector_scratch_bytes(gamma) - ops::dflash2_selector_lists_bytes(gamma);
        let mut lists = vec![0u8; ops::dflash2_selector_lists_bytes(gamma)];
        gpu.copy_d2h(sel.scratch.offset(lists_at), &mut lists)?;
        let mut proj = vec![0u8; gamma * rank * 2];
        gpu.copy_d2h(sel.projected, &mut proj)?;
        let (mut anchor, mut ban) = ([0u8; 4], [0u8; 4]);
        gpu.copy_d2h(sel.anchor, &mut anchor)?;
        if !sel.ban_depth.is_null() {
            gpu.copy_d2h(sel.ban_depth, &mut ban)?;
        }
        let vals: Vec<f32> = words(&lists[..gamma * kmax * 4])
            .map(f32::from_le_bytes)
            .collect();
        let ids: Vec<u32> = words(&lists[gamma * kmax * 4..])
            .map(u32::from_le_bytes)
            .collect();
        let proj: Vec<f32> = proj
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        // Draft position d is the selector's mask row d + 1.
        let lat = Lattice {
            anchor: u32::from_le_bytes(anchor),
            ban_depth: u32::from_le_bytes(ban),
            end_ids,
            ids: (1..gamma)
                .map(|r| ids[r * kmax..r * kmax + top_k].to_vec())
                .collect(),
            unary: (1..gamma)
                .map(|r| vals[r * kmax..r * kmax + top_k].to_vec())
                .collect(),
        };
        fn row_of(cb: &[half::bf16], t: u32, vocab: usize, rank: usize) -> &[half::bf16] {
            let t = if (t as usize) < vocab { t as usize } else { 0 };
            &cb[t * rank..(t + 1) * rank]
        }
        let mut score = |d: usize, pred: u32| -> Vec<f32> {
            let h = &proj[(d + 1) * rank..(d + 2) * rank];
            let p = row_of(pred_cb, pred, vocab, rank);
            let ctx: Vec<f32> = p.iter().zip(h).map(|(a, b)| a.to_f32() * b).collect();
            lat.ids[d]
                .iter()
                .zip(&lat.unary[d])
                .map(|(&id, &u)| {
                    if (id as usize) >= vocab {
                        return f32::NEG_INFINITY;
                    }
                    let s = row_of(succ_cb, id, vocab, rank);
                    u + ctx.iter().zip(s).map(|(c, w)| c * w.to_f32()).sum::<f32>()
                })
                .collect()
        };
        Ok(Some(best_first(&lat, rows, vocab as u32, &mut score)))
    }
}
