// SPDX-License-Identifier: AGPL-3.0-only

//! One warm native follower: prove the current restore on EVERY rank, then
//! acquire it in inline chunk 0 before any model work can evict snapshots.
use super::*;
use spark_runtime::prefix_cache::PrefixCache;

pub(in crate::model) const EP_CMD_NATIVE_PREFIX_PROBE: u32 = 0xFFFF_FFE5;
const MAX_PROBE_TOKENS: usize = 1_048_576;

fn bounded_payload(len: usize, max_blocks: usize, bs: usize, scratch: usize) -> bool {
    len > 0
        && len <= MAX_PROBE_TOKENS
        && len <= max_blocks.saturating_mul(bs)
        && len.saturating_mul(4) <= scratch
}

fn resident_only(cache: &dyn PrefixCache, tokens: &[u32], bs: usize, adapter: u64) -> bool {
    let Some(tier) = cache.nvme().filter(|tier| tier.is_enabled()) else {
        return true;
    };
    let plan = tier.plan_restore(tokens, bs, adapter);
    if plan.disk.is_empty() {
        return true;
    }
    // Exactly the declined-restore path: release the temporary plan's pins.
    // No I/O or new blocks, so completion cannot return an unadopted block.
    let unused = tier.complete_restore(tokens, bs, adapter, &plan, &[], false);
    assert!(unused.is_empty(), "declined prefix probe adopted no blocks");
    false
}

fn paired_target_snapshot(
    cache: &dyn PrefixCache,
    tokens: &[u32],
    bs: usize,
    adapter: u64,
    session: u64,
    expected: (usize, usize),
) -> Option<(usize, usize)> {
    let matched = cache.peek_matched_tokens(tokens, bs, adapter);
    let depth = cache
        .nvme()?
        .snapshot_anchor_depth(tokens, matched, session, adapter);
    if matched != expected.0 || depth < expected.1 || depth == 0 {
        return None;
    }
    cache
        .snapshot_at(tokens, depth, adapter)
        .map(|snap| (snap, depth))
}

fn agreed_credit(
    local: Option<usize>,
    min: impl FnOnce(u32) -> Result<u32>,
) -> Result<Option<usize>> {
    let credit = min(local.unwrap_or(0) as u32)? as usize;
    Ok((credit > 0).then_some(credit))
}

impl TransformerModel {
    fn native_prefix_local_credit(
        &self,
        owner: &SequenceState,
        tokens: &[u32],
        session: u64,
        expected_depth: usize,
    ) -> Option<usize> {
        if !self.native_glm_mtp || owner.disable_mtp || !owner.prefix_lookup_skip
            || owner.adapter_id != self.adapter_id_for_slot(owner.adapter_slot)
            || !owner.vision_cache_key.is_empty() || self.tokens_have_vision_pad(tokens)
            || self.mla_prefill_needs_full_recompute() || !self.prefix_cache.is_active()
            || crate::det_trace::on()
            // Native sparse-index sequences cannot be swapped between probe
            // and start_new_requests; other model layouts get no credit.
            || self.kv_cache.lock().sparse_index_config().is_none()
        {
            return None;
        }
        let bs = self.kv_cache.lock().block_size();
        let depth = expected_depth;
        if bs != 16
            || depth < owner.marconi_skip_to
            || depth == 0
            || depth >= tokens.len()
            || depth < crate::model::mtp_carry::marconi_min_tokens()
            || owner.cached_prefix_tokens != owner.cached_prefix_blocks * bs
            || !resident_only(self.prefix_cache.as_ref(), tokens, bs, owner.adapter_id)
        {
            return None;
        }
        let shared = owner
            .prefix_ref_tokens
            .iter()
            .take(owner.cached_prefix_tokens)
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count()
            / bs
            * bs;
        let matched = self
            .prefix_cache
            .peek_matched_tokens(tokens, bs, owner.adapter_id);
        let shared = shared.min(matched);
        if shared < depth {
            return None;
        }
        let (snap, selected_depth) = paired_target_snapshot(
            self.prefix_cache.as_ref(),
            tokens,
            bs,
            owner.adapter_id,
            session,
            (matched, depth),
        )?;
        if selected_depth != expected_depth
            || (!self.ssm_snapshots.has_aux(snap) && self.requires_aux_state())
        {
            return None;
        }
        Some(shared / bs)
    }

    pub(in crate::model) fn native_prefix_admission_blocks_impl(
        &self,
        owner: &SequenceState,
        tokens: &[u32],
        session: u64,
        private_blocks: usize,
    ) -> Result<Option<usize>> {
        let bs = self.kv_cache.lock().block_size();
        if !self.multi_rank_protocol_active()
            || !self.ep_protocol_v2
            || !bounded_payload(
                tokens.len(),
                self.max_blocks_per_seq as usize,
                bs,
                self.buffers.sizes().scratch,
            )
        {
            return Ok(None);
        }
        // A preceding fork may have saved a deeper shared checkpoint since
        // this owner acquired its prefix. Use today's paired anchor, bounded
        // by the pinned common prefix below. Never go shallower: the scheduler
        // reserved private replay from the owner's original restore depth.
        let matched = self
            .prefix_cache
            .peek_matched_tokens(tokens, bs, owner.adapter_id);
        let Some((_, depth)) = paired_target_snapshot(
            self.prefix_cache.as_ref(),
            tokens,
            bs,
            owner.adapter_id,
            session,
            (matched, owner.marconi_skip_to),
        ) else {
            return Ok(None);
        };
        let local = self.native_prefix_local_credit(owner, tokens, session, depth);
        let Some(proposer) = self.proposer.as_ref() else {
            return Ok(None);
        };
        if local.is_none()
            || proposer
                .native_prefix_free_blocks()
                .is_none_or(|free| free < private_blocks)
            || !self
                .native_mtp_prefix
                .lock()
                .iter()
                .any(|entry| matches(entry, tokens, depth, owner.adapter_id))
        {
            return Ok(None);
        }
        // All validation precedes the command. Maximum payload is 4 MiB.
        self.ep_broadcast_seq_and_cmd(
            owner.slot_idx.try_into()?,
            EP_CMD_NATIVE_PREFIX_PROBE,
            true,
        )?;
        for word in [
            tokens.len() as u32,
            depth as u32,
            session as u32,
            (session >> 32) as u32,
        ] {
            self.ep_broadcast_u32(word)?;
        }
        self.ep_broadcast_tokens(tokens)?;
        agreed_credit(local, |credit| self.ep_min_u32(credit))
    }

    pub(in crate::model) fn native_prefix_receive_probe(
        &self,
        owner: Option<&SequenceState>,
    ) -> Result<bool> {
        let len = self.ep_broadcast_u32(0)? as usize;
        let depth = self.ep_broadcast_u32(0)? as usize;
        let lo = self.ep_broadcast_u32(0)? as u64;
        let hi = self.ep_broadcast_u32(0)? as u64;
        anyhow::ensure!(
            bounded_payload(
                len,
                self.max_blocks_per_seq as usize,
                self.kv_cache.lock().block_size(),
                self.buffers.sizes().scratch
            ) && depth < len,
            "native prefix probe exceeds model/payload bounds"
        );
        let tokens = self.ep_broadcast_tokens(&vec![0; len])?;
        let local = owner.and_then(|owner| {
            self.native_prefix_local_credit(owner, &tokens, lo | (hi << 32), depth)
        });
        // A missing worker slot/snapshot must still vote, never return early
        // and leave the head's collective waiting. Worker owns no predictor.
        agreed_credit(local, |credit| self.ep_min_u32(credit))?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::radix_tree::RadixTree;

    #[test]
    fn a_missing_worker_snapshot_or_owner_vetoes_the_credit() {
        assert_eq!(agreed_credit(Some(65_436), |_| Ok(0)).unwrap(), None);
        let mut voted = false;
        assert_eq!(
            agreed_credit(None, |local| {
                voted = true;
                assert_eq!(local, 0);
                Ok(local)
            })
            .unwrap(),
            None
        );
        assert!(voted, "missing owner must still join the collective");
        assert_eq!(
            agreed_credit(Some(100), |local| Ok(local.min(99))).unwrap(),
            Some(99)
        );
    }

    #[test]
    fn tp4_shared_prefix_credit_requires_all_four_current_snapshots() {
        let blocks = 65_436;
        for missing in 0..4 {
            let mut ranks = [blocks; 4];
            ranks[missing] = 0;
            let local = (ranks[0] > 0).then_some(ranks[0] as usize);
            let credit = agreed_credit(local, |head| {
                assert_eq!(head, ranks[0]);
                Ok(*ranks.iter().min().unwrap())
            })
            .unwrap();
            assert_eq!(credit, None, "rank {missing} must veto prefix reuse");
        }
        assert_eq!(
            agreed_credit(Some(blocks as usize), |head| {
                Ok([head, blocks, blocks, blocks].into_iter().min().unwrap())
            })
            .unwrap(),
            Some(blocks as usize)
        );
    }

    #[test]
    fn probe_payload_is_bounded_before_allocation_or_send() {
        assert!(bounded_payload(1_048_576, 65_537, 16, 4_194_304));
        assert!(!bounded_payload(0, 65_537, 16, 4_194_304));
        assert!(!bounded_payload(1_048_577, 65_537, 16, 8_388_608));
        assert!(!bounded_payload(1_048_576, 65_535, 16, 4_194_304));
        assert!(!bounded_payload(1_048_576, 65_537, 16, 4_194_303));
    }

    #[test]
    fn enabled_nvme_accepts_resident_prefix_and_declines_fault_in_without_leaking_pins() {
        use spark_runtime::prefix_cache::NvmePrefixTier;
        let tree = RadixTree::new();
        assert!(tree.enable(8));
        let tokens = vec![7; 64];
        tree.insert(&tokens, &[1, 2, 3, 4], &[], 16, 0, 0);
        tree.release(&tokens, 16, 0);
        assert!(resident_only(&tree, &tokens, 16, 0));
        assert_eq!(tree.evict(1).physical, vec![4]);
        assert!(!resident_only(&tree, &tokens, 16, 0));
        assert!(!resident_only(&tree, &tokens, 16, 0));
        assert_eq!(
            tree.evict(1).physical,
            vec![3],
            "declined probes must release pins"
        );
    }

    #[test]
    fn advancing_shared_anchor_can_admit_while_original_owner_is_live() {
        let tree = RadixTree::new();
        let mut tokens = vec![7; 67];
        tree.insert_with_snapshot(&tokens[..48], &[1, 2, 3], &[], 16, 0, 0, 0, 0);
        tree.insert(&tokens[..64], &[1, 2, 3, 4], &[], 16, 0, 0);
        let owner_depth = 48;
        // A completed suffix prefill adds the next shared checkpoint while
        // the original owner still records the older restore boundary.
        tree.insert_with_snapshot(&tokens[..64], &[1, 2, 3, 4], &[], 16, 1, 0, 0, 0);
        tokens[66] = 9; // A sibling differs only after the pinned prefix.
        assert_eq!(
            paired_target_snapshot(&tree, &tokens, 16, 0, 0, (64, owner_depth)),
            Some((1, 64))
        );
        // A shallower candidate cannot borrow the old continuation budget.
        assert_eq!(
            paired_target_snapshot(&tree, &tokens, 16, 0, 0, (64, 80)),
            None
        );
    }

    #[test]
    fn credit_requires_the_current_selected_resident_exact_target_anchor() {
        let tree = RadixTree::new();
        let tokens = vec![7; 67];
        tree.insert_with_snapshot(&tokens[..48], &[1, 2, 3], &[], 16, 0, 0, 0, 0);
        tree.insert(&tokens[..64], &[1, 2, 3, 4], &[], 16, 0, 0);
        assert_eq!(
            paired_target_snapshot(&tree, &tokens, 16, 0, 0, (64, 48)),
            Some((0, 48))
        );
        // A worker still must select exactly the anchor the head proved.
        tree.insert_with_snapshot(&tokens[..64], &[1, 2, 3, 4], &[], 16, 1, 0, 0, 0);
        assert_eq!(
            paired_target_snapshot(&tree, &tokens, 16, 0, 0, (64, 48))
                .filter(|&(_, depth)| depth == 48),
            None
        );
        // Nor can another adapter borrow the same token string's target KV.
        assert_eq!(
            paired_target_snapshot(&tree, &tokens, 16, 99, 0, (64, 64)),
            None
        );
        assert_eq!(
            paired_target_snapshot(&tree, &tokens, 16, 0, 0, (48, 64)),
            None
        );
    }
}
