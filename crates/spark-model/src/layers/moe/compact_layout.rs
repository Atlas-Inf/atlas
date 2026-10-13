// SPDX-License-Identifier: AGPL-3.0-only

/// Map global expert slots onto a compact slab containing only locally owned
/// experts. Remote EP placeholders remain `None` in the global pointer table.
pub(super) fn compact_slot_map(
    locally_owned: impl IntoIterator<Item = bool>,
) -> (usize, Vec<Option<usize>>) {
    let mut local_count = 0usize;
    let slots = locally_owned
        .into_iter()
        .map(|owned| {
            owned.then(|| {
                let slot = local_count;
                local_count += 1;
                slot
            })
        })
        .collect();
    (local_count, slots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ep_slots_compact_only_locally_owned_experts() {
        let (local_count, slots) = compact_slot_map([true, false, true, false, false, true]);
        assert_eq!(local_count, 3);
        assert_eq!(slots, vec![Some(0), None, Some(1), None, None, Some(2)]);
    }

    #[test]
    fn empty_ep_rank_requires_no_slab_storage() {
        let (local_count, slots) = compact_slot_map([false, false]);
        assert_eq!(local_count, 0);
        assert_eq!(slots, vec![None, None]);
    }
}
