// SPDX-License-Identifier: AGPL-3.0-only

use super::NgramProposer;

#[test]
fn request_local_lookup_uses_only_the_supplied_history_and_configured_length() {
    let mut proposer = NgramProposer::new(4)
        .with_chain_and_cache(5, None)
        .prompt_lookup_only();
    assert_eq!(
        proposer.propose_chain(&[1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4]),
        [5, 6, 7, 1, 2]
    );
    assert_eq!(
        proposer.propose_chain(&[9, 8, 7, 6, 5, 4, 9, 8, 7, 6]),
        [5, 4, 9, 8, 7]
    );
    assert!(proposer.table.is_empty());
    assert!(proposer.cache_path.is_none());
}

#[test]
fn request_local_lookup_handles_empty_short_unmatched_and_observed_other_requests() {
    let mut proposer = NgramProposer::new(4)
        .with_chain_and_cache(5, None)
        .prompt_lookup_only();
    for history in [&[][..], &[1], &[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6]] {
        assert!(proposer.propose_chain(history).is_empty());
    }
    proposer.observe(&[1, 2, 3, 4], 99);
    assert!(proposer.propose_chain(&[8, 9, 1, 2, 3, 4]).is_empty());
    assert!(proposer.table.is_empty());
}
