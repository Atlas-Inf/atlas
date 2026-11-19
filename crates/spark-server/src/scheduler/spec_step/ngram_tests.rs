// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn native_proposal_does_not_run_lookup_and_disabled_native_uses_bounded_lookup() {
    assert_eq!(
        finish_proposal(2, Ok(Some(vec![11, 22])), || panic!("lookup after native")).unwrap(),
        [11, 22]
    );
    assert_eq!(
        finish_proposal(2, Ok(None), || vec![3, 4, 5]).unwrap(),
        [3, 4]
    );
    assert!(finish_proposal(2, Ok(None), Vec::new).unwrap().is_empty());
    assert!(
        finish_proposal(0, Ok(Some(Vec::new())), || panic!("lookup after native"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn native_errors_and_incorrect_counts_never_fall_back_to_lookup() {
    let error = finish_proposal(2, Err(anyhow::anyhow!("injected draft failure")), || {
        panic!("lookup after native failure")
    })
    .unwrap_err();
    assert_eq!(error.to_string(), "injected draft failure");
    assert!(finish_proposal(2, Ok(Some(vec![1])), || panic!("lookup after bad count")).is_err());
}

#[test]
fn draft_count_obeys_configuration_and_reserves_one_anchor_row() {
    for (max, limit, position, expected) in [
        (5, None, 57, 5),
        (2, Some(64), 57, 2),
        (5, Some(64), 62, 1),
        (5, Some(64), 63, 0),
        (5, Some(64), 64, 0),
        (0, Some(64), 57, 0),
        (5, Some(64), usize::MAX, 0),
    ] {
        assert_eq!(draft_count(max, limit, position), expected);
    }
}
