// SPDX-License-Identifier: AGPL-3.0-only

use super::row_contract::*;

fn contract() -> LightningRowContract {
    LightningRowContract::new(4, 3).unwrap()
}

fn synthetic() -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let logits = vec![
        vec![0.0, 10.0, 0.0],
        vec![0.0, 2.0, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let w1 = vec![vec![0.0], vec![-2.0], vec![2.0]];
    let w2 = vec![vec![0.0], vec![1.0], vec![-1.0]];
    (logits, w1, w2)
}

#[test]
fn query_rows_are_anchor_then_three_masks() {
    assert_eq!(contract().query_rows(42, 990), vec![42, 990, 990, 990]);
}

#[test]
fn gamma_and_k_drift_fail_closed() {
    assert!(
        LightningRowContract::new(8, 3)
            .unwrap_err()
            .to_string()
            .contains("gamma")
    );
    assert!(
        LightningRowContract::new(4, 4)
            .unwrap_err()
            .to_string()
            .contains("num_drafts")
    );
}

#[test]
fn markov_sampling_is_depth_serial_and_anchor_is_unbiased() {
    let (logits, w1, w2) = synthetic();
    let sampled = contract().markov_sample(&logits, &w1, &w2, 0).unwrap();
    assert_eq!(sampled, vec![1, 1, 2, 1]);

    let same_previous_bug = vec![1, 1, 0, 0];
    assert_ne!(sampled, same_previous_bug);

    let anchor_logits = vec![
        vec![0.0, 1.0, 0.5],
        vec![0.0, 2.0, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let anchor = contract()
        .markov_sample(&anchor_logits, &w1, &w2, 1)
        .unwrap()[0];
    assert_eq!(anchor, 1, "row 0 must use raw logits, never Markov bias");
}

#[test]
fn reorder_keeps_bonus_out_of_verify_drafts() {
    let proposal = contract().reorder_and_split(&[9, 1, 2, 3]).unwrap();
    assert_eq!(proposal.drafts, vec![1, 2, 3]);
    assert_eq!(proposal.bonus, 9);
    assert_eq!(
        contract().verify_input(7, &proposal).unwrap(),
        vec![7, 1, 2, 3]
    );
}

#[test]
fn accepted_prefix_zero_through_three_projects_exact_tokens_and_length() {
    let proposal = DsparkProposal {
        drafts: vec![1, 2, 3],
        bonus: 99,
    };
    let target_rows = [10, 11, 12, 13];
    let expected = [
        (vec![10], 101),
        (vec![1, 11], 102),
        (vec![1, 2, 12], 103),
        (vec![1, 2, 3, 13], 104),
    ];
    for (accepted, (tokens, len)) in expected.into_iter().enumerate() {
        let projection = contract()
            .project_commit(100, accepted, &proposal, &target_rows)
            .unwrap();
        assert_eq!(projection.committed_tokens, tokens);
        assert_eq!(projection.new_seq_len, len);
        assert!(!projection.committed_tokens.contains(&proposal.bonus));
    }
}

#[test]
fn malformed_rows_matrices_tokens_and_prefix_are_rejected() {
    let c = contract();
    assert!(c.reorder_and_split(&[1, 2, 3]).is_err());
    assert!(
        c.verify_input(
            0,
            &DsparkProposal {
                drafts: vec![1, 2],
                bonus: 3,
            },
        )
        .is_err()
    );
    assert!(
        c.project_commit(
            0,
            4,
            &DsparkProposal {
                drafts: vec![1, 2, 3],
                bonus: 4,
            },
            &[5, 6, 7, 8],
        )
        .is_err()
    );

    let (mut logits, w1, w2) = synthetic();
    logits[2].pop();
    assert!(c.markov_sample(&logits, &w1, &w2, 0).is_err());

    let (mut logits, w1, w2) = synthetic();
    logits[1][1] = f32::NAN;
    assert!(c.markov_sample(&logits, &w1, &w2, 0).is_err());

    let (logits, w1, w2) = synthetic();
    assert!(c.markov_sample(&logits, &w1, &w2, 3).is_err());
}

#[test]
fn greedy_ties_choose_the_lowest_token_id() {
    let logits = vec![vec![1.0, 1.0]; 4];
    let w1 = vec![vec![0.0], vec![0.0]];
    let w2 = w1.clone();
    assert_eq!(
        contract().markov_sample(&logits, &w1, &w2, 0).unwrap(),
        vec![0, 0, 0, 0]
    );
}

#[test]
fn sequence_length_overflow_is_rejected() {
    let proposal = DsparkProposal {
        drafts: vec![1, 2, 3],
        bonus: 4,
    };
    assert!(
        contract()
            .project_commit(usize::MAX, 0, &proposal, &[5, 6, 7, 8])
            .is_err()
    );
}
