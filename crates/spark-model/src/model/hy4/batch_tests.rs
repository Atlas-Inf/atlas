// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn verify_checkpoint_discards_rejected_cache_rows_and_replays_only_the_prefix() {
    for accepted in [1, 2, 4] {
        let m = model();
        let stream = m.default_stream();
        let mut seq = m.alloc_sequence().unwrap();
        m.prefill(&[1, 2], &mut seq, stream).unwrap();
        m.checkpoint_ssm_states(&mut seq).unwrap();
        m.decode_verify(&[3, 4, 5, 6], &mut seq, stream).unwrap();
        seq.tokens.truncate(2 + accepted);
        seq.seq_len = seq.tokens.len();
        m.commit_accepted_prefix(&mut seq, accepted, 4).unwrap();
        {
            let requests = m.borrow_dependent().requests.lock();
            let data = requests[&seq.slot_idx].borrow_dependent();
            assert_eq!(data.history, seq.tokens);
            assert_eq!(data.layers[0].attn.committed_len(), seq.seq_len);
        }
        m.checkpoint_ssm_states(&mut seq).unwrap();
        m.decode_verify(&[7, 8], &mut seq, stream).unwrap();
        seq.seq_len -= 1;
        seq.tokens.pop();
        m.commit_accepted_prefix(&mut seq, 1, 2).unwrap();
        assert_eq!(seq.tokens.last(), Some(&7));
    }
}

#[test]
fn invalid_verify_commit_requires_checkpoint_and_an_exact_accepted_prefix() {
    let m = model();
    let stream = m.default_stream();
    let mut seq = m.alloc_sequence().unwrap();
    assert!(m.commit_accepted_prefix(&mut seq, 1, 1).is_err());
    m.checkpoint_ssm_states(&mut seq).unwrap();
    m.decode_verify(&[1, 2], &mut seq, stream).unwrap();
    for (accepted, rows) in [(0, 2), (3, 2), (1, 0), (1, 3), (1, 2)] {
        assert!(m.commit_accepted_prefix(&mut seq, accepted, rows).is_err());
        assert_eq!(seq.tokens, [1, 2]);
    }
    seq.tokens.pop();
    seq.seq_len -= 1;
    m.commit_accepted_prefix(&mut seq, 1, 2).unwrap();
    assert_eq!(
        m.borrow_dependent().requests.lock()[&seq.slot_idx]
            .borrow_dependent()
            .history,
        [1]
    );
}

#[test]
fn multi_row_prefill_and_verify_preserve_history_and_expose_all_logits() {
    let m = model();
    let stream = m.default_stream();
    let mut seq = m.alloc_sequence().unwrap();
    assert!(
        m.prefill_chunk(&[1, 2, 3, 4], &mut seq, 0, 2, false, stream)
            .unwrap()
            .is_null()
    );
    assert_eq!(seq.seq_len, 2);
    assert_eq!(
        m.prefill_chunk(&[1, 2, 3, 4], &mut seq, 2, 2, true, stream)
            .unwrap(),
        m.logits_buffer_ptr()
    );
    let ids = m.decode_verify(&[5, 6], &mut seq, stream).unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(m.logits_buffer_ptr(), m.borrow_dependent().multi.logits);
    assert_eq!(seq.tokens, [1, 2, 3, 4, 5, 6]);
    let requests = m.borrow_dependent().requests.lock();
    let data = requests[&seq.slot_idx].borrow_dependent();
    assert_eq!(data.history, seq.tokens);
    assert_eq!(data.layers[0].attn.committed_len(), 6);
    drop(requests);
    m.decode(7, &mut seq, stream).unwrap();
    assert_eq!(m.logits_buffer_ptr(), m.borrow_dependent().logits);
}

#[test]
fn verify_rejects_empty_invalid_or_over_capacity_before_state_advances() {
    let m = model();
    let stream = m.default_stream();
    let mut seq = m.alloc_sequence().unwrap();
    for tokens in [vec![], vec![1, 64], vec![1; 9]] {
        assert!(m.decode_verify(&tokens, &mut seq, stream).is_err());
        assert_eq!(seq.seq_len, 0);
        assert!(
            m.borrow_dependent().requests.lock()[&seq.slot_idx]
                .borrow_dependent()
                .history
                .is_empty()
        );
    }
    m.prefill(&[1, 2], &mut seq, stream).unwrap();
    seq.seq_len = 1;
    assert!(m.decode_verify(&[3, 4], &mut seq, stream).is_err());
    assert_eq!(
        m.borrow_dependent().requests.lock()[&seq.slot_idx]
            .borrow_dependent()
            .history,
        [1, 2]
    );
}

#[test]
fn logits_are_skipped_on_nonfinal_chunks_and_single_sequence_capacity_is_enforced() {
    let m = model();
    let stream = m.default_stream();
    let mut seq = m.alloc_sequence().unwrap();
    let first = m
        .prefill_chunk(&[1, 2, 3], &mut seq, 0, 2, false, stream)
        .unwrap();
    assert!(first.is_null());
    assert_eq!(
        m.borrow_dependent()
            .head_rows
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    let last = m
        .prefill_chunk(&[1, 2, 3], &mut seq, 2, 1, true, stream)
        .unwrap();
    assert!(!last.is_null());
    assert_eq!(
        m.borrow_dependent()
            .head_rows
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
    assert!(
        m.decode_verify(&[4, 5, 6, 7, 8, 9], &mut seq, stream)
            .is_err()
    );
    assert_eq!(seq.seq_len, 3);
}
