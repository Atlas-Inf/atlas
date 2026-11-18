// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::hy4_reference::paged::{IndexTail, PagedLayout, TailKey};

#[test]
fn pool_softmax_is_over_members_per_channel_not_channels_or_tokens() {
    let values: Vec<_> = (0..4).flat_map(|j| vec![j as f32 + 1.; 128]).collect();
    let zero = vec![0.; 512];
    let norm = vec![2.; 128];
    let pooled = gated_pool(&values, &zero, &zero, &norm, 1e-5).unwrap();
    assert_eq!(pooled, vec![2.; 128]);
    let mut gates = zero.clone();
    for d in (0..128).step_by(2) {
        gates[d] = 1000.;
        gates[3 * 128 + d + 1] = 1000.;
    }
    let varying = gated_pool(&values, &gates, &zero, &norm, 1e-5).unwrap();
    assert!(varying[0] < varying[1]);
    assert!(gated_pool(&[], &zero, &zero, &norm, 1e-5).is_err());
    assert!(gated_pool(&values, &zero, &zero, &norm, 0.).is_err());
}
#[test]
fn normalized_hadamard_power_of_two_scaling_and_mqa_relu_order() {
    let mut impulse = vec![0.; 128];
    impulse[0] = 1.;
    let (q, scale) = fwht_quantize(&impulse).unwrap();
    assert!(q.iter().all(|b| *b == q[0]));
    assert_eq!(scale.log2().fract(), 0.);
    let (zero, zs) = fwht_quantize(&vec![0.; 128]).unwrap();
    assert_eq!(zero, vec![0; 128]);
    assert!(zs > 0.);
    let query = vec![(vec![56; 128], 1.), (vec![184; 128], 1.)];
    let expected = 128. / (2f32 * 128.).sqrt();
    assert_eq!(
        score(&query, &[1., 1.], &(vec![56; 128], 1.)).unwrap(),
        expected
    );
    assert!(fwht_quantize(&[0.]).is_err());
    assert!(fwht_quantize(&[f32::NAN; 128]).is_err());
    assert!(score(&[], &[], &(vec![56; 128], 1.)).is_err());
}
#[test]
fn select_historical_groups_then_sort_and_append_causal_tail() {
    let ids = select_groups(&[1., 9., 2., 100.], 3, 3).unwrap();
    assert_eq!(ids, [0, 1, 2]);
    let best = select_groups(&[1., 9., 2., 100.], 3, 2).unwrap();
    assert_eq!(best, [1, 2]);
    assert_eq!(
        expand_groups(&best, 13, true).unwrap(),
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, -1]
    );
    assert_eq!(
        expand_groups(&[-1, -1], 0, true).unwrap(),
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    );
    assert_eq!(expand_groups(&best, 13, false).unwrap(), vec![-1; 12]);
    assert_eq!(select_groups(&[], 0, 3).unwrap(), [-1; 3]);
    assert_eq!(select_groups(&[1., 1., 1.], 3, 2).unwrap(), [0, 1]);
    assert!(select_groups(&[], 1, 1).is_err());
    assert!(select_groups(&[], 0, 0).is_err());
    assert!(expand_groups(&[0, 0], 8, true).is_err());
}
fn key(n: usize) -> TailKey {
    TailKey {
        value: vec![n as f32; 128],
        position: [n as i64; 3],
    }
}
#[test]
fn chunked_tail_matches_full_prefix_and_rejected_drafts_do_not_corrupt_carry() {
    let keys: Vec<_> = (0..11).map(key).collect();
    let mut full = IndexTail::default();
    let expected = full.append(0, &keys).unwrap();
    let mut chunks = IndexTail::default();
    let mut actual = chunks.append(0, &keys[..3]).unwrap();
    let mut draft = chunks.clone();
    draft.append(3, &keys[3..]).unwrap();
    assert_eq!(chunks.committed_len(), 3);
    actual.extend(chunks.append(3, &keys[3..6]).unwrap());
    actual.extend(chunks.append(6, &keys[6..]).unwrap());
    assert_eq!(actual, expected);
    assert_eq!(chunks.committed_len(), 11);
    let before = chunks.committed_len();
    assert!(chunks.append(10, &[key(11)]).is_err());
    assert_eq!(chunks.committed_len(), before);
    assert!(
        chunks
            .append(
                11,
                &[TailKey {
                    value: vec![],
                    position: [11; 3]
                }]
            )
            .is_err()
    );
    assert_eq!(chunks.committed_len(), before);
}
#[test]
fn paged_latent_and_pool_addressing_respects_request_tables_and_circular_reuse() {
    assert_eq!(PagedLayout::window_pages(512, 64).unwrap(), 9);
    let ring = PagedLayout {
        page_size: 64,
        physical_pages: 9,
        table: (0..9).collect(),
        circular: true,
    };
    let mut tags = vec![usize::MAX; 9 * 64];
    for p in 0usize..2048 {
        tags[ring.slot(p).unwrap()] = p;
        for j in p.saturating_sub(511)..=p {
            assert_eq!(tags[ring.slot(j).unwrap()], j);
        }
    }
    let mut layout = PagedLayout {
        page_size: 64,
        physical_pages: 4,
        table: vec![2, 0],
        circular: false,
    };
    assert_eq!(layout.slot(0).unwrap(), 128);
    assert_eq!(layout.slot(63).unwrap(), 191);
    assert_eq!(layout.slot(64).unwrap(), 0);
    assert!(layout.slot(128).is_err());
    layout.circular = true;
    assert_eq!(layout.slot(128).unwrap(), 128);
    layout.table[0] = 4;
    assert!(layout.slot(0).is_err());
    layout.page_size = 0;
    assert!(layout.slot(0).is_err());
}
